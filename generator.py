# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import control_flow_ops, tensor_array_ops
from tqdm import tqdm
from attention_decoder import attention_decoder
from rollout import roll_attention_decoder

FLAGS = tf.app.flags.FLAGS


class Generator(object):
    """A class to represent a sequence-to-sequence model for text summarization.
       Supports both baseline mode, pointer-generator mode, and coverage
    """
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None],
                                                          name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        if FLAGS.seqgan:
            self.D_reward = tf.placeholder(shape=[hps.batch_size, FLAGS.max_dec_steps], dtype=tf.float32)
            self.rouge_reward = tf.placeholder(shape=[hps.batch_size, 1], dtype=tf.float32)

        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps],
                                                name='dec_padding_mask')
        if hps.mode == "decode" and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        if FLAGS.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        if FLAGS.seqgan:
            feed_dict[self.D_reward] = batch.batch_reward
            feed_dict[self.rouge_reward] = batch.batch_rouge_reward
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=seq_len,
                                                                                swap_memory=True)
            # concatenate the forwards and backwards states
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def _add_decoder(self, inputs):
        """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

        Args:
          inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

        Returns:
          outputs: List of tensors; the outputs of the decoder
          out_state: The final state of the decoder
          attn_dists: A list of tensors; the attention distributions
          p_gens: A list of scalar tensors; the generation probabilities
        """
        hps = self._hps
        self.dec_cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True,
                                                initializer=self.rand_unif_init)
        if hps.mode == "decode" and hps.coverage:
            prev_coverage = self.prev_coverage
        else:
            prev_coverage = None
        outputs, out_state, attn_dists, \
        p_gens, state_list, coverage = attention_decoder(inputs,
                                                         self._dec_in_state,
                                                         self._enc_states,
                                                         self._enc_padding_mask,
                                                         self.dec_cell,
                                                         initial_state_attention=(hps.mode == "decode"),
                                                         pointer_gen=hps.pointer_gen,
                                                         use_coverage=hps.coverage,
                                                         prev_coverage=prev_coverage)
        return outputs, out_state, attn_dists, p_gens, state_list, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists, p_gens, multi_batch_num=1):
        """Calculate the final distribution, for the pointer-generator model

        Args:
          vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
          attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

        Returns:
          final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution'):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
            extended_vsize = self._vocab.size() + self._max_art_oovs
            extra_zeros = tf.zeros((self._hps.batch_size*multi_batch_num, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                                    vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

            batch_nums = tf.range(0, limit=self._hps.batch_size*multi_batch_num)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            if multi_batch_num == 1:
                _enc_batch_extend_vocab = self._enc_batch_extend_vocab
            else:
                _enc_batch_extend_vocab = tf.concat([self._enc_batch_extend_vocab]*multi_batch_num, axis=0)

            attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
            indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
            shape = [self._hps.batch_size*multi_batch_num, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                                    attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

            final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                           zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
        https://www.tensorflow.org/get_started/embedding_viz
        Make the vocab metadata file, then make the projector config file pointing to it."""
        train_dir = os.path.join(FLAGS.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        # self._vocab.write_metadata(vocab_metadata_path)  # write metadata file
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize = self._vocab.size()  # size of the vocabulary

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim],
                                                 dtype=tf.float32, initializer=self.trunc_norm_init)
                if hps.mode == "train" or hps.mode == "pretrain":
                    self._add_emb_vis(self.embedding)  # add to tensorboard

                emb_enc_inputs = tf.nn.embedding_lookup(self.embedding, self._enc_batch)
                emb_dec_inputs = [tf.nn.embedding_lookup(self.embedding, x)
                                  for x in tf.unstack(self._dec_batch, axis=1)]

            # Add the encoder.
            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self._enc_states = enc_outputs

            # Reduce the final encoder hidden state to the right size to be the initial decoder hidden state
            self._dec_in_state = self._reduce_states(fw_st, bw_st)

            # Add the decoder.
            with tf.variable_scope('decoder'):
                self.decoder_outputs, self._dec_out_state, \
                self.attn_dists, self.p_gens, self.state_list, self.coverage = self._add_decoder(emb_dec_inputs)

            # Add the output projection to obtain the vocabulary distribution
            with tf.variable_scope('output_projection'):
                self.w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
                self.v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                # vocab_scores is the vocabulary distribution before applying softmax.
                # Each entry on the list corresponds to one decoder step
                vocab_scores = []
                for i, output in enumerate(self.decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, self.w, self.v))  # apply the linear layer

                # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                # The words are in the order they appear in the vocabulary file.
                vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

            # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
            if FLAGS.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists, p_gens=self.p_gens)
            else:  # final distribution is just vocabulary distribution
                final_dists = vocab_dists

            if hps.mode in ['pretrain', 'train', 'eval']:
                # Calculate the loss
                self.output_token = []
                self.output_sample_token = []
                with tf.variable_scope('loss'):
                    # Calculate the loss per step
                    # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                    loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                    sample_loss_per_step = []
                    batch_nums = tf.range(0, limit=hps.batch_size)  # shape (batch_size)
                    for dec_step, dist in enumerate(final_dists):
                        targets = self._target_batch[:, dec_step]  # The indices of the target words. shape (batch_size)
                        indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                        gold_probs = tf.gather_nd(dist,
                                                  indices)  # shape (batch_size). prob of correct words on this step
                        _, max_prob_ind = tf.nn.top_k(dist, 1)
                        self.output_token.append(max_prob_ind)  # shape [[ batch_size, 1]* max_dec_steps]

                        sample_probs, sample_prob_ind = self._get_token_prob(dist, mode='sample')
                        self.output_sample_token.append(sample_prob_ind)

                        losses = -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.))
                        loss_per_step.append(losses)
                        sample_loss = tf.log(tf.clip_by_value(sample_probs, 1e-10, 1.))
                        sample_loss_per_step.append(sample_loss)

                    # Apply dec_padding_mask and get loss
                    self._ML_loss = self._mask_and_avg(loss_per_step, self._dec_padding_mask)
                    sample_loss_with_reward = tf.expand_dims(self.rouge_reward, dim=1) * tf.stack(sample_loss_per_step, axis=1)
                    self._RL_loss = self._mask_and_avg(tf.unstack(sample_loss_with_reward, axis=1), self._dec_padding_mask)

                    if hps.seqgan:
                        _roll_mask = [0.] * hps.max_dec_steps
                        for i in range(1, hps.max_dec_steps, 10):
                            _roll_mask[i] = 1.
                        roll_mask = tf.constant([_roll_mask] * hps.batch_size)
                        loss_with_reward = self.D_reward*tf.stack(loss_per_step, axis=1)*roll_mask
                        self._GAN_loss = 1. *self._mask_and_avg(tf.unstack(loss_with_reward, axis=1), self._dec_padding_mask)

                    tf.summary.scalar('loss', self._ML_loss)
                    tf.summary.scalar('sample_loss', self._RL_loss)
                    tf.summary.scalar('rouge_reward', tf.reduce_mean(self.rouge_reward))
                    if hps.seqgan:
                        tf.summary.scalar('D_reward_loss', self._GAN_loss)
                        tf.summary.scalar('D_reward', tf.reduce_mean(self.D_reward))

                    self._total_loss = 0.01*(self._ML_loss + self._GAN_loss) + 0.99*self._RL_loss
                    tf.summary.scalar('total_loss', self._total_loss)
        if hps.mode == "decode":
            # We run decode beam search mode one decoder step at a time
            assert len(final_dists) == 1
            final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
            self._topk_log_probs = tf.log(topk_probs)

    def _mask_and_avg(self, values, padding_mask):
        """Applies mask to values then returns overall average (a scalar)

        Args:
          values: a list length max_dec_steps containing arrays shape (batch_size).
          padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

        Returns:
          a scalar
        """
        dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
        values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
        values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
        return tf.reduce_mean(values_per_ex)  # overall average

    def _get_token_prob(self, dist, num_samples=1, mode='argmax'):
        if mode == 'argmax':
            _, max_prob_ind = tf.nn.top_k(dist, 1)
            return _, max_prob_ind
        elif mode == 'sample':
            targets = tf.cast(tf.multinomial(tf.log(dist), num_samples=num_samples), tf.int32)
            batch_nums = tf.range(0, limit=self._hps.batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            indices = tf.stack((batch_nums, targets), axis=2)
            sample_prob = tf.gather_nd(dist, indices)
            return sample_prob, targets

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)

        # Apply optimizer
        optimizer = tf.train.AdamOptimizer(self._hps.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=self.global_step, name='train_step')

    def _rollout(self):
        hps = self._hps
        with tf.variable_scope("seq2seq/decoder", reuse=True) as scope:
            expanded_oov_embedding = self.embedding[0] * tf.ones([self._max_art_oovs, 1])
            embedding_extend = tf.concat([self.embedding, expanded_oov_embedding], axis=0)

            def reward_recurrence(time_step, output_token, given_number, out_state):
                input_token = tf.unstack(tf.squeeze(output_token.read(time_step+given_number-1)), axis=0)
                dec_inut = [tf.nn.embedding_lookup(embedding_extend, ids=input_single_token)
                            for input_single_token in input_token]
                enc_state = self._enc_states
                enc_pad = self._enc_padding_mask

                outputs, out_state, attn_dists, p_gens \
                    = roll_attention_decoder(decoder_inputs=dec_inut,
                                             initial_state=out_state,
                                             encoder_states=enc_state,
                                             enc_padding_mask=enc_pad,
                                             cell=self.dec_cell,
                                             initial_state_attention=True,
                                             pointer_gen=hps.pointer_gen)
                # vocab_scores is the vocabulary distribution before applying softmax.
                # Each entry on the list corresponds to one decoder step.
                vocab_scores = []
                for i, output in enumerate(outputs):
                    vocab_scores.append(tf.nn.xw_plus_b(output, self.w, self.v))  # apply the linear layer
                # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                # The words are in the order they appear in the vocabulary file.
                vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
                final_dists = self._calc_final_dist(vocab_dists, attn_dists, p_gens=p_gens)
                sample_token = [tf.cast(tf.multinomial(tf.log(final_dist), num_samples=1), dtype=tf.int32)
                                for final_dist in final_dists]
                output_token = output_token.write(time_step+given_number, sample_token)

                return time_step+1, output_token, given_number, out_state

            def run_once():
                rollout_token = []
                # modifying
                for given_number in tqdm(range(1, hps.max_dec_steps, 10)):
                    out_state = [self.state_list[given_number]]*(int(hps.rollout/2))

                    output_token = tensor_array_ops.TensorArray(dtype=tf.int32, size=FLAGS.max_dec_steps,
                                                                dynamic_size=False, infer_shape=True,
                                                                clear_after_read=False)
                    for i in range(given_number):
                        output_token = output_token.write(i, [self.output_token[i]]*(int(hps.rollout/2)))

                    time_step, output_token, given_number, out_state = control_flow_ops.while_loop(
                            cond=lambda time_step, _1, _2, _3: time_step < hps.max_dec_steps - given_number,
                            body=reward_recurrence,
                            loop_vars=(np.int32(0),
                                       output_token,
                                       given_number,
                                       out_state)
                    )
                    output_token_array = []
                    for n in range(FLAGS.max_dec_steps):
                        output_token_array.append(output_token.read(n))
                    rollout_token.append(tf.concat(output_token_array, axis=-1))
                return tf.stack(rollout_token)

            rollout_output_token_all = []
            rollout_nums = 2
            for _ in range(rollout_nums):
                rollout_output_token_all.append(run_once())
            self.rollout_output_token_all = tf.transpose(tf.concat(rollout_output_token_all, axis=1),
                                                         perm=[1, 0, 2, 3])

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train'or self._hps.mode == 'pretrain':
            self._add_train_op()
            self._rollout()
        self._summaries = tf.summary.merge_all()
        tf.logging.info('Time to build graph: %i seconds', time.time() - t0)

    def run_train_step(self, sess, batch):
        """Runs one training iteration.
           Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._ML_loss,
            'global_step': self.global_step,
            'output_summary_token': self.output_token,
            'output_sample_token': self.output_sample_token
        }
        return sess.run(to_return, feed_dict)

    def run_summary_token_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'output_summary_token': self.output_token
        }
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration.
           Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._ML_loss,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
          sess: Tensorflow session.
          batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
          enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
          dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True)  # feed the batch into the placeholders
        (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state,
                                                            self.global_step], feed_dict)  # run the encoder

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch
        # so we just take the top row.
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        return enc_states, dec_in_state

    def run_rollout_step(self, sess, batch):
        """Runs one training iteration.
           Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'rollout_token': self.rollout_output_token_all
        }
        return sess.run(to_return, feed_dict)

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
        """For beam search decoding. Run the decoder for one step.
        Args:
          sess: Tensorflow session.
          batch: Batch object containing single example repeated across the batch
          latest_tokens: Tokens to be fed as input into the decoder for this timestep
          enc_states: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }

        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(beam_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        if FLAGS.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage