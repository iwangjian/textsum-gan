# -*-coding:utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import os
import time

import numpy as np
from tqdm import tqdm
from collections import namedtuple
from sklearn.metrics import accuracy_score
import util
from batcher import Batcher
from data import Vocab
from generator import SummarizationModel
from discriminator import Discriminator
from data_loader import Dataloader
from rouge import Rouge

tf.logging.set_verbosity(tf.logging.INFO)

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
PRE_EPOCH_NUM = 2  # pretrain G

#########################################################################################
#  Generator Parameters
#########################################################################################
FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '',
                           'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, \
                            i.e. take the current checkpoint, and use it to produce one summary for each example in \
                            the  dataset, write the summaries to file and then get ROUGE scores for the whole dataset. \
                             If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, \
                             use it to produce summaries for randomly-chosen examples and log the results to screen, \
                             indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

tf.app.flags.DEFINE_string('pretrain_dis_data_path', 'Dis_train_data.npz','path for the pretrain dis')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('dis_batch_size', 256, 'batch size for pretrain discriminator')

tf.app.flags.DEFINE_integer('max_enc_steps', 550, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 60, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 3,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. \
                            If the vocabulary file contains fewer words than this number, or if this number is \
                            set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.app.flags.DEFINE_boolean('seqgan', True, 'If False disable seqgan')
tf.app.flags.DEFINE_boolean('pretrain_discriminator', True, 'If False disable seqgan')

tf.app.flags.DEFINE_integer('rollout', 8, 'Size of rollout number')
tf.app.flags.DEFINE_integer('basegpu', 0, 'base gpu index')


# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False,
                            'Use coverage mechanism. Note, the experiments reported in the ACL paper train \
                            WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. \
                             i.e. to reproduce the results in the ACL paper, turn this off for most of training then \
                              turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0,
                          'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False,
                            'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. \
                            Your current training model will be copied to a new version \
                             (same name with _cov_init appended) that will be ready to run with coverage flag turned on, \
                              for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be \
                             used for further training. Useful for early stopping, or if your training checkpoint \
                              has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")


# GPU config, if set log_device_placement=True, all the variables will be copied to GPU, otherwise CPU
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def restore_best_model():
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.global_variables_initializer())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.global_variables()
                            if "Adagrad" not in v.name and 'discriminator' not in v.name and 'beta' not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "train")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to {}...".format(new_fname))
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()


def build_seqgan_graph(hps, vocab):
    with tf.device('/gpu:0'):
        generator = SummarizationModel(hps, vocab)


    #########################################################################################
    #  Discriminator  Hyper-parameters
    #########################################################################################

    dis_filter_sizes = [2, 3, 4, 5]
    dis_num_filters = [100, 100, 100, 100]
    dis_l2_reg_lambda = 0.2

    # the vocab of dis = gen?
    print('Build Discriminator Graph...')
    with tf.device('/gpu:0'):
        discriminator = Discriminator(sequence_length=hps.max_dec_steps,
                                      num_classes=2,
                                      vocab_size=FLAGS.vocab_size,
                                      embedding_size=hps.emb_dim,
                                      filter_sizes=dis_filter_sizes,
                                      num_filters=dis_num_filters,
                                      pretrained_path=False,
                                      l2_reg_lambda=dis_l2_reg_lambda)
    return generator, discriminator


def setup_training(generator, discriminator, generator_batcher, discriminator_batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if FLAGS.restore_best_model:
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=5)  # keep 3 checkpoints at a time

    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                             save_model_secs=60,  # checkpoint every 60 secs
                             global_step=generator.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info("Created session.")
    try:
        if FLAGS.pretrain_discriminator:
            pre_train_discriminator(discriminator, sess_context_manager)

        run_training(generator, discriminator, generator_batcher, discriminator_batcher, summary_writer, sess_context_manager)
        #model, batcher, sess_context_manager, sv, summary_writer)  # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def pre_train_discriminator(discriminator, sess_context_manager):
    #####################  Pretrain Discriminator ##################################
    dis_train_data_loader = Dis_dataloader(FLAGS.dis_batch_size, FLAGS.vocab_size)
    dis_test_data_loader = Dis_dataloader(FLAGS.dis_batch_size, FLAGS.vocab_size)

    print("Pre-train Discriminator")

    pretrain_dis_data = np.load(FLAGS.pretrain_dis_data_path)
    pos_summary, neg_summary = pretrain_dis_data['pos_summary_idx'], pretrain_dis_data['neg_summary_idx']
    positive_train_summary = []
    negative_train_summary = []
    positive_eval_summary = []
    negative_eval_summary = []

    ##############################################################################
    #############      Prepare Train and Eval data  ##############################

    for i in range(len(pos_summary)):
        if i < 143800:
            positive_train_summary.append(pos_summary[i][:FLAGS.max_dec_steps])
            negative_train_summary.append(neg_summary[i][:FLAGS.max_dec_steps])
        else:
            positive_eval_summary.append(pos_summary[i][:FLAGS.max_dec_steps])
            negative_eval_summary.append(neg_summary[i][:FLAGS.max_dec_steps])

    ##############################################################################
    #############      Training       ############################################
    train_max_epoch = 15
    sess = sess_context_manager
    for epoch in tqdm(range(train_max_epoch)):
        dis_train_data_loader.load_data(positive_train_summary, negative_train_summary)
        dis_train_data_loader.reset_pointer()
        for it in range(dis_train_data_loader.num_batch):
            x_batch, y_batch = dis_train_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: 0.5
            }
            sess.run(discriminator.train_op, feed)

        dis_test_data_loader.load_data(positive_eval_summary, negative_eval_summary)
        dis_test_data_loader.reset_pointer()
        acc_list = []

        for it in range(dis_test_data_loader.num_batch):
            x_batch, y_batch = dis_test_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: 1.0
            }
            pred = sess.run(discriminator.predictions, feed)
            target = np.where(np.array(y_batch) == 1)[-1] #np.concatenate(y_batch)
            acc_list.append(accuracy_score(y_pred=pred, y_true=target))
        eval_acc = np.mean(acc_list)

        print('Pretrain epoch:{}, Eval accuracy: {}'.format(epoch, eval_acc))
        #if eval_acc >= 0.8:
        #    break


def run_training(generator, discriminator, generator_batcher, discriminator_batcher, summary_writer, sess_context_manager):
    print('#########################################################################')
    print('Start Adversarial Training...')

    with sess_context_manager as sess:
        D_rewards = np.zeros((FLAGS.batch_size, FLAGS.max_dec_steps))
        rouge_rewards = np.zeros((FLAGS.batch_size, 1))

        while True:
            # Train the generator for one step
            for it in range(1):
                batch = generator_batcher.next_batch()
                batch.batch_reward = D_rewards
                batch.batch_rouge_reward = rouge_rewards

                tf.logging.info('running training step...')
                t0 = time.time()
                result_train = generator.run_train_step(sess, batch)

                t1 = time.time()
                tf.logging.info('seconds for training step: %.3f', t1 - t0)
                loss = result_train['loss']
                tf.logging.info('Generator train loss: %f', loss)  # print the loss to screen

                summaries = result_train['summaries']
                train_step = result_train['global_step']
                summary_writer.add_summary(summaries, train_step)  # write the summaries

                rg = Rouge()

                gtruth_token = batch.target_batch
                output_sample_token = np.transpose(np.squeeze(result_train['output_sample_token']))
                output_argmax_token = np.transpose(np.squeeze(result_train['output_summary_token']))

                def remove_eos(input_text):

                    _input_text_eos = np.where(input_text == 3)[0]
                    if len(_input_text_eos) != 0:
                        cliped_text = input_text[:_input_text_eos[0]]
                    else:
                        cliped_text = input_text
                    return ' '.join(map(str, cliped_text))

                rouge_rewards = []

                for gt, sample, argmax in zip(gtruth_token, output_sample_token, output_argmax_token):
                    _gt = remove_eos(gt)
                    _sample = remove_eos(sample)
                    _argmax = remove_eos(argmax)

                    r_baseline = rg.get_scores(_gt, _argmax)[0]['rouge-l']['f']
                    r_sample = rg.get_scores(_gt, _sample)[0]['rouge-l']['f']
                    rouge_rewards.append(r_baseline - r_sample)

                rouge_rewards = np.reshape(rouge_rewards, [FLAGS.batch_size, 1])
                tf.logging.info('RL reward for rouge-L: %.3f', np.mean(rouge_rewards))

                tf.logging.info('running rollout step...')
                t0 = time.time()
                result_rollout = generator.run_rollout_step(sess, batch)
                t1 = time.time()
                tf.logging.info('seconds for rollout step: %.3f', t1 - t0)

                rollout_output = result_rollout['rollout_token']  # shape [rollout_num, seqlen(this is number of roll), batch_size, seq_len]
                given_number_of_rollout = rollout_output.shape[1]

                # calculate D_reward
                print("start to calculate D_rewards")
                _feed_output_token = np.reshape(rollout_output, [-1, FLAGS.max_dec_steps])

                feed_output_token = []
                for sent in _feed_output_token:
                    index_list = np.where(sent == 3)[0]
                    if len(index_list) != 0:
                        ind = index_list[0]
                        new_sent = np.concatenate([sent[:ind + 1], np.ones(100 - ind - 1)])
                        feed_output_token.append(new_sent)
                    else:
                        new_sent = np.array(sent, dtype=np.int32)
                        feed_output_token.append(new_sent)

                feed_output_token = np.array(feed_output_token)
                feed_output_token = feed_output_token.reshape((len(feed_output_token), -1))
                print("feed_out_token.shape:", feed_output_token.shape)
                '''
                clip_index = np.where(feed_output_token > FLAGS.vocab_size-1)
                index_x = clip_index[0]
                index_y = clip_index[1]
                for i in range(len(index_x)):
                    feed_output_token[index_x[i]][index_y[i]] = 0
                '''
                if feed_output_token.shape[1] > 1:
                    for i in range(len(feed_output_token)):
                        clip_index = np.where(np.array(feed_output_token[i]) > FLAGS.vocab_size - 1)
                        for idx in clip_index:
                            feed_output_token[i][idx] = 0

                    # update
                    ypred_for_auc = []
                    for feed_output_token_small in np.split(feed_output_token, FLAGS.rollout):
                        feed = {discriminator.input_x: feed_output_token_small,
                                discriminator.dropout_keep_prob: 1.0
                                }
                        # ypred_for_auc: [rollout_num * seqlen(this is number of roll) * batch_size, 2]
                        ypred_for_auc.append(sess.run(discriminator.ypred_for_auc, feed))
                    ypred_for_auc = np.concatenate(ypred_for_auc)
                    ypred = np.array([item[1] for item in ypred_for_auc])
                    framed_yred = np.reshape(ypred, [FLAGS.rollout, given_number_of_rollout, FLAGS.batch_size])
                    rewards = np.transpose(np.sum(framed_yred, 0)) / (
                                1.0 * FLAGS.rollout)  # [batch_size, output_max_len// 20]
                    if np.std(rewards) != 0.:
                        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                    D_rewards = np.zeros((FLAGS.batch_size, FLAGS.max_dec_steps))
                    print("rewards.shape:", rewards.shape)

                    for count, i in enumerate(
                            range(1, FLAGS.max_dec_steps, int(FLAGS.max_dec_steps / rewards.shape[1]))):
                        D_rewards[:, i] = rewards[:, count]

                else:
                    tmp = []
                    for i in range(len(feed_output_token)):
                        tmp.append(feed_output_token[i][0])
                    feed_output_token = np.array(tmp).copy()
                    print("feed-new:", feed_output_token.shape)
                    print("Filter out!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            # Train the discriminator
            print("Start to train the Discriminator!")
            for _ in tqdm(range(5)):
                batch = discriminator_batcher.next_batch()
                res = generator.run_summary_token_step(sess, batch)
                _output_argmax_summary = res['output_summary_token']
                _output_argmax_summary = np.transpose(np.squeeze(_output_argmax_summary))  # [batch_size, max_dec_steps]
                gtruth_data = batch.target_batch  # [batch_size, max_dec_steps]; format: [[], [], ...]

                output_argmax_summary = []
                for sent in _output_argmax_summary:
                    index_list = np.where(sent == 3)[0]
                    if len(index_list) != 0:
                        ind = index_list[0]
                        new_sent = np.concatenate([sent[:ind + 1], np.ones(FLAGS.max_dec_steps - ind - 1)])
                        output_argmax_summary.append(new_sent)
                    else:
                        output_argmax_summary.append(sent)
                output_argmax_summary = np.array(output_argmax_summary)

                positive_examples = []
                negative_examples = []
                for ele in gtruth_data:
                    positive_examples.append(ele)
                for ele in output_argmax_summary:
                    negative_examples.append(ele)
                dis_data_loader = Dis_dataloader(FLAGS.batch_size, FLAGS.vocab_size)

                max_epoch = 3

                for epoch in range(max_epoch):
                    dis_data_loader.load_data(positive_examples, negative_examples)
                    dis_data_loader.reset_pointer()
                    for it in range(dis_data_loader.num_batch):
                        x_batch, y_batch = dis_data_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: 0.5
                        }
                        _ = sess.run(discriminator.train_op, feed)


def prepare_hps_vocab():
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                   'pointer_gen', 'seqgan', 'rollout']
    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    return hps, vocab


def main(unused_argv):
    # prints a message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("Problem with flags: %s" % unused_argv)

    hps, vocab = prepare_hps_vocab()

    generator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
    discriminator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    if hps.mode == 'train':
        generator, discriminator = build_seqgan_graph(hps, vocab)
        setup_training(generator, discriminator, generator_batcher, discriminator_batcher)
    elif hps.mode == 'decode':
        # The model is configured with max_dec_steps=1 because we only ever run one step of
        # the decoder at a time (to do beam search).
        decode_model_hps = hps._replace(max_dec_steps=1)
        generator = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(generator, generator_batcher, vocab)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of train/decode")


if __name__ == '__main__':
    tf.app.run()






