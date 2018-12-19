# -*-coding:utf-8 -*-
import os
from collections import namedtuple
from batcher import Batcher
from data import Vocab
from generator import Generator
from discriminator import Discriminator
from decode import BeamSearchDecoder
import trainer as trainer
import util
import tensorflow as tf


# GPU config
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

tf.logging.set_verbosity(tf.logging.INFO)

# Data dirs
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles.\
                           Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('log_root', 'log', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('pretrain_dis_data_path', 'Dis_train_data.npz','path for the pretrain dis')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of pretrain/train/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, \
                            i.e. take the current checkpoint, and use it to produce one summary for each example in \
                            the  dataset, write the summaries to file and then get ROUGE scores for the whole dataset. \
                             If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, \
                             use it to produce summaries for randomly-chosen examples and log the results to screen, \
                             indefinitely.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('dis_batch_size', 256, 'batch size for pretrain discriminator')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 30,
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
tf.app.flags.DEFINE_integer('rollout', 24, 'Size of rollout number')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.app.flags.DEFINE_boolean('seqgan', True, 'If False disable seqgan')
tf.app.flags.DEFINE_boolean('pretrain_discriminator', True, 'If False disable seqgan')

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
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
FLAGS = tf.app.flags.FLAGS


def prepare_hps():
    hparam_list = ['mode', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps',
                   'coverage', 'cov_loss_wt', 'pointer_gen', 'seqgan', 'rollout','lr',
                   'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',]
    hps_dict = {}
    for key, val in FLAGS.__flags.items():
        if key in hparam_list:
            hps_dict[key] = val
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    return hps


def restore_best_model():
    """Load best-model file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring best-model for training...")

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


def build_seqgan_graph(hps, vocab):
    print('Build generator graph...')
    with tf.device('/gpu:0'):
        generator = Generator(hps, vocab)
        generator.build_graph()

    print('Build discriminator graph...')
    with tf.device('/gpu:0'):
        # TODO: Settings in args
        dis_filter_sizes = [2, 3, 4, 5]
        dis_num_filters = [100, 100, 100, 100]
        dis_l2_reg_lambda = 0.2
        discriminator = Discriminator(sequence_length=hps.max_dec_steps,
                                      num_classes=2,
                                      vocab_size=FLAGS.vocab_size,
                                      embedding_size=hps.emb_dim,
                                      filter_sizes=dis_filter_sizes,
                                      num_filters=dis_num_filters,
                                      pretrained_path=False,
                                      l2_reg_lambda=dis_l2_reg_lambda)
    return generator, discriminator


def setup_training(mode, generator, discriminator, generator_batcher, discriminator_batcher):
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if FLAGS.restore_best_model:
        restore_best_model()

    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    supervisor = tf.train.Supervisor(logdir=train_dir,
                                     is_chief=True,
                                     saver=saver,
                                     summary_op=None,
                                     save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                                     save_model_secs=60,  # checkpoint every 60 secs
                                     global_step=generator.global_step)
    summary_writer = supervisor.summary_writer
    sess_context_manager = supervisor.prepare_or_wait_for_session(config=util.get_config())

    try:
        if mode == "pretrain":
            trainer.pretrain_generator(generator, generator_batcher, summary_writer, sess_context_manager)
        elif mode == "train":
            if FLAGS.pretrain_discriminator:
                trainer.pretrain_discriminator(discriminator, sess_context_manager)
            trainer.adversarial_train(generator, discriminator, generator_batcher, discriminator_batcher,
                                      summary_writer, sess_context_manager)
        else:
            raise ValueError("Caught invalid value of mode!")
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        supervisor.stop()


def main(args):
    # prints a message if you've entered flags incorrectly
    if len(args) != 1:
        raise Exception("Problem with flags: %s" % args)

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode != 'decode':
        raise Exception("The single_pass flag should only be True in decode mode")
    hps = prepare_hps()
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    generator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
    discriminator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    if hps.mode == "pretrain" or hps.mode == "train":
        generator, discriminator = build_seqgan_graph(hps, vocab)
        setup_training(hps.mode, generator, discriminator, generator_batcher, discriminator_batcher)
    elif hps.mode == 'decode':
        # The model is configured with max_dec_steps=1 because we only ever run one step of
        # the decoder at a time (to do beam search).
        decode_model_hps = hps
        decode_model_hps.max_dec_steps = 1
        generator = Generator(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(generator, generator_batcher, vocab)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of pretrain/train/decode")


if __name__ == '__main__':
    tf.app.run()
