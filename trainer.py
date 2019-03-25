# -*-coding:utf-8 -*-
import os
import time
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_loader import Dataloader
from rouge import Rouge
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def pretrain_generator(generator, generator_batcher, summary_writer, session):
    """Pretrain generator before adversarial training"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    while True:
        batch = generator_batcher.next_batch()
        t0 = time.time()
        result_train = generator.run_train_step(session, batch)
        global_step = result_train["global_step"]
        loss = result_train["loss"]
        summaries = result_train['summaries']
        train_step = result_train['global_step']
        summary_writer.add_summary(summaries, train_step)
        print("global step: %d train loss: %.3f time: %.3f s" % (global_step, loss, time.time() - t0))
        if loss < 1e-2:
            break


def pretrain_discriminator(discriminator, sess):
    print("Pretrain discriminator...")
    dis_train_loader = Dataloader(FLAGS.dis_batch_size, FLAGS.vocab_size)
    dis_val_loader = Dataloader(FLAGS.dis_batch_size, FLAGS.vocab_size)
    pretrain_dis_data = np.load(FLAGS.pretrain_dis_data_path)
    pos_summary = pretrain_dis_data['pos_summary_idx']
    neg_summary = pretrain_dis_data['neg_summary_idx']
    assert len(pos_summary) == len(neg_summary)

    train_max_epoch = 20  # max training epochs
    val_num = 1000  # number of validation samples
    pos_train = []
    neg_train = []
    pos_val = []
    neg_val = []
    val_select = random.sample(list(range(0, len(pos_summary))), val_num)
    for i in range(len(pos_summary)):
        if i in val_select:
            pos_val.append(pos_summary[i][:FLAGS.max_dec_steps])
            neg_val.append(neg_summary[i][:FLAGS.max_dec_steps])
        else:
            pos_train.append(pos_summary[i][:FLAGS.max_dec_steps])
            neg_train.append(neg_summary[i][:FLAGS.max_dec_steps])
    print("length train:", len(pos_train))
    print("length val:", len(pos_val))

    for epoch in tqdm(range(train_max_epoch)):
        # training process
        dis_train_loader.load_data(pos_train, neg_train)
        dis_train_loader.reset_pointer()
        for it in range(dis_train_loader.num_batch):
            x_batch, y_batch = dis_train_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: 0.5
            }
            sess.run(discriminator.train_op, feed)

        # validation process
        dis_val_loader.load_data(pos_val, neg_val)
        dis_val_loader.reset_pointer()
        acc_list = []
        for it in range(dis_val_loader.num_batch):
            x_batch, y_batch = dis_val_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: 1.0
            }
            pred = sess.run(discriminator.predictions, feed)
            target = np.where(np.array(y_batch) == 1)[-1] # np.concatenate(y_batch)
            acc_list.append(accuracy_score(y_pred=pred, y_true=target))
        eval_acc = np.mean(acc_list)
        print('pretrain epoch:{}, eval accuracy: {}'.format(epoch, eval_acc))


def adversarial_train(generator, discriminator, generator_batcher, discriminator_batcher,
                      summary_writer, sess_context_manager):
    print("Start adversarial training...")
    with sess_context_manager as sess:
        D_rewards = np.zeros((FLAGS.batch_size, FLAGS.max_dec_steps))
        rouge_rewards = np.zeros((FLAGS.batch_size, 1))

        while True:
            # Train generator for one step
            print("Start to train generator...")
            batch = generator_batcher.next_batch()
            batch.batch_reward = D_rewards
            batch.batch_rouge_reward = rouge_rewards
            t0 = time.time()
            result_train = generator.run_train_step(sess, batch)
            loss = result_train['loss']
            summaries = result_train['summaries']
            train_step = result_train['global_step']
            summary_writer.add_summary(summaries, train_step)
            print("train step: %d train loss: %.3f time: %.3fs" % (train_step, loss, time.time() - t0))

            rouge_rewards = []
            target_token = batch.target_batch
            output_sample_token = np.transpose(np.squeeze(result_train['output_sample_token']))
            output_argmax_token = np.transpose(np.squeeze(result_train['output_summary_token']))
            rouge = Rouge()
            for target, sample, argmax in zip(target_token, output_sample_token, output_argmax_token):
                target_ = remove_eos(target)
                sample_ = remove_eos(sample)
                argmax_ = remove_eos(argmax)
                if len(argmax_) > 0:
                    r_baseline = rouge.get_scores(argmax_, target_)[0]["rouge-l"]["f"]
                else:
                    r_baseline = 0
                if len(sample_) > 0:
                    r_sample = rouge.get_scores(sample_, target_)[0]["rouge-l"]["f"]
                else:
                    r_sample = 0
                #print("r_baseline:", r_baseline)
                #print("r_sample:", r_sample)
                rouge_rewards.append(r_baseline - r_sample)
            rouge_rewards = np.reshape(rouge_rewards, [FLAGS.batch_size, 1])
            print("RL reward for rouge-L: %.3f" % np.mean(rouge_rewards))

            print("running rollout step...")
            t0 = time.time()
            result_rollout = generator.run_rollout_step(sess, batch)
            rollout_output = result_rollout['rollout_token']  # shape [rollout_num, seqlen(this is number of roll), batch_size, seq_len]
            print("rollout step: %.3fs" % (time.time() - t0))

            # calculate D_reward
            print("start to calculate D_rewards")
            feed_output_token = []
            rollout_output = np.reshape(rollout_output, [-1, FLAGS.max_dec_steps])
            for sent in rollout_output:
                index_list = np.where(sent == 3)[0]
                if len(index_list) != 0:
                    ind = index_list[0]
                    new_sent = np.concatenate([sent[:ind + 1], np.ones(FLAGS.max_dec_steps - ind - 1)])
                    feed_output_token.append(new_sent)
                else:
                    feed_output_token.append(sent)
            feed_output_token = np.array(feed_output_token)
            feed_output_token[feed_output_token > FLAGS.vocab_size-1] = 0

            # update
            ypred_for_auc = []
            for token in np.split(feed_output_token, FLAGS.rollout):
                feed = {discriminator.input_x: token,
                        discriminator.dropout_keep_prob: 1.0}
                ypred_auc = sess.run(discriminator.ypred_for_auc, feed)  # shape: [rollout_num * seqlen(this is number of roll) * batch_size, 2]
                ypred_for_auc.append(ypred_auc)
            ypred_for_auc = np.concatenate(ypred_for_auc)
            ypred = np.array([item[1] for item in ypred_for_auc])
            ypred = np.reshape(ypred, [FLAGS.rollout, -1, FLAGS.batch_size])
            rewards = np.transpose(np.sum(ypred, 0)) / (1.0 * FLAGS.rollout)  # [batch_size, output_max_len// 20]

            if np.std(rewards) != 0.:
                rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            D_rewards = np.zeros([FLAGS.batch_size, FLAGS.max_dec_steps])
            for count, i in enumerate(range(1, FLAGS.max_dec_steps, 10)):
                D_rewards[:, i] = rewards[:, count]
            print("D_rewards:", D_rewards.shape)

            # Train discriminator
            print("Start to train discriminator...")
            for _ in tqdm(range(5)):
                batch = discriminator_batcher.next_batch()
                result = generator.run_summary_token_step(sess, batch)
                output_summary_token = result['output_summary_token']
                output_summary_token = np.transpose(np.squeeze(output_summary_token))  # [batch_size, max_dec_steps]
                ground_truth = batch.target_batch  # [batch_size, max_dec_steps]
                output_summary = []
                for sent in output_summary_token:
                    index_list = np.where(sent == 3)[0]
                    if len(index_list) != 0:
                        ind = index_list[0]
                        new_sent = np.concatenate([sent[:ind + 1], np.ones(FLAGS.max_dec_steps - ind - 1)])
                        output_summary.append(new_sent)
                    else:
                        output_summary.append(sent)
                output_summary = np.array(output_summary)

                max_epoch = 3
                dis_loader = Dataloader(FLAGS.batch_size, FLAGS.vocab_size)
                pos_train = [ground_truth[i] for i in range(len(ground_truth))]
                neg_train = [output_summary[i] for i in range(len(output_summary))]
                for _ in range(max_epoch):
                    dis_loader.load_data(pos_train, neg_train)
                    dis_loader.reset_pointer()
                    # train for 1 epoch
                    for it in range(dis_loader.num_batch):
                        x_batch, y_batch = dis_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: 0.5
                        }
                        sess.run(discriminator.train_op, feed)


def remove_eos(input_text):
    _input_text_eos = np.where(input_text == 3)[0]
    if len(_input_text_eos) != 0:
        cliped_text = input_text[:_input_text_eos[0]]
    else:
        cliped_text = input_text
    return " ".join(list(map(str, cliped_text)))
