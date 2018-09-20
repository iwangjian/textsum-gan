# -*- coding:utf-8 -*-
import numpy as np
import os


train_pos_path = './data/decode/reference/'
train_neg_path = './data/decode/decoded/'
vocab_path = './data/vocab'

# Generate train samples for GAN training
dis_train_file = './data/discriminator_train_data.npz'


def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r') as vf:
        for idx, line in enumerate(vf):
            w = line.split()[0]
            vocab[w] = idx
    return vocab


def gen_dis_sample(train_pos_path, train_neg_path, vocab_path, n_dim=100, res_file='discriminator_train_data.npz'):
    # load vocabs
    vocab_map = load_vocab(vocab_path)
    print("vocab length:", len(vocab_map))

    # load positive samples
    train_pos = []
    for file in os.listdir(train_pos_path):
        with open(os.path.join(train_pos_path, file), 'r') as rf:
            content = rf.read().strip('.').split()
            text = list(map(lambda x: vocab_map[x] if x in vocab_map else 0, content))
            if len(text) <= n_dim:
                sample = np.pad(text, (0, n_dim-len(text)), mode='constant', constant_values=1).astype(np.int)
            else:
                sample = text[:n_dim]
            train_pos.append(sample)
    print("positives:", len(train_pos))

    # load negative samples
    train_neg = []
    for file in os.listdir(train_neg_path):
        with open(os.path.join(train_neg_path, file), 'r') as rf:
            content = rf.read().strip('.').split()
            text = list(map(lambda x: vocab_map[x] if x in vocab_map else 0, content))
            if len(text) <= n_dim:
                sample = np.pad(text, (0, n_dim-len(text)), mode='constant', constant_values=1).astype(np.int)
            else:
                sample = text[:n_dim]
            train_neg.append(sample)
    print("negatives:", len(train_neg))
    np.savez(res_file, pos_summary_idx=np.array(train_pos), neg_summary_idx=np.array(train_neg))


if __name__ == '__main__':
    gen_dis_sample(train_pos_path, train_neg_path, vocab_path,  res_file=dis_train_file)