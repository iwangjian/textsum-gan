# -*- coding:utf-8 -*-
import numpy as np
import os
import argparse


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
    print("positive samples:", len(train_pos))

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
    print("negative samples:", len(train_neg))
    np.savez(res_file, pos_summary_idx=np.array(train_pos), neg_summary_idx=np.array(train_neg))
    print("file saved: ", res_file)


def main(args):
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    train_pos_path = args.decode_dir + "/reference"
    train_neg_path = args.decode_dir + "/decoded"
    res_file = os.path.join(args.data_dir, "discriminator_train_data.npz")
    gen_dis_sample(train_pos_path, train_neg_path, args.vocab_path, res_file=res_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="program to generate both positive and negative samples")
    parser.add_argument('--data_dir', required=True, help="directory of data")
    parser.add_argument('--decode_dir', required=True, help="root of the decoded directory")
    parser.add_argument('--vocab_path', required=True, help="path of the vocabulary file")
    args = parser.parse_args()
    main(args)
