import numpy as np


class Dataloader():
    def __init__(self, batch_size, vocab_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.vocab_size = vocab_size

    def load_data(self, positive_examples, negative_examples):
        # Load data
        self.sentences = np.array(positive_examples + negative_examples)
        # the elements in sentences which are larger than max_decoded_steps is replaced by 0
        clip_index = np.where(self.sentences > self.vocab_size-1)
        index_x = clip_index[0]; index_y = clip_index[1]
        for i in range(len(index_x)):
            self.sentences[index_x[i]][index_y[i]] = 0

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0