import os
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

class Model(tf.keras.Model):
    def __init__(self):
        self.batch_size = 32
        self.embedding_size = 192
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08)
        self.embedding = #idk how to define embeddings here
        self.lstm_layer = tf.keras.layers.LSTM(192, return_sequences=True, return_state=True)
        self.encoder = tf.keras.layers.Bidirectional(self.lstm_layer, merge_mode='sum', input_shape =(self.batch_size, self.embedding_size))
        self.decoder_lstm = tf.keras.layers.LSTM(192, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')
        #add query matrix weights initialization
        pass

    def call(self, encoder_input, decoder_input):
        encoder_embeddings = self.embedding(encoder_input)
		whole_seq_output, final_memory_state, final_carry_state = self.encoder(inputs=encoder_embeddings, initial_state=None)
		decoder_embeddings = self.embedding(decoder_input)
		whole_seq_output, final_memory_state, final_carry_state = self.decoder_lstm(inputs=decoder_embeddings, initial_state=(final_memory_state,final_carry_state))
        #can compute attention stuff here
		decoder_probs = self.dense(whole_seq_output)
        g = 0.5
        pointer_probs = decoder_probs
		return tf.math.add(tf.math.multiply(g, decoder_probs), tf.math.multiply((1-g), pointer_probs))
        #need to add attention and pointer stuff

	def cross_entropy_loss_function(self, prbs, labels):
		return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))

    def bleu_score(self, references, candidates):
        return corpus_bleu(references, candidates)

