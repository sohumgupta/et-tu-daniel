import os
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, LSTMCell

class Seq2Seq(tf.keras.Model):
	def __init__(self, embeddings, vocab_size, window_size):
		super(Seq2Seq, self).__init__()

		#hyperparameters
		self.batch_size = 100
		self.embedding_size = 192
		self.hidden_state = 192
		self.window_size = window_size
		self.vocab_size = vocab_size
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08)

		#model layers
		# self.embedding = Embedding(self.vocab_size, self.embedding_size, embeddings_initializer = tf.keras.initializers.Constant(embeddings), name="embedding_layer")
		self.embedding = Embedding(self.vocab_size, self.embedding_size, name="embedding_layer")
		# self.embedding.trainable = False
		self.lstm_layer = LSTM(self.hidden_state, return_sequences=True, return_state=True, name="lstm_layer")
		self.encoder = Bidirectional(self.lstm_layer, merge_mode='sum', input_shape =(self.batch_size, self.embedding_size), name="encoder")
		self.decoder_lstm = LSTM(self.hidden_state, return_sequences=True, return_state=True, name="decoder_lstm")
		self.dense = Dense(self.vocab_size, activation='softmax', name="dense")
	
	def call(self, encoder_input, decoder_input):
		encoder_embeddings = self.embedding(encoder_input)
		whole_seq_output_enc, final_memory_state_enc_left, final_carry_state_enc_left, final_memory_state_enc_right, final_carry_state_enc_right = self.encoder(inputs=encoder_embeddings, initial_state=None)
		final_memory_state_enc = final_memory_state_enc_left + final_memory_state_enc_right
		final_carry_state_enc = final_carry_state_enc_left + final_carry_state_enc_right
		decoder_embeddings = self.embedding(decoder_input)
		whole_seq_output, final_memory_state, final_carry_state = self.decoder_lstm(inputs=decoder_embeddings, initial_state=(final_memory_state_enc, final_carry_state_enc))
		probs = self.dense(whole_seq_output)

		return probs
		
	def loss_function(self, prbs, labels, mask):
		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		return tf.reduce_sum(loss * mask)

	def bleu_score(self, references, candidates):
		return corpus_bleu(references, candidates)

