import os
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, LSTMCell

class Model(tf.keras.Model):
	def __init__(self, embeddings, vocab_size, window_size):
		super(Model, self).__init__()

		#hyperparameters
		self.batch_size = 100
		self.embedding_size = 192
		self.hidden_state = 192
		self.window_size = window_size
		self.vocab_size = vocab_size
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08)

		#call without pointer
		self.embedding = Embedding(self.vocab_size, self.embedding_size, embeddings_initializer = tf.keras.initializers.Constant(embeddings), name="embedding_layer")
		self.embedding.trainable = False
		self.lstm_layer = LSTM(self.hidden_state, return_sequences=True, return_state=True, name="lstm_layer")
		self.encoder = Bidirectional(self.lstm_layer, merge_mode='sum', input_shape =(self.batch_size, self.embedding_size), name="encoder")
		self.decoder_lstm = LSTM(self.hidden_state, return_sequences=True, return_state=True, name="decoder_lstm")
		# need to edit
		self.dense = Dense(self.vocab_size, activation='softmax', name="dense")

		#call with pointer
		self.query_weights = Dense(self.hidden_state, use_bias=False, name="query_weights") #dense layer? no bias?
		self.sentinel_vec = tf.Variable(tf.random.truncated_normal([1, self.hidden_state], stddev=.1), name="sentinel_vec")
		self.a_bias = tf.Variable(tf.random.truncated_normal([self.window_size, self.window_size + 1], stddev=.1), name="a_bias")    

	def build_ptr_prob(self, num_sentences, words_in_sentence, encoder_input, betas):
		#num_sentences, words_in_sentence
		#num_sentences, words_in_sentence
		#num_sentences, 1, vocab_size
		# index_mult = np.arange(num_sentences).reshape((-1, 1))
		# index_mult_repeated = np.repeat(index_mult, words_in_sentence, axis=1)

		# encoder_indices = encoder_input.numpy()
		# indices  = encoder_indices * index_mult_repeated

		# p_t_ptr = np.zeros([num_sentences, 1, self.vocab_size])

		# np.put(p_t_ptr, indices, betas.flatten())

		p_t_ptr_empty = tf.zeros([num_sentences, self.vocab_size])

		reshape_encoder_input = tf.cast(tf.reshape(encoder_input, [num_sentences, words_in_sentence, 1]), dtype=tf.int32)
		per_sentence_index = tf.cast(tf.broadcast_to(tf.reshape(tf.range(0, num_sentences), [num_sentences, 1, 1]), [num_sentences, words_in_sentence, 1]), dtype=tf.int32) 
		indices = tf.reshape(tf.concat([per_sentence_index, reshape_encoder_input], 2), [-1, 2])

		p_t_ptr = tf.tensor_scatter_nd_add(p_t_ptr_empty, indices, tf.reshape(betas, [-1]))

		return p_t_ptr
		
		# for i in range(num_sentences):
		#     for j in range(words_in_sentence):
		#         vocab_index = encoder_input[i, j]
		#         p_t_ptr[i, 0, vocab_index] = betas[i, j] 
		
		# # return tf.convert_to_tensor(p_t_ptr)
		# return p_t_ptr

	def call(self, encoder_input, decoder_input):
		num_sentences = encoder_input.shape[0]

		#equivalent to number of hidden states, could also use self.window_size
		num_words_in_sentence = encoder_input.shape[1]

		#embeddings for input and labels
		encoder_embeddings = self.embedding(encoder_input)
		decoder_embeddings = self.embedding(decoder_input)

		#outputs for the encoder
		#whole_seq_output_enc [num_sentences, num_words_in_sentence, self.hidden_size]
		whole_seq_output_enc, final_memory_state_enc_left, final_carry_state_enc_left, final_memory_state_enc_right, final_carry_state_enc_right = self.encoder(inputs=encoder_embeddings, initial_state=None)
		final_memory_state_enc = final_memory_state_enc_left + final_memory_state_enc_right
		final_carry_state_enc = final_carry_state_enc_left + final_carry_state_enc_right

		#I need to concat each sentinel vector with each sentence group in the f_att vector
		#f_att doesn't change with each decoder iteration, so create it outside the loop
		#f_att is [num_sentences, num_words_in_sentence + 1, self.hidden_state]
		broadcasted_sentinel = tf.broadcast_to(self.sentinel_vec, [num_sentences, 1, self.hidden_state])
		f_att = tf.concat([whole_seq_output_enc, broadcasted_sentinel], 1)

		#these variables eventually become the input for each decoder iteration
		#fmsd = h_dec_(t-1) technically, the last HIDDEN (memory) state, not carry, that we use for all our calcs
		final_memory_state_dec = final_memory_state_enc
		final_carry_state_dec = final_carry_state_enc

		probs = tf.zeros([num_sentences, 0, self.vocab_size])

		#looping for each Decoder iteration
		for i in range(0, num_words_in_sentence - 1):
			#[num_sentences, self.hidden_state]
			queries = self.query_weights(final_memory_state_dec)
			reshaped_queries = tf.reshape(queries, [num_sentences, 1, self.hidden_state]) # [32, 1, 192]
			broadcasted_queries = tf.broadcast_to(reshaped_queries, [num_sentences, num_words_in_sentence + 1, self.hidden_state]) # [32, 102, 192]

			#creation of a, works entire batch simultaneously, should auto-broadcast self.a_bias
			#a is [num_sentences, num_words_in_sentence + 1]
			a_before_sum = tf.math.tanh(f_att * broadcasted_queries) # [32, 102, 192]
			a_before_bias = tf.reduce_sum(a_before_sum, 2)
			# a_before_bias = tf.reshape(tf.reduce_sum(a_before_sum, 2), [num_sentences, 1, -1]) # [32, 102, 192]
			# a_squeezed = tf.squeeze(a_before_bias)
			a = tf.nn.softmax(a_before_bias + self.a_bias[i])

			betas = a[:, 0:num_words_in_sentence]
			gs = tf.reshape(a[:, num_words_in_sentence], [num_sentences, 1])
			#tf.subtract
			one_minus_g = tf.subtract(tf.reshape(tf.ones(num_sentences), [num_sentences, 1]), gs)

			p_t_ptr = self.build_ptr_prob(num_sentences, num_words_in_sentence, encoder_input, betas)

			#c_t is [num_sentences, self.hidden_state]
			betas = tf.expand_dims(betas, 2)
			c_t = tf.reduce_sum(whole_seq_output_enc * betas, 1)
		
			sliced_dec_emb = decoder_embeddings[:, i, :]
			concat_emb = tf.expand_dims(tf.concat([sliced_dec_emb, c_t], 1), 1)

			_, final_memory_state_dec, final_carry_state_dec = self.decoder_lstm(inputs=concat_emb, initial_state=(final_memory_state_dec, final_carry_state_dec))

			h_dec_t = tf.concat([final_memory_state_dec, c_t], 1)

			#p_t_lstm is hopefully [num_sentences, self.vocab_size]
			p_t_lstm = self.dense(h_dec_t)

			gs_mult = tf.broadcast_to(gs, [num_sentences, self.vocab_size])
			one_minus_g_mult = tf.broadcast_to(one_minus_g, [num_sentences, self.vocab_size])
			
			probs_t = gs_mult * p_t_lstm + one_minus_g_mult * p_t_ptr
			probs_t = tf.expand_dims(probs_t, 1)

			probs = tf.concat([probs, probs_t], 1)
		
		return probs
		
	def loss_function(self, prbs, labels, mask):
		return tf.reduce_mean(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask))

	def bleu_score(self, references, candidates):
		return corpus_bleu(references, candidates)

