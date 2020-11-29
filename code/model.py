import os
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

class Model(tf.keras.Model):
    def __init__(self):
        #hyperparameters
        self.batch_size = 32
        self.embedding_size = 192
        self.hidden_state = 32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08)

        #call without pointer
        self.embedding = #idk how to define embeddings here
        self.lstm_layer = tf.keras.layers.LSTM(self.hidden_state, return_sequences=True, return_state=True)
        self.encoder = tf.keras.layers.Bidirectional(self.lstm_layer, merge_mode='sum', input_shape =(self.batch_size, self.embedding_size))
        self.decoder_lstm = tf.keras.layers.LSTM(self.hidden_state, return_sequences=True, return_state=True)
        # need to edit
        self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

        #call with pointer
        self.query_weights = tf.keras.layers.Dense(self.hidden_state, use_bias=False) #dense layer? no bias?
        self.sentinel_vec = tf.variable(tf.random.truncated_normal([self.hidden_state], stddev=.1))

        pass

    def call(self, encoder_input, decoder_input):
        encoder_embeddings = self.embedding(encoder_input)
		whole_seq_output, final_memory_state, final_carry_state = self.encoder(inputs=encoder_embeddings, initial_state=None)
		decoder_embeddings = self.embedding(decoder_input)
		whole_seq_output, final_memory_state, final_carry_state = self.decoder_lstm(inputs=decoder_embeddings, initial_state=(final_memory_state, final_carry_state))
        #can compute attention stuff here
		decoder_probs = self.dense(whole_seq_output)
        g = 0.5
        pointer_probs = decoder_probs
		return tf.math.add(tf.math.multiply(g, decoder_probs), tf.math.multiply((1-g), pointer_probs))
        #need to add attention and pointer stuff

    def call_with_pointer(self, encoder_input, decoder_input):
        num_sentences = encoder_input.shape[0]
        num_words_in_sentence = encoder_input.shape[1]

        #embeddings for input and labels
        encoder_embeddings = self.embedding(encoder_input)
        decoder_embeddings = self.embedding(decoder_input)

        #input and output for encoder and decoder
		whole_seq_output_enc, final_memory_state_enc, final_carry_state_enc = self.encoder(inputs=encoder_embeddings, initial_state=None)
		whole_seq_output_dec, final_memory_state_dec, final_carry_state_dec = self.decoder_lstm(inputs=decoder_embeddings, initial_state=(final_memory_state_enc, final_carry_state_enc))
        
        # query vector, basically put all the decoder hidden states through a dense layer
        #[num_sentences, num_words_in_sentence, self.hidden_state]
        queries = self.query_weights(whole_seq_output_enc)
        broadcasted_queries = tf.broadcast_to(queries, [num_sentences, num_words_in_sentence, num_words_in_sentence + 1, self.hidden_state])

        #broadcasting the sentinel vector, p sure I don't need to do this?
        #creating the f_att vector, but I need to concat all the hidden states with their own sentinel
        #this concat is per sentence group
        twodee_sentinel = tf.variable([[self.sentinel_vec]])
        broadcasted_sentinel = tf.repeat(twodee_sentinel, [num_sentences], axis=0)

        #[num_sentences, num_words_in_sentence + 1, self.hidden_state]
        f_att = tf.concat([whole_seq_output_enc, broadcasted_sentinel], 1)

        a_before_sum = tf.zeros([num_sentences, num_words_in_sentence, num_words_in_sentence + 1, self.hidden_state])

        

        for k in range()
            for i in range(num_words_in_sentence):
                for j in range(self.hidden_state):
                a_before_sum[:, :, :, j] = f_att[:, :, j] * queries[:, :, j] 
                
        
		decoder_probs = self.dense(whole_seq_output)
        g = 0.5
        pointer_probs = decoder_probs
		return tf.math.add(tf.math.multiply(g, decoder_probs), tf.math.multiply((1-g), pointer_probs))
        #need to add attention and pointer stuff

	def cross_entropy_loss_function(self, prbs, labels):
		return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs))

    def bleu_score(self, references, candidates):
        return corpus_bleu(references, candidates)

