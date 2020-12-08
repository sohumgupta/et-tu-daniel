import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dot, Flatten
from tensorflow.keras.optimizers import Adam
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Word2Vec(tf.keras.Model):
	def __init__(self, vocab_size):
		super(Word2Vec, self).__init__()

		# Hyperparameters + learning rate
		self.learning_rate =  1e-3
		self.optimizer = Adam(learning_rate=self.learning_rate)
		self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

		# Model architecture
		self.embedding_size = 192
		self.target_embeddings = Embedding(vocab_size, self.embedding_size, name="target_embeddings")
		self.context_embeddings = Embedding(vocab_size, self.embedding_size, name="context_embeddings")
		self.dot = Dot(axes=(3, 2))
		self.flatten = Flatten()

	def call(self, pair):
		target, context = pair
		target_embedding = self.target_embeddings(target)
		context_embedding = self.context_embeddings(context)
		dotted = self.dot([context_embedding, target_embedding])
		return self.flatten(dotted)