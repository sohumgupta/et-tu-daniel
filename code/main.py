import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

import os
from os import listdir
from os.path import isfile, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io

from embeddings import train_embeddings, read_embeddings, write_embeddings, find_closest_words
from model_preprocess import preprocess
from model import Model

def embeddings_call():
	embeddings_path = '../embeddings'
	
	# Train embeddings if necessary
	if 'embeddings.tsv' not in listdir(embeddings_path) or 'names.tsv' not in listdir(embeddings_path):
		print(f"Training word embeddings...")
		embeddings, word2int, int2word = train_embeddings()
		write_embeddings(embeddings, word2int, embeddings_path)
	else:
		print(f"Word embeddings found.")

	# Read embeddings from files
	embeddings, names = read_embeddings(embeddings_path)
	word2int = {word:i for (i, word) in enumerate(names)}

	# Test embeddings by finding closest words
	# find_closest_words(embeddings, word2int, names, "joy")
	# find_closest_words(embeddings, word2int, names, "happy")
	# find_closest_words(embeddings, word2int, names, "father")
	# find_closest_words(embeddings, word2int, names, "son")
	# find_closest_words(embeddings, word2int, names, "thou")
	# find_closest_words(embeddings, word2int, names, "you")
	find_closest_words(embeddings, word2int, names, "kill")
	find_closest_words(embeddings, word2int, names, "marry")
	find_closest_words(embeddings, word2int, names, "stand")
	find_closest_words(embeddings, word2int, names, "sit")
	return embeddings

def train(model, train_modern, train_original, padding_index):

	size = len(train_modern)
	indices = np.arange(size)
	shuffled_indices = tf.random.shuffle(indices)
	train_modern = tf.gather(train_modern, shuffled_indices, None, axis=0, batch_dims=0)
	train_original = tf.gather(train_original, shuffled_indices, None, axis=0, batch_dims=0)
	for i in range(size//model.batch_size):
		batch_inputs = tf.gather(train_modern, indices[i*model.batch_size:(i+1)*model.batch_size], None, axis=0, batch_dims=0)
		batch_labels = tf.gather(train_original, indices[i*model.batch_size:(i+1)*model.batch_size], None, axis=0, batch_dims=0)
		batch_decoder_inputs = batch_labels[:, :-1]
		batch_labels = batch_labels[:, 1:]
		mask = tf.where(tf.equal(batch_labels, padding_index), tf.zeros(tf.shape(batch_labels)), tf.ones(tf.shape(batch_labels)))
		with tf.GradientTape() as tape:
			probs = model.call_with_pointer(batch_inputs, batch_decoder_inputs)
			loss = model.loss_function(probs, batch_labels, mask)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	pass

def test(model, test_modern, test_original, vocab, padding_index):

	size = len(test_modern)
	indices = np.arange(size)
	shuffled_indices = tf.random.shuffle(indices)
	test_modern = tf.gather(test_modern, shuffled_indices, None, axis=0, batch_dims=0)
	test_original = tf.gather(test_original, shuffled_indices, None, axis=0, batch_dims=0)
	pred_sentences = np.zeros((test_modern.shape))
	for i in range(size//model.batch_size):
		batch_inputs = tf.gather(test_modern, indices[i*model.batch_size:(i+1)*model.batch_size], None, axis=0, batch_dims=0)
		batch_labels = tf.gather(test_original, indices[i*model.batch_size:(i+1)*model.batch_size], None, axis=0, batch_dims=0)
		batch_decoder_inputs = batch_labels[:, :-1]
		batch_labels = batch_labels[:, 1:]
		probs = model.call(batch_inputs, batch_decoder_inputs)
		probs = tf.reshape(tf.argmax(probs, axis=2), [-1])
		probs = tf.make_ndarray(probs)
		#need to get sentences to calculate bleu score
		pred_sentences = np.vstack((pred_sentences, sentences))
		mask = tf.where(tf.equal(batch_labels, padding_index), tf.zeros(tf.shape(batch_labels)), tf.ones(tf.shape(batch_labels)))
		loss = model.loss_function(probs, batch_labels, mask)

def main():
	embeddings = embeddings_call()
	modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx = preprocess("../data/train_modern.txt", "../data/train_original.txt", "../data/test_modern.txt", "../data/test_original.txt")
	model = Model(embeddings, len(vocab))
	train(model, modern_train_idx, original_train_idx, padding_index)
	test(model, modern_test_idx, original_test_idx, vocab, padding_index)
	# TODO:
	# 1) Check format of embeddings matrix
	# 2) Get padding index 
	# 3) Create embedding layer in model 
	# 4) Create sentences to calculate BLEU score in test 

if __name__ == '__main__':
	main()