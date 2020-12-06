import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

import os
from os import listdir
from os.path import isfile, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io

from preprocess import get_sentences, pad_sentences, vectorize_sentences
from embeddings import get_embeddings
from model import Model

def train(model, train_modern, train_original, padding_index):

	size = len(train_modern)
	indices = np.arange(size)
	shuffled_indices = tf.random.shuffle(indices)
	train_modern = tf.gather(train_modern, shuffled_indices, None, axis=0, batch_dims=0)
	train_original = tf.gather(train_original, shuffled_indices, None, axis=0, batch_dims=0)
	print(f"batch size: {size//model.batch_size}")
	for i in range(5):
		print(i)
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
	print(f"batch size: {size//model.batch_size}")
	for i in range(5):
		print(i)
		batch_inputs = tf.gather(test_modern, indices[i*model.batch_size:(i+1)*model.batch_size], None, axis=0, batch_dims=0)
		batch_labels = tf.gather(test_original, indices[i*model.batch_size:(i+1)*model.batch_size], None, axis=0, batch_dims=0)
		batch_decoder_inputs = batch_labels[:, :-1]
		batch_labels = batch_labels[:, 1:]
		probs = model.call_with_pointer(batch_inputs, batch_decoder_inputs)
		probs = tf.reshape(tf.argmax(probs, axis=2), [-1])
		probs = tf.make_ndarray(probs)
		#need to get sentences to calculate bleu score
		# pred_sentences = np.vstack((pred_sentences, sentences))
		mask = tf.where(tf.equal(batch_labels, padding_index), tf.zeros(tf.shape(batch_labels)), tf.ones(tf.shape(batch_labels)))
		loss = model.loss_function(probs, batch_labels, mask)

def preprocess_model(modern_train, original_train, modern_test, original_test, modern_valid, original_valid):
	"""
	Input: respective file paths
	Output: modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, embeddings, sentence_length
	"""
	# train
	modern_train_sentences, modern_train_length = get_sentences([modern_train, modern_valid])
	original_train_sentences, original_train_length = get_sentences([original_train, original_valid])
	max_train_length = max(modern_train_length, original_train_length)

	# test
	modern_test_sentences, modern_test_length = get_sentences([modern_test])
	original_test_sentences, original_test_length = get_sentences([original_test])
	max_test_length = max(modern_test_length, original_test_length)

	max_length = max(max_train_length, max_test_length)

	# padding sentences
	modern_train_sentences, original_train_sentences = pad_sentences(modern_train_sentences, max_length), pad_sentences(original_train_sentences, max_length)
	modern_test_sentences, original_test_sentences = pad_sentences(modern_test_sentences, max_length), pad_sentences(original_test_sentences, max_length)

	# get embeddings
	embeddings, vocab, idx = get_embeddings()

	# constructing IDs
	modern_train_idx = vectorize_sentences(vocab, modern_train_sentences)
	modern_test_idx = vectorize_sentences(vocab, modern_test_sentences)

	original_train_idx = vectorize_sentences(vocab, original_train_sentences)
	original_test_idx = vectorize_sentences(vocab, original_test_sentences)

	return modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, vocab["*pad*"], embeddings, max_length

def main():
	modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, padding_index, embeddings, sentence_length = preprocess_model("../data/train_modern.txt", "../data/train_original.txt", "../data/test_modern.txt", "../data/test_original.txt", "../data/valid_modern.txt", "../data/valid_original.txt")
	model = Model(embeddings, len(vocab), sentence_length + 2)
	train(model, modern_train_idx, original_train_idx, padding_index)
	test(model, modern_test_idx, original_test_idx, vocab, padding_index)
	# TODO:
	# 1) Check format of embeddings matrix
	# 2) Get padding index 
	# 3) Create embedding layer in model 
	# 4) Create sentences to calculate BLEU score in test 

if __name__ == '__main__':
	main()