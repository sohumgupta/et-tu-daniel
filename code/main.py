import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
import argparse

import os
from os import listdir
from os.path import isfile, join
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io

from preprocess import get_sentences, pad_sentences, vectorize_sentences, PAD_TOKEN
from embeddings import get_embeddings
from model import Model
from seq2seq import Seq2Seq

def train(model, train_modern, train_original, padding_index, idx):
	size = train_modern.shape[0]
	indices = np.arange(size)
	shuffled_indices = tf.random.shuffle(indices)
	train_modern = train_modern[shuffled_indices]
	train_original = train_original[shuffled_indices]

	num_batches = size // model.batch_size
	
	for i in range(num_batches):
		batch_inputs = train_modern[i*model.batch_size:(i+1)*model.batch_size]
		batch_labels = train_original[i*model.batch_size:(i+1)*model.batch_size]
		
		batch_inputs = batch_inputs[:, 1:]
		batch_decoder_inputs = batch_labels[:, :-1]
		batch_labels = batch_labels[:, 1:]

		# print(f"batch input: {' '.join([idx[x] for x in batch_inputs[0]])}")
		# print(f"batch decoder inputs: {' '.join([idx[x] for x in batch_decoder_inputs[0]])}")
		# print(f"batch labels: {' '.join([idx[x] for x in batch_labels[0]])}")
 
		mask = tf.where(tf.equal(batch_labels, padding_index), tf.zeros(tf.shape(batch_labels)), tf.ones(tf.shape(batch_labels)))
		with tf.GradientTape() as tape:
			probs = model.call(batch_inputs, batch_decoder_inputs)
			loss = model.loss_function(probs, batch_labels, mask)

		# pred_argmax = tf.argmax(probs[0], axis=-1)
		# print(f"predicted output: {' '.join([idx[x] for x in pred_argmax.numpy()])}")
		# print()

		if i % 5 == 0:
			print(f"  ↳  Loss for batch {i}/{num_batches}: {loss}")
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	pass

def test(model, test_modern, test_original, idx, padding_index):
	size = test_modern.shape[0]
	indices = np.arange(size)
	shuffled_indices = tf.random.shuffle(indices)
	test_modern = test_modern[shuffled_indices]
	test_original = test_original[shuffled_indices]

	pred_sentences = np.empty((test_modern.shape[0], test_modern.shape[1]-1), dtype='<U50')
	input_sentences = np.empty((test_modern.shape[0], test_modern.shape[1]-1), dtype='<U50')
	
	num_batches = size // model.batch_size

	for i in range(num_batches):
		print(f"  ↳  Testing batch {i + 1}/{num_batches}")
		batch_inputs = test_modern[i*model.batch_size:(i+1)*model.batch_size]
		batch_labels = test_original[i*model.batch_size:(i+1)*model.batch_size]

		batch_inputs = batch_inputs[:, :-1]
		batch_decoder_inputs = batch_labels[:, :-1]
		batch_labels = batch_labels[:, 1:]

		probs = model.call(batch_inputs, batch_decoder_inputs)
		probs = tf.reshape(tf.argmax(probs, axis=2), [-1])
		probs = probs.numpy()

		words_pred = np.array([idx[x] for x in probs], dtype='<U50')
		sentences_pred = np.reshape(words_pred, batch_decoder_inputs.shape)
		inputs = tf.reshape(batch_inputs, [-1])
		inputs = inputs.numpy()
		input_words = [idx[x] for x in inputs]
		sentences_input = np.reshape(input_words, batch_inputs.shape)
		pred_sentences[i*model.batch_size:(i+1)*model.batch_size] = sentences_pred
		input_sentences[i*model.batch_size:(i+1)*model.batch_size] = sentences_input

	pred_sentences = pred_sentences.tolist()
	input_sentences = input_sentences.tolist()

	f = open("input_sentences.txt", "a")
	f.truncate(0)
	for sentence in input_sentences:
		sentence_str = " ".join(sentence)
		f.write(sentence_str + "\n")
	f.close()

	f = open("pred_sentences.txt", "a")
	f.truncate(0)
	for sentence in pred_sentences:
		sentence_str = " ".join(sentence)
		f.write(sentence_str + "\n")
	f.close()

	# bleu_score = model.bleu_score(input_sentences, input_sentences)
	# return bleu_score

def preprocess_model(modern_train, original_train, modern_test, original_test, modern_valid, original_valid, window_size=None):
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
	if window_size != None:
		max_length = window_size

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

	return modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, vocab[PAD_TOKEN], embeddings, max_length

def parse_args():
	parser = argparse.ArgumentParser(description='Shakespearizing Modern English!')
	parser.add_argument('architecture', type=str, help='Model architecture to use (SEQ2SEQ or POINTER)')
	args = parser.parse_args()
	if (args.architecture not in ['SEQ2SEQ', 'POINTER']):
		parser.error("Achitecture must be one of SEQ2SEQ or POINTER.")
	return args

def main():
	args = parse_args()
	architecture = args.architecture

	window_size = 15
	modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, padding_index, embeddings, sentence_length = preprocess_model(
		"../data/train_modern.txt", "../data/train_original.txt", 
		"../data/test_modern.txt", "../data/test_original.txt", 
		"../data/valid_modern.txt", "../data/valid_original.txt", 
		window_size)

	modern_train_idx, original_train_idx = np.array(modern_train_idx), np.array(original_train_idx)
	original_test_idx, modern_test_idx = np.array(original_test_idx), np.array(modern_test_idx)

	if architecture == 'POINTER':
		model = Model(embeddings, len(vocab), sentence_length + 2)
	elif architecture == 'SEQ2SEQ':
		model = Seq2Seq(embeddings, len(vocab), sentence_length + 2)

	# train model
	NUM_EPOCHS = 1
	for e in range(NUM_EPOCHS):
		print(f"\nTraining Epoch {e+1}/{NUM_EPOCHS}...")
		# train(model, modern_train_idx, original_train_idx, padding_index, idx)
		train(model, modern_train_idx, modern_train_idx, padding_index, idx)

	print(f"\nTesting...")
	# bleu_score = test(model, modern_test_idx, original_test_idx, idx, padding_index)
	bleu_score = test(model, modern_test_idx, modern_test_idx, idx, padding_index)
	# print(f"\nBLEU SCORE: {bleu_score}")

if __name__ == '__main__':
	main()