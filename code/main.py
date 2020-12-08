import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
import argparse

import os
from os import listdir
from os.path import isfile, join
import io
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocess import get_sentences, pad_sentences, vectorize_sentences, PAD_TOKEN
from embeddings import get_embeddings
from model import Copy
from seq2seq import Seq2Seq

def train(model, train_modern, train_original, padding_index):
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
 
		mask = tf.where(tf.equal(batch_labels, padding_index), tf.zeros(tf.shape(batch_labels)), tf.ones(tf.shape(batch_labels)))
		with tf.GradientTape() as tape:
			probs = model.call(batch_inputs, batch_decoder_inputs)
			loss = model.loss_function(probs, batch_labels, mask)

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
	label_sentences = np.empty((test_modern.shape[0], test_modern.shape[1]-1), dtype='<U50')
	
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
		labels = tf.reshape(batch_labels, [-1])
		labels = labels.numpy()
		label_words = [idx[x] for x in labels]
		sentences_label = np.reshape(label_words, batch_labels.shape)
		pred_sentences[i*model.batch_size:(i+1)*model.batch_size] = sentences_pred
		label_sentences[i*model.batch_size:(i+1)*model.batch_size] = sentences_label

	pred_sentences = pred_sentences.tolist()
	label_sentences = label_sentences.tolist()

	return pred_sentences, label_sentences

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
	parser.add_argument('architecture', type=str, help='Model architecture to use (SEQ2SEQ or COPY)')
	args = parser.parse_args()
	if (args.architecture not in ['SEQ2SEQ', 'COPY']):
		parser.error("Achitecture must be one of SEQ2SEQ or COPY.")
	return args

def write_sentences(path, sentences):
	f = open(path, "a")
	f.truncate(0)
	for sentence in sentences:
		sentence_str = " ".join(sentence)
		f.write(sentence_str + "\n")
	f.close()

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

	if architecture == 'COPY':
		model = Copy(embeddings, len(vocab), sentence_length + 2)
	elif architecture == 'SEQ2SEQ':
		model = Seq2Seq(embeddings, len(vocab), sentence_length + 2)

	# train model
	NUM_EPOCHS = 5
	for e in range(NUM_EPOCHS):
		print(f"\nTraining Epoch {e+1}/{NUM_EPOCHS}...")
		train(model, modern_train_idx, original_train_idx, padding_index)

	print(f"\nTesting...")
	predictions, labels = test(model, modern_test_idx, original_test_idx, idx, padding_index)
	print(f"\nCalculating BLEU score...\n")

	results_path = '../results'
	pred_path = join(results_path, f'pred_sentences.{args.architecture.lower()}.e{NUM_EPOCHS}.txt')
	labels_path = join(results_path, f'label_sentences.{args.architecture.lower()}.e{NUM_EPOCHS}.txt')
	write_sentences(pred_path, predictions)
	write_sentences(labels_path, labels)

	subprocess.call(f"./multi-bleu.perl {labels_path} < {pred_path}", shell=True)

if __name__ == '__main__':
	main()