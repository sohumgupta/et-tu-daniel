import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

import os
from os import listdir
from os.path import isfile, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io

from embeddings import train_embeddings

def find_closest_words(embeddings, word2int, names, word, NUM_WORDS=10):
	word_ind = word2int[word]
	word_embedding = embeddings[word_ind]
	word_embedding = np.reshape(word_embedding, (1, -1))

	normed_embeddings = tf.math.l2_normalize(embeddings, axis=1)
	normed_word = tf.math.l2_normalize(word_embedding, axis=1)

	similarities = tf.matmul(normed_word, tf.transpose(normed_embeddings, [1, 0]))
	best_words = np.argsort(-similarities[0])[:NUM_WORDS]

	print(f"words most similar to \"{word}\"")
	print("--")
	for (i, word) in enumerate(best_words):
		print(f"{i + 1}. {names[word]}")
	print()

def read_embeddings(embeddings_path):
	embeddings = np.loadtxt(join(embeddings_path, 'embeddings.tsv'), dtype=np.float32, delimiter='\t', converters=None) 
	names = np.loadtxt(join(embeddings_path, 'names.tsv'), dtype=np.unicode_, delimiter='\t', converters=None) 

	return embeddings, names

def write_embeddings(embeddings, word2int, embeddings_path):
	embeddings_file = io.open(join(embeddings_path, 'embeddings.tsv'), 'w', encoding='utf-8')
	names_file = io.open(join(embeddings_path, 'names.tsv'), 'w', encoding='utf-8')

	for word, index in word2int.items():
		embedding = embeddings[index] 
		embeddings_file.write('\t'.join([str(x) for x in embedding]) + "\n")
		names_file.write(word + "\n")
	embeddings_file.close()
	names_file.close()

def embeddings():
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
	find_closest_words(embeddings, word2int, names, "joy")
	find_closest_words(embeddings, word2int, names, "father")
	find_closest_words(embeddings, word2int, names, "thou")

def train(model, train_modern, train_original):

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
		with tf.GradientTape() as tape:
			probs = model.call_with_pointer(batch_inputs, batch_decoder_inputs)
			loss = model.loss_function(probs, batch_labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	pass

def test(model, test_modern, test_original, vocab):

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
		probs = tf.argmax(probs, axis=2)
		probs = tf.make_ndarray(probs)
		sentences = 
		pred_sentences = np.vstack((pred_sentences, sentences))
		loss = model.loss_function(probs, batch_labels)
		acc = model.accuracy_function(probs, batch_labels, mask)
		words = tf.math.count_nonzero(mask, dtype='float32')
		tot_loss += loss
		tot_acc += acc * words
		tot_words += words
	perp = tf.math.exp(tot_loss/tot_words)
	acc = tot_acc/tot_words
	return perp,acc


def main():
	embeddings()

if __name__ == '__main__':
	main()