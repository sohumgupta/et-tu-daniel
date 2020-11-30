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

def main():
	embeddings()

if __name__ == '__main__':
	main()