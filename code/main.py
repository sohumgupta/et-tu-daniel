import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

import os
from os import listdir
from os.path import isfile, join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io

from embeddings import train_embeddings, read_embeddings, write_embeddings, find_closest_words

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

def main():
	embeddings()

if __name__ == '__main__':
	main()