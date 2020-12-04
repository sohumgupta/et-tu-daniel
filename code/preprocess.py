import numpy as np
import tensorflow as tf

from embeddings import get_embeddings

PAD_TOKEN = "*pad*"
STOP_TOKEN = "*stop*"
START_TOKEN = "*start*"

def vectorize_sentences(vocab, sentences):
	return np.stack([[vocab[word] for word in sentence] for sentence in sentences])

def get_words(sentence):
	punctuation = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
	for punc in punctuation:
		sentence = sentence.replace(punc, " ") 
	words = sentence.strip().split()
	words = [w.lower() for w in words]
	return words

def get_sentences(file_paths):
	all_sentences = []
	max_length = 0
	for file_path in file_paths:
		file = open(file_path)
		for line in file:
			sentence = get_words(line)
			max_length = max(max_length, len(sentence))
			all_sentences.append(sentence)
	return all_sentences, max_length

def pad_sentences(sentences, sentence_length, pad_token=PAD_TOKEN, start_token=START_TOKEN, stop_token=STOP_TOKEN):
	for (i, sentence) in enumerate(sentences):
		sentences[i] = [start_token] + sentence + [stop_token] + ([pad_token] * (sentence_length - len(sentence)))
	return sentences

def construct_vocab(sentences):
	all_words = set()
	for sentence in sentences:
		for word in sentence:
			all_words.add(word)

	all_words = list(all_words)
	int2word = {i:word for (i, word) in enumerate(all_words)}
	word2int = {word:i for (i, word) in enumerate(all_words)}
	return word2int, int2word

def preprocess(modern_train, original_train, modern_test, original_test, modern_valid, original_valid):
	"""
	Input: respective file paths
	Output: modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, embeddings
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

	return modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, vocab[PAD_TOKEN], embeddings

preprocess("../data/train_modern.txt", "../data/train_original.txt", "../data/test_modern.txt", "../data/test_original.txt", "../data/valid_modern.txt", "../data/valid_original.txt")