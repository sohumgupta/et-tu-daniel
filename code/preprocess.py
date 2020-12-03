import numpy as np
import tensorflow as tf

def vectorize_sentences(sentences, word2int):
	sentence_vectors = []
	for sentence in sentences:
		vectorized = [word2int[word] for word in sentence]
		sentence_vectors.append(np.array(vectorized))
	return sentence_vectors

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

def pad_sentences(sentences, sentence_length, pad_token=''):
	for (i, sentence) in enumerate(sentences):
		if len(sentence) < sentence_length:
			padding = [pad_token for _ in range(sentence_length - len(sentence))]
			sentences[i] = sentences[i] + padding
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