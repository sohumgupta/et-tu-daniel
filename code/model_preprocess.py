PAD_TOKEN = "*pad*"
STOP_TOKEN = "*stop*"
START_TOKEN = "*start*"
UNK_TOKEN = "*unk*"
WINDOW_SIZE = 14

from preprocess import get_words, construct_vocab
import numpy as np

def pad_sentences(modern, original, sentence_length):
	MODERN_padded_sentences = []
	MODERN_sentence_lengths = []
	for line in modern:
		padded_MODERN = line[:sentence_length]
		padded_MODERN += [STOP_TOKEN] + [PAD_TOKEN] * (sentence_length - len(padded_MODERN)-1)
		MODERN_padded_sentences.append(padded_MODERN)

	ORIGINAL_padded_sentences = []
	ORIGINAL_sentence_lengths = []
	for line in original:
		padded_ORIGINAL = line[:sentence_length]
		padded_ORIGINAL = [START_TOKEN] + padded_ORIGINAL + [STOP_TOKEN] + [PAD_TOKEN] * (sentence_length - len(padded_ORIGINAL))
		ORIGINAL_padded_sentences.append(padded_ORIGINAL)

	return MODERN_padded_sentences, ORIGINAL_padded_sentences

def convert_to_id(vocab, sentences):
	return np.stack([[vocab[word] for word in sentence] for sentence in sentences])

# slightly modified from 'preprocess.py'
def get_sentences(file_path):
	sentences = []
	max_length = 0
	file = open(file_path)
	for line in file:
		sentence = get_words(line)
		max_length = max(max_length, len(sentence))
		sentences.append(sentence)
	return sentences, max_length

def preprocess(modern_train, original_train, modern_test, original_test, valid_modern, valid_original):
	"""
	Input: respective file paths
	Output: modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx
	"""
	# train
	modern_train_sentences, modern_train_length = get_sentences(modern_train)
	original_train_sentences, original_train_length = get_sentences(original_train)
	max_train_length = max(modern_train_length, original_train_length)

	# test
	modern_test_sentences, modern_test_length = get_sentences(modern_test)
	original_test_sentences, original_test_length = get_sentences(original_test)
	max_test_length = max(modern_test_length, original_test_length)
	
	
	# valid
	valid_modern_sentences, valid_modern_length = get_sentences(valid_modern)
	valid_original_sentences, valid_original_length = get_sentences(valid_original)
	max_valid_length = max(valid_modern_length, valid_original_length)

	max_length = max(max_train_length, max(max_test_length, max_valid_length))

	# padding sentences
	modern_train_sentences, original_train_sentences = pad_sentences(modern_train_sentences, original_train_sentences, max_length)
	modern_test_sentences, original_test_sentences = pad_sentences(modern_test_sentences, original_test_sentences, max_length)
	valid_modern_sentences, valid_original_sentences = pad_sentences(valid_modern_sentences, valid_original_sentences, max_length)

	# constructing vocab
	vocab, idx = construct_vocab(original_train_sentences + valid_modern_sentences + original_test_sentences + modern_train_sentences + modern_test_sentences + valid_original_sentences)

	# constructing IDs
	modern_train_idx = convert_to_id(vocab, modern_train_sentences)
	modern_test_idx = convert_to_id(vocab, modern_test_sentences)

	original_train_idx = convert_to_id(vocab, original_train_sentences)
	original_test_idx = convert_to_id(vocab, original_test_sentences)

	return modern_train_idx, modern_test_idx, original_train_idx, original_test_idx, vocab, idx, vocab[PAD_TOKEN]
