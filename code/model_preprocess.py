PAD_TOKEN = "*pad*"
STOP_TOKEN = "*stop*"
START_TOKEN = "*start*"
UNK_TOKEN = "*unk*"
WINDOW_SIZE = 14

from preprocess import pad_sentences, get_words, construct_vocab

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

def preprocess(modern, original, modern_test, original_test):
	# train
	modern_train_sentences, modern_train_length = get_sentences(modern_train)
	original_train_sentences, original_train_length = get_sentences(original_train)

	max_train_length = max(modern_train_length, original_train_length)

	modern_train_sentences = pad_sentences(modern_train_sentences, max_train_length)
	original_train_sentences = pad_sentences(original_train_sentences, max_train_length)

	# test
	modern_test_sentences, modern_test_length = get_sentences(modern_test)
	original_test_sentences, original_test_length = get_sentences(original_test)

	max_test_length = max(modern_test_length, original_test_length)

	modern_test_sentences = pad_sentences(modern_test_sentences, max_test_length)
	original_test_sentences = pad_sentences(original_test_sentences, max_test_length)

	modern_vocab, modern_idx = construct_vocab(modern_train_sentences)
	original_vocab, original_idx = construct_vocab(original_train_sentences)


# get_data("../data/test_modern.txt", "../data/test_original.txt")

