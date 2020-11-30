import numpy as np
import tensorflow as tf
from preprocess import construct_vocab, get_sentences, vectorize_sentences, pad_sentences
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from word2vec import Word2Vec
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_training_data(sentences, vocab_size, negative_samples, window_size):
	targets, contexts, labels = [], [], []

	sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

	for sentence in sentences:
		positive_examples, _ = tf.keras.preprocessing.sequence.skipgrams(
			sentence,
			vocabulary_size=vocab_size,
			sampling_table=sampling_table,
			window_size=window_size,
			negative_samples=0
		)

		for target, context in positive_examples:
			context_class = np.reshape(np.array([context]), (-1, 1))
			negative_examples, _, _ = tf.random.log_uniform_candidate_sampler(
				true_classes=context_class, num_true=1, 
				num_sampled=negative_samples, unique=True, range_max=vocab_size, 
			)

			negative_examples = np.reshape(negative_examples, (-1, 1))
			example_contexts = np.concatenate((context_class, negative_examples), axis=0)
			example_labels = np.array([1] + [0] * negative_samples)

			targets.append(target)
			contexts.append(example_contexts)
			labels.append(example_labels)

	return targets, contexts, labels

def train_embeddings():
	data_path = '../data'
	all_paths = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]

	print(f"Preprocessing for embeddings...")
	all_sentences, max_length = get_sentences(all_paths)
	all_sentences = pad_sentences(all_sentences, sentence_length=max_length)
	word2int, int2word = construct_vocab(all_sentences)
	print(f"Vocab Size: {len(word2int)}")
	sentence_vectors = vectorize_sentences(all_sentences, word2int)
	print(f"Number of Sentences: {len(all_sentences)}")

	VOCAB_SIZE = len(word2int)
	SENTENCE_LENGTH = max_length
	WINDOW_SIZE = 2
	NEGATIVE_SAMPLES = 10

	print(f"Generating training data...")
	targets, contexts, labels = get_training_data(
		sentences=sentence_vectors, 
		window_size=WINDOW_SIZE, 
		negative_samples=NEGATIVE_SAMPLES, 
		vocab_size=VOCAB_SIZE)
	print(f"Number of Training Examples: {len(labels)}")

	BATCH_SIZE = 1000
	NUM_EPOCHS = 5

	dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
	dataset = dataset.shuffle(VOCAB_SIZE).batch(BATCH_SIZE, drop_remainder=True)

	word2vec = Word2Vec(VOCAB_SIZE)
	word2vec.compile(optimizer=word2vec.optimizer, loss=word2vec.loss, metrics=['accuracy'])
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
	word2vec.fit(dataset, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])

	return word2vec.get_layer('target_embeddings').get_weights()[0], word2int, int2word