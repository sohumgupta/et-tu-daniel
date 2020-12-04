import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from word2vec import Word2Vec
import io

from preprocess import construct_vocab, get_sentences, vectorize_sentences, pad_sentences

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

def train_embeddings(NEGATIVE_SAMPLES, NUM_EPOCHS):
	data_path = '../data'
	all_paths = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]

	print(f"Preprocessing for embeddings...")
	all_sentences, max_length = get_sentences(all_paths)
	all_sentences = pad_sentences(all_sentences, sentence_length=max_length)
	word2int, int2word = construct_vocab(all_sentences)
	print(f"Vocab Size: {len(word2int)}")
	sentence_vectors = vectorize_sentences(word2int, all_sentences)
	print(f"Number of Sentences: {len(all_sentences)}")

	VOCAB_SIZE = len(word2int)
	SENTENCE_LENGTH = max_length
	WINDOW_SIZE = 2

	print(f"Generating training data...")
	targets, contexts, labels = get_training_data(
		sentences=sentence_vectors, 
		window_size=WINDOW_SIZE, 
		negative_samples=NEGATIVE_SAMPLES, 
		vocab_size=VOCAB_SIZE)
	print(f"Number of Training Examples: {len(labels)}")

	BATCH_SIZE = 1000

	dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
	dataset = dataset.shuffle(VOCAB_SIZE).batch(BATCH_SIZE, drop_remainder=True)

	word2vec = Word2Vec(VOCAB_SIZE)
	word2vec.compile(optimizer=word2vec.optimizer, loss=word2vec.loss, metrics=['accuracy'])
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
	word2vec.fit(dataset, epochs=NUM_EPOCHS, callbacks=[tensorboard_callback])

	return word2vec.get_layer('target_embeddings').get_weights()[0], word2int, int2word

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

def get_embeddings():
	embeddings_path = '../embeddings'
	
	# Train embeddings if necessary
	if 'embeddings.tsv' not in listdir(embeddings_path) or 'names.tsv' not in listdir(embeddings_path):
		print(f"Training word embeddings...")
		embeddings, word2int, int2word = train_embeddings(10, 5)
		write_embeddings(embeddings, word2int, embeddings_path)
	else:
		print(f"Word embeddings found.")

	# Read embeddings from files
	embeddings, names = read_embeddings(embeddings_path)
	word2int = {word:i for (i, word) in enumerate(names)}
	int2word = {i:word for (i, word) in word2int.items()}
	return embeddings, word2int, int2word
