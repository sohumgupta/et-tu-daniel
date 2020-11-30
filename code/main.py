import numpy as np
import tensorflow as tf

def main():
	#2, 1, 2
	first = tf.constant([[[1, 2]],[[1, 2]]], dtype=tf.float32)
	#2, 1, 1
	mult = tf.constant([[[.5]],[[.5]]])

	final = first * mult

	print(final.shape)
	print(final)


if __name__ == '__main__':
	main()