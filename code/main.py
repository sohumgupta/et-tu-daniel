import numpy as np
import tensorflow as tf

def main():
	first = tf.constant([[[1]], [[2]], [[3]]])

	# 3, 1
	# [
	# 	[
	# 		[1]
	# 	], 
	# 	[
	# 		[2]
	# 	], 
	# 	[
	# 		[3]
	# 	]
	# ]

	# [
	# 	[
	# 		[
	# 			[1], 
	# 			[1]
	# 		]
	# 	], 
	# 	[
	# 		[2],
	# 		[2]
	# 	], 
	# 	[
	# 		[3],
	# 		[3]
	# 	]
	# ]

	broadcasted = tf.broadcast_to(first, [3, 2, 1])

	print(first.shape)
	print(first)
	print(broadcasted.shape)
	print(broadcasted)

if __name__ == '__main__':
	main()