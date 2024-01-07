import tensorflow as tf

#Creating a tensor
tensor_a = tf.constant([1, 2, 3])
print(tensor_a)

#Tensor operations
tensor_b = tf.constant([4, 5, 6])
result = tf.add(tensor_a, tensor_b)
print(result.numpy())