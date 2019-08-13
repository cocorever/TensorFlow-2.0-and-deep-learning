import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# x: [60k, 28, 28]
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
# 
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

y = tf.one_hot(y, depth=10)
# 创建数据集，取 batch
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
# 添加迭代器
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# 初识化 w 和 b，转化为 tf.Variable 可求梯度
# [b ,784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out] [dim_out]
# 减小方差防止梯度爆炸
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
for epoch in range(10): # 对整个数据集迭代
	for step, (x, y) in enumerate(train_db): # 对 batch 迭代
		# x:[128, 28, 28]
		# y:[128, 10]

		# x:[128, 28, 28] => [128, 784]
		x = tf.reshape(x, [-1, 28*28])

		with tf.GradientTape() as tape: # 只接受 tf.variable
			# x: [128, 784]
			# output1 = x @ w1 + b1
			# [128, 784] @ [784, 256] + [256] => [128, 256] + [256]
			output1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
			output1 = tf.nn.relu(output1)
			output2 = output1 @ w2 + b2
			output2 = tf.nn.relu(output2)
			output = output2 @ w3 + b3

			# 计算 loss
			# mse = mean(sum((y-out)^2))
			# [128, 10]
			loss = tf.square(y - output)
			# mean: scalar
			loss = tf.reduce_mean(loss)

		# 计算梯度
		grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
		# w1 = w1 - lr * w1_grad
		w1.assign_sub(lr * grads[0])
		b1.assign_sub(lr * grads[1])
		w2.assign_sub(lr * grads[2])
		b2.assign_sub(lr * grads[3])
		w3.assign_sub(lr * grads[4])
		b3.assign_sub(lr * grads[5])
	
	print('训练', epoch, '次 - loss:', float(loss))