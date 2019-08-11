import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets

# TensorFlow 会打印一些无关的语句，这个语句用于消除无关的部分
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# TensorFlow 提供了 datasets 的包可以下载一些常见的数据集，包括 MNIST
# 这个数据集函数会返回两个元组，第一个元组是 60k 训练集，第二个元组是 10k 的测试集
# x 的形状是 [60k,28,28]，y 的形状是 [60k]，还未经过 one-hot 编码
(x, y), (x_val, y_val) = datasets.mnist.load_data() 
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
# one_hot 编码
y = tf.one_hot(y, depth=10)
print('x 的形状：', x.shape, 'y 的形状：', y.shape) # (60000, 28, 28) (60000, 10)
# 使用 batch() 控制每次训练的图片数量，也就是 input [a, 784] 中的 a，这里设置每次输入 100 张图片
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(100)
# 计算 w @ x + b，只需要给 Dense() 相应的参数，activation 就是激活函数，前面的数字是向量降维后的长度
model = keras.Sequential([
	layers.Dense(512, activation='relu'),
	layers.Dense(256, activation='relu'),
	layers.Dense(10)])
# 使用 optimizers 类可以简单地定义梯度下降函数 x' = x - lr * δloss/δx
optimizer = optimizers.SGD(learning_rate=0.001)

def train_epoch(epoch):
	# 一个 step 等于使用 batch 个样本训练一次；，每次输入 100 张，需要 600 个 step
	for step, (x, y) in enumerate(train_dataset):
		# 计算 h1, h2, output
		with tf.GradientTape() as tape:
			# 将 input 的形状从 [a, 28, 28] 变为 [a, 784]
			x = tf.reshape(x, (-1, 28*28))
			# 应用模型，计算 output，得到一个 [a, 10] 的输出
			out = model(x)
			# 定义损失函数 loss
			loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
		# 打印优化前的误差
		if epoch == 0 and step == 0:
			print('优化前的 loss:', loss.numpy())
		# 更新 w1,b1,w2,b2,w3,b3，只需要将 loss 函数和模型的参数传入梯度函数 gradient() 可以得到 loss 对于各个参数的偏导数
		grads = tape.gradient(loss, model.trainable_variables)
		# 将 loss 对于各参数以及其偏导数传入由 optimizers 类得到的函数，计算 Gradient Descent
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
	# 每 10 次 epoch 打印误差
	if (epoch + 1) % 10 == 0:
		print('训练', epoch + 1, '次的 loss:', loss.numpy())

def train():
	# 训练 30 次，一个 epoch 等于使用训练集中的全部样本训练一次
	for epoch in range(30):
		train_epoch(epoch)

if __name__ == '__main__':
	train()
