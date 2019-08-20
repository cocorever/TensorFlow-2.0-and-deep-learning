import  os
import  tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.random.normal([4,10])
# 求最大值、最小值的位置
print(tf.argmin(a).shape)
# (10,)
print(tf.argmax(a, axis=1).shape)
# (4,)

y = tf.constant([3,6,9])
out = tf.constant([[0.1,0,0.2,0.7,0,0,0,0,0,0],[0,0,0,0,0,0,0.8,0,0,0.2],
	[0,0,0,0,0,0,0,0,0.9,0.1]], dtype=tf.float32)
# argmax/min 返回的元素类型是 int64，需要转化成 int32 才能与 y 比较
out = tf.cast(tf.argmax(out, axis=1), dtype=tf.int32)
print(out, out.shape)
# tf.Tensor([3 6 8], shape=(3,), dtype=int32) (3,)
# 比较预测结果和真实结果，只留下相同结果标为 1 并求和
correct = tf.reduce_sum(tf.cast(tf.equal(out,y), dtype=tf.int32))
print(correct) # tf.Tensor(2, shape=(), dtype=int32)
accuracy = correct/y.shape[0]
print(accuracy) # tf.Tensor(0.6666666666666666, shape=(), dtype=float64)

b = range(10, 15)
print(tf.sort(b, direction='DESCENDING'))
# tf.Tensor([14 13 12 11 10], shape=(5,), dtype=int32)
print(tf.argsort(b, direction='DESCENDING'))
# tf.Tensor([4 3 2 1 0], shape=(5,), dtype=int32)
# 只求最大的 k 个值
res = tf.math.top_k(b, 2)
print(res.values, res.indices)
# tf.Tensor([14 13], shape=(2,), dtype=int32) tf.Tensor([4 3], shape=(2,), dtype=int32)

c = tf.ones([2,2])
print(tf.norm(c))
# tf.Tensor(2.0, shape=(), dtype=float32)
# 使用第二范数原公式验证，结果相同。
print(tf.sqrt(tf.reduce_sum(tf.square(c))))
# tf.Tensor(2.0, shape=(), dtype=float32)
print(tf.norm(c, ord=2, axis=1))
# tf.Tensor([1.4142135 1.4142135], shape=(2,), dtype=float32)
print(tf.sqrt(tf.reduce_sum(tf.square(c[0]))), tf.sqrt(tf.reduce_sum(tf.square(c[1]))))
# tf.Tensor(1.4142135, shape=(), dtype=float32) tf.Tensor(1.4142135, shape=(), dtype=float32)
# 第一范数
print(tf.norm(c, ord=1))
# tf.Tensor(4.0, shape=(), dtype=float32)
# 使用第一范数原公式验证，结果相同。
print(tf.reduce_sum(tf.abs(c)))
# tf.Tensor(4.0, shape=(), dtype=float32)

