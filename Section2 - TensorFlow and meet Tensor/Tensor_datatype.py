import  os
import  tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

b = tf.range(10)
print(b)
# tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)
# 正序下标从 0 开始，倒序从 -1 开始
print(b[0], b[-10], b[9], b[-1])
# tf.Tensor(0, shape=(), dtype=int32) tf.Tensor(0, shape=(), dtype=int32) tf.Tensor(9, shape=(), dtype=int32) tf.Tensor(9, shape=(), dtype=int32)
# 数字 0 的下标可以用 0 或 -10 表示，数字 9 的下标可以用 9 或 -1 表示
print(b[0:6])
# tf.Tensor([0 1 2 3 4 5], shape=(6,), dtype=int32)
# 索引包含一个冒号 [start:end] 表示取下标区间为 [start,end) 的元素
print(b[6:], b[:6])
# tf.Tensor([6 7 8 9], shape=(4,), dtype=int32) tf.Tensor([0 1 2 3 4 5], shape=(6,), dtype=int32)
# 如果不填 start，默认从第一个元素开始取，如果不填 end，默认取到最后一个元素，[:] 表示都取
print(b[:-3], b[-7:])
# tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32) tf.Tensor([3 4 5 6 7 8 9], shape=(7,), dtype=int32)
# 不管下标是正是负，单冒号的索引总是由左向右取值