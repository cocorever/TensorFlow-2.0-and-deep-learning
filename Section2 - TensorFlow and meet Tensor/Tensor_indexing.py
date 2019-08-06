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
print(b[1:6:], b[6:1:-1])
# tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32) tf.Tensor([6 5 4 3 2], shape=(5,), dtype=int32)
# 索引包含两个冒号 [start:end:step]，step 表示每几个元素取值一次
print(b[1:6:2], b[-1:-6:-2])
# tf.Tensor([1 3 5], shape=(3,), dtype=int32) tf.Tensor([9 7 5], shape=(3,), dtype=int32)
# step 为正表示正向取，end 大于 start，为负表示负向取，start 大于 end

c = tf.random.normal([2,4,6,8,10])
print(c.shape)
# 随机初始化一个五维的 Tensor ，shape=(2, 4, 6, 8, 10)
print(c[0,:,:,:,0].shape, c[0,...,0].shape)
# (4, 6, 8) (4, 6, 8)
print(c[:,:,:,1,1].shape, c[...,1,1].shape)
# (2, 4, 6) (2, 4, 6)

d = tf.random.normal([4,36,8])
print(d.shape)
# (4, 36, 8)
dd = tf.gather(d, axis=1, indices=[4,24,8,9])
print(dd.shape)
# (4, 4, 8)
# axis=1 表示操作学生维度，indices 存储取值元素的下标
# 在实现学生自定义取值的基础上可以继续自定义取课程
ddd = tf.gather(dd, axis=2, indices=[6,4,2])
print(ddd.shape)
# (4, 4, 3)
# axis=1 表示操作课程维度

print(tf.gather_nd(d, [[0,0],[1,1],[2,2]]).shape)
# (3, 8) 表示取 1 班的 1 号，2 班的 2 号，3 班的 3号
print(tf.gather_nd(d, [[0,0,0],[1,1,1],[2,2,2]]).shape)
# (3,) 表示取 1 班的 1 号的第 1 门课，2 班的 2 号的第 2 门课，3 班的 3号的第 3 门课

print(tf.boolean_mask(ddd, mask=[[True,False,False,False],[False,True,False,False],[False,False,True,False],[False,False,False,True]]))
# tf.Tensor(
# [[-1.5180492  -1.27997     0.44133297]
#  [-1.482573   -0.7758471   1.1777923 ]
#  [ 0.97029454 -0.7003514   0.74898523]
#  [-0.05706852  0.835683    0.52352333]], shape=(4, 3), dtype=float32)