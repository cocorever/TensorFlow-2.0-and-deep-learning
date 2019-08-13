import  os
import  tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# 调用 tf.reshape 改变 Tensor 的形状，改变的是视图
a = tf.random.normal([4,28,28,3])
print(a.shape, a.ndim)
# (4, 28, 28, 3) 4
aa = tf.reshape(a, [4,784,3])
print(aa.shape, aa.ndim)
# (4, 784, 3) 3
aa = tf.reshape(a, [4,-1,3])
print(aa.shape, aa.ndim)
# (4, 784, 3) 3
aaa = tf.reshape(a, [4,784*3])
print(aaa.shape, aaa.ndim)
# (4, 2352) 2
aaa = tf.reshape(a, [4,-1])
print(aaa.shape, aaa.ndim)
# (4, 2352) 2
aaaa = tf.reshape(aaa, a.shape)
print(aaaa.shape, aaaa.ndim)
# (4, 28, 28, 3) 4

# 调用 tf.transpose 交换维度，改变的是数据
b = tf.random.normal([2,4,6,8])
print(b.shape)
# (2, 4, 6, 8)
bb = tf.transpose(b)
print(bb.shape)
# (8, 6, 4, 2)
bbb = tf.transpose(b,perm=[0,2,1,3])
print(bbb.shape)
# (2, 6, 4, 8)

# 调用 tf.expand_dims 增加维度 
c = tf.random.normal([4,36,8])
print(tf.expand_dims(c, axis=0).shape)
# (1, 4, 36, 8)

# 调用 tf.squeeze 减少维度
cc = tf.random.normal([1,4,36,1])
print(tf.squeeze(cc, axis=0).shape)
# (4, 36, 1)
print(tf.squeeze(cc).shape)
# (4, 36)

# 调用 tf.broadcast_to 扩张维度
d = tf.random.normal([4,28,28,3])
dd = tf.random.normal([28,3])
dd = tf.expand_dims(dd, axis=0)
dd = tf.expand_dims(dd, axis=0)
print(dd.shape)
# (1, 1, 28, 3)
ddd = tf.broadcast_to(dd, d.shape)
print(ddd.shape)
# (4, 28, 28, 3)
print((d + ddd).shape)
# (4, 28, 28, 3)