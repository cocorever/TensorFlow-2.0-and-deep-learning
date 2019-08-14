import  os
import  tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.random.normal([4,36,8])
aa = tf.ones([4,36,8])
aaa = tf.stack([a,aa], axis=0)
print(aaa.shape)
# (2, 4, 36, 8)
# tf.stack 会先增加一个维度，然后在新维度上合并
aaaa = tf.stack([a,aa], axis=2)
print(aaaa.shape)
# (4, 36, 2, 8)

b = tf.ones([2,3,36,8])
bb = tf.zeros([2,1,36,8])
bbb = tf.concat([b,bb], axis=1)
print(bbb.shape)
# (2, 4, 36, 8)

c = tf.random.normal([4,36,8])
cc = tf.unstack(c, axis=0)
print(len(cc), cc[0].shape)
# 4 (36, 8)
ccc = tf.unstack(c, axis=2)
print(len(ccc), ccc[0].shape)
# 8 (4, 36)

dd = tf.split(c, axis=2, num_or_size_splits=2)
print(len(dd), dd[0].shape, dd[1].shape)
# 2 (4, 36, 4) (4,36,4)
ddd = tf.split(c, axis=2, num_or_size_splits=[1,3,4])
print(len(ddd), ddd[0].shape, ddd[1].shape, ddd[2].shape)
# 3 (4, 36, 1) (4, 36, 3) (4, 36, 4)