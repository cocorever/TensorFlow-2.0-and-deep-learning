import  os
import  tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.random.normal([8])
print(a)
# tf.Tensor([ 0.19498333  0.05311483 -0.31021473  0.9116785  -0.48776302 -1.105898 -1.2519776  -0.77230555], shape=(8,), dtype=float32)
print(tf.pad(a, [[2,0]]))
# tf.Tensor([ 0.          0.          0.19498333  0.05311483 -0.31021473  0.9116785 -0.48776302 -1.105898   -1.2519776  -0.77230555], shape=(10,), dtype=float32)
print(tf.pad(a, [[1,1]]))
# tf.Tensor([ 0.          0.19498333  0.05311483 -0.31021473  0.9116785  -0.48776302 -1.105898   -1.2519776  -0.77230555  0.        ], shape=(10,), dtype=float32)
print(tf.pad(a, [[0,2]]))
# tf.Tensor([ 0.19498333  0.05311483 -0.31021473  0.9116785  -0.48776302 -1.105898 -1.2519776  -0.77230555  0.          0.        ], shape=(10,), dtype=float32)

b = tf.random.normal([2,2])
print(b)
# tf.Tensor(
# [[1.772424  0.2548454]
 # [0.7038152 1.4971926]], shape=(2, 2), dtype=float32)
print(tf.pad(b, [[1,1],[1,1]]))
# tf.Tensor(
# [[0.        0.        0.        0.       ]
 # [0.        1.772424  0.2548454 0.       ]
 # [0.        0.7038152 1.4971926 0.       ]
 # [0.        0.        0.        0.       ]], shape=(4, 4), dtype=float32)