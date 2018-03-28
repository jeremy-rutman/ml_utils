import tensorflow as tf

x = tf.nn.conv2d(tf.ones([1,1,10,1]), tf.ones([1,5,1,1]), strides=[1, 1, 1, 1], padding='SAME')
with tf.Session() as sess:
    y=sess.run(x)
	print(y)
# this should output a tensor of shape (1,1,10,1) with
#[3,4,5,5,5,5,5,5,4,3]


assert y==[3,4,5,5,5,5,5,5,4,3],'tf answer not as expected'