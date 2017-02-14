import tensorflow as tf

input_data=[1,5,3,7,8,10,12]
label_data=[0,0,0,1,0]

INPUT_SIZE=7
HIDDEN1_SIZE=10
HIDDEN2_SIZE=8
CLASSES=5


x=tf.placeholder(tf.float32, shape=[None, INPUT_SIZE]) #shape=[batchSize, dimension]
y_=tf.placeholder(tf.float32, shape=[None, CLASSES])

feed_dict={x:input_data, y_:label_data}

# Building Model
W_h1 =tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32) # truncated_normal: Outputs random values from a normal distribution
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32)
hidden1=tf.matmul(x, W_h1) +b_h1

W_h2 =tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32)
b_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32)
hidden2 =  tf.matmul(hidden1,W_h2) + b_h2

W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32)
b_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32)
y = tf.matmul(hidden2, W_o) +b_o



# Training
cost= tf
