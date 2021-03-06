import tensorflow as tf

input_data=[[1,5,3,7,8,10,12],[5, 8, 10, 3, 9, 7,1]]
label_data=[[0,0,0,1,0], [1, 0 ,0, 0, 0]]

INPUT_SIZE=7
HIDDEN1_SIZE=10
HIDDEN2_SIZE=8
CLASSES=5
Learning_Rate=0.05

# shape must be matched to data dimension
x=tf.placeholder(tf.float32, shape=[None, INPUT_SIZE]) #shape=[batchSize, dimension]
y_=tf.placeholder(tf.float32, shape=[None, CLASSES])

tensor_map={x:input_data, y_:label_data}

# Building Model
W_h1 =tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32) # truncated_normal: Outputs random values from a normal distribution
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32)
hidden1=tf.matmul(x, W_h1) +b_h1
hidden1=tf.sigmoid(hidden1)

W_h2 =tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32)
b_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32)
hidden2 =  tf.matmul(hidden1,W_h2) + b_h2
hidden2=tf.sigmoid(hidden2)

W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32)
b_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32)
y = tf.matmul(hidden2, W_o) +b_o
y=tf.sigmoid(y)



# Training
cost= tf.reduce_mean(-y_*tf.log(y)-(1-y_)*tf.log(1-y))
train= tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)
sess = tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for i in range(100):
    _, loss = sess.run([train, cost], feed_dict=tensor_map)
    print ("Step: ", i)
    print ("loss: ", loss)

sess.close()