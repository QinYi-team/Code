'''
Name:Gated Dual Attention Unit neural network.
Author:Yi Qin; Dingliang Chen
Date:2020/12/14
Paper:Gated dual attention unit neural networks for remaining useful life prediction of rolling bearings.
Environment:Python 3.7.4, tensorflow 1.13.1, numpy 1.19.2, matplotlib 3.1.1, scipy 1.5.2.
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
# Network weight initialization (standard initialization method)
def initialize_weights(n_input, n_hidden, n_output):
    all_weights = dict()
    # weight initialization of reset gate
    all_weights['ur'] = tf.Variable(tf.random_uniform((n_hidden, n_input), minval=-1 / np.sqrt(n_input), maxval=1 / np.sqrt(n_input)), name='ur')
    all_weights['wr'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)), name='wr')
    all_weights['br'] = tf.Variable(tf.zeros([n_hidden, 1]), name='br')
    # weight initialization of update gate
    all_weights['uz'] = tf.Variable(tf.random_uniform((n_hidden, n_input), minval=-1 / np.sqrt(n_input), maxval=1 / np.sqrt(n_input)), name='uz')
    all_weights['wz'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)), name='wz')
    all_weights['bz'] = tf.Variable(tf.zeros([n_hidden, 1]), name='bz')
    # weight initialization of candidate state
    all_weights['uh'] = tf.Variable(tf.random_uniform((n_hidden, n_input), minval=-1 / np.sqrt(n_input), maxval=1 / np.sqrt(n_input)), name='uh')
    all_weights['wh'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)), name='wh')
    all_weights['bh'] = tf.Variable(tf.zeros([n_hidden, 1]), name='bh')
    # weight initialization of attention gate 1
    all_weights['ua1'] = tf.Variable(tf.random_uniform((n_hidden, n_input), minval=-1 / np.sqrt(n_input), maxval=1 / np.sqrt(n_input)), name='ua1')
    all_weights['wa1'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)), name='wa1')
    all_weights['v'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)), name='v')
    all_weights['ua2'] = tf.Variable(tf.random_uniform((n_hidden, n_input), minval=-1 / np.sqrt(n_input), maxval=1 / np.sqrt(n_input)),name='ua2')
    all_weights['wa2'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)),name='wa2')
    # weight initialization of attention gate 2
    all_weights['uaa1'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)),name='uaa1')
    all_weights['waa1'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)),name='waa1')
    all_weights['baa1'] = tf.Variable(tf.zeros([n_hidden, 1]), name='baa1')
    all_weights['uaa2'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)),name='uaa2')
    all_weights['waa2'] = tf.Variable(tf.random_uniform((n_hidden, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)),name='waa2')
    all_weights['baa2'] = tf.Variable(tf.zeros([n_hidden, 1]), name='baa2')
    # weight initialization from hidden layer to output layer
    all_weights['w'] = tf.Variable(tf.random_uniform((n_output, n_hidden), minval=-1 / np.sqrt(n_hidden), maxval=1 / np.sqrt(n_hidden)), name='w')
    all_weights['b'] = tf.Variable(tf.zeros([n_output, 1]), name='b')
    return all_weights
# Builds GDAU cell structure
def gdaumodel(gdau_input,gdau_state,n_input, n_hidden, n_output):
    weights = initialize_weights(n_input, n_hidden, n_output)
    r = tf.sigmoid(tf.matmul(weights['ur'], gdau_input) + tf.matmul(weights['wr'], gdau_state) + weights['br']) # The output of the reset gate
    z = tf.sigmoid(tf.matmul(weights['uz'], gdau_input) + tf.matmul(weights['wz'], gdau_state) + weights['bz']) # The output of the update gate
    candidate = tf.tanh(tf.matmul(weights['uh'], gdau_input) + tf.matmul(weights['wh'], tf.multiply(r,gdau_state)) + weights['bh']) # The output of the candidate state
    at = tf.matmul(weights['v'],tf.tanh(tf.matmul(weights['ua1'],gdau_input) + tf.matmul(weights['wa1'],gdau_state)))
    ut = tf.nn.softmax(at, axis=0)
    h = tf.tanh(tf.multiply(ut,(tf.matmul(weights['ua2'],gdau_input) + tf.matmul(weights['wa2'],gdau_state)))) # The output of the attention gate 1
    a1 = tf.sigmoid(tf.matmul(weights['uaa1'],r) + tf.matmul(weights['waa1'],z) + weights['baa1'])
    a2 = tf.tanh(tf.matmul(weights['uaa2'], r) + tf.matmul(weights['waa2'], z) + weights['baa2'])
    a = tf.multiply(a1, a2) # The output of the attention gate 2
    ht = tf.multiply((1-z), gdau_state) + tf.multiply(z ,candidate)/2 + tf.multiply(z, h)/2 + a # The output of hidden layer
    output = tf.matmul(weights['w'], ht) + weights['b'] # The output of output layer
    return output, ht

path = r'Bearing1_1.mat'
mat_data = scio.loadmat(path)
data = mat_data['T1']
data = np.transpose(np.array(data,dtype='float32')) # Read RMS data
test_data_num = 1725 # Defines the length of test data
pre_num = 100 # Defines the number of prediction points
threshold = 1.903 # Define value of the preset threshold,and the specific circumstances are different.
### Set the number of network units ###
input_num = 120 # Defines the cell number of the input layer
output_num = 1 # Defines the number of cells in the output layer
hidden_num = int(np.floor(np.sqrt(input_num+output_num))) + 10 # Defines the number of cells in the hidden layer
train_data_num = test_data_num - pre_num # Determine the length of training data
data1 = data[0:train_data_num] # Defines training data
max_data = np.max(data1)
min_data = np.min(data1)
data2 = (data1 - min_data)/(max_data - min_data) # linear normalization
y1 = np.zeros([input_num + 1, train_data_num - input_num],dtype='float32') # Define the training matrix
for i in range(input_num + 1):
    for j in range(train_data_num - input_num):
        y1[i,j] = data2[i+j] # The training matrix is constructed according to the number of cells in the input layer
train_data = y1[0:input_num] # Determine input matrix
test_data = y1[input_num] # Determine ouput vector
learning_rate = 0.06 # learing rate
train_epochs = 1000 # Maximum number of iterations
y2 = np.zeros(test_data_num + 200,dtype='float32')
y3 = np.zeros(test_data_num + 200,dtype='float32') # Predefine output vector
batch_loss = np.zeros(train_data_num - input_num,dtype='float32')
loss = np.zeros(train_epochs,dtype='float32') # Predefine loss vector
init_state = np.zeros([hidden_num,1],dtype='float32') # Initializes the hidden layer state
# Defines placeholders
x = tf.placeholder(tf.float32,[input_num,1]) # Input of gdau model
y_test = tf.placeholder(tf.float32,[output_num,1]) # Output of gdau model
state = tf.placeholder(tf.float32,[hidden_num,1]) # state of the hidden layer
# Builds GDAU neural network structure
y_pred, new_state = gdaumodel(x,state,input_num,hidden_num,output_num) # Output of output layer and state of hidden layer
cost = tf.reduce_mean(tf.pow(y_pred - y_test, 2)) # Error calculation
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # cost optimization

with tf.Session() as sess:
    init = tf.global_variables_initializer() # Initializes variables
    sess.run(init)
    # The section of  network training learning
    for epoch in range(train_epochs):
        for i in range(train_data_num - input_num):
            batch_x = train_data[:,i]
            batch_x = batch_x.reshape(input_num, 1)
            batch_y = test_data[i]
            batch_y = batch_y.reshape(output_num, 1)
            _, h1 = sess.run([optimizer, new_state], feed_dict={x: batch_x, y_test: batch_y, state: init_state})
            c = sess.run(cost, feed_dict={x: batch_x, y_test: batch_y, state: init_state})
            y2[i] = sess.run(y_pred, feed_dict={x: batch_x, state: init_state})
            y3[i] = y2[i] * (max_data - min_data) + min_data  # Inverse linear normalization
            init_state = h1
            batch_loss[i] = c
            
        mean_loss = np.mean(batch_loss) # loss of each epoch

        print("Epoch:", '%04d/1000' % (epoch + 1), "Loss=", "{:.9f}".format(mean_loss))
        loss[epoch] = mean_loss
        
    # COST figure
    fig1 = plt.figure(num = 1,figsize = (30,18),dpi = 150)
    plt.plot(loss)
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    plt.legend(loc = 'center left', prop = font)
    plt.xlabel('The number of iterations', font)
    plt.ylabel('Prediction error', font)
    plt.show()
    
    # The section of prediction
    for j in range(train_data_num - input_num,test_data_num + 200):
        batch_x = y2[j- input_num : j] # Take the reciprocal output of the NN as input
        batch_x = batch_x.reshape(input_num, 1)
        h2 = sess.run(new_state,feed_dict = {x:batch_x, state:init_state})
        y2[j] = sess.run(y_pred, feed_dict = {x:batch_x, state:init_state})
        y3[j] = y2[j] * (max_data - min_data) + min_data
        init_state = h2
        
    for k in range(train_data_num - input_num,test_data_num + 200):
        if np.abs(y3[k]) >= threshold:
            shouming = k + input_num + 1 # Determine Life (represent by health characteristic point number)
            print("Remaining useful life is ",shouming)
            break
        
# figure of degradation tracking
fig2 = plt.figure(num = 2, figsize = (30,18), dpi = 150)
plt.scatter(np.arange(input_num + 1,test_data_num + 1), data[input_num:test_data_num], c = 'b', s = 7, label = 'Actual value')
plt.plot(np.arange(input_num + 1, train_data_num + 1), y3[0:train_data_num - input_num], 'r', linewidth = 3.2,label = 'Training value')
plt.scatter(np.arange(train_data_num + 1, test_data_num + 1), y3[train_data_num - input_num:test_data_num - input_num], c = 'k', s = 7,label = 'Predictive value')
x11 = test_data_num
y11 = threshold
plt.plot([1, x11, ], [y11, y11, ], 'g', linewidth = 3.2, label = 'Failure threshold')
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
plt.legend(loc = 'center left', prop = font)
new_ticks = np.arange(0, 2000, 200)
plt.xticks(new_ticks)
plt.tick_params(labelsize = 12)
plt.xlabel('Serial number of the characteristic point', font)
plt.ylabel('RMS', font)
plt.show()
