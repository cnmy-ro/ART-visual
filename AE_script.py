from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as layers


########################### Helper functions ##################################

def save_model(sess, file_path):
    saver = tf.train.Saver()
    saver.save(sess, file_path)

def load_model(sess, file_path):
    loader = tf.train.import_meta_graph(file_path + "AE.ckpt.meta")
    loader.restore(sess, file_path + "AE.ckpt")
    return sess
           
###########################  MODEL  ###########################################

X = tf.placeholder(tf.float32, (None, 28, 28, 1))
# encoder
net1 = layers.conv2d(X, 32, [5, 5], stride=2, padding='SAME')
net2 = layers.conv2d(net1, 16, [5, 5], stride=2, padding='SAME')
enc = layers.conv2d(net2, 8, [5, 5], stride=1, padding='SAME')

# decoder
net3 = layers.conv2d_transpose(enc, 16, [5, 5], stride=1, padding='SAME')
net4 = layers.conv2d_transpose(net3, 32, [5, 5], stride=2, padding='SAME')
dec_img = layers.conv2d_transpose(net4, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)

############################ TRAINING #########################################

lr = 0.001
batch_size = 100
n_epochs = 5

input_data = pd.read_csv("datasets/MNIST/mnist_test.csv").values[:,1:]
input_data = np.reshape(input_data, (-1,28,28,1))
input_data = input_data/255
#input_data = np.expand_dims(input_data, 3)

#input_data = resize_batch(input_data)
#print(inputs.shape)

loss = tf.reduce_mean(tf.square(dec_img - X))
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

n_batches = floor( input_data.shape[0]/batch_size )

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_list=[]
for e in range(n_epochs):
    for t in range(n_batches):
        img_batch = input_data[t:t+batch_size]
        #print("img_batch type: ", img_batch.shape)
        #img_batch = resize_batch(img_batch)
        _, loss_value = sess.run( [train_step, loss], 
                                  feed_dict={X:img_batch} )
        loss_list.append(loss_value)
        #if t%100 == 0:
        print("Epoch: {}, Batch: {} Loss: {}".format(str(e+1), str(t+1), str(loss_value)))
    #loss_list.append(loss_value)
        
plt.plot(loss_list)
plt.show()

############################ Visualization ####################################

n = np.random.randint(0,100)

sample_img = np.expand_dims(input_data[n],0)
reconstructed_img = sess.run( dec_img,
                              feed_dict={X:sample_img}  )


encoded_img = sess.run( enc,
                        feed_dict={X:sample_img}  )
encoded_img = np.squeeze(encoded_img)

reconstructed_img = np.squeeze(reconstructed_img)
sample_img = np.squeeze(sample_img)

plt.imshow(sample_img)
plt.title("original")
plt.show()

plt.imshow(reconstructed_img)
plt.title("reconstructed")
plt.show()


#save_model(sess,"models/TF-AE-1/AE_1.ckpt")  # save model


sess = load_model(sess, "models/TF-AE-2/") # load model
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("Placeholder:0")
AE_out = graph.get_tensor_by_name("Conv2d_transpose_2/Tanh:0")
rec_img = sess.run(AE_out, feed_dict = {X: sample_img})
plt.imshow(np.squeeze(rec_img))
plt.title("reconstructed")
plt.show()

sess.close()

'''
op = sess.graph.get_operations()
tensor_list = [m.values() for m in op]
'''