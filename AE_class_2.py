from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as layers

############################# Helper Fns ######################################

def AE(X, encode=False):
    # ENCODER
    # input shape = -1 x 32 x 32 x 1  (padded 28 x 28)
    net = layers.conv2d(X, 32, [5, 5], stride=2, padding='SAME')
    net = layers.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    enc = layers.conv2d(net, 8, [5, 5], stride=4, padding='SAME')

    if encode is True:
        return enc # encoded shape = -1 x 2 x 2 x 8 
    # Basically, 784 pixel values get cut down to 32 values of the encoding
   
    # DECODER
    net = layers.conv2d_transpose(enc, 16, [5, 5], stride=4, padding='SAME')
    net = layers.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    dec_img = layers.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return dec_img


def data_preprocess(input_data):
    input_data = np.reshape(input_data, (-1,28,28))
    input_data = zero_pad(input_data)
    input_data = np.reshape(input_data, (-1,32,32,1))
    input_data = input_data/255
    return input_data

def zero_pad(imgs, new_size=32):
    
    imgs_new = np.insert(imgs, 27, 0, axis = 1)
    imgs_new = np.insert(imgs_new, 27, 0, axis = 1)
    imgs_new = np.insert(imgs_new, 0, 0, axis = 1)
    imgs_new = np.insert(imgs_new, 0, 0, axis = 1)    
    
    imgs_new = np.insert(imgs_new, 27, 0, axis = 2)
    imgs_new = np.insert(imgs_new, 27, 0, axis = 2)
    imgs_new = np.insert(imgs_new, 0, 0, axis = 2)
    imgs_new = np.insert(imgs_new, 0, 0, axis = 2)
    
    return imgs_new

################################# AE class ####################################

class AutoEncoder:
    
    def __init__(self):
        self.sess = tf.Session()
        self.X = None
        self.enc_img = None
        self.dec_img = None
        
    
    
    def train(self, input_data, lr, batch_size, n_epochs):
        self.X = tf.placeholder(tf.float32, (None, 32, 32, 1))
        self.enc_img = AE(self.X, True)
        self.dec_img = AE(self.X, False)
        
        input_data = data_preprocess(input_data)
        
        loss = tf.reduce_mean(tf.square(self.dec_img - self.X))
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        
        n_batches = floor( input_data.shape[0]/batch_size )

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        loss_list=[]
        for e in range(n_epochs):
            for t in range(n_batches):
                img_batch = input_data[t:t+batch_size]
                
                _, loss_value = self.sess.run( [train_step, loss], 
                                               feed_dict={self.X:img_batch} )
                loss_list.append(loss_value)
                print("Epoch: {}, Batch: {} Loss: {}".format(str(e+1), str(t+1), str(loss_value)))
            #loss_list.append(loss_value)
                
        plt.plot(loss_list)
        plt.title("Loss plot (lr={}, batch_size={}, n_epochs={})".format(str(lr), str(batch_size), str(n_epochs)))
        plt.show()
        
        
    
    def autoencode(self, img, encode=False):
       if encode is False:
           output = self.sess.run( self.dec_img,
                                   feed_dict={self.X:img} )
       else:
           output = self.sess.run( self.enc_img,
                                   feed_dict={self.X:img} )
           
       return output
   
    
    def save_model(self, file_path):
        saver = tf.train.Saver()
        saver.save(self.sess, file_path)
    
    def load_model(self, file_path):
        self.sess = tf.Session()
        loader = tf.train.import_meta_graph(file_path + "AE.ckpt.meta")
        loader.restore(self.sess, file_path + "AE.ckpt")
        
        #graph = tf.get_default_graph()
        graph = self.sess.graph
        self.X = graph.get_tensor_by_name("Placeholder:0")
        self.enc_img = graph.get_tensor_by_name("Conv_2/Relu:0")
        self.dec_img = graph.get_tensor_by_name("Conv2d_transpose_2/Tanh:0")
        
    
    
################################# TRAINING ####################################

if __name__ == '__main__':

    input_data = pd.read_csv("datasets/MNIST/mnist_train.csv").values
    
    
    model = AutoEncoder()
    lr = 0.001
    batch_size = 64
    n_epochs = 5
    
    #model.load_model("models/TF-AE-4/")        # load model
    
    model.train(input_data, lr, batch_size, n_epochs)

    ################################# visualization ###########################
    n = np.random.randint(0,100)
    sample_img = input_data[n][1:]
    plt.imshow(sample_img.reshape(28,28))
    plt.title("Original image")
    plt.show()
    sample_img = input_data[n]
    sample_img = data_preprocess(np.expand_dims(sample_img,0))
    reconstructed_img = model.autoencode(sample_img, False)
    plt.imshow(np.squeeze(reconstructed_img))
    plt.title("Reconstructed image")
    plt.show()
 
    ################################ save model ###############################
    
   # model.save_model("models/TF-AE-4/AE.ckpt") # save model
    
    #model.sess.close()


