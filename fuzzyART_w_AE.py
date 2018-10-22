from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf

import fuzzy_ART
import AE_class


fuzzy_ART_MODE = "TRAIN"
results_dir = "results/fuzzyART_w_AE-1/"

################################ Helper Functions #############################

def get_data():
    mnist = pd.read_csv("datasets/MNIST/mnist_test.csv").values    
    data = {}
    for i in range(0,10):
        mask = mnist[:,0]==i
        temp = mnist[mask]
        temp = temp/255
        np.random.shuffle(temp)
        temp = temp[0:5, 1:]      # 5 images of each digit
        data[str(i)] = temp
    return data

def preprocess_for_AE(data_dict):
    data_array = []
    for i in range(0,10):
        data_array.append( data_dict[str(i)] )
    data_array = np.array(data_array).reshape(10*5,784)
    data_array = data_array.reshape(-1, 28, 28)
    data_array = np.expand_dims(data_array, 3)

    return np.array(data_array)


def preprocess_for_ART(data_array):
    data_array = data_array.reshape(-1,7*7*8)
    return data_array
    

def train_fuzzy_ART(model, data_array):
        
    for i in range(data_array.shape[0]):  
        Z, k = model.train(data_array[i])
        
        enc_learned_wts = Z[:392].reshape(8,49)
        #plt.subplot(10,5,i+1)
        plt.imshow(enc_learned_wts) #display learned expectations
        #plt.imshow(data_digit[i].reshape(28,28))
        plt.title("Viz. of learned encoded digit: {}, sample:{}".format(str(floor(i/5)),str(i%5 + 1)))
        #plt.savefig( results_dir+"encoded_wts/"+"{}_{}.png".format(str(floor(i/5)),str(i%5 + 1)) )
        plt.show()
         
        print("CLASS: ", k)
    
    return model


################################  Load Models  ################################
if fuzzy_ART_MODE is "TEST":
    # Load pre-trained fuzzy art model
    ART_rho = 0.72  # vigilance parameter
    ART_model = fuzzy_ART.fuzzy_ART(7*7*8, 
                                    c_max=20, 
                                    rho=ART_rho, 
                                    alpha=0.00001,
                                    beta=1)
    ART_model.load_params("models/fuzzyART_weights_1")
    
elif fuzzy_ART_MODE is "TRAIN":
    ART_rho = 0.975  # vigilance parameter
    ART_model = fuzzy_ART.fuzzy_ART(7*7*8, 
                                    c_max=20, 
                                    rho=ART_rho, 
                                    alpha=0.00001, 
                                    beta=1)

# Load Tensorflow AutoEncoder model
AE_model = AE_class.AutoEncoder()
AE_model.load_model("models/TF-AE-3/")

##############################  Import test data  #############################

if fuzzy_ART_MODE is "TEST":
    test_data = pd.read_csv("datasets/MNIST/mnist_test.csv").values
    test_data = test_data[:,1:]
    n = np.random.randint(0,1000)
    
    
############################## or import train data ###########################

elif fuzzy_ART_MODE is "TRAIN":
    '''
    In this case, the fuzzy ART model will be trained on the data encoded by the
    Autoencoder
    '''    
    ART_train_data = get_data()
    
    preprocessed_data =  preprocess_for_AE(ART_train_data)
    encoded_train_data = AE_model.autoencode(preprocessed_data, True)
    
    encoded_train_data = preprocess_for_ART(encoded_train_data)
    ART_model = train_fuzzy_ART(ART_model, encoded_train_data)