from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf

import fuzzy_ARTMAP
import AE_class_2 




fuzzy_ART_MODE = "TRAIN"
results_dir = "results/fuzzyART_w_AE-2/"



################################ Helper Functions #############################

def get_data():
    mnist = pd.read_csv("datasets/MNIST/mnist_test.csv").values    
    data = {}
    labels = {}
    for i in range(0,10):
        mask = mnist[:,0]==i
        digit_imgs = mnist[mask]  
        
        np.random.shuffle(digit_imgs)
        
        digit_labels = digit_imgs[0:5, 0]
        digit_labels = one_hot_encode(digit_labels)
        
        digit_imgs = digit_imgs[0:5, 1:]      # 5 images of each digit
        digit_imgs = digit_imgs/255
        
        data[str(i)] = digit_imgs
        labels[str(i)] = digit_labels
    return data, labels

def one_hot_encode(labels):
    #print(labels[0])
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    #print("one hot shape: ",one_hot_labels.shape)
    return one_hot_labels

def preprocess_for_AE(data_dict, label_dict):
    data_array = []
    label_array = []
    for i in range(0,10):
        data_array.append( data_dict[str(i)] )
        label_array.append( label_dict[str(i)] )
        
    data_array = np.array(data_array).reshape(10*data_dict[str(i)].shape[0], 784)
    data_array = AE_class_2.data_preprocess(data_array)
    label_array = np.array(label_array).reshape(10*label_dict[str(i)].shape[0], 10)

    return data_array, label_array


def preprocess_for_ART(data_array):
    data_array = data_array.reshape(-1,32)
    return data_array
    

def train_fuzzy_ARTMAP(model, data_array, label_array):

    for i in range(data_array.shape[0]):  
        #print("one hot shape: ",label_array.shape)
        Z, k = model.train(data_array[i], label_array[i])
        
        #enc_learned_wts = Z[:32].reshape(4,8)
        plt.imshow(Z) #display learned expectations
        plt.title("Viz. of learned encoded digit: {}, sample:{}".format(str( floor( i/(data_array.shape[0]/10) ) ),
                                                                        str( i%(data_array.shape[0]/10) + 1)) )
        #plt.savefig( results_dir+"encoded_wts/"+"{}_{}.png".format(str(floor(i/5)),str(i%5 + 1)) )
        plt.show()
         
        print("CLASS: ", k)
    
    return model

def test(model):
    test_data = pd.read_csv("datasets/MNIST/mnist_test.csv").values
    np.random.shuffle(test_data)
    test_data = test_data[:1001, :]
    test_data_labels = np.array(test_data[:,0])
    test_data = test_data[:,1:]
    
    test_data = test_data.reshape(-1, 28, 28)
    test_data = np.expand_dims(test_data, 3)

    test_data = AE_class_2.data_preprocess(test_data)
    encoded_test_data = AE_model.autoencode(test_data, True)
    
    encoded_test_data = preprocess_for_ART(encoded_test_data)
    
    ART_output = []
    for i in range(0,encoded_test_data.shape[0]):
        op = ART_model.infer( encoded_test_data[i] )
        #if op[0] is None and op[1] is None:
            #op = [0, ART_model.N]
        op = np.argmax(op)
        
        ART_output.append(op)
    ART_output = np.array(ART_output)
    
    accuracy = np.mean( np.equal(test_data_labels, ART_output) ) * 100
    
    return accuracy, ART_output, test_data_labels


def Purity(model):
    _,  ART_output, test_data_labels = test(ART_model)
    
    #combined = np.vstack([ART_output, test_data_labels]).T
    categories = {}
    purity_list = []
    for i in range(np.max(ART_output)+1):

        mask = ART_output==i
        temp = test_data_labels[mask]  # labels of data belonging to category i
        
        categories[str(i)] = temp
        if temp.shape[0] > 0:
            most_freq_label = np.argmax( np.bincount(temp) )
            purity_for_curr_category = np.mean( temp==most_freq_label ) *100
        else:
            purity_for_curr_category = 0
        purity_list.append( purity_for_curr_category)

    purity = np.mean( np.array(purity_list) )    
    return purity, categories
    
   
############### Load pre-trained Tensorflow AutoEncoder model #################
AE_model = AE_class_2.AutoEncoder()
AE_model.load_model("models/TF-AE/")

#####################################  TEST  ##################################

if fuzzy_ART_MODE is "TEST":
    # Load pre-trained fuzzy art model
    ART_rho = 0.95  # vigilance parameter
    ART_model = fuzzy_ARTMAP.simplified_fuzzy_ARTMAP(32, 
                                                    c_max=20, 
                                                    rho=ART_rho, 
                                                    alpha=0.00001,
                                                    beta=1)
    ART_model.load_params("models/fuzzyART_w_AE-2_weights")
 
    accuracy, _, _ = test(ART_model)
    print("Test accuracy: ", accuracy)


    
################################# or TRAIN ####################################

elif fuzzy_ART_MODE is "TRAIN":
    '''
    In this case, the fuzzy ART model will be trained on the data encoded by the
    pretrained Autoencoder
    '''    
    np.random.seed(0)
    #initialize a fuzzy ART model for training
    ART_a_rho = 0.9 # vigilance parameter
    map_field_rho = 0.8
    ART_model = fuzzy_ARTMAP.simplified_fuzzy_ARTMAP(32, 10, 
                                                    c_max_a=50, 
                                                    rho_a=ART_a_rho,
                                                    rho_ab=map_field_rho,
                                                    alpha=0.000001, 
                                                    beta=1)
    
    ART_train_data, one_hot_labels = get_data()
    
    preprocessed_data, one_hot_labels =  preprocess_for_AE(ART_train_data, one_hot_labels)
    encoded_train_data = AE_model.autoencode(preprocessed_data, True)
    
    encoded_train_data = preprocess_for_ART(encoded_train_data)
    
    n_epochs = 1
    acc_list = []
    #purity_list = []
    for e in range(n_epochs):
        ART_model = train_fuzzy_ARTMAP(ART_model, encoded_train_data, one_hot_labels)
        accuracy, ART_output, _ = test(ART_model)  
        #purity, Categories = Purity(ART_model)
        
        print("Test accuracy at epoch {}: {}".format(e+1, accuracy))
        #print("Purity at epoch {}: {}".format(e+1, purity))
        acc_list.append(accuracy)
        #purity_list.append(purity)
        
        
    if n_epochs > 1:
        plt.plot(acc_list)
        plt.show()
        
        
    '''
    n = np.random.randint(encoded_train_data.shape[0])
    sample = encoded_train_data[n]
    inference = ART_model.infer(sample)
    '''    
    #ART_model.save_params("models/fuzzyART_w_AE-2_weights")

    