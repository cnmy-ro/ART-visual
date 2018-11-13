"""
Fuzzy ARTMAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

############################### helper functions ##############################

def minimum(A,B):
    ans=[]
    for i in range(len(A)):
        #print("I and W sizes: ", len(A), len(B))
        ans.append(min([A[i],B[i]]))
        
    ans = np.array(ans)
    return ans
    
def complement_code(X):
        I = np.hstack((X, 1-X))
        return I
   
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
    print(labels[0])
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

###################################  class  ###################################
class simplified_fuzzy_ARTMAP:
    
    def __init__(self, X_size, label_size, c_max_a, rho_a, rho_ab, alpha=0.00001, beta=1):
        self.M_a = X_size    # input vector size
        self.M_ab = label_size    # input label vector size
        
        self.c_max_a = c_max_a # max categories for ART-a
        
        
        self.rho_a = rho_a    # vigilance parameter for ART-a
        self.rho_a_baseline = rho_a
        self.rho_ab = rho_ab  # vigilance parameter for map field
        self.alpha = alpha # choice parameter
        self.beta = beta   # learning rate
        
        self.N_a = 0         # no. of categories of ART_a initialized to zero
        
        self.W_a = np.ones( (c_max_a, self.M_a*2) ) # initialize W_a with 1s
        self.W_ab = np.ones( (self.M_ab, c_max_a) ) # initialize W_ab with 1s
        
        self.X_ab = np.zeros( (self.M_ab,) )
    
    def train(self, X, one_hot_labels, rho_a_inc_rate=0.001):
        A = complement_code(X)   # shape of X = Mx1, shape of I = 2Mx1
        B = one_hot_labels
         
        self.rho_a =  self.rho_a_baseline
         
        T = []
        for i in range(self.N_a):           
            T.append( np.sum(minimum(A,self.W_a[i,:])) / (self.alpha+np.sum(self.W_a[i,:])) ) # calculate output
        
        J_list = np.argsort(np.array(T))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
        
        for J in J_list:
            # Checking for resonance in ART-a  ---
            X_a_mod = np.sum(minimum(A,self.W_a[J,:])) 
                
            while X_a_mod >= self.rho_a * np.sum(A): # resonance occured in ART-a 
                
                #####  match tracking  #####
                # Checking for resonance in the MAP FIELD  ---
                   
                self.X_ab_mod = np.sum(minimum( B,self.W_ab[:,J] ))
                
                if self.X_ab_mod > self.rho_ab * np.sum(B): # resonance occurs in the MAP FIELD
                    # weight update
                    self.W_a[J,:] = self.beta*minimum(A,self.W_a[J,:]) + (1-self.beta)*self.W_a[J,:] # weight update of ART-a
                    K = np.argmax( B )
                    self.W_ab[:,J] = 0
                    self.W_ab[K,J] = 1
                    return self.W_ab, K
                
                else: # NO resonance in the MAP FIELD
                    self.rho_a += rho_a_inc_rate
       
        if self.N_a < self.c_max_a:    # no resonance occured in ART-a, create a new category
            n = self.N_a
            self.W_a[n,:] = self.beta*minimum(A,self.W_a[n,:]) + (1-self.beta)*self.W_a[n,:] # weight update
            self.N_a += 1
            
            K = np.argmax( B )
            self.W_ab[:,n] = 0
            self.W_ab[K,n] = 1
            return self.W_ab, K
        
        if self.N_a >= self.c_max_a:
            print("ART-a memory error!")
            return None, None
    
    def infer(self, X):
        A = complement_code(X)   
        T = []
        for i in range(self.N_a):           
            T.append( np.sum(minimum(A,self.W_a[i,:])) / (self.alpha+np.sum(self.W_a[i,:])) ) # calculate output        
        J_list = np.argsort(np.array(T))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
        J = J_list[0] # maximum activation
        #return self.W[J,:], J
        X_ab = self.W_ab[:,J] 
        return X_ab
        '''
        for J in J_list:
            # Checking for resonance ---
            d = np.sum(minimum(I,self.W[J,:])) / np.sum(I)
            if d >= self.rho: # resonance occured
                return self.W[J,:], J
            return None, None
        '''
'''
    
class fuzzy_ARTMAP:
    
    def __init__(self, X_size, label_size, c_max_a, c_max_b, rho_a, rho_b, rho_ab, alpha=0.00001, beta=1):
        self.M_a = X_size    # input vector size
        self.M_b = label_size    # input label vector size
        
        self.c_max_a = c_max_a # max categories for ART-a
        self.c_max_b = c_max_b # max categories for ART-b
        
        self.rho_a = rho_a    # vigilance parameter for ART-a
        self.rho_b = rho_b    # vigilance parameter for ART-b
        self.rho_ab = rho_ab  # vigilance parameter for map field
        self.alpha = alpha # choice parameter
        self.beta = beta   # learning rate
        
        self.N_a = 0         # no. of categories of ART_a initialized to zero
        
        self.W_a = np.ones( (c_max_a, self.M_a*2) ) # initialize W_a with 1s
        self.W_b = np.ones( (c_max_b, self.M_b*2) ) # initialize W_b with 1s
        self.W_ab = np.ones( (c_max_b, c_max_a) ) # initialize W_ab with 1s
    
    def train(self, X):
        I = complement_code(X)   # shape of X = Mx1, shape of I = 2Mx1
        
        T = []
        for i in range(self.N):           
            T.append( np.sum(minimum(I,self.W[i,:])) / (self.alpha+np.sum(self.W[i,:])) ) # calculate output
        
        J_list = np.argsort(np.array(T))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
        for J in J_list:
            # Checking for resonance ---
            d = np.sum(minimum(I,self.W[J,:])) / np.sum(I)
            if d >= self.rho: # resonance occured
                self.W[J,:] = self.beta*minimum(I,self.W[J,:]) + (1-self.beta)*self.W[J,:] # weight update
                return self.W[J,:], J
       
        if self.N < self.c_max:    # no resonance occured, create a new category
            k = self.N
            self.W[k,:] = self.beta*minimum(I,self.W[k,:]) + (1-self.beta)*self.W[k,:] # weight update
            self.N += 1
            return self.W[k,:], k
        
        return None, None

'''