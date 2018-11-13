"""
Fuzzy ARTMAP
"""

import numpy as np
import pandas as pd


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

##################################  CLASSES  ##################################
'''
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
                   
                self.X_ab_mod = np.sum( minimum( B,self.W_ab[:,J] ) )
                
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
        #J_list = np.argsort(np.array(T))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
        #J = J_list[0] # maximum activation
        J = np.argmax(np.array(T))
        X_ab = self.W_ab[:,J] 
        return X_ab
        
        
        
        
        for J in J_list:
            # Checking for resonance ---
            d = np.sum(minimum(I,self.W[J,:])) / np.sum(I)
            if d >= self.rho: # resonance occured
                return self.W[J,:], J
            return None, None
        
'''
   
class fuzzy_ARTMAP:
    
    def __init__(self, X_size, label_size, c_max_a, c_max_b, rho_a, rho_b, rho_ab, alpha=0.00001, beta=1):
        self.M_a = X_size    # input vector size
        self.M_b = label_size    # input label vector size
        
        self.c_max_a = c_max_a # max categories for ART-a
        self.c_max_b = c_max_b # max categories for ART-b
        
        self.rho_a = rho_a    # vigilance parameter for ART-a
        self.rho_a_baseline = rho_a
        self.rho_b = rho_b    # vigilance parameter for ART-b
        self.rho_ab = rho_ab  # vigilance parameter for map field
        self.alpha = alpha # choice parameter
        self.beta = beta   # learning rate
        
        self.N_a = 0         # no. of categories of ART-a initialized to zero
        self.N_b = 0         # no. of categories of ART-b initialized to zero
        
        self.W_a = np.ones( (c_max_a, self.M_a*2) ) # initialize W_a with 1s
        self.W_b = np.ones( (c_max_b, self.M_b) ) # initialize W_b with 1s
        self.W_ab = np.ones( (c_max_b, c_max_a) ) # initialize W_ab with 1s
    
    def train(self, X, one_hot_labels, rho_a_inc_rate=0.001):
        A = complement_code(X)   # shape of X = Mx1, shape of I = 2Mx1
        B = one_hot_labels
        #B = complement_code(one_hot_labels)
        
        self.rho_a =  self.rho_a_baseline
         
        T_a = [] # for ART-a
        T_b = [] # for ART-b
        
        for i in range(self.N_a): # calculate T at F2 in ART-a
            T_a.append( np.sum(minimum(A,self.W_a[i,:])) / (self.alpha+np.sum(self.W_a[i,:])) ) # calculate output
        J_list = np.argsort(np.array(T_a))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
        
        
        for J in J_list:
            # Checking for resonance in ART-a  ---
            X_a_mod = np.sum(minimum(A,self.W_a[J,:])) 
                
            while X_a_mod >= self.rho_a * np.sum(A): # while resonance occured in ART-a 
                T_b = []
                for i in range(self.N_b): # calculate T at F2 in ART-b
                    T_b.append( np.sum(minimum(B,self.W_b[i,:])) / (self.alpha+np.sum(self.W_b[i,:])) ) # calculate output
                #K_list = np.argsort(np.array(T_b))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
                T_b = np.array(T_b)
                Y_b = np.zeros_like( T_b )
                K = np.argmax(T_b)
                Y_b[K] = 1
                
                # Checking for resonance in ART-b  ---
                X_b_mod = np.sum(minimum(B,self.W_b[K,:]))
                
                if X_b_mod >= self.rho_b * np.sum(B):  # resonance occured in ART-b
                    X_ab_mod = np.sum( minimum( Y_b,self.W_ab[:,J] ) ) # calc. MAP FIELD state
                
                elif X_b_mod < self.rho_b * np.sum(B) and self.N_b < self.c_max_b: # NO resonance occured in ART-b
                    X_ab_mod = np.sum( self.W_ab[:,J]  ) # calc. MAP FIELD state
                    
                    n = self.N_b  # create new F2 category in ART-b
                    self.W_b[n,:] = self.beta*minimum(B,self.W_b[n,:]) + (1-self.beta)*self.W_b[n,:] # weight update
                    self.N_b += 1
                    
                elif self.N_b >= self.c_max_b:
                    print("ART-b memory error!")
                    return None, None
                    
                #####  match tracking  #####
                # Checking for resonance in the MAP FIELD  ---
                if X_ab_mod >= self.rho_ab * np.sum(B): # resonance occurs in the MAP FIELD
                    # weight update
                    self.W_a[J,:] = self.beta*minimum(A,self.W_a[J,:]) + (1-self.beta)*self.W_a[J,:] # weight update of ART-a
                    
                    K = np.argmax(T_b)
                    self.W_b[K,:] = self.beta*minimum(B,self.W_b[K,:]) + (1-self.beta)*self.W_b[K,:] # weight update of ART-b
                   
                    self.W_ab[:,J] = 0
                    self.W_ab[K,J] = 1
                    return self.W_ab, K
                
                else: # NO resonance in the MAP FIELD
                    self.rho_a += rho_a_inc_rate # slightly increase rho_a
       
            
        ############ 
        
        
        if self.N_a < self.c_max_a:    # NO resonance occured in ART-a, create a new category
            n = self.N_a   # create new F2 category in ART-a
            self.W_a[n,:] = self.beta*minimum(A,self.W_a[n,:]) + (1-self.beta)*self.W_a[n,:] # weight update
            self.N_a += 1
            
            
            # Checking for resonance in ART-b  ---
            res_B = False
            T_b = []
            for i in range(self.N_b): # calculate T at F2 in ART-b
                T_b.append( np.sum(minimum(B,self.W_b[i,:])) / (self.alpha+np.sum(self.W_b[i,:])) ) # calculate output
            #K_list = np.argsort(np.array(T_b))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
            T_b = np.array(T_b)
            Y_b = np.zeros_like( T_b )
            K_list = np.argsort(T_b)[::-1]
            
            for K in K_list:
                Y_b[K] = 1
                X_b_mod = np.sum(minimum(B,self.W_b[K,:]))
                
                if X_b_mod >= self.rho_b * np.sum(B):  # resonance occured in ART-b
                    res_B = True
                    X_ab_mod = np.sum( Y_b ) # calc. MAP FIELD state
                    
            if res_B is False and self.N_b < self.c_max_b: # NO resonance occured in ART-b
                X_ab_mod = 0
                
                n = self.N_b  # create new F2 category in ART-b
                self.W_b[n,:] = self.beta*minimum(B,self.W_b[n,:]) + (1-self.beta)*self.W_b[n,:] # weight update
                self.N_b += 1
                    
            elif self.N_b >= self.c_max_b:
                print("ART-b memory error!")
                return None, None
            
            # Checking for resonance in the MAP FIELD  ---
            if X_ab_mod >= self.rho_ab * np.sum(B): # resonance occurs in the MAP FIELD
                # weight update
                if T_b.shape[0] != 0:
                    K = np.argmax(T_b)
                else:
                    K = 0
                #self.W_b[K,:] = self.beta*minimum(B,self.W_b[K,:]) + (1-self.beta)*self.W_b[K,:] # weight update of ART-b
                J = self.N_a-1
                self.W_ab[:,J] = 0
                self.W_ab[K,J] = 1
                return self.W_ab, K


        
        if self.N_a >= self.c_max_a:
            print("ART-a memory error!")
            return None, None
        
        print("None trigerred!!!")
        return None, None
    
    
    
    def infer(self, X):
        A = complement_code(X)   
        T = []
        for i in range(self.N_a):           
            T.append( np.sum(minimum(A,self.W_a[i,:])) / (self.alpha+np.sum(self.W_a[i,:])) ) # calculate output        
        
        J = np.argmax(np.array(T))
        X_ab = self.W_ab[:,J] 
        return X_ab
