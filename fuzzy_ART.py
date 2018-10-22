import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class fuzzy_ART:
    
    def __init__(self, X_size, c_max, rho, alpha=0.00001, beta=1):
        self.M = X_size    # input vector size
        self.c_max = c_max # max categories
        self.rho = rho     # vigilance parameter
        self.alpha = alpha # choice parameter
        self.beta = beta   # learning rate
        
        self.N = 0 
        self.W = np.ones( (c_max, self.M*2) )
    
    def train(self, X):
        I = complement_code(X)   # shape = Mx1
        
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
    
    def infer(self, X):
        I = complement_code(X)   # shape = Mx1
        T = []
        for i in range(self.N):           
            T.append( np.sum(minimum(I,self.W[i,:])) / (self.alpha+np.sum(self.W[i,:])) ) # calculate output        
        J_list = np.argsort(np.array(T))[::-1]  # J_list: indices of F2 nodes with decreasing order of activations        
        for J in J_list:
            # Checking for resonance ---
            d = np.sum(minimum(I,self.W[J,:])) / np.sum(I)
            if d >= self.rho: # resonance occured
                return self.W[J,:], J
 
            
def minimum(A,B):
    ans=[]
    for i in range(len(A)):
        ans.append(min([A[i],B[i]]))
        
    ans = np.array(ans)
    return ans
    
def complement_code(X):
        I = np.hstack((X, 1-X))
        return I
   
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


###############################################################################
if __name__ == '__main__':

    np.random.seed(1)
    
    data = get_data()
    
    '''
    #data check
    for k in range(0,10):
        digit = data[str(k)][:, 1:]
        for i in range(digit.shape[0]):              
            sample = digit[i].reshape(28,28)
            plt.imshow(sample)
            plt.show()
    '''
    
    rho = 0.72 
    model = fuzzy_ART(28*28, c_max=100, rho=rho, alpha=0.00001, beta=1)
    
    for k in range(0,10):
        
        data_digit = data[str(k)]  # images of one digit at a time
        
        for i in range(data_digit.shape[0]):  
            Z, k = model.train(data_digit[i])
            
            plt.imshow(Z[:784].reshape(28,28)) #display learned expectations
            #plt.imshow(data_digit[i].reshape(28,28))
            plt.show()
            print("CLASS: ", k)
   
    #Inference
    digit = 9
    sample_no = 2
    sample = data[str(digit)][sample_no]
    plt.imshow(sample[:784].reshape(28,28))
    plt.show()            
    print("Inferred category: ", model.infer(sample)[1] )
    