import numpy as np
import matplotlib.pyplot as plt
import utilities as utils           ## this module contains necessory items we will be using here



# I wanted to do some more expermint so I wrote a general  classifer.

# For the purpose of Assigment we will only use 'Softmax classifier' with L2 regularization.
class mySoftmax:
    """ This is my NN class which will define all the structure and funtionalities of my FCN. It has four methods :
        1. __init__   :  initialization of weights and biases
        2. forward    :  Perform forward pass of FCN
        3. loss       :  Calculate loss function after forward pass
        4. backward   :  Perform backward pass of FCN after loss calculation and calculate all the gradients of weights and biases
        5. update     :  Update the weights and biases of the FCN 
    """
    
    def __init__(self, layers , reg_lambda):
        
        """ 
        args :::::     layer = [input_features, hidden layer1 , hidden layer2 , ......, outputs]
                    ie. number of nodes in each layer to define a fully connected neural network architechture
        
        """
        self.reg_lambda = reg_lambda  # regularizing parameter
        self.m = 0

        # We are using only SOFTMAX CLASSIFIER so our layer will be in format ::: [ input_features, out_features]
        self.layers = layers


        self.weights, self.biases = utils.initialization(layers)     ## initialization of weights and biases in proper dimesnions
        self.forward_cache_list = []    ## this caches list will be used for saving the values during forward pass to
                                        ## reuse them during backward pass

        L = len(self.weights)
        # initializing momentum terms which will be used in update equations
        self.v = {}
        for l in range(L):
	        self.v["dW" + str(l+1)] = np.zeros(self.weights['W' + str(l+1)].shape)
	        self.v["db" + str(l+1)] = np.zeros(self.biases['b' + str(l+1)].shape)


        
        
    def forward(self, xdata):
        
        """ Performs forward pass for FCN of given architecture defined in __init__()
        
        args ::::       xdata is input matrix in shape of (D * m)
                            D = no. of input features
                            m = no. of training examples
        return ::::      aL  is the final output of softmax or sigmoid layer after forward pass
        """
        self.forward_cache_list = []   # some values which may be required during backpropogaation
        A = xdata                      # input data   ie. A0
        L = len(self.weights)          # L is total number of layers present in then FCNN
        
        for i in range(1, L):
            
            Aprev = A                  # output of previous layer after activation function
            #following function does forward pass upto second last layer using relu activation function
            A, cache = utils.forward(Aprev, self.weights["W" + str(i)], self.biases["b" + str(i)], "relu")
            self.forward_cache_list.append(cache)  # saving some items for later use in back prop
        
       
        # following will calculate forward pass for last layer using sigmoid / softmax (depending upon arguments)
        AL, cache = utils.forward(A, self.weights["W" + str(L)], self.biases["b" + str(L)], "softmax")
        self.forward_cache_list.append(cache)      # saving some items for later use in back prop
        
        return AL
    
    def loss(self, AL, ydata):
        
        """ computes cross entropy loss with L2 regularization between predicted probability scores (AL) and 
            ground truth labels(ydata)
            
        args ::::       AL    -   output of last sigmoid or softmax activation function in forward pass
                        ydata - ground truth labels
                     
        return ::::     cross entropy loss between AL 
            
            
            
        """
        
        m = ydata.shape[1]
        self.m = m

        #loss = -(np.dot(ydata, np.log(AL).T) + np.dot((1-ydata), np.log(1-AL).T)) / m
        cross_entropy_loss = -(np.sum(np.log(AL+1e-10) * ydata)) / m   # this is for softmax and above one is for sigmoid
        regularization_loss = 0

        for l in range(len(self.weights)):

        	regularization_loss += np.sum(np.square(self.weights['W' + str(l+1)])) 
        regularization_loss *= self.reg_lambda /(2*m)
        
        return np.float(cross_entropy_loss+regularization_loss)
    
    
    def backward(self, AL, ydata ):
            """ This method will perform backpropagation and calculate the gradients of all weights and biases
            
            args ::::    AL    - output of the softmax/sigmoid layer
                         ydata - groundtruth labels
            
            return ::::  gradients - dictionary containing all the gradients
            """
            
            gradients = {}                     # taking and empty dictionary to save gradients later
            L = len(self.weights)              # Total layers in FCNN
            m = ydata.shape[1]                 # Total number of samples in datasets
            ydata = ydata.reshape(AL.shape)    # making sure that they have same shape
            
            
            
            #dAL = -(ydata / AL) + ((1-ydata) / (1-AL))  # initialization of loss gradient wrt out for sigmoid at last layer
            dAL = (AL - ydata)                           # same initialization but for softmax at last layer
            

            # for Lth layer the back propogation will be diffent as others have relu activation function and this
            # one will have either softmax or sigmoid.
            
            present_cache = self.forward_cache_list[L-1]  ## count starts from 0 so Lth layer cache is in L-1 index
            
            # below function will weight and bias gradients and previous layer gradient input and well will
            # save them in 'gradients' dictionary
            temp1, temp2, temp3  = utils.backward(dAL, present_cache, 'softmax')
            gradients["dA" + str(L-1)] = temp1
            gradients["dW" + str(L)] = temp2
            gradients["db" + str(L)] = temp3                                   
            
            
            # now remaining layer all have relu action so we can iterate over a loop to do the same procdure as above
            
            # loop from L-2 to 0 in index of forward_cache_list .... in layers it is from L-1 to 1
            for i in reversed(range(L-1)):
                present_cache = self.forward_cache_list[i]
                temp1, temp2, temp3 = utils.backward(gradients["dA" + str(i+1)], present_cache, 'relu')
                # saving up all the gradients in 'gradients' dictionary
                gradients["dA" + str(i)] = temp1
                gradients["dW" + str(i+1)] = temp2
                gradients["db" + str(i+1)] = temp3
            return gradients
    
    def update(self, gradients, lr, beta=0.9):
        
        """ this method performs weigth step using momentumfor given gradients learning rate and beta
        
        agrs :::    gradients -  a dictionary which contains all the gradients(dAprev, dW, db)
        
        returns ::: lr - learning rate
                ::: beta - momentum term mulitplier
        """
        
        #  WE UPDATE WEIGTS USING L2 REGULARIZATION
        L = len(self.weights)
        
        for l in range(L):
        
	        # computing velocities
	        self.v["dW" + str(l+1)] = beta * self.v["dW" + str(l+1)] + (1-beta) * gradients['dW' + str(l+1)]
	        self.v["db" + str(l+1)] = beta * self.v["db" + str(l+1)] + (1-beta) * gradients['db' + str(l+1)]
	        
	        # update parameters using momentum and regularization 'reg_lambda'

	        constant = 1 - (lr * self.reg_lambda)/ self.m

	        self.weights["W" + str(l+1)] = constant * self.weights["W" + str(l+1)] - lr * self.v['dW' + str(l+1)] 
	        self.biases["b" + str(l+1)] -=  lr * self.v['db' + str(l+1)] # we don't regularize biases
