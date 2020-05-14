from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


""" SVM support vector machine is linear classifier that is it basically just draws a line between what classfies as A and what classifies as B.
We have Weight matrix that we initialize randomly and optimize it continuously to increase our accuracy and we are usiing gradient descent here to find 
the derivative of the loss function wrt to the weight and then add this gradient *step size(learning rate) and optimise our W so that each row of W vaguely
represents the template of the correct the class i.e no of rows in W = no.of classes that exist."""

"""what I am not able to understand the gradient and the loss function and what is the dW optimisation """

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero same shape as W
    # compute the loss and the gradient
    #print("shape of W which is are weight matrix: ",W.shape)
    #print("shape of X which contains the minibatch of data:",X.shape)
    num_classes = W.shape[1]  #because W is 3073*10 no of classes =10
    num_train = X.shape[0]  #no of training examples 500*3073 #we are using minibatch SGD stochastic gradient descent
    loss = 0.0
    for i in range(num_train): #for each training example ie each row ;we want to find the score and then the loss for all the classes
        scores = X[i].dot(W) #the scores for the first image #the scores array will be an array with 10 scoress
        correct_class_score = scores[y[i]] #y[i] contains the correct label therefore score[y[i]] gives the score of the correct class label
        for j in range(num_classes): #looping over the score for each class there are 10 classes therefore
            if j == y[i]:
                continue
            #margin here is nothing but the loss function    
            margin = scores[j] - correct_class_score + 1 # note delta = 1 ; calculating the loss
            if margin > 0: #implementation of the condition of svm loss
                # we are implementing the formula we get after we take the derivative of the loss function wrt to the weight
            	dW[:, y[i]] -= X[i, :] #update of the class which is the correct class label 
            	dW[:, j] += X[i, :] #update the class which we are currently finding the loss value for
            	loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train #average over all the training examples when we use W as the weight matrix.
    #dW.transpose()
    # Add regularization to the loss.
    dW /= num_train
    # Add regularization to the loss.
    #loss += 0.5 * np.sum(W * W)
    dW += reg* W #regualarization term
    loss += reg * np.sum(W * W)  #the regularization term is lambda*(double summation of W**2)
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0] 
    #scores = X.dot(W)
    #correct_class_score = scores[np.arange(num_train), y]
    #print(correct_class_score.shape)
    scores = X.dot(W)
    #print(x=np.array ([x0,]*n)scores.shape) 500 * 10 for each example 10 scores
    """correct_answers = y
    for i in range(500):
    	k = y[i]
    	correct_answers[i] = X[i][k]
    correct_answers = np.repeat(correct_answers,10,axis=0)
    correct_answers = np.reshape(correct_answers,(500,10))	
    #(scores.transpose() - y).transpose()
    #for broadcast remember that the trailing dimension must be the same
    print(correct_answers)
    margin = np.maximum(0,scores - correct_answers + 1)	
    loss = np.sum(margin)
    print("loaaass")"""
    #scores = np.sum(scores , axis =1)	
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #loss /= num_train
    #loss += reg * np.sum(W * W)  #the regularization term is lambda*(double summation of W**2)    
    #pass
    #"""
    dW = np.zeros(W.shape) 
    num_train = X.shape[0]
    delta = 1.0
    scores = X.dot(W) #500*10
    #y is 500*1
    correct_class_score = scores[np.arange(num_train), y]
    #so here the values np.arange(num_train),y act as 
    #print(y)
    #essentially it takes correct class and makes it a 2d matrix with each row having the value of the correct class to match the dimensions
    #scores 500*10 correct class scores = 500*1
    margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + delta)
    #we make the value of loss of the corect class = 0 else it will have value of delta
    #np.arange(num_train) ,y give the co-ordinate of the correct class of the ith image
    margins[np.arange(num_train), y] = 0
    print(margins.shape)
    #print(dW[:,y])
    #print(X.shape)
    """ tranx = X.T
    dW[:,y] -= 20*tranx[:,np.arange(num_train)]
    dW[:,y] += tranx[:,np.arange(num_train)]
	"""
    loss = np.sum(margins)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W.T.dot(W))
    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
    count = np.sum(X_mask, axis=1)
    X_mask[np.arange(num_train), y] = -count
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += np.multiply(W, reg)
   
	
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
	#ss"""
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
