from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """
    
    #IN ALL THE FUNCTIONS TO CALCULATE THE L2 DISTANCE WE ONLY SEND THE DATA FOR WHICH WE NEED TO PREDICT THE LABELS

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))                
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i , :] = np.sqrt(np.sum(np.square(X[i] - self.X_train),axis=1)) #axis =1 means in downward direction
            pass
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dists = np.sqrt(-2*np.dot(X,self.X_train.T) + np.sum(self.X_train**2,axis=1) + np.sum(X**2,axis=1)[:,np.newaxis])
        pass
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            x = np.argsort(dists[i])
            closek = []
            for mmk in range(k):  
                dekh = x[mmk]
                closek.append(self.y_train[dekh])
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            counter = [0]*len(closek)
            for p in range(len(closek)):
              for q in range(len(closek)):
                if closek[p] == closek[q]:
                  counter[p] = counter[p] + 1
            maxi = 0
            for z in range(len(counter)):
              if counter[z] > maxi:
                maxi = z
            y_pred[i] = closek[maxi]
            pass
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred
