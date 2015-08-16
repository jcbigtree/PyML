"""Naive Bayes. Apply Bayes' theorem with strong assumption that features are independent.  
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 3 clause

from mlnp.base import BaseLearner
from distributions import IsotropicGaussian
import numpy as np

   
###################################################################################################
class GaussianNB(BaseLearner):
    """Gaussian Naive Bayes classifier.
    
    Parameters:
    -----------
    class_prior_type : str 'equal' or 'estimated'
        If 'equal', use the same prior for all classes. 
        If 'estimated', (prior for a given class) = (number of samples in the class)
                                                    /(total number of samples)    
                                                    
    Attributes:
    -----------
    n_classes_ : int
        Number of classes
        
    n_features_ : int
        Number of features
    """
    def __init__(self,
                 class_prior_type='equal'):
        
        if not class_prior_type in ('equal','estimated'):
            raise ValueError('class_prior should be either \'equal\' or \'estimated\' ')    
        
        self.class_prior_type = class_prior_type
        self.class_labels = [] 
        self.guassians = []
        self.class_prior = []
        
        # Attributes
        self.n_classes__ = 0
        self.n_features__ = 0
        
       
    def fit(self, X, y, sample_weight=[]):        
        """Train on a train set (X,y).  
        Warning: Should be called by derived classes unless they have their own sanity check.
        
        Parameters:
        -----------
        X : array-like. shape [n_samples, n_features]
            Each row is a sample.
            
        y : array-like shape [n_samples]
            Corresponding label of samples.        
        
        Returns:
        --------
        self : object.
            return the object itself.
        """
        #
        # call base learner first
        #
        super(GaussianNB, self).fit(X,y,sample_weight)
        
        #        
        # Reset all the variables to empty
        #
        self.class_labels = []
        self.class_prior = []
        self.guassians = []
   
        #     
        # then do our own work        
        #
        self.class_labels = list(set(y))        
        self.n_classes_ = len(self.class_labels)
        self.n_features_ = X.shape[1]        
        self.n_samples = X.shape[0]
        
        for c in self.class_labels:
            X_c = X[np.where(y == c)]
            weights_c = self.sample_weight[np.where(y == c)]  
            weights_c /= np.sum(weights_c)
            self.class_prior.append(X_c.shape[0])
            
            #
            # weighted mean and variance
            #
            mean_c = np.sum(X_c*weights_c, axis = 0)
            var_c = np.sum(np.power(X_c - mean_c,2) * weights_c, axis = 0)
                            
            cov_c = np.eye(self.n_features_, dtype = np.float64)
            np.fill_diagonal(cov_c, var_c) 
            gau_c = IsotropicGaussian(mean = mean_c, cov = cov_c)
            self.guassians.append(gau_c)
            
        # normalize prior 
        self.class_prior = np.array(self.class_prior, dtype=float)
        self.class_prior /= np.sum(self.class_prior)
        
        return self


    def predict(self, X):
        """Predict y given a set of samples X.   
            
        Params:
        -------
        X : array-like. shape [n_samples, n_features]
            
        Returns:
        --------
        y : array-like shape [n_samples]                
        """
        log_post_probs = []
        for i in range(self.n_classes_):
            if self.class_prior_type == 'equal':
                log_post_probs.append(self.guassians[i].log_pdf(X) + np.log(1.0/self.n_classes_))
            elif self.class_prior_type == 'estimated':                        
                log_post_probs.append(self.guassians[i].log_pdf(X) + np.log(self.class_prior[i]))             
        
        log_post_probs = np.array(log_post_probs)
        predicted_class_index = np.argmax(log_post_probs, axis=0)
        predicted_class = np.zeros_like(predicted_class_index, dtype=np.int)
        for i in range(self.n_classes_):
            predicted_class[np.where(predicted_class_index == i)] = self.class_labels[i]
       
        #post_prob = post_probs[predicted_class]
        return predicted_class
   


