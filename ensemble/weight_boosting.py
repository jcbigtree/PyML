"""Weight boosting classifiers. 
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 

from mlnp.base import clone
import numbers
import numpy as np
import copy 
import sys

__all__=[
    "BaseWeightBoosting", "AdaBoostClassifier"
]


EPSILON = 1e-10

class BaseWeightBoosting(object):
    """Base class for Weight Boosting. Implement based on sklearn. 
   
    Warning: This class should not be used directly. Use derived class instead.
    
    Parameters:
    -----------
    base_learner.  object 
        At each iteration, a base learner is trained on the weighted samples. In the end, 
        the weighted combination of all the base learner forms the final strong classifier.              
            
    n_learners. int, default 100, optional
        Number of the base learners. Note that if some kind of early-stopping strategies are 
        applied, the actual number of learned base learners can be less than n_learners.
        
    Attributes:
    -----------
    base_learners_: list of object. 
        Learned base learners.
        
    learner_weights_: List of float.
        Weights for combining base learners.
        
    """
    def __init__(self, 
                 base_learner=None,
                 learner_params=tuple(),  # Params passed to base learners
                 n_learners=100, 
                 verbose=False):
        
        #
        # Check sanity                
        #
        if not base_learner or not isinstance(base_learner, object):
            raise ValueError("base_learner must be of the type object")      
        
        if not (isinstance(n_learners,(numbers.Number, np.number)) 
                and n_learners == int(n_learners) 
                and n_learners > 1):
            raise ValueError("n_learners must be an integer and bigger than 1.0")
        
        if not isinstance(verbose, (bool)):
            raise ValueError("verbose must be of the type bool")           
       

        #         
        # Parameters
        #
        self.base_learner=base_learner
        self.n_learners=n_learners
        self.learner_params=learner_params        
        self.verbose=verbose  
                
        self.n_samples = 0   
        self.n_features = 0
        self.iterator = 0       
        self.early_stopping= False
        self.sample_weights = []
        
        #
        # Attributes
        #
        self.base_learners_ = []
        self.learner_weights_ = []        
        
        #
        # Disabled attributes. Shall not be used. Define similar variables in the derived class
        #
        self.staged_exponential_loss_ = [] 
        self.staged_misclassification_rate_ = []    

       
    def _make_learner(self, append = True):
        """Copy and properly initialize a base learner."""
        learner = clone(self.base_learner)
        if append:
            self.base_learners_.append(learner)            
        
        return learner


    def _check_stop_criteria(self):
        """Check all the stop criteria, return True if anyone of them is satified, 
            otherwise return False"""
        return self.early_stopping
    
    
    def _update_sample_weights(self):
        """Update samples' weights. Must be overridden in derived classes."""               
        pass 
        

    def fit(self, X, y):
        """Train on a train set (X,y)
        
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]
            Each row is a sample.
            
        y. array-like shape [n_samples]
            Corresponding label of samples.        
        Returns:
        --------
        self. object.
            return the object itself.
        """
        # Params check
        if not isinstance(X, (np.ndarray, np.generic)) or \
           not isinstance(y, (np.ndarray, np.generic)):
            raise ValueError("Numpy array is the only acceptable format.")
        
        if len(X.shape) != 2: 
            raise ValueError("X must be of shape [n_samples, n_features]")
        
        if len(y.shape) != 1:
            raise ValueError("Y must be of shape [n_samples]")
                
        
        # Initialize sample weights uniformly.
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.sample_weights = np.array([1.0/self.n_samples]*self.n_samples)    
        self.X = X
        self.y = y
        
        
        # Main loop
        self.iterator = 0
        self.early_stopping = False
        while self.iterator < self.n_learners:            
            # Train a new weak learner on the weighted samples.
            learner = self._make_learner()                 

            bl = learner.fit(X,y,sample_weight=self.sample_weights)           
            self.h_y = bl.predict(X)  # base learner response
                      
            # Update sample weights            
            self._update_sample_weights()
           
            # Check if stop criteria is satisfied.
            if self._check_stop_criteria():
                break         
            self.iterator += 1   
            
            # Print intermediate information if desired 
            if self.verbose:
                print 'Boosting ... Iteration ', self.iterator - 1
                    
        self.n_learners = len(self.base_learners_)
        return self


    def predict(self, X):
        """Predict y given a set of samples X.   
            
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]
            
        Returns:
        --------
        y. array-like shape [n_samples]                
        """
        y_sum = 0
        for k in range(len(self.base_learners_)):
            y_sum += self.base_learners_[k].predict(X)*self.learner_weights_[k]
        return np.sign(y_sum)                    
    
    
    def staged_predict(self, X):
        """Predict y given a set of samples X.   
            
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]
            
        Returns:
        --------
        Y. array-like shape [n_samples, n_learners]                
        """
        Y = []
        y_sum = 0
        for k in range(len(self.base_learners_)):
            y_sum += self.base_learners_[k].predict(X)*self.learner_weights_[k]
            Y.append(np.sign(y_sum))
        return Y
                
                
    def score(self, X, y):
        """Calculate the predict error given samples and targets.
        
        Parameters:
        -----------
        X. array-like. shape [n_samples, n_features]            
        y. array-like shape [n_samples]

        Returns:
        --------
        score. float.     
            Successful rate
        """
        y_p = self.predict(X)       
                
        # compute the successful rate        
        successful_rate = 1 - np.sum(np.abs(y_p - y)*0.5)/X.shape[0]
        return successful_rate


    def _compute_loss(self, X, y):        
        y_p = self.predict(X)
        
        # Exponential loss
        exp_loss = np.sum(np.exp(-y_p*y))
        self.staged_exponential_loss_.append(exp_loss)
        
        # Mis-classification rate
        miscls_rate = np.sum(np.abs(y_p - y)*0.5)/X.shape[0]
        self.staged_misclassification_rate_.append(miscls_rate)
         

##################################################################################################
class AdaBoostClassifier(BaseWeightBoosting):
    """Adaboost for classification.
            
    Parameters:
    -----------
    base_learner.  object 
        At each iteration, a base learner is trained on the weighted samples. In the end, 
        the weighted combination of all the base learner forms the final strong classifier.              
            
    n_learners. int, optional (default=100)
        Number of the base learners.
    
    Reference:
    ---------
    Yoav Freund and Robert E. Schapire. 1997. A decision-theoretic generalization of on-line 
    learning and an application to boosting. J. Comput. Syst. Sci. 55, 1 (August 1997), 119-139.
    """
    def __init__(self,
                 base_learner=None,
                 learner_params=tuple(), 
                 n_learners=100,
                 learning_rate=1.0,
                 verbose=False):        
        super(AdaBoostClassifier, self).__init__(base_learner, 
                                                 learner_params, 
                                                 n_learners, 
                                                 verbose)        
       
        self.learning_rate = learning_rate 
        self.alpha = 0              
        self.predict_error = []        
        self.weighted_errors = []
        
        # 
        # Response of all base learners on the samples. 
        #         x0,      x1, ... ,  xn-1 
        #  h_0    
        #  h_1
        #  ...
        #  h_T-1 
        #    
        self.base_learners_response = []
        
        # 
        # Sample weights for all the iterations 
        #         x0,   x1, ... ,  xn-1 
        #  f_0   
        #  f_1
        #  ...
        #  f_T-1 
        # 
        self.staged_sample_weights = []


    def _check_stop_criteria(self):
        """Overridden. Check all the stop criteria, return True if anyone of them is satified, 
        otherwise return False"""
        early_stopping = False
        # Return true, if no improvement any more. 
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        if np.sum(error) < EPSILON:
            early_stopping = True
        
        return early_stopping or super(AdaBoostClassifier, self)._check_stop_criteria()


    def _update_sample_weights(self):
        """Overridden. Update samples' weight."""               
        #
        # Compute weighted error of the latest base learner
        #
        error = np.abs(self.h_y - self.y) * 0.5   # 1, if incorrect. 0, if correct.
        weighted_error = np.inner(error, self.sample_weights)
        
        #
        # Compute base learners' weights
        #
        learner_weight = 0.5 * np.log(
            (1.0 - weighted_error + EPSILON) / (weighted_error + EPSILON)
        )        

        # 
        # Compute sample weights for next iteration
        #
        self.sample_weights *= np.exp(learner_weight * (error - 0.5) * 2.0 * self.learning_rate)
        self.sample_weights /= np.sum(self.sample_weights)
                    
        #
        # Store the changes
        #
        self.learner_weights_.append(learner_weight)
        self.weighted_errors.append(weighted_error)

        self.base_learners_response.append(np.copy(self.h_y))
        self.staged_sample_weights.append(np.copy(self.sample_weights))




  

 

