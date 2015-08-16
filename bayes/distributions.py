"""All kinds of probability distributions
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 

import numpy as np



__all__=[ "Gaussian", "IsotropicGaussian" ]


###################################################################################################
class Gaussian(object):
    """Multivariate Guassian distribution. Default is 1D standard normal distribution N(0,1).
      
    Parameters: 
    -----------
    mean : 1D array-like of length N. default, 0.0
        Mean of N-dimensional Guassian distribution
    
    cov : 2D array-like of shape (N, N) . default, 1.0
        Covariance matrix. Must be symmetric and positive-semidefinite.
    """
    def __init__(self,
                 mean = np.array([0.0]), 
                 cov = np.array([1.0])):
        #Check parameters 
        if not (isinstance(mean, np.ndarray) and len(mean.shape) == 1):   
            raise ValueError("mean must be 1D array.")
        
        if not (isinstance(cov, np.ndarray) and     # array
                len(cov.shape) == 2 and             # 2D
                cov.shape[0] == cov.shape[1]):      # symmetric, ignore 'positive semi-definite'
            raise ValueError("cov must be a 2D symmetric array.")
        
        if mean.shape[0] != cov.shape[0]: # mean and cov should be the same
            raise ValueError("Dimension of mean and cov is not consistent.")
        
        self.sample_weight = []
        self.n_classes = None 
        self.n_features = None
        self.n_samples = None

        # Attributes
        self.mean_ = mean 
        self.cov_ = cov 

        #U,s,V = np.linalg.svd(self.cov_)
        #import matplotlib.pyplot as plt
        #plt.plot(s)
        #plt.show()

        # Internal variables for efficience.
        self.cov_det = np.linalg.det(self.cov_) 
        self.cov_inv = np.linalg.inv(self.cov_)
        
            
    def _diagonal_inv(self, cov):
        """Invert a diagonal matrix. 
        
        Warning: This method should only be used internally. Parameter-checking is ignored.
        
        Params:
        -------
        cov : array-like. of shape [N, N]
            Covariance matrix(diagnoal).
        
        Return:
        -------
        cov_inv : array-like. of shape [N, N]
            Inverse of the input matrix.
        """    
        epsilon = 1e-10
        cov_diag = np.diag(cov) + epsilon  # Regulization to avoid dividing-by-zero
        cov_inv = np.eye(cov.shape[0], cov.shape[1], np.float)
        np.fill_diagonal(cov_inv, 1.0/cov_diag)
        return cov_inv
        
        
    def pdf(self, X):        
        """Compute the PDF of samples. 
        
        Warning: If X is very large, Memory-error might be raised.
                 
        Parameters:
        ----------
        X : array of shape [n_samples, n_features]
            Each row represents a sample. 
            
        Return:
        -------
        PDF of samples
        """        
        # Warning: If X is very large, Memory-error might be raised.         
        X_bar = np.asmatrix(X-self.mean_)
        md = X_bar * np.asmatrix(self.cov_inv) * np.transpose(X_bar)
        #xx = np.dot(np.dot(X-self.mean_, self.cov_inv), np.transpose(X-self.mean_))
        dmt = np.sqrt(self.cov_det)*np.power(2*np.pi,0.5*self.cov_.shape[0])        
        return np.exp(-0.5*md.diagonal())/dmt
    
           
    def log_pdf(self, X):
        """Compute the logrithm of the PDF of samples. 
        
        Warning: If X is very large, Memory-error might be raised.
                 
        Parameters:
        ----------
        X : array of shape [n_samples, n_features]
            Each row represents a sample. 
            
        Return:
        -------
        Logrithm of the PDF of samples
        """        
        return np.log(self.pdf(X))
        
   
###################################################################################################
class IsotropicGaussian(Gaussian):
    """Isotropic Gaussian distribution. 
            
    covariance magtrix - diag[sigma_0, sigma_1, ..., sigma_n] 
    """
    def __init__(self, 
                 mean = np.array([0.0]), 
                 cov = np.array([1.0])):
        """
        Parameters: 
        -----------
        mean : 1D array-like of length N. default, 0.0
            Mean of N-dimensional Guassian distribution
        
        cov : 2D array-like of shape (N, N) . default, 1.0
            Covariance matrix. Must be symmetric and positive-semidefinite.
        """
        #super(IsotropicGaussian, self).__init__(mean, cov)
    
        if not (isinstance(mean, np.ndarray) and len(mean.shape) == 1):   
            raise ValueError("mean must be 1D array.")
        
        if not (isinstance(cov, np.ndarray) and     # array
                len(cov.shape) == 2 and             # 2D
                cov.shape[0] == cov.shape[1]):      # symmetric, ignore 'positive semi-definite'
            raise ValueError("cov must be a 2D symmetric array.")
        
        if mean.shape[0] != cov.shape[0]: # mean and cov should be the same
            raise ValueError("Dimension of mean and cov is not consistent.")
        
        self.sample_weight = []
        self.n_classes = None 
        self.n_features = None
        self.n_samples = None    
        
        # Attributes
        self.mean_ = mean 
        self.cov_ = cov 
        
        # Internal variables for efficience.
        self.cov_det = np.linalg.det(self.cov_) 
        self.cov_inv = self._diagonal_inv(self.cov_)

        
    def log_pdf(self, X):
        """Compute the log-pdf of samples. 
                 
        Parameters:
        ----------
        X : array of shape [n_samples, n_features]
            Each row represents a sample. 
            
        Return:
        -------
        log-pdf of samples
        """        
        X_bar = X-self.mean_        
        log_denom = -0.5* np.log(2*np.pi*self.cov_.diagonal())
        log_pdf = -0.5 * np.power(X_bar,2) * self.cov_inv.diagonal()
        return np.sum(log_pdf + log_denom, axis=1)

   
