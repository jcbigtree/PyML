"""Mixture models. 
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 3 clause

from mlnp.base import BaseLearner
from distributions import Gaussian
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np


class GaussianMixture(object):
    """Gaussian mixture model optimized by Expectation Mixmization

    Parameters:
    -----------
    n_components: int optional default 1
        Number of Gaussians

    cov_type: str optional default 'diag'. Must be 'diag', or 'full'
        Type of covariance matrix

    max_iters: int optional default  10
        Maximum iterations

    tolerance: float optional default 1e-6
        If improvement of the loss function is smaller than tolerance during two consecutive 
        iterations, stop the iteration

    opt_algo: ["EM"], optional default "EM"
        Optimization algorithm.
       
    verbose: bool optional default False
        If True, print details during iterations
    """
    def __init__(self,
                 n_components=1,
                 cov_type='diag', 
                 max_iters=10,
                 tolerance=1e-6,
                 opt_algo="EM",
                 verbose=False):                
        
        self.components = []                 # Gaussians go here
        self.coef = []
        self.n_components = n_components
        self.cov_type = cov_type.lower()
        self.opt_algo = "EM"
        self.max_iters = max_iters
        self.neg_log_likelihood = None        
        self.staged_loss = []
        self.tolerance = tolerance
        self.verbose = verbose

        self.sigma_epsilon = 1E-10 # A small constant added to sigma to avoid singularity
        self.sigma_corr_factor = 0.1
        
        self.dege_tole = 1E-5 # Degenerate tolerance
        
        #
        # Attributes 
        #
        self.components_ = None
        self.coef_ = None 
        self.n_components_ = None
        

    def __str__(self):
        """Return the details of the mixture model"""
        out_str = "Gaussian Mixtures \n" \
                + " --- Number of components: " + str(self.n_components) + "\n"   \
                + " --- Coef: " + str(self.coef) + "\n"   


        for i in range(self.n_components):
            out_str += " --- Mean: " + str(self.components[i].mean_)  + "\n" 
            out_str += " --- Cov: " + str(self.components[i].cov_)  + "\n"             

        return out_str

        
    def fit(self, X, sample_weight=[]):        
        """Fit a mixture of Gaussian model on X
        
        Parameters:
        -----------
        X : ndarray [n_samples, n_features]
            Each row is a sample.

        Returns:
        --------
        self : object.
            return the object itself.
        """
        #
        # Use uniform weights over samples if sample weights are not specified.
        #
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        if sample_weight == []:
            self.sample_weight = np.ones((self.n_samples, 1), dtype=np.float64)/self.n_samples
            
        elif len(sample_weight.shape) == 1:  
            self.sample_weight = np.reshape(sample_weight, (self.n_samples, 1))
            self.sample_weight.astype(np.float64) 
        
        #
        # Run KMeans to find initial centroids and radiuses 
        #        
        #whitened_samples = whiten(X)
        whitened_samples = X
        centroids, distortion = kmeans(whitened_samples, self.n_components)
        indexes, distortion = vq(whitened_samples, centroids)

        if self.n_components != centroids.shape[0]:
            print 'KMean wasn\'t able to find as many clusters as you wanted'               
            self.n_components = centroids.shape[0] 

        coef = np.zeros((self.n_components,), np.float64)
        
        #
        # Update mixing coefficients, mean and covariance
        #
        cim = np.zeros((self.n_samples,), np.float64) 
        for i in range(self.n_samples):
                cim[i] = np.argmin(np.sum(np.power(X[i,:] - centroids,2), axis=1))

        mus = []
        sigmas = []
        for k in range(self.n_components):
            Xk = X[np.where(cim==k)[0],:]
            coef[k] = Xk.shape[0]/float(self.n_samples)
            Xkb = Xk - centroids[k,:]
            sigmak = np.dot(np.transpose(Xkb), Xkb)/float(self.n_samples-1)
            if self.cov_type == 'diag':
                sigmak = np.diag(np.diag(sigmak))

            #
            # Check the component k is degenerate
            #
            # Quick check
            if np.mean(np.diag(sigmak)) >= self.dege_tole:
                mus.append(centroids[k,:])

                var=np.diag(sigmak)
                mvar = np.sum(var)/len(np.where(var>0)[0])
                sigmak = sigmak * (1-self.sigma_corr_factor) + \
                         self.sigma_corr_factor*mvar*np.eye(sigmak.shape[0], dtype=np.float64)  

            #sigmak += self.sigma_epsilon*np.eye(sigmak.shape[0], dtype=np.float64)              
                sigmas.append(sigmak)
                
        self.components = self._pack_gaussian(mus, sigmas)    
        self.n_components = len(mus)
        self.coef = coef    
        self.staged_loss.append(self._loss(X))
        
        #
        # Main loop: Expectation Maximization
        #
        resp_mat = np.zeros((self.n_samples, self.n_components), np.float64) 
        for iter in range(self.max_iters):

            #
            # E step. Compute responsibility. 
            #
            for k in range(self.n_components):
                resp_mat[:,k] = self.components[k].pdf(X) * self.coef[k]
            resp_mat /= np.reshape(np.sum(resp_mat, axis=1),(self.n_samples,1))
        
            #
            # M step. Re-estimate parameters
            #
            mus = []
            sigmas = []
            NK = np.sum(resp_mat, axis=0)
            self.coef = NK/float(self.n_samples)
            for k in range(self.n_components):     
                r = np.reshape(resp_mat[:,k],(self.n_samples,1)) /NK[k] * self.sample_weight
                r /= np.sum(r)                
                muk = np.sum(r*X, axis=0)
                sigmak = 0.0
                for n in range(self.n_samples):
                    xkb = np.asmatrix(X[n,:] - muk)
                    sigmak += r[n,0]*np.transpose(xkb)*xkb

                #
                # Check if the component k is degenerate
                #
                # Quick check
                if np.mean(np.diag(sigmak)) < self.dege_tole:
                    self._del_component(k)
                    k -= 1
                    continue   
                
                #
                # Correct sigmak
                #
                var=np.diag(sigmak)
                mvar = np.sum(var)/len(np.where(var>0)[0])
                sigmak = sigmak * (1-self.sigma_corr_factor) + \
                         self.sigma_corr_factor*mvar*np.eye(sigmak.shape[0], dtype=np.float64)  

                if self.cov_type == 'diag':
                    sigmak = np.diag(np.diag(sigmak))

                mus.append(muk)
                sigmas.append(sigmak)

            self.components = self._pack_gaussian(mus, sigmas)
            self.staged_loss.append(self._loss(X))
            
            if self.verbose:
                print "Iteration: ", iter, "  Negative-Log-likelihood: ", self.staged_loss[-1]

            #
            # Early stopping if the improvement is tiny.
            #
            if np.abs((self.staged_loss[-1] - self.staged_loss[-2])) < self.tolerance:
                break
                                    
                  
        self.components_ = self.components
        self.coef_ = self.coef
        self.n_components_ = self.n_components

        #
        # Remove very small components
        #
        #self._prune()

        return self        


    def log_likelihood(self,X):
        """Compute the logrithm likelihood of given samples X

        Parameters:
        -----------
        X : ndarray [n_samples, n_features]
            Each row is a sample.

        Returns:
        --------
        ndarray [n_sample,]
        """
        # 
        # Sanity check
        #
        if len(self.components) <= 0: 
            raise ValueError("There are no valid components.")
        
        if not hasattr(self.components[0], 'pdf'):
            raise ValueError("The component model doesn't have \' pdf \' function."
                             + " I can't proceed.")

        val = 1E-10
        for i in range(self.n_components):
            val += self.coef[i]*self.components[i].pdf(X)
        return np.asarray(np.log(val))
    

    def _pack_gaussian(self, mus, sigmas):
        """Update"""
        components = []
        for i in range(len(mus)):
            components.append(Gaussian(mean=mus[i], cov=sigmas[i]))
        return components

    
    def _loss(self, X):
        """Loss function"""
        return -np.sum(self.log_likelihood(X))
    

    def _prune(self):
        """Remove extremely small components"""
        pruned = []
        pruned_coef = []
        try:
            for i in range(self.n_components):
                U,s,V = np.linalg.svd(self.components[i].cov_)                
                if np.min(s) > 2.0*self.sigma_epsilon:
                    pruned.append(self.components[i])
                    pruned_coef.append(self.coef[i])
        except np.linalg.LinAlgError:
            return

        self.components = pruned
        self.coef = np.array(pruned_coef)                           
        self.coef /= np.sum(self.coef)
        self.n_components = len(self.components)

        self.components_ = self.components
        self.coef_ = self.coef
        self.n_components_ = self.n_components
        
    
    def _del_component(self, index):
        """Delete a component"""
        if index < 0 or index > self.components:
            print 'Index out of bound'
            return 
        
        del self.components[index]
        self.n_components -= 1
        print self.n_components


###################################################################################################
class BayesGMM(BaseLearner):
    """Bayes classifier in which Gaussian Mixture Models are used to model classes

    Parameters:
    -----------
    n_components: int optional default 1
        Number of Gaussians

    max_iters: int optional default  10
        Maximum iterations

    tolerance: float optional default 1e-6
        If improvement of the loss function is smaller than tolerance during two consecutive 
        iterations, stop the iteration

    opt_algo: ["EM"], optional default "EM"
        Optimization algorithm.
       
    verbose: bool optional default False
        If True, print details during iterations
    """
    def __init__(self,
                 n_components=1,
                 cov_type='diag',
                 max_iters=10,
                 tolerance=1e-6,
                 opt_algo="EM",
                 verbose=False):        
        #
        # Parameters
        #
        self.n_components = n_components
        self.cov_type = cov_type
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.opt_algo = opt_algo
        self.verbose = verbose

        #
        # Where we store all the gmms
        #
        self.gmms = []       

        
        
    def fit(self, X, y, sample_weight=[]):        
        """Train on a train set (X,y).  
        
        Parameters:
        -----------
        X : ndarray [n_samples, n_features]
            Each row is a sample.
            
        y : ndarray [n_samples]
            Labels of samples.        
        
        Returns:
        --------
        self : object.
            return the object itself.
        """
        super(BayesGMM, self).fit(X,y,sample_weight)
        self.class_labels = list(set(y))
        self.n_classes = len(self.class_labels)
        for k in range(self.n_classes):
            idx = np.where(y==self.class_labels[k])
            gmm = GaussianMixture(n_components=self.n_components, 
                                  cov_type=self.cov_type,
                                  max_iters=self.max_iters,
                                  verbose=self.verbose)

            Xk = X[idx,:][0]
            gmm.fit(Xk, sample_weight=sample_weight)
            self.gmms.append(gmm)

        return self


    def predict(self, X):
        """Predict y given a set of samples X.   
            
        Params:
        -------
        X : ndarray [n_samples, n_features]
            
        Returns:
        --------
        y : ndarray [n_samples]                
        """
        scores = []
        for gmm in self.gmms:
            log_lh = gmm.log_likelihood(X)
            scores.append(log_lh.ravel())

        scores = np.array(scores)
        y = np.array(self.class_labels)[np.argmax(scores,axis=0)]
        return y

        
