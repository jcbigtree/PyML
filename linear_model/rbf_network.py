"""RBF network
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 

import numpy as np
from numpy.linalg import inv
from scipy.cluster.vq import vq, kmeans, whiten
from mlnp.optimize import line_search
from mlnp.base import BaseLearner


__all__=[
    "RBFNetwork"
]

###################################################################################################
class RBFNetwork(BaseLearner):
    """Radial Basis Function Network with adaptive centers
    
    Parameters:
    -----------
    n_basis_functions : int
        Number of basis functions. 
                   
    max_iters : int, optional default 10, used in Ratsch's paper
        Max iterations
        
    reg_cons : float, optional default 1e-6, used in Ratsch's paper
        Regularization constant                     
        
    opt_algo : ["ConjugateGradient", "GradientDescent"], optional default "ConjugateGradient".
        Algorithm used for minimizing the loss function. 

    References:
    -----------
    G. Ratsch, T. Onoda, and K. R. Muller, 'Soft margins for AdaBoost', Machine Learning, 
    vol.42, pp. 287-320, March 2001.    
    """
    def __init__(self,
                 n_basis_functions = 10,
                 max_iters = 10,  
                 reg_cons = 1e-6,
                 err_tolerance = 1e-6,
                 opt_algo = "ConjugateGradient"):         

        self.n_basis_functions = n_basis_functions
        self.max_iters = max_iters
        self.reg_cons = reg_cons
        self.opt_algo = opt_algo.lower()
        self.err_tolerance = err_tolerance
      
        #
        # Basis functions
        #
        self.centroids = []
        self.radiuses = []        
        self.coef = []        
       
        self.X = []
        self.y = []

        #
        # Attributes 
        #
        self.staged_loss_ = None


    def __str__(self):
        """Return the details of the RBFNetwork"""
        rbf_str = "RBF Network with adaptive centers \n" \
                + " --- Number of basis functions: " + str(self.n_basis_functions) + "\n"   \
                + " --- Regularization constant: " + str(self.reg_cons) + "\n" \
                + " --- Centroids: " + str(self.centroids) + "\n" \
                + " --- Radiuses: " + str(self.radiuses) + "\n" \
                + " --- Coef: " + str(self.coef) + "\n"   
        return rbf_str

        
    def fit(self, X, y, sample_weight=[]):        
        """Fit on a train set (X,y).  
        
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
        # Call base learner first
        #
        super(RBFNetwork, self).fit(X,y,sample_weight)
        self.X = X
        self.y = y
        
        #
        # Run KMeans to find initial centroids and radiuses 
        #        
        #whitened_samples = whiten(X)
        whitened_samples = X
        self.centroids, distortion = kmeans(whitened_samples, self.n_basis_functions)
        indexes, distortion = vq(whitened_samples, self.centroids)
        
        # In case that KMean doesn't find as many classes as we want.
        self.n_basis_functions = self.centroids.shape[0] 

        radiuses = []
        for k in range(self.n_basis_functions):         
            dmuk = self.centroids[k,:] - self.centroids
            d = np.sum(dmuk*dmuk, axis=1)           
            radiuses.append(np.min(d[np.where(d>0)]))
        self.radiuses = np.array(radiuses, dtype=np.float64)       

        mu = np.reshape(self.centroids,(1,-1))
        sigma = np.reshape(self.radiuses,(1,-1))
        mu_sigma = np.concatenate((mu, sigma),axis=1)

        #
        # Save all the gradients during evolution, partically for computing congjugate gradient
        # [ dE_du_0,   dE_dsigma_0   ]    <--- basis function 0 
        # [       :  :   :           ]
        # [ dE_du_k-1, dE_dsigma_k-1 ]    <--- basis function k-1             
        #
        staged_grad = []  
        staged_conj_grad = []         
        staged_loss = []

        K = self.n_basis_functions      
        M = self.n_features                    
        G = np.empty((self.n_samples, K), dtype=np.float64)
        for iter in range(self.max_iters):
            #
            # Compute optimal weights given the initial guess of centroids and radiuses            
            #
            mu = np.reshape(mu_sigma[0,0:M*K],(K,M))             
            sigma = mu_sigma[0,M*K:]
            for k in range(self.n_basis_functions):
                for i in range(self.n_samples):
                    x_bar = X[i,:] - np.reshape(mu[k,:],(1,-1))
                    G[i][k] = np.exp(- np.sum(x_bar*x_bar)/(2*sigma[k]*sigma[k]))                   
             
            I = np.eye(K, K, dtype=np.float64)            
            opt_w = np.dot(inv(np.dot(np.transpose(G), G) + 2.0*self.reg_cons/self.n_samples*I), 
                           np.dot(np.transpose(G), y))            
            self.coef = opt_w
            
            #
            # Compute gradients with respect to centroids (mu) and radiuses (sigma)
            #            
            grad = self._regularized_least_square_loss_gradient(rbf_params=mu_sigma,
                                                                X=X,
                                                                y=y)
            staged_grad.append(grad) 

            #
            # Either: Conjugate gradient
            #
            if  self.opt_algo == "conjugategradient":
                #                        
                # Estimate the conjugate direction with Fletcher-Reeves-Polak-Ribiere CG method
                # Reference : Numerical Recipes in C - The Art of Scientific Computing, Page 422
                #
                if iter%self.n_features == 0:  # Restart 
                    conj_grad = -grad
                else:               
                    prev_g = staged_grad[iter-1].ravel()
                    g = grad.ravel()
                    gamma = np.max(np.dot(g-prev_g, g)/np.dot(prev_g, prev_g),0)
                    conj_grad = - g + gamma*staged_conj_grad[iter-1] 
                    
                staged_conj_grad.append(conj_grad)            

                pk = conj_grad
                alpha = line_search(f=self._regularized_least_square_loss, 
                                    gf=self._regularized_least_square_loss_gradient, 
                                    xk=mu_sigma, 
                                    pk=pk,
                                    gfk=grad,
                                    c1=1e-4,
                                    c2=0.1,
                                    rho=0.75,
                                    condition='Armijo')
                    
                mu_sigma += alpha*pk             
                  
            #
            # Or: Gradient Descent
            # 
            if self.opt_algo == "gradientdescent":
                pk = - grad 
                alpha = line_search(f=self._regularized_least_square_loss, 
                                    gf=self._regularized_least_square_loss_gradient, 
                                    xk=mu_sigma, 
                                    pk=pk,
                                    gfk=grad,
                                    c1=1e-4,
                                    c2=0.1,
                                    rho=0.75,
                                    condition='Armijo')

                mu_sigma += alpha*pk
            #
            # Update centroids and radiuses 
            #
            self.centroids = mu_sigma[:,0:-K]
            self.radiuses  = mu_sigma[:,-K:]

            cur_loss = self._regularized_least_square_loss(rbf_params=mu_sigma, 
                                                           X=X, y=y)
            staged_loss.append(cur_loss)            
            
            #
            # Early stop
            #
            if iter > 1 and np.abs(staged_loss[-2] - cur_loss) < self.err_tolerance:
                break
                         
            print iter, " : ", cur_loss, ' ', np.dot(grad.ravel(), pk.ravel())
     
        self.staged_loss_ = staged_loss
        return self


    def predict(self, X): 
        """Predict y given a set of samples X.   
            
        Parameters:
        ----------
        X : array-like. shape [n_samples, n_features]
            
        Returns:
        --------
        y : array-like shape [n_samples]                
        """
        return np.sign(np.tanh(self._rbf(X) - 0.5))        
        

    def _rbf(self, X, rbf_params=None):
        """Calculate output of the RBF function"""
        #
        # Parse rbf_params
        #
        K = self.n_basis_functions      
        M = self.n_features                       

        if rbf_params is None:
            mu = np.reshape(self.centroids,(1,-1))
            sigma = np.reshape(self.radiuses,(1,-1))
            rbf_params = np.concatenate((mu, sigma),axis=1)
      
        rbf_params = np.reshape(rbf_params, (1,-1))        
        mu = np.reshape(rbf_params[0,0:M*K],(K,M))             
        sigma = rbf_params[0,M*K:]
        
        w = self.coef   
        
        b_X = [] # response of basis functions
        for k in range(self.n_basis_functions):
            Xk = X - np.reshape(mu[k,:],(1,-1))
            bk = np.exp(-np.sum(Xk*Xk, axis=1)/(2*sigma[k]*sigma[k]))
            b_X.append(bk*w[k])           
  
        return np.sum(np.array(b_X), axis=0)

    
    def _regularized_least_square_loss(self, rbf_params=None,  X=None, y=None):
        """Calculate the regularized least square loss function, 
        aka regularized residual
        
        Note: rbf_params = 
            [mu0, mu1, ..., muN-1 sigma0, sigma1, ..., sigmaN-1]
        """
        if X is None: X = self.X 
        if y is None: y = self.y    
        if rbf_params is None:
            mu = np.reshape(self.centroids,(1,-1))
            sigma = np.reshape(self.radiuses,(1,-1))
            rbf_params = np.concatenate((mu, sigma),axis=1)
            
        K = self.n_basis_functions      
        M = self.n_features         
        rbf_y = self._rbf(X=X, rbf_params=rbf_params)

        #
        # Compute regularized least square loss
        #
        rls_loss = 0.5*np.mean(np.power(y-rbf_y, 2)) \
                       + 0.5*self.reg_cons/self.n_samples * np.sum(np.power(self.coef,2))
        return rls_loss
    
            
    def _regularized_least_square_loss_gradient(self, rbf_params=None,  X=None, y=None):
        """Calculate the gradient of the regularized least square loss function
        
        Note: rbf_params = 
            [mu0, mu1, ..., muN-1 sigma0, sigma1, ..., sigmaN-1]
        """
        if X is None: X = self.X 
        if y is None: y = self.y
        if rbf_params is None:
            mu = np.reshape(self.centroids,(1,))
            sigma = np.reshape(self.radiuses,(1,))
            rbf_params = np.concatenate((mu, sigma),axis=1)

        K = self.n_basis_functions      
        M = self.n_features    

        mu = np.reshape(rbf_params[0,0:M*K],(K,M))
        sigma = rbf_params[0,M*K:]
        w = self.coef     
        rbf_y = self._rbf(X=X, rbf_params=rbf_params)
        err_y = np.reshape(rbf_y - y, (self.n_samples,1))

        grad_mu = np.zeros((K, M), dtype=np.float64) 
        grad_sigma = np.zeros((K, 1), dtype=np.float64) 
        for k in range(K):
            Xk = X - mu[k,:]
            bk = np.reshape(np.exp(-np.sum(Xk*Xk, axis=1)/(2*np.power(sigma[k],2.0))), 
                            (self.n_samples,1))
            grad_mu[k,:] = np.sum(err_y * bk * w[k] * (Xk/np.power(sigma[k],2.0)), axis=0)                

            wkd = np.reshape((np.sum(Xk*Xk, axis=1)/np.power(sigma[k],3)), 
                            (self.n_samples,1))             
            grad_sigma[k] = np.sum(err_y*bk*w[k]*wkd, axis=0)
                    
        grad_mu = np.reshape(grad_mu,(1,-1))
        grad_sigma = np.reshape(grad_sigma,(1,-1))        
        return np.concatenate((grad_mu, grad_sigma), axis=1)
