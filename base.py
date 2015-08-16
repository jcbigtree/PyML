"""Base functions
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 3 clause


import numpy as np
import copy
import inspect
import warnings
from scipy import sparse

__all__=[
    "BaseLearner", "clone" 
]


###############################################################################
def clone(learner, safe=False):
    """Constructs a new learner with the same parameters.
    Clone does a deep copy of the model in an learner
    without actually copying attached data. It yields a new learner
    with the same parameters that has not been fit on any data.

    Note: Implemented based on SKlearn

    Parameters
    ----------
    learner: learner object, or list, tuple or set of objects
        The learner or group of learners to be cloned

    safe: boolean, optional
        If safe is false, clone will fall back to a deepcopy on objects
        that are not learners.
    """
    learner_type = type(learner)
    # XXX: not handling dictionaries
    if learner_type in (list, tuple, set, frozenset):
        return learner_type([clone(e, safe=safe) for e in learner])
    elif not hasattr(learner, 'get_params'):
        if not safe:
            return copy.deepcopy(learner)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not implement a 'get_params' methods."
                            % (repr(learner), type(learner)))
    
    klass = learner.__class__
    new_object_params = learner.get_params(deep=False)
    for name, param in new_object_params.iteritems():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    return new_object


###################################################################################################
class SkBaseEstimator(object):
    """Part of the Sklearn base estimator. This class is adopted from Sklearn so that methods such 
    as clone can be used.
    """
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args


    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out


    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        
        valid_params = self.get_params(deep=True)
        
        #for key, value in six.iteritems(params):
        for key, value in params.iteritems():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self


    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, self.get_params(deep=False),)
        


###################################################################################################
class BaseLearner(SkBaseEstimator):
    """Base learner for ensembling"""    
    def __init__(self):  
        pass
                  
    
    def fit(self, X, y, sample_weight=[]):
        """Train on a train set (X,y).  
        Warning: Should be called by derived classes unless they have their own sanity check.
        
        Parameters:
        -----------
        X : array-like. shape [n_samples, n_features]
            Each row is a sample.
            
        y : array-like shape [n_samples]
            Corresponding label of samples.        
        
        sample_weight : array-like of shape [n_samples] or [n_samples, 1]
            weights on the sample. If this parameter is set, the base learner will perform a 
            weighted learning.
        Returns:
        --------
        self. object.
            return the object itself.
        """
        # Validate parameters
        if not isinstance(X, (np.ndarray, np.generic)) or \
           not isinstance(y, (np.ndarray, np.generic)):
            raise ValueError("Numpy array is the only acceptable format.")
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X.shape) != 2 or len(y.shape) != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                "X must be of shape [n_samples, n_features], Y must be of shape [n_samples]"
            )
            
        # Must check the sanity of the sample weights
       
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        
        if sample_weight == []:
            self.sample_weight = np.ones((self.n_samples, 1), dtype=np.float64)/self.n_samples
            
        elif len(sample_weight.shape) == 1:  # 1D array
            self.sample_weight = np.reshape(sample_weight, (self.n_samples, 1))
            self.sample_weight.astype(np.float64) 
            
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
        # Do nothing
                
                
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
     
        #        
        # compute the successful rate        
        #
        successful_rate = 1 - np.sum(np.abs(y_p - y)*0.5)/X.shape[0]
        return successful_rate
 

