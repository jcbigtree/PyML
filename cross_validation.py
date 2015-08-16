"""
    For cross-validation. Return indexes of randomly splitted data sets.
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 3 clause

import numpy as np
import numbers
from numbers import Integral

__all__ = [
    "train_validation_test_split", 
    "train_test_split", 
    "kfold_split"
] 


def train_validation_test_split(n_samples):
    """Split a data set into [train_set, validation_set, test_set]
        
    Params
    ------
    n_samples. Integer. 
        Number of samples. 
    Returns
    ------
    sub_sets. List of numpy array.
        A list of Array-like object containing the random indexes. 
    """  
    if n_samples < 3:
        raise ValueError('Number of samples can not be smaller than 3.')
    
    #rnd_idx = np.random.permutation(n_samples)    
    #d = n_samples/3
    #idx_0 = rnd_idx[0:d]
    #idx_1 = rnd_idx[d:d*2]
    #idx_2 = rnd_idx[d*2:]    
    #return idx_0, idx_1, idx_2
    
    sub_sets = _split_data_set(n_samples, 3)
    return sub_sets


def train_test_split(n_samples):
    """Split a data set into [train_set, test_set]
      
    Params
    ------
    n_samples. Integer. 
        Number of samples. 

    Returns
    ------
    sub_sets. List of numpy array.
        A list of Array-like object containing the random indexes. 
    """  
    if (not isinstance(n_samples, (np.integer, Integral)) 
            or n_samples < 2):
        raise ValueError('Number of samples can not be smaller than 2.')
    
    sub_sets = _split_data_set(n_samples, 2)
    return sub_sets


def kfold_split(n_samples, n_folds):
    """Split a data set into [n_folds] for k-fold cross-validation.
      
    Params
    ------
    n_samples. Integer. 
        Number of samples. 

    n_folds. Integer. 
        Number of folds

    Returns
    -------
    sub_sets. List of numpy array.
        A list of Array-like object containing the random indexes. 
    """
    try:
        sub_sets = _split_data_set(n_samples, n_folds)
        return sub_sets
    except ValueError as verr:
        raise    
    

def _split_data_set(n_samples, n_subsets):
    """Private function of generating random indexes of subsets"""
    if (not isinstance(n_samples, (np.integer, Integral))                   
            or not isinstance(n_subsets, (np.integer, Integral))         
            or n_samples < n_subsets or n_samples < 1 or n_subsets < 1):     
       raise ValueError(
                'n_samples, n_subsets must be positive integer. n_samples >= n_subsets')
    
    random_sets = []
    random_idx = np.random.permutation(n_samples)
    d = n_samples / n_subsets
    for i in range(n_subsets):
        random_sets.append(random_idx[i*d:i*d+d])
   
    count = 0
    for k in range(n_subsets*d,n_samples):
        random_sets[count] = np.append(random_sets[count], random_idx[k])
        count += 1 
   
    return random_sets
