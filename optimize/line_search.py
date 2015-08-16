"""Line search
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 3 clause


import numpy as np
import numbers


__all__=[
    "line_search"  
]


def line_search(f, gf, xk, pk, gfk=None, c1=1e-4, c2=0.1, rho=0.9, 
                max_iters=50, condition='Armijo'):
    """Inexact line search : find a relatively good step size 't'
    such that f(x + alpha*p)<f(x).
    
    Parameters:
    -----------
    f: callable object
        Objective funciton to be minimized
    
    gf: callable object 
        Gradient of the objective function          

    xk: ndarray
        Starting point
       
    pk: ndarray
        Search direction

    gfk: ndarray
        Gradient of the objective function at the starting point xk
   
    c1: float, optional default 1e-4
        Control parameter used in Armijo condition. Must be within (0, 0.5] 

    c2: float, optional default 0.1
        Control parameter used in curvature condition. Range (c1, 1)
 
    rho: float, optional default 0.9
        Control the speed of the shrinkage of the step size. Must be within (0, 1) 

    max_iters: float, optional default 50
        Maximum iterations. Stop and return 0 when exceeds the max_iter
    
    condition: str ['Armijo', 'Wolfe', 'StrongWolfe'] default 'Armijo'

    Return:
    -------
    alpha: float
        an acceptable good step size 
        
    Reference:
    ----------
    [1] Convex Optimization I, Professor Stephen Boyd, Stanford University. 
    [2] Line search methods. Chap 3. http://pages.cs.wisc.edu/~ferris/cs730/chap3.pdf
    [3] Wolfe conditions http://en.wikipedia.org/wiki/Wolfe_conditions
    
    Note: Notations here mainly follow the reference [2]
    """
    #
    # A quick sanity check
    #
    if not hasattr(f, '__call__'):
        raise ValueError(
            "f must be callable"
            )       

    if not hasattr(gf, '__call__'):
        raise ValueError(
            "gf must be callable"
            )       
    
    if not isinstance(xk, (np.ndarray)):
        raise ValueError(
            "x must be numpy ndarray"
            )
        
    if not isinstance(pk, (np.ndarray)):
        raise ValueError(
            "p must be numpy ndarray"
            )
    
    if (not isinstance(c1, (numbers.Number, np.number)) or 
        c1 >= 0.5 or c1 <= 0.0):
        raise ValueError(
            "c1 must be number within (0, 0.5]"
            )

    if (not isinstance(c2, (numbers.Number, np.number)) or 
        c2 <= 0.0 or c2 <= c1):
        raise ValueError(
            "c2 must be number within (c1, 1)"
            )

    if (not isinstance(rho, (numbers.Number, np.number)) or 
        rho > 1.0 or rho <= 0.0):
        raise ValueError(
            "rho must be number within (0, 1)"
            )       
    
    if (not isinstance(max_iters, (numbers.Number, np.number)) or 
        max_iters <= 0.0):
        raise ValueError(
            "max_iters must be a positive integer"
            )          
  
    if condition.lower() not in ['armijo', 'wolfe', 'strongwolfe']:
        raise ValueError(
            "condition must be one of ['armijo', 'wolfe', 'strongwolfe'])"
            )
      
    if gfk is None:
        gfk = gf(xk) 
   
    #
    # If search direction is not a descending direction, 
    # return 0 immediately
    #
    if np.dot(gfk.ravel(), pk.ravel()) > 0:
        return 0

    #
    # Main loop
    #
    iter = 0
    alpha = 1
    while iter < max_iters:
        if condition.lower() == 'armijo' \
           and f(xk + alpha*pk) <= f(xk) + c1*alpha*np.dot(gfk.ravel(), pk.ravel()):
            break

        if condition.lower() == 'wolfe' \
           and f(xk + alpha*pk) <= f(xk) + c1*alpha*np.dot(gfk.ravel(), pk.ravel()) \
           and np.dot(pk.ravel(), gf(xk + alpha*pk).ravel())  >=  \
               c2*np.dot(pk.ravel(), gfk.ravel()):  # curvature condition
            break

        if condition.lower() == 'strongwolfe' \
           and f(xk + alpha*pk) <= f(xk) + c1*alpha*np.dot(gfk.ravel(), pk.ravel()) \
           and np.abs(np.dot(pk.ravel(), gf(xk + alpha*pk).ravel()))  \
               <= c2*np.abs(np.dot(pk.ravel(), gfk.ravel())):
            break       

        alpha *= rho
        iter += 1
                 
    return alpha

