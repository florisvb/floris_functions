import numpy as np
import copy


def normalize(array):
    normed_array = norm_array(array)
    return array / normed_array
def norm_array(array):
    normed_array = np.zeros_like(array)
    for i in range(len(array)):
        normed_array[i,:] = np.linalg.norm(array[i])
    return normed_array[:,0]
def diffa(array):
    d = np.diff(array)
    d = np.hstack( (d[0], d) )
    return d
    

###
def iseven(n):
    if int(n)/2.0 == int(n)/2:
        return True
    else:
        return False 
def isodd(n):
    if int(n)/2.0 == int(n)/2:
        return False
    else:
        return True
        
        
###
def interpolate_nan(Array):
    if True in np.isnan(Array):
        array = copy.copy(Array)
        for i in range(2,len(array)):
            if np.isnan(array[i]).any():
                array[i] = array[i-1]
        return array
    else:
        return Array
    
###
def remove_angular_rollover(A, max_change_acceptable):
    array = copy.copy(A)
    for i, val in enumerate(array):
        if i == 0:
            continue
        diff = array[i] - array[i-1]
        if np.abs(diff) > max_change_acceptable:
            factor = np.round(np.abs(diff)/(np.pi))  
            if iseven(factor):
                array[i] -= factor*np.pi*np.sign(diff)
    if len(A) == 2:
        return array[1]
    else:
        return array
