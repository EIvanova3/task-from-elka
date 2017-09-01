
# coding: utf-8

# In[1]:

import numpy as np
from lcmvweights import Lcmvweights
from constraint import Constraint
#from constraint2 import Constraint

class MVDRBeamformer:
    
    def __init__(self, array, speed, freq, angle, N):
        self.array = array
        self.c = speed
        self.fc = freq
        self.angle = angle
        self.N = N
        
    def step(self, rx):
        w = []
        for i in xrange(self.angle.shape[1]):
            constraint = Constraint(self.array, self.c, self.fc, self.angle[:,i], self.N)
            w.append(Lcmvweights(rx, constraint))
        w = np.array(w).transpose()        
        y = np.dot(rx, np.conjugate(w))
        return [y, w]
    
    def __call__(self, rx):
        [y, w] = self.step(rx)
        return [y, w]