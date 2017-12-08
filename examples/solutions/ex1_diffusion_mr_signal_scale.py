# -*- coding: utf-8 -*-
"""
4. Diffusion MR signal
"""
import numpy as np

n = 3 #3D space
D = 2.4 # in (micro m)**2/ms
t = 50 # ms

mean_squared_displacement = 2*n*D*t # in (micro m)**2
print mean_squared_displacement
std_displacement = np.sqrt(mean_squared_displacement) # in (micro m)
print "std of displacements in micro meters:", std_displacement
