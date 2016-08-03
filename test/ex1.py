import bsp
import numpy
import ctypes

ex1 = ctypes.CDLL('ex1.so')
params = numpy.zeros([106])
bsp.tic();
r = bsp.maximize(params, ex1.myFunValue4, ex1.myGradient4)
print("time = ", bsp.toc(), "ms")
print("minimum = ", r)
print("params = ", params)

