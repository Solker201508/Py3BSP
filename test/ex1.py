import bsp
import numpy
import ctypes

ex1 = ctypes.CDLL('./ex1.so')
params = numpy.zeros([106])
bsp.tic();
#r = bsp.minimize(params, ex1.myFunValue1, ex1.myGradient1, penaltyLevel=0.5, penalty="LogSum")
#r = bsp.maximize(params, ex1.myFunValue2, ex1.myGradient2, penalty="LogSum", penaltyLevel=0.5)
#r = bsp.minimize(params, ex1.myFunValue3, ex1.myGradient3, penalty="LogSum", penaltyLevel=0.5)
#r = bsp.maximize(params, ex1.myFunValue4, ex1.myGradient4, penalty="LogSum", penaltyLevel=0.5)
#r = bsp.minimize(params, ex1.myFunValue1, ex1.myGradient1, penalty="L1", penaltyLevel=0.5)
#r = bsp.maximize(params, ex1.myFunValue2, ex1.myGradient2, penalty="L1", penaltyLevel=0.5)
#r = bsp.minimize(params, ex1.myFunValue3, ex1.myGradient3, penalty="L1", penaltyLevel=0.5)
r = bsp.maximize(params, ex1.myFunValue4, ex1.myGradient4, penalty="L1", penaltyLevel=0.5)
print("time = ", bsp.toc(), "ms")
print("minimum = ", r)
print("params = ", params)

