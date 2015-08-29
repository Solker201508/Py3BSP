import bsp
import numpy
import math

# get the 2D coordinates
nProcs = bsp.procCount()
myProcID = bsp.myProcID()
n1Dim = int(math.floor(math.sqrt(nProcs)))
i0 = int(myProcID / n1Dim)
i1 = myProcID % n1Dim

# create the local parts of the global array
bsp.createArray('arr.a1','f8',[10,10])
a1=bsp.asNumpy('arr.a1')
for i in range(10):
    for j in range(10):
        a1[i][j] = i0+i1+i+j

# globalize the local array
bsp.globalize(0,(n1Dim,n1Dim),'arr.a1')

# create an index-set for the request
inds1=bsp.createRegionSet(([[5,5],[2,0]],[[14,14],[11,9]]))

# create a local array to get the requested data
bsp.createArray('arr.a2','f8',[2,10,10])

# request elements to the local array from the global array
bsp.requestTo('arr.a2','arr.a1@global',inds1)
bsp.sync('just for fun!!')

if myProcID == 0:
    a2 = bsp.asNumpy('arr.a2')
    print(a2)


