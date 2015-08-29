import bsp
import numpy
import math

# get the 2D coordinates
nProcs = bsp.procCount()
myProcID = bsp.myProcID()
n1Dim = int(math.floor(math.sqrt(nProcs)))
i0 = myProcID / n1Dim
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
#inds1=bsp.createPointSet('ind.lower1')

# create a local array to store the update data
bsp.createArray('arr.a2','f8',[2,10,10])
a2=bsp.asNumpy('arr.a2')
for i in range(2):
    for j in range(10):
        for k in range(10):
            a2[i][j][k] = 1.0

# request elements to the local array from the global array
bsp.updateFrom('arr.a2','+','arr.a1@global',inds1)
bsp.sync('just for fun!!')

if myProcID == 0:
    print (a1)


