import bsp
import numpy
import math
import sys
import traceback
try:
    # get the 2D coordinates
    nProcs = bsp.procCount()
    myProcID = bsp.myProcID()
    n1Dim = int(math.floor(math.sqrt(nProcs)))
    i0 = int(myProcID / n1Dim)
    i1 = myProcID % n1Dim

    # create the local parts of the global array
    a1=bsp.array('arr.a1','f8',[10,10])
    for i in range(10):
        for j in range(10):
            a1[i][j] = i0+i1+i+j

    # globalize the local array
    bsp.globalize(0,(n1Dim,n1Dim),'arr.a1')
    futureA1=bsp.Future(0,'arr.a1@global')

    # create a local array to get the requested data
    a2=bsp.array('arr.a2','f8',[2,10,10])

    # request elements to the local array from the global array
    a2[1,:,:]=futureA1[5:14,5:14]
    a2[2,:,:]=futureA1[2:11,0:9]

    if myProcID == 0:
        print(a2)
except:
        info = sys.exc_info()
        print('Error: ', info[1])
        traceback.print_tb(info[2])


