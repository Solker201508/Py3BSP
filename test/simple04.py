import bsp
#bsp.toggleVerbose()
nProcs = bsp.procCount()
myProcID = bsp.myProcID()
if myProcID != 0:
    bsp.fromObject(myProcID, 'procID')
    bsp.toProc(0, 'procID')
    bsp.async('one async')
elif myProcID == 0:
    for round in range(3):
        procID = bsp.async('one async')
        iProc = bsp.toObject(bsp.fromProc(procID)['procID'])
        print('%d, %d, %d'%(nProcs, procID, iProc))
