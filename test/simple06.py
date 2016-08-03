import bsp
import numpy as np
#bsp.toggleVerbose()
nProcs = bsp.procCount()
myProcID = bsp.myProcID()
nBatches = 1000
if myProcID == 0:
    for i in range(1, nProcs):
        bsp.addWorker(i)
    bsp.setScheduler(10,5,8)
    bsp.createArray('iLocalBatch','i4',[nProcs])
    bsp.share('iLocalBatch')
    bsp.sync('NOTHING')
    for r in range(nBatches):
        batch = bsp.async('to master', r == nBatches - 1 or (r + 1) % 100 == 0)
        rLocal = []
        for i in batch:
            rLocal.append(bsp.asNumpy('iLocalBatch')[i])
        print('%i: '%r,batch,rLocal)

        bsp.fromObject(r, 'iBatch')
        for i in batch:
            bsp.toProc(i, 'iBatch')
        bsp.async('from master')
else:
    bsp.createArray('iLocalBatch','i4',[1])
    bsp.share('iLocalBatch')
    bsp.sync('NOTHING')
    inds1=bsp.createPointSet([myProcID])
    for s in range(nBatches):
        bsp.asNumpy('iLocalBatch')[0] = s
        bsp.updateFrom('iLocalBatch','=','iLocalBatch',inds1,0)
        bsp.async('to master')
        bsp.async('from master')
        r = bsp.toObject(bsp.fromProc(0)['iBatch'])
        if r == nBatches - 1:
            break
