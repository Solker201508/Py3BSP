import bsp
import os
#bsp.toggleVerbose()
nProcs = bsp.procCount()
myProcID = bsp.myProcID()
nBatches = 1000
if myProcID == 0:
    for i in range(1, nProcs):
        bsp.addWorker(i)
    bsp.setScheduler(10,5,8)
    for r in range(nBatches):
        batch = bsp.async('to master', r == nBatches - 1 or (r + 1) % 100 == 0)
        print('%i: '%r,batch)

        bsp.fromObject(r, 'iBatch')
        for i in batch:
            bsp.toProc(i, 'iBatch')
        bsp.async('from master')
else:
    for s in range(nBatches):
        bsp.fromObject(s, 'iLocalBatch')
        bsp.toProc(0, 'iLocalBatch')
        bsp.async('to master')
        bsp.async('from master')
        r = bsp.toObject(bsp.fromProc(0)['iBatch'])
        if r == nBatches - 1:
            break
