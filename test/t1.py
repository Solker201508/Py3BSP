import bsp
import gc
#gc.set_debug(gc.DEBUG_STATS|gc.DEBUG_LEAK) 
bsp.createArray('ind.lower1','u4',[2,2])
bsp.createArray('ind.upper1','u2',[2,2])
bsp.createArray('ind.lower2','i4',[2])
bsp.createArray('ind.upper2','i4',[2])
bsp.createArray('ind.lower3','i4',[3])
bsp.createArray('ind.upper3','i4',[3])
lower1=bsp.asNumpy('ind.lower1')
upper1=bsp.asNumpy('ind.upper1')
lower1[0][0] = 5; lower1[0][1] = 5
upper1[0][0] = 14; upper1[0][1] = 14
lower1[1][0] = 2; lower1[1][1] = 0
upper1[1][0] = 11; upper1[1][1] = 9
lower2=bsp.asNumpy('ind.lower2')
upper2=bsp.asNumpy('ind.upper2')
lower3=bsp.asNumpy('ind.lower3')
upper3=bsp.asNumpy('ind.upper3')
lower2[0] = 5
lower2[1] = 3
upper2[0] = 6
upper2[1] = 7
lower3[0] = 15
lower3[1] = 10
lower3[2] = 6
upper3[0] = 20
upper3[1] = 12
upper3[2] = 12
inds1=bsp.createRegionSet(('ind.lower1','ind.upper1'))
print('inds1 indcount = ', bsp.indexCount(inds1), ', regcount = ', bsp.regionCount(inds1))
inds2=bsp.createRegionSet(('ind.lower2','ind.upper2'),('ind.lower3','ind.upper3'))
print('inds2 indcount = ', bsp.indexCount(inds2), ', regcount = ', bsp.regionCount(inds2))
inds3=bsp.createPointSet('ind.lower1')
print('inds3 indcount = ', bsp.indexCount(inds3), ', regcount = ', bsp.regionCount(inds3))
inds4=bsp.createPointSet('ind.lower2','ind.lower3')
print('inds4 indcount = ', bsp.indexCount(inds4), ', regcount = ', bsp.regionCount(inds4))
inds5=bsp.createRegionSet(('ind.lower2','ind.upper3'),('ind.lower3','ind.upper2'))
print('inds5 indcount = ', bsp.indexCount(inds5), ', regcount = ', bsp.regionCount(inds5))

inds6=bsp.createRegionSet(([[5,5],[2,0]], [[14,14],[11,9]]))
print('inds6 indcount = ', bsp.indexCount(inds6), ', regcount = ', bsp.regionCount(inds6))
inds7=bsp.createRegionSet(([5,3],[6,7]),([15,10,6],[20,12,12]))
print('inds7 indcount = ', bsp.indexCount(inds7), ', regcount = ', bsp.regionCount(inds7))
inds8=bsp.createPointSet([[5,5],[2,0]])
print('inds8 indcount = ', bsp.indexCount(inds8), ', regcount = ', bsp.regionCount(inds8))
inds9=bsp.createPointSet([5,3],[15,10,6])
print('inds9 indcount = ', bsp.indexCount(inds9), ', regcount = ', bsp.regionCount(inds9))

bsp.createArray('arr.a1','f8',[2,3])
bsp.createArray('arr.a2','u4',[10,10])

bsp.delete(inds1,inds2,inds3,inds4,inds5,inds6,inds7,inds8,inds9)
bsp.delete('arr.a1','arr.a2')
print("Do it again!")
bsp.delete(inds1,inds2,inds3,inds4,inds5,inds6,inds7,inds8,inds9)
bsp.delete('arr.a1','arr.a2')
print("OK!")
