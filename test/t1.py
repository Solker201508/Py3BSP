import bsp
import pickle as pk
a = [(1,2),3]
bsp.fromObject(a, "a")
print("dumps(a)=",pk.dumps(a))
b = bsp.toObject("a")
print(b)

