import torch
import numpy as np
from collections import deque

print("cuda:0" if torch.cuda.is_available() else "cpu")


c = np.array([1, 1,1])


#target_pos = np.array([[c,c],[[3, 3],[4,4]]])
#target_pos2 = np.array([[[5, 5,5],[6,6,6]],[[7, 7],[8,8]]])
#print(target_pos2, target_pos2.reshape(1))
target_rot = np.array([0, 1, 0, 0])
x = np.array([[1,1],[2,2]])
#print(x)
k = np.array([[11],[22]])
n = [0,1]
#print(n)
y = np.zeros(2)
#np.append(y,x,axis= 0)
#np.insert(y,0,x)
for i in range(10):
    print(np.random.normal(0.7,0.1,1))
    np.random.normal
#print(x[:3])

#y = np.hstack((target_pos,target_pos2))
#print(y)

#distance = np.linalg.norm(target_pos-target_pos2, axis =(0))
#print(distance)
#print(np.array([(distance < .2) - 1]).squeeze())
#print("target pos ",target_pos)
#print("target pos 2",target_pos2 )
#target = np.array([target_pos,target_pos2],dtype= object)
#print("target",target, target.shape)


"""
print(target, type(target))
tar = np.array([target_pos,target_pos2,target_pos])
print(tar)

def cr(transition):
    print(*transition,"aaaa",transition,type(transition))
    a,b,c =transition
    print(a)
    print(b)


cr(tar)
"""