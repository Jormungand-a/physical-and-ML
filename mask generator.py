

def iter2(n,r=''):  #recursive to transform it into binary numbrt
    
    if n==0 and len(r)>0:  #last digit, return binary number
        return r
    if len(r)==0 and n==0:
        return '0'
    else:
        r=str(n%2)+r  #according to the remaineder to add 0/1
      
        return iter2(n//2,r)

xx=[]
for i in range(1,1024):
    j=list(iter2(i))
    
    x=len(j)
    for e in range(0,x):
        j[e]=int(j[e])
    b=[0]*(10-x)
    j=b+j
    xx.append(j)

import queue

p=queue.PriorityQueue()
import numpy
#xx=numpy.array(xx)


for i in range(0,1023):
    p.put((sum(xx[i]),xx[i]))
    










