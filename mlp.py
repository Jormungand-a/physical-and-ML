import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy



mask=numpy.array([1,1,1,1,1,1,1,1,1,1])
#mask=numpy.array([0,0,0,0,0,0,1,0,0,0]) #enable this line to use mask

mask=nn.Parameter(torch.tensor(mask).float())

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.L1=nn.Linear(10,1,bias=False)
        self.coe=nn.Parameter(torch.rand(10,1))
        self.w=nn.Parameter(torch.rand([]))
        self.pha=nn.Parameter(torch.rand([]))
        self.e=torch.tensor(2.718)
        self.ew=nn.Parameter(torch.rand([]))
        #self.b=torch.tensor(1.8)
        #self.pha=torch.tensor(-3.1416/6)
        #self.w=torch.tensor(0.583)
        self.b=torch.tensor(1.8)
        
        self.lw=nn.Parameter(torch.rand([]))
        self.lpha=nn.Parameter(torch.rand([]))
        self.cw=nn.Parameter(torch.rand([]))
        self.cpha=nn.Parameter(torch.rand([]))
        self.wx=nn.Parameter(torch.rand([]))
        
        self.law=nn.Parameter(torch.rand([]))

        self.w1=nn.Parameter(torch.rand([]))
        self.w2=nn.Parameter(torch.rand([]))
        self.j1=nn.Parameter(torch.rand([]))
        self.j2=nn.Parameter(torch.rand([]))
        
        self.m1=nn.Parameter(torch.rand([]))
    def forward(self,time,mask):
        
        j=torch.cos(self.w*time[0:600,3:4]+self.pha)
        k=self.e**(-self.ew*time[0:600,4:5])
        kk=(time[0:600,5:6])*torch.cos(self.cw*time[0:600,5:6]+self.cpha)

        last=(time[0:600,7:8])*self.e**(-self.wx*time[0:600,7:8])
        
        rr=torch.cos(time[0:600,6:7]*self.lw+self.lpha)*self.e**(-self.law*time[0:600,6:7])
      
        ee=time[0:1000,8:9]**3
        gg=F.relu((torch.log(time[0:600,9:10]/self.m1+1)))
        e=time[0:600,0:3]
        t=torch.cat((e,j),1)
        v=torch.cat((t,k),1)
        vv=torch.cat((v,kk),1)
        qq=torch.cat((vv,rr),1)
        
        xx=torch.cat((qq,last),1)
      
        rrr=torch.cat((xx,ee),1)
        cv=torch.cat((rrr,gg),1)
        cv=cv*mask
    
        result=self.L1(cv)

        
       
        l=torch.abs(self.L1.weight)
        L1=torch.sum(l)
        
        return result ,L1











label=[]
time=[]
e=numpy.e
'''
for i in range(0,1):
    a=6
    b=4
    ca=10
    w=0.3
    pha=0.4
    for j in range(0,3000,8):
        gg=[]
        f=[]
        for p in range(j,j+16,1):
            line=0.5*a*((p/100)**2)+(1/6)*b*(p/100)**3#-ca*(-numpy.cos(pha)+numpy.cos(pha+w*(p/100))+w*(p/100)*numpy.sin(pha))/w**2
            #line=ca*(-numpy.cos(pha)+numpy.cos(pha+w*(p/100))+w*(p/100)*numpy.sin(pha))/w**2
            gg.append(line)
            f.append(p/100)
        f=numpy.array(f)
        test.append(gg)
        label.append(gg)
        time.append(f)
'''
#simple version
a=10
b=4
pha=0.5
w=1
e=2.718
disp=[]
beta=0.0089
m=0.42
g=9.8
vta=(m*g/beta)**0.5
ta=(m/(beta*g))**0.5
tm=ta*numpy.arctan(11/vta)
for j in range(0,605):
        disp.append(5*(j/100)**2+(2-2*numpy.cos(j/100)))
        ti=[1,j/100,(j/100)**2,j/100,j/100,j/100,j/100,j/100,j/100,j/100]
        '''
        if j/100<tm:
            label.append(vta*numpy.tan((tm-j/100)/ta))
        else:
            label.append(-vta*numpy.tanh(j/100-tm)/ta)
            '''
        label.append(20*numpy.cos(j/50)*2.718**(-j/300))
        time.append(ti)

label=numpy.array(label)
time=numpy.array(time)

x=ResNet()
oo=100
o=torch.optim.Adam(x.parameters(),lr=0.1)
#o=torch.optim.SGD(x.parameters(),lr=0.1)
criterion = nn.MSELoss()
t=time.shape[0]//600
for i in range(0,100000):
    ee=0
    numpy.random.seed(10)
    label=numpy.random.permutation(label)
    numpy.random.seed(10)
    time=numpy.random.permutation(time)

    for j in range(0,t):
        
   

        ti=time[j*600:j*600+600]
        ti=torch.tensor(ti).float()
        ti=torch.nn.Parameter(ti)

        target=label[j*600:j*600+600]
        target=torch.tensor(target).float()
        target=torch.autograd.Variable(target)
        target=target.reshape(600,1)
        
        o.zero_grad()
        output,los=x(ti,mask)
  
      
        los1=criterion(output,target)
        loss=los1+0.01*los
        loss.backward()
        
        o.step()
        ee=ee+float(los1)
    if ee<oo:
        oo=ee
    
    if ee<0.0002:break
        
    if i %1000==0:
        print("total loss in ",str(i)," is "+str(ee))
        print(x.L1.weight)
        
     
        '''
    if i%5000==0:
        if x.coe.requires_grad==False:
            
            x.coe.requires_grad=True
        if x.coe.requires_grad==True:
            
            x.coe.requires_grad=False
        '''


























