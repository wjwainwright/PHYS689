# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

print("Actual: 9.81 m/s")

#Read in the file and decompose matrix into lists
L,dL,t1,t2,t3 = np.genfromtxt("measure_g.csv",delimiter=',',skip_header=2,unpack=True)

#mm to m
L = [a/1000 for a in L]
dL = [a/1000 for a in dL]

tbar=[]
dt=[]
g=[]
dg=[]

#Stepwise
for i in range(len(L)):
    
    #Time
    tList = [t1[i],t2[i],t3[i]]
    tbar.append(np.average(tList)/10)
    dt.append(np.std(tList)/10)
    
    #Gravity
    g.append( 4*np.pi**2*L[i]*tbar[i]**-2 )
    dg.append(np.sqrt( (4*np.pi**2*tbar[i]**-2*dL[i])**2 + (8*np.pi**2*L[i]*dt[i]*tbar[i]**-3)**2 ))
print(f"Stepwise: ({round(np.mean(g),4)} +/- {round(np.mean(dg),4)}) m/s")


#Graphically
plt.figure()
plt.title("Pendulum Data")
plt.xlabel("L")
plt.ylabel(r"$T^2$")
tsquare = [a**2 for a in tbar]
plt.scatter(L,tsquare)
coeff = np.polyfit(L,tsquare,1)
f = np.poly1d(coeff)
fit = [f(a) for a in L]
plt.plot(L,f(L),label=f"{f}")
plt.grid()
plt.legend()
plt.savefig("plot.pdf")

grav = coeff[0]**-1 *4*np.pi**2
dgrav = np.mean([abs(a-b) for a,b in zip(fit,tsquare)])
print(f"Graphically: ({round(grav,4)} +/- {round(dgrav,4)}) m/s")