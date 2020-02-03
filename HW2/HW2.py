# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def gauss(mu,std,n=1000):
    noise = np.random.normal(mu,std,n)
    x = np.linspace(mu-3*std,mu+3*std,n)
    y = st.norm.pdf(x,mu,std)
    
    #Timeseries
    plt.figure()
    plt.title(f"Gaussian Time Series   μ={mu}   σ={std}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.plot(range(len(noise)),noise)
    plt.savefig(f"Gaussian_Timeseries_{mu}_{std}.pdf")
    plt.savefig(f"Gaussuan_Timeseries_{mu}_{std}.png")
    
    #Histogram
    plt.figure()
    plt.title(f"Gaussian   μ={mu}   σ={std}   bins={int(n/30)}")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.hist(noise,bins=int(n/30),density=True,label='Data')
    plt.plot(x,y,color='red',label='Gaussian')
    plt.legend()
    plt.savefig(f"Gaussian_{mu}_{std}.pdf")
    plt.savefig(f"Gaussian_{mu}_{std}.png")

def poisson(mu,n=1000):
    noise = np.random.poisson(mu,n)    
    bins = np.arange(min(noise),max(noise)+1,1)
    
    #Timeseries
    plt.figure()
    plt.title(f"Poisson Time Series   μ={mu}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.plot(range(len(noise)),noise)
    plt.savefig(f"Poisson_Timeseries_{mu}.pdf")
    plt.savefig(f"Poisson_Timeseries_{mu}.png")
    
    
    #Sharp Histogram
    x = np.arange(0,max(noise),0.01)
    y = st.poisson.pmf(x,mu)
    
    plt.figure()
    plt.title(f"Poisson Sharp   μ={mu}   n={1000}")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.hist(noise,bins=bins,align='left',density=True,label='Data')
    plt.xticks(bins)
    plt.plot(x,y,color='red',label='Poisson')
    plt.legend()
    plt.savefig(f"Poisson_Sharp_{mu}.pdf")
    plt.savefig(f"Poisson_Sharp_{mu}.png")
    
    #Fuzzy Histogram
    x = np.arange(0,max(noise),1)
    y = st.poisson.pmf(x,mu)
    
    plt.figure()
    plt.title(f"Poisson Fuzzy   μ={mu}   n={1000}")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.hist(noise,bins=bins,align='left',density=True,label='Data')
    plt.xticks(bins)
    plt.plot(x,y,color='red',label='Poisson')
    plt.legend()
    plt.savefig(f"Poisson_Fuzzy_{mu}.pdf")
    plt.savefig(f"Poisson_Fuzzy_{mu}.png")

#1
gauss(0,1)

#2
gauss(10.345,2.338)

#3
poisson(2)

#4
poisson(3.45)