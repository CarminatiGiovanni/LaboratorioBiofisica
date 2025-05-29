import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from numpy import ndarray, float64
from collections.abc import Callable
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib

def final_val(x,sigma,decimals = 2,exp = 0, udm: str = '') -> str:
    x = np.round(x*np.power(10.0,-exp),decimals)
    sigma = np.round(sigma*np.power(10.0,-exp),decimals)
    if decimals == 0:
        x = int(x)
        sigma = int(sigma)
    return f'{x} ± {sigma} {udm}' if exp == 0 else f'({x} ± {sigma})e{exp} {udm}'

# NOTA: NO SIGMA X
class Interpolazione:
    def __init__(self,X: ndarray[float64], Y: ndarray[float64],
                 sigmaY: ndarray[float64] | float64,
                 model: Callable,guess: ndarray[float64],
                 names: list[str] = None,
                 limits:list[tuple] = None) -> None:
        self.f = model
        self.Y = Y.astype('float64')
        self.X = X.astype('float64')
        self.N = len(X)
        self.names = names
        self.limits = limits

        if type(sigmaY) == np.float64:
            self.sigmaY = np.ones(self.N) * sigmaY
        else:
            self.sigmaY = sigmaY.astype('float64')
                        
        cost = LeastSquares(self.X,self.Y,self.sigmaY,self.f)        

        # 1 = least squares, 0.5 = log likelihood, 0 = chi2

        self.m = Minuit(cost, *guess, name=names)
        self.m.migrad()
        self.m.hesse()
        
        self.chi2 = np.round(self.m.fval,2)
        self.dof = len(X) - len(guess)
        self.rchi2 = np.round(self.chi2/self.dof,2)
        
        self.values = self.m.values.to_dict()
        self.errors = self.m.errors.to_dict()
        self.covariance = np.array(self.m.covariance)
        
        self.pvalue = np.round(sc.stats.chi2.sf(self.chi2,self.dof),2) # sf = survival function = 1 - cdf
        
    def draw(self,xscale='linear',N=1000):
        if xscale == 'log':
            x = np.logspace(np.log10(self.X.min()),np.log10(self.X.max()),N)
        else:
            x = np.linspace(self.X.min(),self.X.max(),N)
        
        return x, self.f(x,*self.values.values())
        
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        # s1 = str(self.m) + '\n\n
        s0 = '----------------- VALORI FIT: -----------------\n'
        s2 = f"dof: {self.dof}\nchi2: {self.chi2}\nchi2 ridotto: {self.rchi2}\npvalue: {self.pvalue}"
        
        exponents = np.array(np.floor(np.log10(np.abs(list(self.values.values())))),dtype=np.int32)
                
        s3 = '\n'.join([n + ': ' + final_val(v,s,3,e) for n,v,s,e in zip(self.names,self.values.values(),self.errors.values(),exponents)])
         
        s4 = '\n------------------------------------------------\n'
        return s0 + s3 + '\n\n' + s2 + s4
    


if __name__ == '__main__':
   
    y = np.array([2.25, 4.25, 6.5, 8.75, 10.75, 12.75])
    x = np.array([10.,20.,30.,40.,50.,60.])
    y_err = np.array([0.21, 0.21, 0.25, 0.21, 0.21, 0.21])


    plt.errorbar(x,y,y_err,fmt='ok')
    plt.show()

    def func1(x, a,b,c): # delta_N = ... * delta_p
        return a*x**2 + b*x + c

    # print(my_minuit)
    fit = Interpolazione(x,y,y_err,func1,[1,2,0],names=['a','b','c'])
        
    # plt.errorbar(x,y,y_err,x_err,fmt='ok')
    # plt.plot(*fit.draw())
    # plt.show()
    
    print(fit)
