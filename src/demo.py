from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from math import log2
from tqdm import tqdm

from linear import probability, probability_exact

N = 103

# Plots for time complexity of aproximation for linear grammar
def steps_iteration(i, k):
    '''
    Number of steps in i-th iteration
    '''
    #4+ \binom{i+k}{k-1} + k\binom{i+k-1}{k-1} + (k+2)\binom{i-1}{k-1}+k\log_2 i \frac{i-1}{k-1}\binom{i-2}{k-2}
    if i < k:
        return 1
    a = binom(i+k,k-1)
    b = (k+2) * binom(i-1, k-1)
    c = k*binom(i+k-1,k-1)
    d = k*log2(i)*(i-1)*binom(i-2,k-2)/(k-1)
    return 4+a+b+c+d

def steps_aproximation(m,k):
    '''
    Number of steps for m iterations
    '''
    ans = 0
    for i in range(2,m+1):
        ans+= steps_iteration(i,k)
    return ans

def steps_formula(k):
    '''
    Number of operations needed for calculations using the exact formula
    '''
    return (2**(k-1))*(10+k)

def error(m,  p=0.5, *qs):
    '''
    Returns estimation of an erorr of aproximation with m iterations
    '''
    if len(qs) == 0:
        Q = 0.5
    else:
        Q = sum(qs)
    return (1-p)*(p**m)*(Q**m)/(1-p*Q)


def eq_qs(k, divisor=1):
    '''
    Returns an array of equal probabilities q_1= ...= q_k for rules 
      V -> x_1 [q_1] | ... | x_k [q_k]

    q_i = 1/(k*divisor)
    '''
    return [1/(k*divisor) for _ in range(k)]


def real_error(m, p=0.5, *qs):
    '''
    Returns real error, made by aproximations with m iterations
    '''
    proba = probability_exact(p, *qs)
    aprox = probability(m, p, *qs)
    return abs(proba - aprox)

def error_vs_estimator(m_min, m_max, p=0.5,filename='error_vs_estimator.pdf', *qs):
    '''
    Plots true error vs estimated error
    '''
    x = [i for i in range(m_min, m_max+1)]
    real_errors = [real_error(m, p, *qs) for m in range(m_min,m_max+1)]
    estimations = [error(m, p, *qs) for m in range(m_min,m_max+1)]
    plt.plot(x, real_errors, color='green', label='Real error')
    plt.plot(x, estimations, color='yellow', label='Estimated error')
    plt.legend()
    plt.savefig('../img/' + filename)
    plt.show()

def plot_err_iter(m_min, m_max, divisor=2, p=0.5, filename='error_over_iterations.pdf'):
    '''
    Plots true error 
    '''
 #   for k, color in tqdm([(5, 'blue'), (10, 'red'), (15, 'green')], total=3):
    for k, color in tqdm([(3, 'blue'), (5, 'red'), (7, 'green')], total=3):
        x = [i for i in range(m_min, m_max + 1)]
        y = [real_error(i, p, *eq_qs(k, divisor)) for i in range(m_min, m_max+1)]
        plt.plot(x, y, color=color, label=f'k={k}')
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Iteration')
    



if __name__== 'main':
    error_vs_estimator(10, 0.5,'error_vs_estimator.pdf', *eq_qs(5))