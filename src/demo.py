from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
import scipy
from math import log2
from tqdm import tqdm
from utils import multinomial, partitions
import matplotlib.cm as cm
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
    for i in range(k,m+k):
        ans+= steps_iteration(i,k)
    return int(ans)

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
    
def plot_sum_elements_dist(i, q1, q2):
    dist = scipy.stats.multinomial(i, [q1, q2])

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # data  k = 2 
    x = partitions(i, 2)
    y = [dist.pmf(par) for par in x]

    ax.bar(x, y)
    



i = 15#25
q1 = 0.3
q2 = 0.3
q3 = 0.3

dist = scipy.stats.multinomial(i, [q1, q2, q3])

# plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# data  k = 2 
pars = partitions(i, 3)
heights = [dist.pmf(par) for par in pars]

x = [p[0] for p in pars] 
y = [p[1] for p in pars] 
z = [0 for _ in pars]
dx = [1 for _ in pars]

# coloringsdsfga asdfaewghrbqare3hbwsreghzarQWE5hbewntnetzuhjn
min_height = min(heights)
max_height = max(heights)
# cmap = cm.get_cmap('jet')
cmap = cm.get_cmap('rainbow')
rgbs = [cmap( (h - min_height)/max_height ) for h in heights ]


ax.bar3d(x, y, z, dx,dx,heights, color=rgbs)
plt.savefig(f'sum_elemets_i{i}_q1{q1}_q2{q2}_q3{q3}.pdf')
plt.show()