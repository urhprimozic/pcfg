from math import factorial
from utils import partitions, partitions_with_zeros
from tqdm import tqdm
def probability(m, p,*qs):
    '''
    Return the probability of parsing any word v, which include exactly len(qs) diffferent sybols x_i, and P(V -> x_i) = qs[i-1]
    Parameters
    --------------
    - m - Number of iterations
    - p = P(S -> V)
    - qs[i] = P(V -> x_i)
    '''
    # initalizing 
    k = len(qs)
    P = 0
    # dictionary of multinomial coeficients
    k_factorial = factorial(k)
    coef = {tuple(1 for _ in range(k)) + (False,) : k_factorial}
    for j in range(k+1):
        key = tuple(0 if l == j else 1 for l in range(k)) + (True,)
        coef[key] = factorial(k)
    
    # iterations
    pi = p**(k-1)
    #for i in range(k, m+k):
    for i in tqdm(range(k, m+k), total=m):
        # iterate over partitions
        ps = partitions_with_zeros(i, k)
        sum_over_partitions = 0
        for par in ps:

            # new coeficients 
            d_coef  = 0
            for j in range(0,k):
                # (l1, ..., lj-1, ..., lk)
                if par[j] == 0:
                    # multinomial coef with negative number is 0
                    continue
                tmp_p = par[:j] + (par[j] -1,) + par[j+1:-1] + (par[-1] or par[j] == 1,)
                if (not par[-1]) and par[j] >= 1:
                    d_coef += coef[tmp_p]
            coef[par] = d_coef
            
            # product of q_i^(l_i)
            prod = 1
            for index, l in enumerate(par[:-1]):
                prod *= qs[index]**l
            # inner sum 
            sum_over_partitions += coef[par]*prod 

        # new pi = p^i
        pi *= p

        # estimation difference
        dP = (1-p)*pi * sum_over_partitions
        # new estimate
        P += dP
    return P

    