from audioop import mul
from math import factorial
from utils import partitions_with_zeros, powerset, even, multinomial, exp, sign, partitions
from tqdm import tqdm
from nltk import PCFG, nonterminals
import matplotlib.pyplot as plt
from queue import Queue
from linear import mode, kappa

def multinomial_aprox_tests(coef, i, *qs, eps=0.01):
    '''
    SHOULD RETURN sum(qs)
    '''
    ans = 0
    eps = 0

    # top is at E[X] = i(q1, ..., qk)
    top = (int(round(i*q, 0)) for q in qs)
    # in queue are elements of the sum, yet to be visited
    # parametrised by partitions (l1, ..., lk), where l1 +...+ lk = i
    q = Queue()
    q.put(mode(i, top))

    visited = set()

    while not q.empty():

        # partition
        partition = q.get()
        if partition in visited:
            continue
        visited.add(partition)

        # get coeficient
        if coef.get(partition) is None:
            d_coef = 0
            
            for j in range(len(partition)):
                if partition[j] == 0:
                    # multinomial coef with negative number is 0
                    continue
                tmp_p = partition[:j] + (partition[j] - 1,) + partition[j+1:]
               
               # update current coefitient
                if coef.get(tmp_p) is None:
                    d_coef = multinomial(*partition, coef=coef)
                    break
                    #coef[tmp_p] = multinomial(*tmp_p, coef=coef)
                d_coef += coef[tmp_p]
           
            coef[partition] = d_coef

         # sum element from the partition
        prod = 1
        for index, l in enumerate(partition):
            prod *= exp(qs[index], l)
        curr = coef[partition]*prod
        ans += curr

        # check if too small
        if curr < eps:
            continue

        # add new partitionss to the queue
        for j in range(len(partition)):
            for k in range(len(partition)):
                if k == j:
                    continue
                new_partition = list(partition)
                new_partition[k] += 1
                new_partition[j] -= 1
                if new_partition[j] < 0:
                    continue
                q.put(tuple(new_partition))
    return ans


def mode_test(*qs, i, coef={}):
    k = len(qs)
    # actuall calculation:
    max_kappa = 0
    max_par = None
    for par in partitions(i, k):
        tmp = kappa(par, *qs , coef=coef)
        if tmp >= max_kappa:
            max_kappa = tmp 
            max_par = par 
    
    # mode:
    sum_q = sum(qs)
    f_mode  = mode(i, *qs, coef=coef)
    f_kappa = kappa(f_mode, *qs, coef=coef)

    # bound_par = (i*q/sumQ_q for q in qs)
    # bound = kappa()
    print('real mode: ', max_par, '   bfs: ', f_mode)
    print('real max value: ', max_kappa, '   bfs: ', f_kappa)

mode_test(0.3, 0.3, 0.3, i=15)