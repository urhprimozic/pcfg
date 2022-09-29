from audioop import mul
from math import factorial
from utils import partitions_with_zeros, powerset, even, multinomial, exp, sign
from tqdm import tqdm
from nltk import PCFG, nonterminals
import matplotlib.pyplot as plt
from queue import Queue
from scipy.special import binom

def probability_deprecated(m: int, p: float, *qs: float) -> float:
    '''
    Return the aproximation of the probability of parsing any word v, which include exactly len(qs) diffferent sybols x_i, and P(V -> x_i) = qs[i-1]

    with the grammar

    E -> E + cV [p] | c [1-p]
Return the aproximation of the probability of parsing any word v, which include exactly len(qs) diffferent sybols x_i, and P(V -> x_i) = qs[i-1]

    with the grammar

    E -> E + cV [p] | c [1-p]

    V -> x_1 [q_1] | ... | x_n [q_n]

    Parameters
    --------------
    - m - Number of iterations
    - p = P(S -> V)
    - qs[i] = P(V -> x_i)

    Returns
    -----------
    Probability of P([w]), where w = c + \sum cx_i

    Example
    ---------
    Probability of parsing an expression c+cx_i in a grammar 

    E -> E + cV [p] | c [1-p]

    V -> x_1 [1]
    >>> probability(10,0.5,1)
    0.49951171875

    E -> E + cV [p] | c [1-p]

    V -> x_1 [1]
    >>> probability(10,0.5,1)
    0.49951171875

    '''
    print('WARNING\nDeprecated version..')


    # initalizing
    k = len(qs)
    P = 0
    # dictionary of multinomial coeficients
    k_factorial = factorial(k)
    coef = {tuple(1 for _ in range(k)) + (False,): k_factorial}

 # prepočas
 #   for par in partitions_with_zeros(k-1, k):
  #      coef[par] = multinomial( *par[:-1])

    # for j in range(k+1):
    #    key = tuple(0 if l == j else 1 for l in range(k)) + (True,)
    #    coef[key] = factorial(k)

    # iterations
    pi = p**(k-1)
    for i in range(k, m+k):
        # for i in tqdm(range(k, m+k), total=m):
        # iterate over partitions
        ps = partitions_with_zeros(i, k)
        sum_over_partitions = 0
        for par in ps:

            # new coeficient for this partition
            d_coef = 0
            for j in range(0, k):
                # (l1, ..., lj-1, ..., lk)
                if par[j] == 0:
                    # multinomial coef with negative number is 0
                    continue
                tmp_p = par[:j] + (par[j] - 1,) + \
                    par[j+1:-1] + (par[-1] or par[j] == 1,)
               # update current coefitient
                if coef.get(tmp_p) is None:
                    coef[tmp_p] = multinomial(*par[:-1], coef=coef)
                d_coef += coef[tmp_p]
            coef[par] = d_coef

            # if partition doesn'0t include 0, we can add it to DeltaP
            if not par[-1]:
                # product of q_i^(l_i)
                prod = 1
                for index, l in enumerate(par[:-1]):
                    prod *= exp(qs[index], l)
                # inner sum
                sum_over_partitions += coef[par]*prod

        # new pi = p^i
        pi *= p

        # estimation difference
        dP = (1-p)*pi * sum_over_partitions
        # new estimate
        P += dP
    return P


def probability_exact(p: float, *qs: float) -> float:
    '''
    Use the exact formula for probability. Exponential time complexity.

    Returns the EXACT probability of parsing any word v, which include exactly len(qs) diffferent sybols x_i, and P(V -> x_i) = qs[i-1]

    with the grammar

    E -> E + cV [p] | c [1-p]

    V -> x_1 [q_1] | ... | x_n [q_n]

    Parameters
    --------------
    - m - Number of iterations
    - p = P(S -> V)
    - qs[i] = P(V -> x_i)

    Returns
    -----------
    Probability of P([w]), where w = c + \sum cx_i

    Example
    ---------
    Probability of parsing an expression c+cx_i in a grammar 

    E -> E + cV [p] | c [1-p]

    V -> x_1 [1]
    >>> probability(0.5,1)
    0.5

    '''
    ans = 0
    for i in powerset(qs):
        ans += even(len(qs) - len(i))*(1-p) / (1-p*sum(i))
    return ans


def integer_maximum_aprox(i, top):
    '''
    A close integer aproximation for (q1, ..., qk), which sums to i

    - top - (q1, ..., qk)
    - i = sum(q_j)
    '''
    ans = list(int(round(q, 0)) for q in top)
    diff = i - sum(ans)
    for index in range(len(ans)):
        if diff == 0:
            break
        ans[index] += sign(diff)
        diff -= sign(diff)
    return tuple(ans)


def multinomial_aprox(coef, i, *qs, gamma, epsilon, p):
    '''
    Sums the elements > epsilon of an inner sum. BFS

    - coef : dictionary of mult. coefifcients 
    '''
    ans = 0

    # top is at E[X] = i(q1, ..., qk)
    sum_q = sum(qs)
    top = (int(round(i*q/sum_q, 0)) for q in qs)
    # in queue are elements of the sum, yet to be visited
    # parametrised by partitions (l1, ..., lk), where l1 +...+ lk = i
    q = Queue()
    q.put(integer_maximum_aprox(i, top))

    visited = set()

    # number of sum elements that will get calculated
    n_possible_partitions = binom(i-1, len(qs) -1)
    n_sum_elements = int(gamma * n_possible_partitions)
    n_visited=0

    while not q.empty():
        # get new  partition
        partition = q.get()
        if partition in visited:
            continue
        visited.add(partition)

        # get new coeficient
        if coef.get(partition) is None:
            coef[partition] = multinomial(*partition, coef=coef)

        # get the element of the sum, coresponding to the partition
        prod = 1
        for index, l in enumerate(partition):
            prod *= exp(qs[index], l)
        # curr - current element of the sum
        curr = coef[partition]*prod
        # add to the inner sum aproximation
        ans += curr

        # update gamma (gamma = procent of elements CALCULATED)
        # alternativa:  max_error_coef = (1-p)*exp(i, p/(1-p))
        max_error_coef = (1-p)*exp(len(qs), p/(1-p))
        
        # first: calculate the number of elements LEFT OUT
        gamma = epsilon/(max_error_coef * curr)
        # second: change to the procent of elements CALCULATED
        gamma = 1 - gamma
        # update number of elements calculated (miight be bigeer)
        n_sum_elements = int(gamma * n_possible_partitions)
        
        # check, if we calculated enough partitions
        if n_visited >= n_sum_elements:
            break
        n_visited += 1

        # add new partitionss to the queue
        for j in range(len(partition)):
            for k in range(len(partition)):
                if k == j:
                    continue
                new_partition = list(partition)
                new_partition[k] += 1
                new_partition[j] -= 1
                if new_partition[k] <= 0 or new_partition[j] <= 0:
                    continue
                q.put(tuple(new_partition))
    return ans


def probability(m: int, p: float, *qs: float, epsilon=0.8) -> float:
    '''
    Return the aproximation of the probability of parsing any word v, which include exactly len(qs) diffferent sybols x_i, and P(V -> x_i) = qs[i-1]

    with the grammar

    E -> E + cV [p] | c [1-p]
Return the aproximation of the probability of parsing any word v, which include exactly len(qs) diffferent sybols x_i, and P(V -> x_i) = qs[i-1]

    with the grammar

    E -> E + cV [p] | c [1-p]

    V -> x_1 [q_1] | ... | x_n [q_n]

    Parameters
    --------------
    - m - Number of iterations
    - p = P(S -> V)
    - qs[i] = P(V -> x_i)
    - eps - maksimal error of aproximation allowed .

    Returns
    -----------
    Probability of P([w]), where w = c + \sum cx_i

    Example
    ---------
    Probability of parsing an expression c+cx_i in a grammar 

    E -> E + cV [p] | c [1-p]

    V -> x_1 [1]
    >>> probability(10,0.5,1)
    0.49951171875

    E -> E + cV [p] | c [1-p]

    V -> x_1 [1]
    >>> probability(10,0.5,1)
    0.49951171875

    '''
    # initalizing
    k = len(qs)
    P = 0

    # dictionary of multinomial coeficients
    coef = {}

    #precision - gamma
    sum_q = sum(qs)
    top = (int(round(k*q/sum_q, 0)) for q in qs)
    top = integer_maximum_aprox(k, top)
    max_element = multinomial(top, coef=coef) # bmultinomial()
    for index, q in enumerate(qs):
        max_element *= exp(q, top[index])

    # TODO: tto je zelooo groba ocena in je posledično gamma zelooooooo velik al mali al kaj
    max_error_coef = (1-p)*exp(k, p/(1-p))
    gamma = epsilon/(max_error_coef * max_element)
    gamma = 1 - gamma


    # p^i
    pi = p**(k-1)

    # iterations
    for i in range(k, m+k):
        # for i in tqdm(range(k, m+k), total=m):
        # iterate over partitions
        sum_over_partitions = multinomial_aprox(coef, i, *qs, gamma=gamma, epsilon=epsilon, p=p)

        # new pi = p^i
        pi *= p

        # estimation difference
        dP = (1-p)*pi * sum_over_partitions
        # new estimate
        P += dP
    return P

'''
grammar = PCFG.fromstring("""
 S -> S '+' 'c' V [0.5] | 'c' [0.5]
 V -> 'x' [0.7] | 'y' [0.3]
 """)
'''

# print(probability(3, 0.5, 0.2,0.2,0.2))