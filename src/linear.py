from audioop import mul
from math import factorial
from math import log
from numpy import partition
from utils import partitions, powerset, even, multinomial, exp, sign
from tqdm import tqdm
from nltk import PCFG, nonterminals
import matplotlib.pyplot as plt
from queue import Queue
from scipy.special import binom
from mpmath import hyp2f1  # hypergeometric function


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


def f_gamma(M, j, p, k, pi):
    '''
    Assertion for futture error 
    Returns f(M,j) from the article. 
    '''
    ans = 0
    if pi is None:
        pi = exp(p, j)
    pi /= p
    for i in range(j, M+k+1):
        pi *= p
        ans += (1-p)*pi*binom(i-1, k-1)
    return ans


def kappa(partition, *qs,  coef):
    prod = 1
    for index, l in enumerate(partition):
        prod *= exp(qs[index], l)
    return prod*multinomial(*qs, coef=coef)


def get_cs_constant_gamma(max_i, M, i, p, k, pi, epsilon, prev_cs):
    '''
    S(i) - the first (1-gamma)% partitions
    '''
    if i > k:
        return prev_cs
    denominator = f_gamma(M, i, p, k, pi) * max_i
    gamma = min(epsilon/denominator, 1)
    return 1-gamma


def get_cs_gamma(max_i, M, i, p, k, pi, epsilon, prev_cs):
    '''
    S(i) - the first (1-gamma_i)% of partitions, closest to the maximum of kappa

    gamma_1 >= gamma_2 >= gamma_3 ... 
    '''
    # extract the previous gamma
    gamma = 1-prev_cs

    denominator = f_gamma(M, i, p, k, pi) * max_i
    if i == k:
        gamma = min(epsilon/denominator, 1)
    else:
        gamma = min(epsilon/denominator, gamma, 1)
    return 1-gamma


def diff_poly(M, j, p, k, pi):
    '''
    TODO
    '''
    ans = 0
    if pi is None:
        pi = exp(p, j)
    pi /= p
    for i in range(j, M+k+1):
        pi *= p
        ans += (1-p)*pi*binom(i-1, k-1)*(i**2-1)/(i**2)
    return ans

def f_poly(M, j, p, k, pi):
    '''
    TODO
    '''
    ans = 0
    if pi is None:
        pi = exp(p, j)
    pi /= p
    for i in range(j, M+k+1):
        pi *= p
        ans += (1-p)*pi*binom(i-1, k-1)/(i**2)
    return ans

def get_cs_decrease_poly(max_i, M, i, p, k, pi, epsilon, prev_cs):
    if i > k:
        # prev_cs = (1-gamma)/(i-1)²
        return prev_cs*((i-1)**2)/(i**2)

    denominator = f_poly(M, i, p, k, pi)
    diff = diff_poly(M, i, p, k, pi)
    gamma = min((epsilon/max_i - diff)/denominator, 1 )
    return (1-gamma)/(i**2)

def diff_exp(M, j, p, k, pi):
    '''
    TODO
    '''
    ans = 0
    if pi is None:
        pi = exp(p, j)
    pi /= p
    for i in range(j, M+k+1):
        pi *= p
        ans += (1-p)*pi*binom(i-1, k-1)*(exp(2,i)-1)/(exp(2,i))
    return ans

def f_exp(M, j, p, k, pi):
    '''
    TODO
    '''
    ans = 0
    if pi is None:
        pi = exp(p, j)
    pi /= p
    for i in range(j, M+k+1):
        pi *= p
        ans += (1-p)*pi*binom(i-1, k-1)/(exp(2,i))
    return ans

def get_cs_decrease_exp(max_i, M, i, p, k, pi, epsilon, prev_cs):
    if i > k:
        # prev_cs = (1-gamma)/(i-1)²
        return prev_cs*((i-1)**2)/(i**2)

    denominator = f_exp(M, i, p, k, pi)
    diff = diff_exp(M, i, p, k, pi)
    gamma =min( (epsilon/max_i - diff)/denominator, 1)
    return (1-gamma)/exp(2,i)


def multinomial_aprox(coef: dict, i: int, *qs, epsilon: float, p: float, M: int, pi: float, prev_computed_size: float, get_computed_size):
    '''
    Sums the first get_computed_size% elements  of an inner sum using BFS

    - coef : dictionary of mult. coefifcients 
    - i - current iteration
    - get_computed_size(max_i, M, i, p, k, pi, epsilon, prev_cs): - function that returns percent of elements computed in this iteration

    Returns
    ----------
    sum_of_elemets, new_gamma, Ai (aprox of the error made)
    '''
    ans = 0
    k = len(qs)
    # top is at E[X] = i(q1, ..., qk)
    sum_q = sum(qs)
    top = (int(round(i*q/sum_q, 0)) for q in qs)
    top = integer_maximum_aprox(i, top)
    # in queue are elements of the sum, yet to be visited
    # parametrised by partitions (l1, ..., lk), where l1 +...+ lk = i
    q = Queue()
    q.put(top)

    visited = set()

    # get maximal element
    maximal_element = kappa(top, *qs, coef=coef)
    # update minimal element calculated
    minimal_element = maximal_element
    # EPSILON SHOULD ALREADY BE REDUCED BY A
    computed_size = get_computed_size(
        maximal_element, M, i, p, k, pi, epsilon, prev_computed_size)
    error_size = 1 - computed_size

    # number of sum elements that will get calculated
    n_possible_partitions = binom(i-1, len(qs) - 1)
    # increased by 1, so at least 1 will get calculated
    n_sum_elements = int(computed_size * n_possible_partitions) + 1

    while not q.empty():
        # get new  partition
        partition = q.get()
        if partition in visited:
            continue

        # this was unvisited, change to visited
        visited.add(partition)

        # check if we can make any more moves
        if n_sum_elements <= 0:
            break
        n_sum_elements -= 1

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

        # update minimal element
        minimal_element = min(minimal_element, curr)
        # TODO also update maximal and change gamma accordnilgnli

        # add new partitionss to the queue
        for j in range(len(partition)):
            for jj in range(len(partition)):
                if jj == j:
                    continue
                new_partition = list(partition)
                new_partition[jj] += 1
                new_partition[j] -= 1
                if new_partition[j] <= 0:
                    continue
                q.put(tuple(new_partition))

    # get better aprox for past error
    Ai = (1-p)*pi * error_size * binom(i-1, k-1) * minimal_element
    # alternativa:  Ai = (1-p)*pi * gamma * binom(i-1, k-1) * maximal_element

    return ans, computed_size, Ai

def n_iter(epsilon, p, *qs):
    '''
    Returns minimal M, such that the finite sum for i=k, k+1, ..., M
    differs from infinite sum i >= k for less than epsilon.
    '''
    Q = sum(qs)
    pQ = p*Q
    M = log(epsilon * (1-pQ)/(1-p)) / log(pQ)
    return int(M) + 1

def probability(p: float, *qs: float, epsilon=0.0001, get_computed_size=get_cs_gamma, verbose=0):
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
    - epsilon - margin for error, caused by 2nd degree aproximation
    - exponent - C (each iteration, we divide the number of computed elements by i^C)

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
    '''
    # The error, done by only computing finite number of elemets, is at most epsilon / 2
    # the error, done by leaving elemnts of the inner sum is at most epsilon /2 
    # --> the total error < epsilon
    epsilon /= 2
    # get number of iterations
    m = n_iter(epsilon, p, *qs)

    # initalizing
    k = len(qs)
    P = 0
    # aproximation of an error, caused by skiping elements up to this point
    A = 0
    # dictionary of multinomial coeficients
    coef = {}

    # set gamma to whatever
    computed_size = 1

    # p^i
    pi = exp(p, k-1)  # p**(k-1)

    # iterations
    for i in range(k, m+k):
        # for i in tqdm(range(k, m+k), total=m):
        # iterate over partitions

        # decrease the margin for error
        epsilon -= A

        # new pi = p^i
        pi *= p

        # get inner sum aproximation, the new gamm, anbd the new A_i
        sum_over_partitions, computed_size, Ai = multinomial_aprox(
            coef, i, *qs, epsilon=epsilon, p=p, M=m, pi=pi, prev_computed_size=computed_size, get_computed_size=get_computed_size)

        # update A
        A += Ai

        # estimation difference
        dP = (1-p)*pi * sum_over_partitions
        # new estimate
        P += dP
        if verbose:
            print('dp = ', dP, 'Ai = ', Ai)
    return P
