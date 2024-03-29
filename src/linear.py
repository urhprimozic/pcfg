from math import log
from utils import powerset, even, multinomial, exp, sign, is_positive
from queue import Queue, PriorityQueue
from scipy.special import binom
from typing import Callable, Tuple


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


def mode(i, *qs, coef):
    '''
    Maximum of Kappa on partitions

    - top - (q1, ..., qk)
    - i = sum(q_j)
    '''
    k = len(qs)
    sum_q = sum(qs)
    top = (int(round(i*q/sum_q, 0)) for q in qs)
    ans = list(int(round(q, 0)) for q in top)
    diff = i - sum(ans)
    for index in range(len(ans)):
        if diff == 0:
            break
        ans[index] += sign(diff)
        diff -= sign(diff)
    # some close approximation
    ans =  tuple(ans)
    # bfs with priority queue
    q = PriorityQueue()
    q.put((-kappa(ans, *qs, coef=coef), ans ))
    max_kappa = 0
    visited=set()
    while not q.empty():
        value, par = q.get()
        # fix value to be positive
        value = -value
        visited.add(par)
        # update kappa only if this partition is strictly positive
        #(wqe dont want too look at partitions with zero )
        if is_positive(par):
            if value >= max_kappa:
                ans = par
                max_kappa = value
        # check neighbours 
        for j in range(k):
            for jj in range(k):
                if jj == j:
                    continue
                new_par = list(par)
                new_par[jj] += 1
                new_par[j] -= 1
                new_par = tuple(new_par)
                # check if the new partition has more zeros than before
                if new_par[j] <= 0:
                    continue
                if new_par in visited:
                    continue
                # check if new par is higher or equal
                new_value = kappa(new_par, *qs, coef=coef)
                if new_value >= value:
                    q.put((-new_value, new_par))
    return ans



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
    return prod*multinomial(*partition, coef=coef)


def get_cs_constant_gamma(max_i, M, i, p, k, pi, epsilon, prev_cs):
    '''
    S(i) - the first (1-gamma)% partitions
    '''
    if i > k:
        return prev_cs
    denominator = f_gamma(M, i, p, k, pi) * max_i
    gamma = min(epsilon/denominator, 1)
    return gamma, 1-gamma


def get_cs_gamma(max_i, M, i, p, k, pi, epsilon, prev_cs):
    '''
    S(i) - the first (1-gamma_i)% of partitions, closest to the maximum of kappa

    gamma_1 >= gamma_2 >= gamma_3 ... 
    '''
    gamma, _ = prev_cs

    denominator = f_gamma(M, i, p, k, pi) * max_i
    if i == k:
        gamma = min(epsilon/denominator, 1)
    else:
        gamma = min(epsilon/denominator, gamma, 1)
    return gamma, 1-gamma


def get_cs_gamma_uniform(max_i, M, i, p, k, pi, epsilon, prev_cs):
    gamma, _ = prev_cs
    if i == k:
        gamma = epsilon / M / (1 - p) / max_i / pi
    else:
        gamma = gamma * (i - k) / p / i
    # it is important to keep gamma as high as possible,
    # so do not say gamma = max(1, gamma)
    cs = max(0.0, 1.0 - gamma)
    return gamma, cs


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
    gamma, cs = prev_cs
    if i > k:
        # cs = (1-gamma)/(i-1)²
        new_cs = cs*((i-1)**2)/(i**2)
        return 1 - new_cs, new_cs

    denominator = f_poly(M, i, p, k, pi)
    diff = diff_poly(M, i, p, k, pi)
    gamma = min((epsilon/max_i - diff)/denominator, 1)
    return gamma, (1-gamma)/(i**2)


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
        ans += (1-p)*pi*binom(i-1, k-1)*(exp(2, i)-1)/(exp(2, i))
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
        ans += (1-p)*pi*binom(i-1, k-1)/(exp(2, i))
    return ans


def get_cs_decrease_exp(max_i, M, i, p, k, pi, epsilon, prev_cs):
    gamma, cs = prev_cs
    if i > k:
        # cs = (1-gamma)/(i-1)²
        new_cs = cs*((i-1)**2)/(i**2)
        return 1 - new_cs, new_cs

    denominator = f_exp(M, i, p, k, pi)
    diff = diff_exp(M, i, p, k, pi)
    gamma = min((epsilon/max_i - diff)/denominator, 1)
    return gamma, (1-gamma)/exp(2, i)


def multinomial_aprox(
        coef: dict,
        i: int, *qs,
        epsilon: float,
        p: float,
        M: int,
        pi: float,
        prev_computed_size: Tuple[float, float],
        get_computed_size: Callable[
            [float, int, int, float, int, float, float, Tuple[float, float]],
            Tuple[float, float]
        ],
        verbose: int
):
    '''
    Sums the first get_computed_size% elements  of an inner sum using BFS

    - coef : dictionary of mult. coefficients
    - i - current iteration
    - get_computed_size(max_i, M, i, p, k, pi, epsilon, prev_gama_and_cs):
            - function that returns the pair gamma, computed size
              (upper bound for proportion of skipped elements, proportion of elements computed in this iteration)

    Returns
    ----------
    sum_of_elemets, new_gamma, Ai (aprox of the error made)
    '''
    ans = 0
    k = len(qs)
    # maximum of kappa
    top = mode(i, *qs, coef=coef)
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
    gamma_and_computed_size = get_computed_size(
        maximal_element, M, i, p, k, pi, epsilon, prev_computed_size)
    computed_size = gamma_and_computed_size[1]
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

    return ans, gamma_and_computed_size, Ai


def adaptive_multinomial_aprox(coef: dict, i: int, *qs, epsilon: float, verbose):
    '''
    Sum the biggest elements of the inner sum, until the error is smaller than epsilon

    Returns
    ----------
    sum_of_elemets, new_gamma, Ai (aprox of the error made)
    '''
    ans = 0
    k = len(qs)
    # top is at E[X] = i(q1, ..., qk)
    top = mode(i, *qs, coef=coef)
    maximal_element =  kappa(top, *qs, coef=coef)
    # parametrised by partitions (l1, ..., lk), where l1 +...+ lk = i
    q = PriorityQueue()
    q.put((-maximal_element,top))

    visited = set()

    # update minimal element calculated
    minimal_element = maximal_element

    # error estimation:
    n_partitions = binom(i-1, k-1)
    n_visited = 0

    while not q.empty():
        # get new  partition
        value, partition = q.get()
        value = -value

        # add the partition
        ans += value

        if partition in visited:
            continue

        # this was unvisited, change to visited
        visited.add(partition)
        n_visited += 1

        # update minimal element
        minimal_element = min(minimal_element, value)

        # recalculate the error:
        error_estimation = minimal_element*(n_partitions - n_visited)
        if error_estimation < epsilon:
            break

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
                new_partition = tuple(new_partition)
                new_value = kappa(new_partition, *qs, coef=coef)
                q.put((-new_value, new_partition))
    # print(f"    Computed size: {n_visited / n_partitions}")
    return ans,  epsilon - error_estimation 


def n_iter(epsilon, p, *qs):
    '''
    Returns minimal M, such that the finite sum for i=k, k+1, ..., M
    differs from infinite sum i >= k for less than epsilon.
    '''
    Q = sum(qs)
    pQ = p*Q
    M = log(epsilon * (1-pQ)/(1-p)) / log(pQ)
    return int(M) + 1


def probability(p: float, *qs: float, epsilon=0.0001, get_computed_size=get_cs_gamma, verbose=0, adaptive=0, m=None):
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
    # get number of iterations
    if m is None:
        # The error, done by only computing finite number of elemets, is at most epsilon / 2
        # the error, done by leaving elements of the inner sum is at most epsilon /2
        # --> the total error < epsilon
        epsilon /= 2
        m = n_iter(epsilon, p, *qs)
        if verbose:
            print(f'Number of iterations: {m}')

    # initalizing
    k = len(qs)
    P = 0
    # approximation of an error, caused by skiping elements up to this point
    A = 0
    # dictionary of multinomial coeficients
    coef = {}

    # set gamma to whatever
    gamma_and_computed_size = (0.0, 1.0)

    # p^i
    pi = exp(p, k-1)  # p**(k-1)
    epsilon0 = epsilon
    # iterations
    for i in range(k, m+k):
        # for i in tqdm(range(k, m+k), total=m):
        # iterate over partitions

        # decrease the margin for error
        epsilon -= A

        # new pi = p^i
        pi *= p

        # get inner sum approximation, the new gamma, anb the new A_i
        if adaptive:
            j = m+k-1-i
            Ai = 0
            
            sum_over_partitions, epsilon = adaptive_multinomial_aprox(
                coef, j, *qs, epsilon=epsilon, verbose=verbose)
        else:
            if i == k and epsilon0 != epsilon:
                raise ValueError(f"Sorry, no go. Need epsilon0 = epsilon if get_computed_size is uniform")
            sum_over_partitions, gamma_and_computed_size, Ai = multinomial_aprox(
                coef, i, *qs, epsilon=epsilon, p=p, M=m, pi=pi,
                prev_computed_size=gamma_and_computed_size, get_computed_size=get_computed_size, verbose=verbose
            )
            if verbose:
                print(f"   Computed size at i = {i}: {gamma_and_computed_size[1]}")

        # update A
        A += Ai

        # estimation difference
        dP = (1 - p) * pi * sum_over_partitions
        # new estimate
        P += dP
    return P
