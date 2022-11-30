from utils import  multinomial, exp, partitions, eq_qs, partitions_with_zeros
from queue import Queue
from linear import mode, kappa
import numpy as np
from tqdm import tqdm

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
    # print('real mode: ', max_par, '   bfs: ', f_mode)
    # print('real max value: ', max_kappa, '   bfs: ', f_kappa)
    settings= {'qs': qs, 'i': i}
    return {'settings' : settings, 'real_mode': max_par, 'f_mode' : f_mode, 'real_value': max_kappa, 'f_value' : f_kappa}

def run_mode_test(ks, number_of_i, divisors = [1,2,5]):
    print('Testing for q1=q1=..=qn...')
    results = []
    errors = []
    for k in tqdm(ks, total=len(ks)):
        for divisor in divisors:
            qs = eq_qs(k, divisor=divisor)
            iss = [k+j*5 for j in range(1, number_of_i+1)]
            for i in iss:
                #print: "testing with qs = {qs} and i = i"
                result = mode_test(*qs, i=i)
                results.append(result)
                # check for error
                if result['real_value'] != result['f_value']:
                    print(f'ERROR! for qs={qs}, i={i}')
                    print('real_value: ', result['real_value'], 'f_value: ', result['f_value'])
                    errors.append(result)
    print('testing random values of qj')
    for k in tqdm(ks, total=len(ks)):
        qs = np.random.rand(k)
        qs = qs / np.linalg.norm(qs)
        qs = qs/sum(qs)
        iss = [k+j*5 for j in range(1, number_of_i+1)]
        for i in iss:
            #print: "testing with qs = {qs} and i = i"
            result = mode_test(*qs, i=i)
            results.append(result)
            # check for error
            if result['real_value'] != result['f_value']:
                print(f'ERROR! for qs={qs}, i={i}')
                print('real_value: ', result['real_value'], 'f_value: ', result['f_value'])
                errors.append(result)
    return results, errors
