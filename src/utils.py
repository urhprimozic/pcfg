from functools import lru_cache
from itertools import chain, combinations

def partitions(i,k):
    @lru_cache(maxsize=None)
    def f(i,k):
        if i < k or k <= 0:
            return []
        if k==1:
            return [(i,)]
        if i==k:
            return [tuple(1 for _ in range(k))]
        ans = []
        for j in range(1, i-k+2):
            for p in f(i-j, k-1):
                new_partition = (j,) + p
                ans.append(new_partition)
        return ans 
        
    return f(i,k)


def partitions_with_zeros(i,k):
    '''
    Returns all the partitions of i in k integers (might be zero).
    Format:
    p[:-1] - partiotion
    p[-1] - True if it includes zero 
    '''
    @lru_cache(maxsize=None)
    def f(i,k):
        if k <= 0:
            return []
        if k==1:
            return [(i,i==0)]
        ans = []
        for j in range(0, i+1):
            for p in f(i-j, k-1):
                new_partition = (j,) + p[:-1] + (p[-1] or (j == 0),)
                ans.append(new_partition)
        return ans 
    return f(i,k)



def powerset(s):
    '''
    Returns powerset
    '''
    s = list(s)
    return chain.from_iterable(combinations(s, i) for i in range(len(s)+1))

def even(i :int):
    '''
    Returns 1 if even, -1 otherwise
    '''
    if i % 2 == 0:
        return 1
    return -1
