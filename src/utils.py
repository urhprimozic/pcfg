from functools import lru_cache

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
    clean = []
    zeros = []
    for p in f(i,k):
        if p[-1]:
            zeros.append(p[:-1])
        else:
            clean.append(p[:-1])
    return clean, zeros

