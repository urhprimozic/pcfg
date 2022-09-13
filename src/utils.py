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


def partitions_bottom_up(i, k):
    '''
   Yields partitions of i on k elements with all the elements of partition at least 1
    '''
    if i < k:
        return None 
    memo = []
    x = i-k+1
    y = 1
    ans = [(i,)]
    while not (x == i and y == k):
        x += 1
    raise NotImplementedError
    return ans
