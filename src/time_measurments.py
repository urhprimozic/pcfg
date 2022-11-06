from utils import eq_qs
import time
from timeit import timeit
from linear import probability, probability_exact
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np

def measure_time(p: float, *qs: float, epsilon=0.0001, adaptive=0, verbose=0):
    start = time.time()
    probability(p, *qs, epsilon=epsilon, adaptive=adaptive)
    end = time.time()
    exect_time = end - start
 #   exect_time = timeit(probability(p, *qs, epsilon=epsilon, adaptive=adaptive)) 
    if verbose:
        print(f'probability({p}, k={k}, eps={epsilon}) done in {exect_time}')
    return exect_time

def measure_exact(p, *qs, verbose=0):
    start = time.time()
    probability(p, *qs)
    end = time.time()
    if verbose:
        print('exact done in ', end - start)
    return end - start 

ks = [5,10,20,30]
p = 0.5
############## running time (epsilon)
epsilons = np.linspace(1e-8, 0.1, 10)

# nomral probabiliy
data = []
for k in tqdm(ks, total=len(ks)):
    data.append([measure_time(p, *eq_qs(k), epsilon= eps, adaptive=0) for eps in epsilons])

# adaptive approach
data_adaptive = []
for k in tqdm(ks, total=len(ks)):
    data_adaptive.append([measure_time(p, *eq_qs(k), epsilon= eps, adaptive=1) for eps in epsilons])

# exact formula
data_exact=[]
for k in tqdm(ks, total=len(ks)):
    data_exact.append(measure_exact(p, *eq_qs(k)))



# save data
with open('../data/time_probability.pickle', 'wb') as f:
    pickle.dump(data, f)
with open('../data/time_probability_adaptive.pickle', 'wb') as f:
    pickle.dump(data_adaptive, f)
with open('../data/time_probability_exact.pickle', 'wb') as f:
    pickle.dump(data_exact, f)

for i in range(len(ks)):
    plt.plot(epsilons, data[i], label=f'k={ks[i]}')
plt.plot()
plt.yscale('log')
plt.legend()
plt.show()