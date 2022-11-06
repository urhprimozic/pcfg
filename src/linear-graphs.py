from utils import eq_qs
import time
from linear import probability, probability_exact
import matplotlib.pyplot as plt
from tqdm import tqdm

def measure_time(p: float, *qs: float, epsilon=0.0001, adaptive=0):
    start = time.time()
    probability(p, *qs, epsilon=epsilon, adaptive=adaptive)
    end = time.time()
    print ('done in ', end-start)
    return end-start
    

ks = [5,10,20,30]
p = 0.5
### running time (epsilon)
epsilons = [10**(-i) for i in range(1, 8)]
data = []
# ni verobsa: data = [[measure_time(p, *eq_qs(k), epsilon= eps, adaptive=0) for eps in epsilons] for k in ks ]
for k in tqdm(ks, total=len(ks)):
    data.append([measure_time(p, *eq_qs(k), epsilon= eps, adaptive=0) for eps in epsilons])


for i in range(len(ks)):
    plt.plot(epsilons, data[i], label=f'k={ks[i]}')
plt.legend()
plt.show()