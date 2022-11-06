import matplotlib.pyplot as  plt
from linear import n_iter
from utils import eq_qs
import numpy as np

ps = [0.2, 0.5, 0.8]
k = 10
epsilons = np.linspace(1e-8, 0.1, 1500)
for p in ps:
    qs = eq_qs(k)
    y = [n_iter(eps, p, *qs) for eps in epsilons]
    plt.plot(epsilons, y, label=f'p={p}')
plt.title('Number of iterations with respect to epsilon')
plt.xlabel('Epsilon')
plt.legend()
plt.ylabel('Number of iterations')
plt.savefig('../img/n_iter(time).png')
plt.show()