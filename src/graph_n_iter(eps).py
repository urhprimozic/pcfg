import matplotlib.pyplot as  plt
from linear import n_iter
from utils import eq_qs
import numpy as np

p = 0.5
qs = eq_qs(10)

epsilons = np.linspace(1e-8, 0.1, 1500)
y = [n_iter(eps, p, *qs) for eps in epsilons]
plt.plot(epsilons, y)
plt.title('Number of iterations with respect to epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Number of iterations')
plt.savefig('../img/n_iter(time).png')
plt.show()