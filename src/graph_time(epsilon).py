import pickle
import matplotlib.pyplot as plt
import numpy as np

epsilons = np.linspace(1e-8, 0.1, 10)
ks = [5,10,20,30]
ks_exact = [5,10,20]
with open('../data/time_probability.pickle', 'rb') as f:
    data = pickle.load(f)
with open('../data/time_probability_adaptive.pickle', 'rb') as f:
    data_adaptive = pickle.load(f)
with open('../data/time_probability_exact.pickle', 'rb') as f:
    data_exact = pickle.load(f)

print('Aproximation times:')
print(data)
print('Adaptive aproximation  times:')
print(data_adaptive)
print('Exact times:')
print(data_exact)

colors = ['green', 'orange', 'blue', 'red']

for i in range(len(ks)):
    plt.plot(epsilons, data[i], label=f'k={ks[i]}', color=colors[i])
for i in range(len(ks_exact)):
    plt.axhline(data_exact[i], 0.045, 0.955, linestyle='dotted',  label=f'exact formula, k={ks[i]}',color=colors[i])
    # plt.plot(0, data_exact[i], 'o',  label=f'exact formula, k={ks[i]}')
plt.plot()
plt.yscale('log')
plt.legend()
plt.title('Execution time of aproximation')
plt.xlabel('Epsilon')
plt.ylabel('Execution time')
plt.savefig('../img/executions.png')
plt.show()

for i in range(len(ks)):
    plt.plot(epsilons, data_adaptive[i], label=f'k={ks[i]}',color=colors[i])
for i in range(len(ks_exact)):
    plt.axhline(data_exact[i], 0.045, 0.95, linestyle='dotted',  label=f'exact formula, k={ks[i]}',color=colors[i])
    # plt.plot(0, data_exact[i], 'o',  label=f'exact formula, k={ks[i]}')
plt.yscale('log')
plt.plot()
plt.legend()
plt.title('Execution time of adaptive aproximation')
plt.xlabel('Epsilon')
plt.ylabel('Execution time')
plt.savefig('../img/executions_adaptive.png')
plt.show()

for i in range(len(ks)):
    plt.plot(epsilons, data[i], label=f'$\gamma$, k={ks[i]}',color=colors[i])
for i in range(len(ks)):
    plt.plot(epsilons, data_adaptive[i], linestyle='dotted', label=f'Adaptive, k={ks[i]}',color=colors[i])

plt.yscale('log')
plt.plot()
plt.legend()
plt.title('Execution time of aproximations')
plt.xlabel('Epsilon')
plt.ylabel('Execution time')
plt.savefig('../img/executions_adaptive_vs_normal.png')
plt.show()