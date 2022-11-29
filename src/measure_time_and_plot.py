from utils import eq_qs
from timeit import timeit
from linear import probability, probability_exact, get_cs_gamma, get_cs_gamma_uniform
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
import gc
import os


def number_of_iterations_for_timeit(statement, global_values):
    minimal_repetitions = 2
    maximal_repetitions = 10 ** 6
    expected_time = 3
    t = timeit(statement, globals=global_values, number=1)
    if t > 0:
        return int(max(minimal_repetitions, expected_time / t))
    else:
        return maximal_repetitions


def measure_approximation(p: float, *qs: float, epsilon=0.0001, adaptive=False, verbose=0, cs_getter=get_cs_gamma):
    statement = "probability(p, *qs, epsilon=epsilon, adaptive=adaptive, get_computed_size=get_computed_size)"
    global_values = {
        "probability": probability,
        "p": p, "qs": qs, "epsilon": epsilon,
        "adaptive": adaptive,
        "get_computed_size": cs_getter
    }
    number = number_of_iterations_for_timeit(statement, global_values)
    print(f"Will execute '{statement}' {number}-times.")
    exact_time = timeit(statement, globals=global_values, number=number) / number
    if verbose:
        qs_str = [f"{q:.3f}" for q in qs[:3]]
        if len(qs) > 3:
            qs_str.append(f"<{len(qs) - 3} other values>")
        print(f'probability({p:.2e}, qs={qs_str}, eps={epsilon:.2e}) done in {exact_time}')
    return exact_time


def measure_exact(p, *qs, verbose=0):
    statement = "probability_exact(p, *qs)"
    global_values = {"probability_exact": probability_exact, "p": p, "qs": qs}
    number = number_of_iterations_for_timeit(statement, global_values)
    exact_time = timeit(statement, globals=global_values, number=number) / number
    if verbose:
        print(f"Will execute '{statement}' {number}-times.")
        print('exact done in ', exact_time)
    return exact_time


def compare(out_dir_data, out_dir_img, linear_epsilons=True):
    os.makedirs(out_dir_data, exist_ok=True)
    os.makedirs(out_dir_img, exist_ok=True)

    ks = [5, 10, 20, 30]
    ks_exact = [5, 10, 20]
    p = 0.5

    # running time (epsilon)
    if linear_epsilons:
        epsilons = np.linspace(1e-8, 0.1, 10)
    else:
        epsilons = np.logspace(-8, -1, 10)
    print("Using epsilons", epsilons)
    print("Standard approximation")
    data = []
    for k in tqdm(ks):
        data.append([measure_approximation(p, *eq_qs(k), epsilon=eps, adaptive=False, verbose=1) for eps in epsilons])

    print("adaptive approach")
    data_adaptive = []
    for k in tqdm(ks):
        data_adaptive.append([measure_approximation(p, *eq_qs(k), epsilon=eps, adaptive=True) for eps in epsilons])

    print("uniform approach")
    data_uniform = []
    for k in tqdm(ks):
        data_uniform.append(
            [
                measure_approximation(
                    p, *eq_qs(k), epsilon=eps, adaptive=False, cs_getter=get_cs_gamma_uniform
                )
                for eps in epsilons
            ]
        )

    print("exact formula")
    data_exact = []
    for k in tqdm(ks_exact):
        if k == 5:
            measure_exact(p, *eq_qs(k))
        gc.collect()
        data_exact.append(measure_exact(p, *eq_qs(k)))
        print('k=', k, 'time: ', data_exact[-1])

    # save data
    scale = "lin" if linear_epsilons else "log"
    with open(os.path.join(out_dir_data, f'time_probability_{scale}.pickle'), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(out_dir_data, f'time_probability_adaptive_{scale}.pickle'), 'wb') as f:
        pickle.dump(data_adaptive, f)
    with open(os.path.join(out_dir_data, f'time_probability_uniform_{scale}.pickle'), 'wb') as f:
        pickle.dump(data_uniform, f)
    with open(os.path.join(out_dir_data, f'time_probability_exact_{scale}.pickle'), 'wb') as f:
        pickle.dump(data_exact, f)

    print('Aproximation times:')
    print(data)
    print('Adaptive aproximation  times:')
    print(data_adaptive)
    print("Uniform approximation times:")
    print(data_uniform)
    print('Exact times:')
    print(data_exact)

    colors = ['green', 'orange', 'blue', 'red']

    for i in range(len(ks)):
        plt.plot(epsilons, data[i], label=f'k={ks[i]}', color=colors[i])
    for i in range(len(ks_exact)):
        plt.axhline(data_exact[i], 0.045, 0.955, linestyle='dotted', label=f'exact formula, k={ks[i]}', color=colors[i])
        # plt.plot(0, data_exact[i], 'o',  label=f'exact formula, k={ks[i]}')
    plt.plot()
    plt.yscale('log')
    plt.legend()
    plt.title('Execution time of aproximation')
    plt.xlabel('Epsilon')
    plt.ylabel('Execution time')
    plt.savefig(os.path.join(out_dir_img, f'executions_{scale}.png'))
    plt.show()

    for i in range(len(ks)):
        plt.plot(epsilons, data_adaptive[i], label=f'k={ks[i]}', color=colors[i])
    for i in range(len(ks_exact)):
        plt.axhline(data_exact[i], 0.045, 0.95, linestyle='dotted', label=f'exact formula, k={ks[i]}', color=colors[i])
        # plt.plot(0, data_exact[i], 'o',  label=f'exact formula, k={ks[i]}')
    plt.yscale('log')
    plt.plot()
    plt.legend()
    plt.title('Execution time of adaptive aproximation')
    plt.xlabel('Epsilon')
    plt.ylabel('Execution time')
    plt.savefig(os.path.join(out_dir_img, f'executions_adaptive_{scale}.png'))
    plt.show()

    for i in range(len(ks)):
        plt.plot(epsilons, data[i], label=f'$\gamma$, k={ks[i]}', color=colors[i])
    for i in range(len(ks)):
        plt.plot(epsilons, data_adaptive[i], linestyle='dotted', label=f'Adaptive, k={ks[i]}', color=colors[i])
    for i in range(len(ks)):
        plt.plot(epsilons, data_uniform[i], linestyle='dashed', label=f'Uniform, k={ks[i]}', color=colors[i])

    plt.yscale('log')
    plt.plot()
    plt.legend()
    plt.title('Execution time of aproximations')
    plt.xlabel('Epsilon')
    plt.ylabel('Execution time')
    plt.savefig(os.path.join(out_dir_img, f'executions_adaptive_vs_normal_vs_uniform_{scale}.png'))
    plt.show()


if __name__ == "__main__":
    compare("../data2", "../img2", True)
    compare("../data2", "../img2", False)
