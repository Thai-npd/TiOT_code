import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
import os
import seaborn as sns


def eTiOT(X,Y, eps = 0.1):
    if len(X) < 1000:
        subprob_tol = 10**-7
    else:
        subprob_tol = 0.01
    return TiOT_lib.eTiOT(X,Y, eps = eps, subprob_tol=subprob_tol, timing=True)

def eTAOT(X,Y):
    return TiOT_lib.eTAOT(X,Y, eps = 0.1,  timing=True)

def TiOT(X,Y):
    return TiOT_lib.TiOT(X,Y, timing=True, detail_mode=True)

def TAOT(X,Y):
    return TiOT_lib.TAOT(X,Y, timing=True)

def save_result(results, result_file):
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)

def read_result(result_file):
    df = pd.read_csv(result_file)
    results = df.to_dict(orient='list')
    return results

def generate_two_gaussian(n, noise=0.01, seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    peak_positions = [0.5 * n, 0.9 * n]  
    base_widths = [0.01 * n, 0.02 * n]     
    base_heights = [1.2, 0.2]             
    y1 = np.zeros_like(t, dtype=float)
    for p, h, w in zip(peak_positions, base_heights, base_widths):
        y1 += h * np.exp(-(t - p) ** 2 / (2 * w ** 2))
    y1 += rng.normal(0, noise, size=n)
    y2 = np.zeros_like(t, dtype=float)
    for p, h, w in zip(peak_positions, base_heights, base_widths):
        p_shift = p - 0.4 * n 
        h_var   = h * rng.uniform(0.9, 1.1)          
        w_var   = w * rng.uniform(0.9, 1.1)       
        y2 += h_var * np.exp(-(t - p_shift) ** 2 / (2 * w_var ** 2))
    y2 += rng.normal(0, noise, size=n)
    return y1, y2

def plot_runtime(results, plot_file):
    lengths = results['len']
    metric_names = [k for k in results.keys() if k != 'len']
    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 7))
    plt.grid(False)
    markers = ['s', 'o',  '^', 'D', 'v', 'P', 'X']
    i = 0
    colors1 = ['#6d36ab', '#ff7f0e', '#1f77b4', '#2ca02c', '#d62728'] 
    name_dict = {'TiOT':   'TiOT', 'eTiOT':  'eTiOT', 'TAOT':   'OT', 'eTAOT':  'eOT'}
    for name in metric_names:
        plt.plot(lengths[:len(results[name])], results[name], label = name_dict[name], color = colors1[i], linewidth=1.75, marker = markers[i], markersize = 12)
        i+=1

    plt.tick_params(axis="both", which="major", labelsize=21, bottom=True, left=True)
    plt.xticks([length for length in lengths if length > 1000])
    plt.xlabel("Series' lengths", fontsize = 21)
    plt.ylabel("Running time (s)", fontsize = 21)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=25,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300) 
    plt.show()

def runtime_experiment(RUN = True):
    dataset_name = 'Gaussian'
    lengths = [100, 300, 500, 3000, 6000, 9000, 12000] 
    metrics = [TiOT, TAOT, eTiOT,  eTAOT] 
    metric_names = [metric.__name__ for metric in metrics]
    results = {**{'len': lengths}, **{name: [] for name in metric_names}}
    result_file = os.path.join('Experimental_outputs', "runningtime_data", f"Results runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})" + ".csv")
    plot_file = os.path.join('Experimental_outputs', "runningtime_data", f"Plot runtime_graph {dataset_name}(size {lengths[0]} to {lengths[-1]})"  +  ".pdf")

    if RUN:
        for length in lengths:
            X, Y = generate_two_gaussian(n = length)
            print(f"Start length {length}")
            for metric in metrics:
                if metric == TiOT and length > 900:
                    print(f"  ---> TiOT is skipped")
                    results[metric.__name__].append(None)
                    continue
                print(f"  ---> Start metric {metric.__name__}")
                results[metric.__name__].append(metric(X,Y)[-1])
        save_result(results, result_file)
        plot_runtime(results, plot_file)
    else:
        results = read_result(result_file)
        plot_runtime(results, plot_file)

def deviation_experiment(RUN = True):
    size = 100
    seeds = [i for i in range(100)]
    lamdas = [2, 10, 50, 100] 
    eps_list = 1/np.array(lamdas)
    plot_file = os.path.join('Experimental_outputs', "runningtime_data", 'boxplot' + ".pdf")
    result_file = os.path.join('Experimental_outputs', "runningtime_data", 'boxplot' + ".csv")
    result_w_file = os.path.join('Experimental_outputs', "runningtime_data", 'boxplot_w' + ".csv")
    results = dict()
    results_w = dict()
    if RUN :
        for eps in eps_list:
            deviation = []
            w_deviation = []
            print(f"Start experimenting with eps = {eps}")
            for seed in seeds:
                X, Y = generate_two_gaussian(n = size, seed=seed)
                emd, plan_emd,  w_emd, t = TiOT(X,Y)
                sinkhorn, plan_sink, w_sink,t  = eTiOT(X,Y, eps=eps)
                print(f"At seed {seed}: emd = {emd} with w = {w_emd}, sinkhorn = {sinkhorn} with w = {w_sink}")
                deviation.append(np.abs(sinkhorn - emd) / emd)
                w_deviation.append(np.abs(w_emd - w_sink) / w_emd )
            results[eps] = deviation
            results_w[eps] = w_deviation
    else:
        results = read_result(result_file)
        results_w = read_result(result_w_file)

    save_result(results, result_file)
    save_result(results_w, result_w_file)

    positions = np.arange(1, len(lamdas) + 1)
    width = 0.3

    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(False)
    bp1 = ax.boxplot(list(results.values()), positions=positions - width/2, widths=0.25,
                    patch_artist=False,
                    boxprops=dict(color='#B22222', linewidth=1.2),
                    capprops=dict(color='#B22222', linewidth=1.2),
                    whiskerprops=dict(color='#B22222', linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3, markerfacecolor='#B22222', markeredgecolor='#B22222'),
                    medianprops=dict(color='#B22222', linewidth=1.2))

    bp2 = ax.boxplot(list(results_w.values()), positions=positions + width/2, widths=0.25,
                    patch_artist=False,
                    boxprops=dict(color='#008080', linewidth=1.2, linestyle='--'),
                    capprops=dict(color='#008080', linewidth=1.2, linestyle='--'),
                    whiskerprops=dict(color='#008080', linewidth=1.2, linestyle='--'),
                    flierprops=dict(marker='s', markersize=3, markerfacecolor='#008080', markeredgecolor='#008080'),
                    medianprops=dict(color='#008080', linewidth=1.2))

    # Axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(lamdas)
    ax.tick_params(axis="both", which="major", labelsize=22, bottom=True, left=True)
    ax.set_ylabel('Distribution of deviation', fontsize=21, labelpad=12)
    ax.set_xlabel(r'$1/\varepsilon $', fontsize=21, labelpad=12)
    ax.legend(
    [bp2["boxes"][0], bp1["boxes"][0]],
    [
        r"$\frac{|w^*_{\varepsilon} - w^*|}{w^*}$",
        r"$\frac{| \langle C(w^*_{\varepsilon}), \pi^*_{\varepsilon} \rangle - \langle C(w^*), \pi^* \rangle |}{\langle C(w^*), \pi^* \rangle}$"
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1), 
    ncol=2,
    fontsize=33,
    frameon=False                  
)
    plt.yscale("log") 
    plt.tight_layout()
    plt.savefig(plot_file , dpi=300)
    plt.show()


if __name__ == "__main__":
    runtime_experiment()
    deviation_experiment()
