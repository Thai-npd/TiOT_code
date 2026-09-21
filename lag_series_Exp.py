import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import TiOT_lib
from TiOT_lib import TiOT, TAOT
import os
import seaborn as sns


def plot_graph(results, plot_file, index_name, x_label, y_label):
    indices = results[index_name]
    names = [k for k in results.keys() if k != index_name]

    sns.set(style="whitegrid", context="paper")
    plt.figure(figsize=(9, 7))
    plt.grid(False) 
    markers = ['o', 's', 'v', '^',  'P', 'X']
    linestyles = ['-', '--', '-.', ':', '-', '-', '-', '-']
    colors1 = ["#090311", "#d3a20e", "#109510", '#B22222']
    colors2 = ['#B22222', '#008080', "#582c05", '#e377c2']

    if index_name == r'$\ell$':
        colors = colors1
        markers = [None for _ in range(len(markers))]
        legend_size = 25
        ncol = 2
    elif index_name == 'w':
        colors = colors2
        linestyles = [None for _ in range(len(linestyles))]
        legend_size = 25
        ncol = 2
    handles = []
    labels = []

    for i, name in enumerate(names):
        (line,) = plt.plot(
            indices,
            results[name],
            label=name,
            linewidth=1.75,
            linestyle=linestyles[i],
            color=colors[i],
            marker=markers[i],
            markersize=10, 
        )
        handles.append(line)
        labels.append(name)

    plt.xlabel(x_label, fontsize=23)
    plt.ylabel(y_label, fontsize=23)

    plt.tick_params(axis="both", which="major", labelsize=21, bottom=True, left=True)
    plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=legend_size,
        ncol=ncol,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight") 
    plt.show()


def save_result(results, result_file):
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)

def read_result(result_file):
    df = pd.read_csv(result_file)
    results = df.to_dict(orient='list')
    return results

def safe_sqrt(x):
    # This function allow safe sqrt and will be needed when numerical error causing the distance to be negative. 
    return np.sqrt(np.clip(x, 0, None))

def dist_lag_exp(RUN = True):
    start = 0
    lags = range(0, 730)
    length = 365
    file_path = 'DailyDelhiClimateTrain.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    w_list = [0.2, 0.5, 0.8]
    metric_names = (
    [r"$\mathcal{D}_2(x^{(\ell)}, x^{(0)})$"]
    + [rf"$\mathcal{{W}}_{{2, {w}}}(x^{{(\ell)}}, x^{{(0)}})$" for w in w_list]
)
    results = {**{r'$\ell$' : lags}, **{name: [] for name in metric_names}}

    result_file = os.path.join('Experimental_outputs', "lag_series_data", f"Results distances of original with lag series {file_path}(lag {lags[0]} to {lags[-1]})_sqrt.csv")
    plot_file = os.path.join('Experimental_outputs', "lag_series_data", f"Plot distances of original with lag series {file_path}(lag {lags[0]} to {lags[-1]})_sqrt.pdf")

    if RUN:
        for lag in lags:
            results[r"$\mathcal{D}_2(x^{(\ell)}, x^{(0)})$"].append(safe_sqrt(TiOT(np.array(df['meantemp'].iloc[start:start + length]), np.array(df['meantemp'].iloc[start + lag:start + lag+length]))[0]))
            for w in w_list:
                results[rf"$\mathcal{{W}}_{{2, {w}}}(x^{{(\ell)}}, x^{{(0)}})$"].append(safe_sqrt(TAOT(np.array(df['meantemp'].iloc[start:start + length]), np.array(df['meantemp'].iloc[start + lag:start + lag+length]), w = w)[0]))
            print(f"Done Lag = {lag}")
        save_result(results, result_file)
        plot_graph(results, plot_file, r'$\ell$', r'$\ell$', 'Distance')
    else:
        results = read_result(result_file)
        plot_graph(results, plot_file, r'$\ell$', r'$\ell$', 'Distance')

def dist_w_exp(RUN = True):
    file_path = 'DailyDelhiClimateTrain.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    lags = [30 ,90,180,270] 
    w_list = [0.1 * i for i in range(11)]
    x = [df['meantemp'].iloc[:365]]
    result_file = os.path.join('Experimental_outputs', "lag_series_data", f"Results TAOT distances with w on {file_path}(lag {lags[0]} to {lags[-1]})_sqrt.csv")
    plot_file = os.path.join('Experimental_outputs', "lag_series_data", f"Plot TAOT distances with w on {file_path}(lag {lags[0]} to {lags[-1]})_sqrt.pdf")

    if RUN:
        for lag in lags:
            x.append(df['meantemp'].iloc[lag:lag + 365])
        results = {**{'w' : w_list}, **{rf'$\ell= {lag}$': [] for lag in lags}}
        for i ,lag in zip(range(1, len(x)), lags):
            for w in w_list:
                results[rf'$\ell= {lag}$'].append(np.sqrt(TiOT_lib.TAOT(x[0].to_list(), x[i].to_list(), w = w)[0]))
            print(f'Complete lag = {lag}')
        save_result(results, result_file)
        plot_graph(results, plot_file, 'w', r"$w$", rf"$\mathcal{{W}}_{{2, w}}$")
    else:
        results = read_result(result_file)
        plot_graph(results, plot_file, 'w', r"$w$", rf"$\mathcal{{W}}_{{2, w}}$")


if __name__ == "__main__":
    dist_lag_exp()
