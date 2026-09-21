# eTiOT-selected-w experiment, with timing.
#
# Two metrics are run through the identical kNN pipeline and timed separately:
#
#   eTAOT2  : w is SELECTED automatically, per eps, by solving eTiOT on sampled
#             cross-class train pairs and averaging the returned optimal w*_i with
#             distance-decreasing weights (closest cross-class pairs weigh most).
#             The 1-NN then uses TiOT_lib.eTAOT2 = eTAOT with costmatrix2
#             (w*spatial + (1-w)*temporal, median-normalized) at the fixed w'(eps).
#             Training cost = w-select + eps cross-validation. NO w tuning at all.
#
#   oriTAOT : the TAOT-benchmark form (costmatrix0: spatial + w*temporal, median-
#             normalized) at the PRE-TUNED w taken from the reference table. Its
#             measured training cost covers only the eps cross-validation -- the w
#             grid search the TAOT paper performs (19 w values, each requiring the
#             same CV) is reported as an extrapolation (x10 / x19).
#
# The report file records accuracies of both metrics plus a timing section.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import pandas as pd
import TiOT_lib
import multiprocessing
from itertools import combinations
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy import stats
from tqdm import tqdm

# Sentinel distance for a failed (non-finite) OT solve. Larger than any real distance, so
# such a pair is never selected as a nearest neighbour.
DIST_INF = 1e12
N_WORKERS = 16


def eTAOT2_metric(X1, X2, eps, w):
    """eTAOT2: costmatrix2 = median-normalized (w*spatial + (1-w)*temporal)."""
    return TiOT_lib.eTAOT2(X1, X2, w=w, eps=eps)[0]

def oriTAOT_metric(X1, X2, eps, w):
    """TAOT-benchmark form: costmatrix0 = median-normalized (spatial + w*temporal)."""
    return TiOT_lib.eTAOT(X1, X2, w=w, eps=eps, costmatrix=TiOT_lib.costmatrix0)[0]

def build_metric(metric_name, eps, w):
    if metric_name == 'eTAOT2':
        return partial(eTAOT2_metric, eps=eps, w=w)
    elif metric_name == 'oriTAOT':
        return partial(oriTAOT_metric, eps=eps, w=w)
    raise ValueError(f"Unknown metric_name: {metric_name}")


def process_data(dataset_name):
    train_file = os.path.join("time_series_datasets", dataset_name, dataset_name + "_TRAIN.txt")
    test_file = os.path.join("time_series_datasets", dataset_name, dataset_name + "_TEST.txt")

    with open(train_file, "r") as file:
        data = np.array([line.strip().split() for line in file], dtype=float)

    Y_train = data[:, 0]
    X_train = data[:, 1:]

    with open(test_file, "r") as file:
        data_test = np.array([line.strip().split() for line in file], dtype=float)

    Y_test = data_test[:, 0]
    X_test = data_test[:, 1:]

    return [X_train, Y_train, X_test, Y_test]


def confidence_interval(values, confidence=0.95):
    """95% confidence interval of the mean using the Student-t distribution."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = arr.mean()
    if n < 2:
        return mean, 0.0, mean, mean
    sem = arr.std(ddof=1) / np.sqrt(n)
    half_width = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, half_width, mean - half_width, mean + half_width


def cross_val_error_precomputed(M, y, seed, cv=3):
    """Mean 1-NN cross-validation error with distances read from a precomputed N x N
    matrix (shuffled KFold, so the folds are fixed by the seed)."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in kf.split(M):
        knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
        knn.fit(M[np.ix_(train_idx, train_idx)], y[train_idx])
        y_pred = knn.predict(M[np.ix_(test_idx, train_idx)])
        scores.append(accuracy_score(y[test_idx], y_pred))
    return 1 - float(np.mean(scores))


def select_eps_candidates(errors, eps_list):
    """(smallest_eps, largest_eps) among the eps achieving the minimum validation error."""
    best_err = min(errors)
    tied = [eps for eps, err in zip(eps_list, errors) if err == best_err]
    return min(tied), max(tied)


# ---------------------------------------------------------------------------
# w selection from eTiOT on sampled cross-class pairs.
# ---------------------------------------------------------------------------
def sample_class_pairs(Y_train, n_sample, sample_seed=0):
    """Sample k train indices per class (fixed RNG so every eps and every CV seed sees
    the same pairs), and return the list of (i, j) index pairs: for each unordered class
    pair (A, B), every sampled point of A to every sampled point of B (one direction).

    Per-class count: k = min(class size, ceil(0.2 * n_train / n_classes), n_sample).
    The 20%-budget term keeps the TOTAL sampled points ~0.2*n_train, so the number of
    cross-class pairs (~C(C,2) * k^2 <= (0.2 n)^2 / 2) stays bounded regardless of the
    class count -- without it, many-class datasets (e.g. Adiac, 37 classes) explode to
    C(C,2) * n_sample^2 pairs. The n_sample cap keeps large few-class datasets at the
    proven-sufficient constant budget (w' is stable in sample count)."""
    rng = np.random.default_rng(sample_seed)
    classes = np.unique(Y_train)
    k_budget = int(np.ceil(0.2 * len(Y_train) / len(classes)))
    sampled = {}
    for c in classes:
        idx = np.flatnonzero(Y_train == c)
        k = min(len(idx), k_budget, n_sample)
        sampled[c] = rng.choice(idx, size=k, replace=False)

    pairs = [(i, j)
             for ca, cb in combinations(classes, 2)
             for i in sampled[ca] for j in sampled[cb]]
    counts = {c: len(v) for c, v in sampled.items()}
    return pairs, counts


def combine_w(dists, ws):
    """Weighted average of the per-pair optimal w*_i with weights DECREASING in the
    distance d*_i: weight_i = (1 - d_i / sum_j d_j) / (N - 1). A proper convex
    combination; the closest cross-class pair gets the largest weight."""
    d = np.asarray(dists, dtype=float)
    w = np.asarray(ws, dtype=float)
    n = len(d)
    if n == 1:
        return float(w[0])
    total = d.sum()
    if total <= 0:  # all distances zero -> no preference, plain average
        return float(w.mean())
    weights = (1.0 - d / total) / (n - 1)
    return float(weights @ w)


# ---------------------------------------------------------------------------
# Worker side. The dataset is loaded once per worker via the pool initializer.
# ---------------------------------------------------------------------------
_DATA = None

def _init_worker(data):
    global _DATA
    _DATA = data

def _pair_task(task):
    """One eTiOT solve between two TRAIN points; returns (distance, optimal w)."""
    eps, i, j = task
    X = _DATA[0]
    d, _, w = TiOT_lib.eTiOT(X[i], X[j], eps=eps)
    return d, w

def _row_task(task):
    """One row of the train pairwise distance matrix at a fixed (metric, eps, w)."""
    metric_name, eps, w, i = task
    X = _DATA[0]
    metric = build_metric(metric_name, eps, w)
    xi = X[i]
    n = len(X)
    row = np.zeros(n)
    for j in range(n):
        if j != i:
            row[j] = metric(xi, X[j])
    return row

def _test_task(task):
    """1-NN prediction for a single test sample at a fixed (metric, eps, w)."""
    metric_name, eps, w, i = task
    X_train, Y_train, X_test = _DATA[0], _DATA[1], _DATA[2]
    metric = build_metric(metric_name, eps, w)
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(X_train, Y_train)
    return knn.predict([X_test[i]])[0]


def save_report(dataset_name, eps_list, seeds, n_sample, class_counts, w_info,
                metric_reports, timing, report_file):
    lines = []
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"eps grid: {eps_list}")
    lines.append(f"Seeds: {list(seeds)}")
    lines.append(f"w-selection (eTAOT2 only): eTiOT on sampled cross-class pairs, "
                 f"n_sample={n_sample}, sampled per class: {class_counts}")
    lines.append("")

    lines.append("Selected w' per eps (from eTiOT, distance-weighted average):")
    w_df = pd.DataFrame(
        {"w'": [w_info[eps]['w'] for eps in eps_list],
         "finite pairs": [f"{w_info[eps]['n_ok']}/{w_info[eps]['n_total']}" for eps in eps_list]},
        index=[f"eps={e}" for e in eps_list],
    )
    lines.append(w_df.to_string())
    lines.append("")

    for metric_label, rep in metric_reports.items():
        lines.append(f"===== Metric: {metric_label} =====")
        val_df = pd.DataFrame(
            {f"seed{s}": rep['val_errors'][s] for s in seeds},
            index=[f"eps={e}" for e in eps_list],
        )
        lines.append("Validation errors (rows = eps, cols = seed):")
        lines.append(val_df.to_string())
        lines.append("")

        for rule_label, eps_key, final_key in [
            ("tie-break = smallest eps", 'eps_small', 'final_small'),
            ("tie-break = largest eps",  'eps_large', 'final_large'),
        ]:
            eps_str = ", ".join(f"seed{s}={rep[eps_key][s]}" for s in seeds)
            finals = [rep[final_key][s] for s in seeds]
            mean, half_width, lo, hi = confidence_interval(finals)
            lines.append(f"--- {rule_label} ---")
            lines.append(f"  Selected eps per seed  : {eps_str}")
            lines.append(f"  Final test error / seed: {finals}")
            lines.append(f"  Mean final test error  : {mean:.6f}")
            lines.append(f"  95% confidence interval: [{lo:.6f}, {hi:.6f}]  (mean +/- {half_width:.6f})")
            lines.append("")

    # ---- timing section ----
    t_wsel = timing['w_select']
    lines.append(f"===== Timing (seconds, {N_WORKERS} workers) =====")
    lines.append(f"w-select (eTiOT, all {len(eps_list)} eps): {t_wsel:.2f}")
    for key in [k for k in ['eTAOT2', 'oriTAOT'] if ('cv_matrix', k) in timing]:
        cv = timing[('cv_matrix', key)] + timing[('cv_lookup', key)]
        lines.append(f"{key}:")
        lines.append(f"  cv distance matrices : {timing[('cv_matrix', key)]:.2f}")
        lines.append(f"  cv fold lookups      : {timing[('cv_lookup', key)]:.2f}")
        lines.append(f"  test phase           : {timing[('test', key)]:.2f}")
        if key == 'eTAOT2':
            lines.append(f"  TRAINING total (w-select + eps CV) = {t_wsel + cv:.2f}")
        else:
            lines.append(f"  TRAINING total (eps CV only; w pre-tuned offline) = {cv:.2f}")
            lines.append(f"  extrapolated w-grid tuning: x10 grid = {10*cv:.2f}, "
                         f"x19 grid (TAOT paper) = {19*cv:.2f}")
    cv2 = timing[('cv_matrix', 'eTAOT2')] + timing[('cv_lookup', 'eTAOT2')]
    cv0 = (timing[('cv_matrix', 'oriTAOT')] + timing[('cv_lookup', 'oriTAOT')]
           if ('cv_matrix', 'oriTAOT') in timing else 0.0)
    if cv0 > 0:
        lines.append(f"ratio: eTAOT2 training / oriTAOT eps-CV-only = {(t_wsel+cv2)/cv0:.2f}")
        lines.append(f"ratio: eTAOT2 training / oriTAOT x10 grid    = {(t_wsel+cv2)/(10*cv0):.2f}")
        lines.append(f"ratio: eTAOT2 training / oriTAOT x19 grid    = {(t_wsel+cv2)/(19*cv0):.2f}")
    lines.append("")

    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {report_file}")


def experiment_TiOTselect_kNN(dataset_name, w_TAOT, n_sample=10, sample_seed=0,
                              seeds=range(1, 6), eps_list=None):
    """w_TAOT: the pre-tuned oriTAOT w from the reference table (TAOT parametrization,
    cost = spatial + w*temporal). eTAOT2's w needs no input -- it is selected by eTiOT.

    Pass w_TAOT=None for datasets outside the TAOT paper's table, where no tuned w exists:
    the oriTAOT arm is then skipped entirely rather than run at an arbitrary (untuned) w,
    which would produce a misleading accuracy column. Only the eTAOT2 arm is reported.

    eps_list: the eps grid to cross-validate over; defaults to 0.01..0.1 in steps of 0.01.
    Pass a single-element list (e.g. [0.01]) to FIX eps instead of cross-validating it --
    the entropic eps is a numerical regularization scale, not a data-dependent temporal
    trade-off like TAOT's w or DTW's window, so CV over it is arguably not worth its cost.
    With one eps the two tie-break rules coincide and the run costs ~1/10 of the full grid.
    The w-selection is unchanged: it simply runs at that eps only."""
    seeds = list(seeds)
    if eps_list is None:
        eps_list = [round(0.01 * i, 2) for i in range(1, 11)]
    eps_name = (f" (eps={eps_list[0]})" if len(eps_list) == 1
                else f" ({eps_list[0]} to {eps_list[-1]})")
    result_file = os.path.join('Experimental_outputs', "kNN_data", "saved_results",
                               "Results on " + dataset_name + eps_name + '.txt')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    data = process_data(dataset_name=dataset_name)
    Y_train, Y_test = data[1], data[3]
    n_train = len(Y_train)
    n_test = len(Y_test)
    timing = {}

    pairs, class_counts = sample_class_pairs(Y_train, n_sample, sample_seed)
    pair_tasks = [(eps, i, j) for eps in eps_list for (i, j) in pairs]

    print(f"[{dataset_name}] n_train={n_train}, n_test={n_test}; w-selection: "
          f"{len(pairs)} cross-class pairs (per class: {class_counts}) x {len(eps_list)} eps "
          f"-> {len(pair_tasks)} eTiOT solves ({N_WORKERS} workers)")

    with multiprocessing.Pool(N_WORKERS, initializer=_init_worker, initargs=(data,)) as pool:
        # ---- Phase 0 (eTAOT2 only): eTiOT on sampled pairs -> w'(eps), timed ----
        t0 = time.perf_counter()
        pair_results = list(tqdm(pool.imap(_pair_task, pair_tasks, chunksize=4),
                                 total=len(pair_tasks), desc="w-select (eTiOT)"))
        timing['w_select'] = time.perf_counter() - t0

        w_info = {}
        for k, eps in enumerate(eps_list):
            res = pair_results[k * len(pairs):(k + 1) * len(pairs)]
            # A non-finite distance means the eTiOT solve failed (entropic underflow at
            # small eps); its w is meaningless, so the pair is dropped from the average.
            ok = [(d, w) for d, w in res if np.isfinite(d)]
            if ok:
                w_sel = combine_w([d for d, _ in ok], [w for _, w in ok])
            else:
                w_sel = 0.5
                print(f"WARNING: eps={eps}: all {len(res)} eTiOT solves diverged; "
                      f"falling back to w'=0.5")
            w_info[eps] = {'w': w_sel, 'n_ok': len(ok), 'n_total': len(res)}
        print("  w'(eps): " + ", ".join(f"{eps}->{w_info[eps]['w']:.4f}" for eps in eps_list))

        metrics = [
            {'key': 'eTAOT2',  'label': "eTAOT2 (w from eTiOT)",
             'metric_name': 'eTAOT2',  'w_of_eps': {eps: w_info[eps]['w'] for eps in eps_list}},
        ]
        if w_TAOT is not None:
            metrics.append(
                {'key': 'oriTAOT', 'label': f"oriTAOT (w={w_TAOT})",
                 'metric_name': 'oriTAOT', 'w_of_eps': {eps: w_TAOT for eps in eps_list}})

        val_errors, eps_small, eps_large, chosen = {}, {}, {}, {}
        for m in metrics:
            key = m['key']
            # With a single eps there is nothing to select, so the whole train x train
            # matrix phase (whose only purpose is to score eps candidates) is skipped.
            if len(eps_list) == 1:
                timing[('cv_matrix', key)] = 0.0
                timing[('cv_lookup', key)] = 0.0
                val_errors[key] = {s: [float('nan')] for s in seeds}
                for s in seeds:
                    eps_small[(key, s)] = eps_large[(key, s)] = eps_list[0]
                chosen[key] = [eps_list[0]]
                continue

            # ---- Phase 1a: train pairwise distance matrices, timed per metric ----
            row_tasks = [(m['metric_name'], eps, m['w_of_eps'][eps], i)
                         for eps in eps_list for i in range(n_train)]
            t0 = time.perf_counter()
            rows = list(tqdm(pool.imap(_row_task, row_tasks, chunksize=1),
                             total=len(row_tasks), desc=f"cv matrix ({key})"))
            timing[('cv_matrix', key)] = time.perf_counter() - t0

            matrices = {eps: np.zeros((n_train, n_train)) for eps in eps_list}
            for (metric_name, eps, w, i), row in zip(row_tasks, rows):
                matrices[eps][i] = row
            for eps, M in matrices.items():
                bad = ~np.isfinite(M)
                n_bad = int(bad.sum())
                if n_bad:
                    print(f"WARNING: {key} eps={eps}: {n_bad} non-finite distances of "
                          f"{n_train * (n_train - 1)} pairs -> treated as +inf (never a neighbour).")
                    M[bad] = DIST_INF

            # ---- Phase 1b: CV = lookups into the matrices, timed per metric ----
            t0 = time.perf_counter()
            val_errors[key] = {s: [] for s in seeds}
            for s in seeds:
                for eps in eps_list:
                    val_errors[key][s].append(
                        cross_val_error_precomputed(matrices[eps], Y_train, seed=s))
            timing[('cv_lookup', key)] = time.perf_counter() - t0

            # ---- Phase 1c: eps tie-break candidates per seed ----
            for s in seeds:
                lo, hi = select_eps_candidates(val_errors[key][s], eps_list)
                eps_small[(key, s)] = lo
                eps_large[(key, s)] = hi
            chosen[key] = sorted({eps_small[(key, s)] for s in seeds} |
                                 {eps_large[(key, s)] for s in seeds})

        # ---- Phase 2: final test error at each selected eps, timed per metric ----
        test_error = {}
        for m in metrics:
            key = m['key']
            test_tasks = [(m['metric_name'], eps, m['w_of_eps'][eps], i)
                          for eps in chosen[key] for i in range(n_test)]
            t0 = time.perf_counter()
            test_preds = list(tqdm(pool.imap(_test_task, test_tasks, chunksize=1),
                                   total=len(test_tasks), desc=f"test ({key})"))
            timing[('test', key)] = time.perf_counter() - t0
            pos = 0
            for eps in chosen[key]:
                preds = test_preds[pos:pos + n_test]
                pos += n_test
                test_error[(key, eps)] = 1 - accuracy_score(Y_test, preds)

    metric_reports = {}
    for m in metrics:
        key = m['key']
        rep = {'val_errors': val_errors[key],
               'eps_small': {s: eps_small[(key, s)] for s in seeds},
               'eps_large': {s: eps_large[(key, s)] for s in seeds},
               'final_small': {s: test_error[(key, eps_small[(key, s)])] for s in seeds},
               'final_large': {s: test_error[(key, eps_large[(key, s)])] for s in seeds}}
        metric_reports[m['label']] = rep

    save_report(dataset_name, eps_list, seeds, n_sample, class_counts, w_info,
                metric_reports, timing, result_file)


if __name__ == "__main__":
    # The 63 datasets of Table 1, in the order of error_comparison_by_n_over_len.pdf
    # (increasing n_train / length). Second argument: the pre-tuned TAOT w from the TAOT paper.
    # eps is fixed to 0.01. Uncomment the datasets to run.
    experiment_TiOTselect_kNN('CinCECGTorso', 10, eps_list=[0.01])
    # experiment_TiOTselect_kNN('BeetleFly', 0.3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('BirdChicken', 0.1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ShapeletSim', 2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('DiatomSizeReduction', 0.2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('OliveOil', 0.6, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Symbols', 0.8, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Beef', 6, eps_list=[0.01])
    # experiment_TiOTselect_kNN('FaceFour', 5, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Coffee', 2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Car', 0.8, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ToeSegmentation2', 0.8, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Herring', 0.2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Meat', 0.9, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ArrowHead', 3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ToeSegmentation1', 0.1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ECGFiveDays', 5, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Worms', 1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('WormsTwoClass', 8, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Lightning7', 0.9, eps_list=[0.01])
    # experiment_TiOTselect_kNN('CBF', 1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('MoteStrain', 1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Wine', 9, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Ham', 0.7, eps_list=[0.01])
    # experiment_TiOTselect_kNN('TwoLeadECG', 0.1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('SonyAIBORobotSurface1', 2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('GunPoint', 0.3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Computers', 0.6, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Trace', 0.3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('SonyAIBORobotSurface2', 10, eps_list=[0.01])
    # experiment_TiOTselect_kNN('OSULeaf', 3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('RefrigerationDevices', 0.2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ScreenType', 0.2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('SmallKitchenAppliances', 4, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Earthquakes', 7, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Yoga', 0.2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Plane', 0.5, eps_list=[0.01])
    # experiment_TiOTselect_kNN('InsectWingbeatSound', 10, eps_list=[0.01])
    # experiment_TiOTselect_kNN('WordSynonyms', 3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ECG200', 3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ShapesAll', 0.8, eps_list=[0.01])
    # experiment_TiOTselect_kNN('CricketX', 3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('CricketY', 3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('CricketZ', 5, eps_list=[0.01])
    # experiment_TiOTselect_kNN('FacesUCR', 3, eps_list=[0.01])
    # experiment_TiOTselect_kNN('FiftyWords', 2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Adiac', 0.1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Strawberry', 0.2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ItalyPowerDemand', 7, eps_list=[0.01])
    # experiment_TiOTselect_kNN('MedicalImages', 4, eps_list=[0.01])
    # experiment_TiOTselect_kNN('SwedishLeaf', 0.9, eps_list=[0.01])
    # experiment_TiOTselect_kNN('MiddlePhalanxTW', 0.4, eps_list=[0.01])
    # experiment_TiOTselect_kNN('DistalPhalanxOutlineAgeGroup', 1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('DistalPhalanxTW', 0.5, eps_list=[0.01])
    # experiment_TiOTselect_kNN('MiddlePhalanxOutlineAgeGroup', 0.2, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ProximalPhalanxOutlineAgeGroup', 0.1, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ProximalPhalanxTW', 0.7, eps_list=[0.01])
    # experiment_TiOTselect_kNN('SyntheticControl', 4, eps_list=[0.01])
    # experiment_TiOTselect_kNN('Wafer', 8, eps_list=[0.01])
    # experiment_TiOTselect_kNN('DistalPhalanxOutlineCorrect', 0.4, eps_list=[0.01])
    # experiment_TiOTselect_kNN('MiddlePhalanxOutlineCorrect', 0.5, eps_list=[0.01])
    # experiment_TiOTselect_kNN('ProximalPhalanxOutlineCorrect', 0.7, eps_list=[0.01])
    # experiment_TiOTselect_kNN('TwoPatterns', 6, eps_list=[0.01])
