"""Plot 1-NN test error per dataset for TiOT-select, oriTAOT and the UCR / TAOT-paper baselines.

One line per method over the 63 datasets of the TAOT paper's table. Our method and oriTAOT use a
fixed eps (default 0.01); ED / DTW errors come from the UCR 2018 archive summary. The published WDTW
errors from the TAOT paper (Zhang, Tang & Corpetti, IEEE Access 2020, Tables 1-2) are off by
default and added with --wdtw.

Usage (from the repository root):
    .venv/bin/python Tools/plot_error_comparison.py                   # sorted by n_train / length
    .venv/bin/python Tools/plot_error_comparison.py --sort n_train --out plot.pdf

Fixed-eps errors are read from Experimental_outputs/kNN_data/saved_results: the
dedicated run "Results on X (eps=E).txt" when it exists, otherwise the test error of a CV seed that
selected E in "Results on X (0.01 to 0.1).txt" (identical, since the test phase is deterministic).
"""

import argparse
import csv
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TOOLS_DIR)
RESULTS_DIR = os.path.join(REPO_DIR, "Experimental_outputs", "kNN_data", "saved_results")
UCR_SUMMARY = os.path.join(TOOLS_DIR, "data", "UCR_DataSummary.csv")

# Published WDTW 1-NN errors from the TAOT paper, Tables 1-2 (3 decimals, as published).
WDTW = {
    'SyntheticControl': .010, 'GunPoint': .027, 'CBF': .009, 'OSULeaf': .479, 'SwedishLeaf': .173,
    'FiftyWords': .253, 'Trace': .000, 'TwoPatterns': .000, 'Wafer': .003, 'FaceFour': .125,
    'Lightning7': .274, 'ECG200': .130, 'Adiac': .366, 'Yoga': .153, 'Plane': .000, 'Car': .217,
    'Beef': .300, 'Coffee': .000, 'OliveOil': .167, 'CinCECGTorso': .075,
    'DiatomSizeReduction': .036, 'ECGFiveDays': .138, 'FacesUCR': .078, 'ItalyPowerDemand': .043,
    'MedicalImages': .263, 'MoteStrain': .142, 'SonyAIBORobotSurface1': .255,
    'SonyAIBORobotSurface2': .154, 'Symbols': .049, 'TwoLeadECG': .111, 'CricketX': .210,
    'CricketY': .238, 'CricketZ': .246, 'InsectWingbeatSound': .431, 'ArrowHead': .183,
    'BeetleFly': .300, 'BirdChicken': .250, 'Ham': .429, 'Herring': .453,
    'ProximalPhalanxOutlineAgeGroup': .195, 'ProximalPhalanxOutlineCorrect': .213,
    'ProximalPhalanxTW': .260, 'ToeSegmentation1': .219, 'ToeSegmentation2': .115,
    'DistalPhalanxOutlineAgeGroup': .225, 'DistalPhalanxOutlineCorrect': .237,
    'DistalPhalanxTW': .268, 'Earthquakes': .292, 'MiddlePhalanxOutlineAgeGroup': .260,
    'MiddlePhalanxOutlineCorrect': .292, 'MiddlePhalanxTW': .414, 'ShapeletSim': .244,
    'Wine': .426, 'WordSynonyms': .249, 'Computers': .416, 'Meat': .067,
    'RefrigerationDevices': .592, 'ScreenType': .589, 'ShapesAll': .192,
    'SmallKitchenAppliances': .347, 'Strawberry': .062, 'Worms': .552, 'WormsTwoClass': .376,
}
TAOT63 = sorted(WDTW)

# (label, key, colour, linewidth). Colours: validated categorical palette, fixed order.
SERIES = [
    # {eps} and {eq} are filled in at plot time (fixed eps, equation number of the w* rule in the paper).
    (r"eTiOT ($\varepsilon = {eps}$, $w = w^*$ from Eq. ({eq}))", "ours", "#2a78d6", 2.6),
    (r"eTAOT ($\varepsilon = {eps}$, grid-search $\omega$)", "oriTAOT", "#eb6834", 1.6),
    ("DTW (learned w)", "dtw_lw", "#1f1f1f", 1.6),
    ("DTW (w = 100)", "dtw100", "#eda100", 1.6),
    ("WDTW (published)", "wdtw", "#e87ba4", 1.6),
    ("ED", "ed", "#008300", 1.6),
]

SORT_KEYS = {
    "n_over_len": lambda r: r["n_train"] / r["length"],
    "n_train": lambda r: r["n_train"],
    "length": lambda r: r["length"],
    "ours": lambda r: r["ours"],
}


def parse_report(path):
    """{metric: {'small'|'large': {'eps': [...], 'errs': [...], 'mean': float}}}, parsed line by line."""
    out, metric, rule = {}, None, None
    with open(path) as f:
        for line in f:
            m = re.match(r"===== Metric: (\w+)", line)
            if m:
                metric, rule = m.group(1), None
                out[metric] = {}
                continue
            if line.startswith("===== Timing"):
                metric = None
                continue
            if metric is None:
                continue
            if "tie-break = smallest eps" in line:
                rule = "small"
                out[metric][rule] = {}
            elif "tie-break = largest eps" in line:
                rule = "large"
                out[metric][rule] = {}
            elif rule is not None:
                m = re.match(r"\s*Selected eps per seed\s*:\s*(.+)", line)
                if m:
                    out[metric][rule]["eps"] = [float(e) for e in re.findall(r"seed\d=([\d.]+)", m.group(1))]
                m = re.match(r"\s*Final test error / seed: \[(.*)\]", line)
                if m:
                    out[metric][rule]["errs"] = [float(x) for x in m.group(1).split(",")]
                m = re.match(r"\s*Mean final test error\s*:\s*([\d.]+)", line)
                if m:
                    out[metric][rule]["mean"] = float(m.group(1))
    return out


def error_at_eps(report, metric, eps):
    """Test error of `metric` at `eps` from any seed that selected it, else None."""
    for rule in ("small", "large"):
        r = report.get(metric, {}).get(rule, {})
        for e, err in zip(r.get("eps", []), r.get("errs", [])):
            if abs(e - eps) < 1e-12:
                return err
    return None


def fixed_eps_error(dataset, metric, eps):
    fixed = os.path.join(RESULTS_DIR, f"Results on {dataset} (eps={eps}).txt")
    if os.path.exists(fixed):
        err = error_at_eps(parse_report(fixed), metric, eps)
        if err is not None:
            return err
    cv = os.path.join(RESULTS_DIR, f"Results on {dataset} (0.01 to 0.1).txt")
    return error_at_eps(parse_report(cv), metric, eps) if os.path.exists(cv) else None


def load_rows(eps):
    ucr = {}
    with open(UCR_SUMMARY, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ucr[r["Name"].strip()] = {k.strip(): (v or "").strip() for k, v in r.items()}
    rows, missing = [], []
    for name in TAOT63:
        u = ucr[name]
        row = {
            "name": name, "n_train": int(u["Train"]), "length": int(u["Length"]),
            "ours": fixed_eps_error(name, "eTAOT2", eps),
            "oriTAOT": fixed_eps_error(name, "oriTAOT", eps),
            "ed": float(u["ED (w=0)"]), "dtw_lw": float(u["DTW (learned_w)"].split()[0]),
            "dtw100": float(u["DTW (w=100)"]), "wdtw": WDTW[name],
        }
        if row["ours"] is None or row["oriTAOT"] is None:
            missing.append(name)
        rows.append(row)
    return rows, missing


def plot(rows, sort, eps, out, include_wdtw=False, eq="?"):
    rows = sorted(rows, key=SORT_KEYS[sort])
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(16, 7.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    series = [s for s in SERIES if include_wdtw or s[1] != "wdtw"]
    # Tuning-based methods are dashed and drawn after (on top of) the solid tuning-free lines.
    tuning_based = {"dtw_lw", "oriTAOT"}
    for label, k, colour, lw in sorted(series, key=lambda s: s[1] in tuning_based):
        dashed = k in tuning_based
        ax.plot(x, [r[k] for r in rows], color=colour, lw=lw, label=label.replace("{eps}", f"{eps:g}").replace("{eq}", eq),
                alpha=1.0 if k == "ours" else 0.9, linestyle=(0, (4, 2)) if dashed else "-",
                solid_capstyle="round", dash_capstyle="butt", zorder=4 if dashed else (3 if k == "ours" else 2))
    if sort == "n_over_len":
        ticks = [f"{r['name']}  ({r['n_train']}/{r['length']}={r['n_train'] / r['length']:.2f})" for r in rows]
    elif sort == "n_train":
        ticks = [f"{r['name']}  (n={r['n_train']})" for r in rows]
    elif sort == "length":
        ticks = [f"{r['name']}  (len={r['length']})" for r in rows]
    else:
        ticks = [r["name"] for r in rows]
    ax.set_xticks(list(x))
    ax.set_xticklabels(ticks, rotation=90, fontsize=7, color="#444")
    ax.set_ylabel("1-NN test error", fontsize=11, color="#222")
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0, max(max(r[k] for r in rows) for _, k, _, _ in series) * 1.08)
    ax.grid(axis="y", color="#e4e2de", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c6c0")
    ax.tick_params(colors="#666", labelsize=8)
    # Legend in two rows (tuning-based methods, then tuning-free methods). Each row is its own
    # single-row legend, so all entries of a row share one baseline; the bold row headers are text
    # placed to the left, and both legends start right after the wider header so they line up.
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    handles = {line.get_label(): line for line in ax.get_lines()}
    label_of = {k: label.replace("{eps}", f"{eps:g}").replace("{eq}", eq) for label, k, _, _ in series}
    legend_rows = [
        ("Tuning-based methods:", ["dtw_lw", "oriTAOT"]),
        ("Tuning-free methods:", ["ed", "dtw100", "ours"]),
    ]
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()
    headers = [ax.text(0, 1, h, transform=ax.transAxes, fontsize=13, color="#222",
                       va="center", ha="left", clip_on=False) for h, _ in legend_rows]
    header_w = max(float(np.diff(to_axes.transform(t.get_window_extent(renderer))[:, 0])[0]) for t in headers)
    placed = []
    for r, ((_, keys), header) in enumerate(zip(legend_rows, headers)):
        # Legend objects are created directly: ax.legend() would also register the last row as the
        # axes legend, so that row would be drawn twice (visibly bolder text).
        legend = Legend(ax, [handles[label_of[k]] for k in keys], [label_of[k] for k in keys],
                           loc="lower left", bbox_to_anchor=(header_w + 0.012, 1.0 + 0.075 * (len(legend_rows) - 1 - r)),
                           frameon=False, fontsize=13, ncol=len(keys), handlelength=2.5,
                           handletextpad=0.6, columnspacing=1.8, borderaxespad=0)
        for t in legend.get_texts():
            t.set_color("#222")
        legend.set_in_layout(False)   # keep the plot full width; the legend fits inside the figure
        ax.add_artist(legend)
        header.set_in_layout(False)
        placed.append((legend, header))
    fig.canvas.draw()   # legend labels only get their final positions when drawn
    for legend, header in placed:
        box = to_axes.transform(legend.get_texts()[0].get_window_extent(renderer))
        header.set_position((0, box[:, 1].mean()))   # vertically centred on the row's labels
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eps", type=float, default=0.01, help="fixed eps of our method and oriTAOT")
    parser.add_argument("--sort", choices=sorted(SORT_KEYS), default="n_over_len")
    parser.add_argument("--wdtw", action="store_true", help="also plot the published WDTW line")
    parser.add_argument("--eq", default="?", help="equation number of the w* selection rule, shown in the legend")
    parser.add_argument("--out", default=None, help="output file (.pdf/.png); default under Experimental_outputs")
    args = parser.parse_args()

    rows, missing = load_rows(args.eps)
    if missing:
        raise SystemExit(f"No eps={args.eps} result for {len(missing)} dataset(s): {', '.join(missing)}")
    out = args.out or os.path.join(REPO_DIR, "Experimental_outputs", "kNN_data",
                                   f"error_comparison_by_{args.sort}.pdf")
    plot(rows, args.sort, args.eps, out, include_wdtw=args.wdtw, eq=args.eq)
    print(f"Saved {out} ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
