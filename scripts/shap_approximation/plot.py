import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pddshap.experiments import load_experiment_config

_DESC = """
Reads a run directory written by run_experiment.py
(OUT_DIR/YYYYMMDDhhmmss_shap_approximation/{metrics,runtimes}.csv) and plots,
for every dataset: R2 and Spearman rank correlation of PDD-SHAP against each
baseline explainer, the PDD-SHAP surrogate training time, and the explanation
runtime of every method.

By default, the most recent run directory under OUT_DIR is used; pass
--run-dir to pick a specific one.
"""

_METHOD_ORDER = ["pdd_shap", "sampling", "permutation", "kernel"]
_METHOD_LABELS = {
    "pdd_shap": "PDD-SHAP",
    "sampling": "Sampling",
    "permutation": "Permutation",
    "kernel": "Kernel",
}


def _method_palette(methods: list[str]) -> dict:
    colors = sns.color_palette("colorblind", n_colors=len(_METHOD_ORDER))
    base = dict(zip(_METHOD_ORDER, colors))
    return {m: base[m] for m in methods}


def _grouped_bar(
    df: pd.DataFrame, value_col: str, title: str, ylabel: str, path: pathlib.Path,
    log_scale: bool = False,
) -> None:
    methods = [m for m in _METHOD_ORDER if m in df["method"].unique()]
    palette = _method_palette(methods)
    datasets = sorted(df["dataset"].unique())

    fig, ax = plt.subplots(figsize=(1.6 * max(len(datasets), 3), 4))
    sns.barplot(
        data=df, x="dataset", y=value_col, hue="method",
        order=datasets, hue_order=methods, palette=palette, ax=ax,
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [_METHOD_LABELS[m] for m in methods], title=None, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(exp_out_dir: pathlib.Path) -> None:
    metrics = pd.read_csv(exp_out_dir / "metrics.csv")
    runtimes = pd.read_csv(exp_out_dir / "runtimes.csv")

    plots_dir = exp_out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white")

    _grouped_bar(
        metrics, "r2", "PDD-SHAP vs. baselines: R2", "R2",
        plots_dir / "r2.png",
    )
    _grouped_bar(
        metrics, "spearman", "PDD-SHAP vs. baselines: Spearman rank correlation", "Spearman correlation",
        plots_dir / "spearman.png",
    )

    train_times = runtimes[runtimes["method"] == "pdd_shap"].copy()
    train_times["train_time (s)"] = pd.to_numeric(train_times["train_time (s)"])
    _grouped_bar(
        train_times, "train_time (s)", "PDD-SHAP surrogate training time", "Time (s, log scale)",
        plots_dir / "train_time.png", log_scale=True,
    )

    _grouped_bar(
        runtimes, "explain_time (s)", "Explanation runtime by method", "Time (s, log scale)",
        plots_dir / "runtime.png", log_scale=True,
    )
    print(f"Plots written to {plots_dir}")


def _find_latest_run_dir(out_dir: pathlib.Path) -> pathlib.Path:
    candidates = sorted(out_dir.glob("*_shap_approximation"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_shap_approximation run directory found under {out_dir}. "
            "Run run_experiment.py first, or pass --run-dir explicitly."
        )
    return candidates[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESC, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default=None,
        help="Path to an alternative experiment config YAML, overriding the "
             "default config/shap_approximation.yaml. Only used to resolve "
             "OUT_DIR when --run-dir is not given.",
    )
    parser.add_argument(
        "-r", "--run-dir", type=pathlib.Path, default=None,
        help="Specific OUT_DIR/YYYYMMDDhhmmss_shap_approximation/ run directory "
             "to plot. Defaults to the most recent one under OUT_DIR.",
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        cfg = load_experiment_config(pathlib.Path(__file__).parent, config_path=args.config)
        run_dir = _find_latest_run_dir(pathlib.Path(cfg["out_dir"]))

    make_plots(run_dir)
