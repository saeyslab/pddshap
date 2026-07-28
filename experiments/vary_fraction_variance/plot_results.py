import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from experiments.vary_fraction_variance.result_reader import ResultReader

_DESC = "TODO"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESC)
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        help="directory containing datasets and trained models",
    )
    parser.add_argument(
        "-r",
        "--result-dir",
        type=str,
        help="directory containing results from vary_fraction_variance",
    )
    parser.add_argument("-o", "--out-dir", type=str, help="directory to write plots to")
    parser.add_argument(
        "-m", "--model", type=str, help="the model to use when extracting results"
    )
    parser.add_argument(
        "-b", "--baseline", type=str, help="the baseline (shap explainer) to compare to"
    )
    args = parser.parse_args()

    result_reader = ResultReader(
        args.data_dir, args.result_dir, args.model, args.baseline
    )
    os.makedirs(args.out_dir, exist_ok=True)
    prog = tqdm(total=5)

    # Plot training and inference runtimes
    inference_df = pd.DataFrame(columns=["method", "dataset", "runtime"])
    train_df = pd.DataFrame(columns=["method", "dataset", "runtime"])
    for ds in result_reader.datasets:
        runtimes = result_reader.get_runtimes(ds)

        # Add a row for the baseline
        bl_entry = pd.DataFrame.from_dict(
            {
                "method": [args.baseline],
                "dataset": [ds],
                "runtime": [runtimes["baseline"]],
            }
        )
        inference_df = pd.concat([inference_df, bl_entry], ignore_index=True)

        fracs = runtimes["pddshap"].keys()
        # Get inference time for each PDD-SHAP config
        pdd_shap_inference_times = pd.DataFrame.from_dict(
            {
                "method": [f"PDD-SHAP: {frac[0] + '.' + frac[1:]}" for frac in fracs],
                "dataset": [ds] * len(fracs),
                "runtime": [runtimes["pddshap"][frac]["inference"] for frac in fracs],
            }
        )
        inference_df = pd.concat(
            [inference_df, pdd_shap_inference_times], ignore_index=True
        )

        # Get training time for each PDD-SHAP config
        train_times = pd.DataFrame.from_dict(
            {
                "method": [f"PDD-SHAP: {frac[0] + '.' + frac[1:]}" for frac in fracs],
                "dataset": [ds] * len(fracs),
                "runtime": [runtimes["pddshap"][frac]["training"] for frac in fracs],
            }
        )
        train_df = pd.concat([train_df, train_times])

    fig, ax = plt.subplots()
    g = sns.barplot(data=inference_df, x="dataset", y="runtime", hue="method", ax=ax)
    g.set_yscale("log")
    g.set_xticklabels(g.get_xticklabels(), rotation=30, ha="right")
    g.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2)
    fig.savefig(
        os.path.join(args.out_dir, "runtime_inference.png"), bbox_inches="tight"
    )
    prog.update(1)

    fig, ax = plt.subplots()
    g = sns.barplot(data=train_df, x="dataset", y="runtime", hue="method", ax=ax)
    g.set_yscale("log")
    g.set_xticklabels(g.get_xticklabels(), rotation=30, ha="right")
    g.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2)
    fig.savefig(os.path.join(args.out_dir, "runtime_training.png"), bbox_inches="tight")
    prog.update(1)

    # Box plots and bar plots for each metric
    for metric in ["pearson", "spearman", "r2"]:
        df = pd.DataFrame(columns=["dataset", "method", "score"])
        for ds in result_reader.datasets:
            scores = result_reader.get_score(ds, metric)
            for frac in sorted(scores.keys()):
                frac_label = frac[0] + "." + frac[1:]
                frac_scores = scores[frac].flatten()
                frac_df = pd.DataFrame.from_dict(
                    {
                        "dataset": [ds] * len(frac_scores),
                        "method": [f"PDD-SHAP {frac_label}"] * len(frac_scores),
                        "score": frac_scores,
                    }
                )
                df = pd.concat([df, frac_df], ignore_index=True)

        fig, ax = plt.subplots()
        g = sns.boxplot(data=df, x="dataset", y="score", hue="method", ax=ax)
        g.set_xticklabels(g.get_xticklabels(), rotation=30, ha="right")
        g.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2)
        fig.savefig(os.path.join(args.out_dir, f"{metric}.png"), bbox_inches="tight")

        fig, ax = plt.subplots()
        g = sns.boxplot(
            data=df, x="dataset", y="score", hue="method", ax=ax, showfliers=False
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=30, ha="right")
        g.legend(bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2)
        fig.savefig(
            os.path.join(args.out_dir, f"{metric}_nofliers.png"), bbox_inches="tight"
        )
        prog.update(1)
