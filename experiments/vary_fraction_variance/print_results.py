import argparse

import numpy as np

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

    for ds in result_reader.datasets:
        print(ds)
        print("#" * 80)
        runtimes = result_reader.get_runtimes(ds)

        print(f"Permutation: {runtimes['baseline']:.3e} s/row")
        for key in runtimes["pddshap"]:
            print(f"{key}:")
            for score_fn in ["pearson", "spearman", "r2"]:
                scores = result_reader.get_score(ds, score_fn)
                print(
                    f"\t{score_fn}: {np.average(scores[key]):.3f} (median: {np.median(scores[key]):.3f})"
                )
            print(f"\tTraining: {runtimes['pddshap'][key]['training']:.3f}s")
            print(f"\tInference: {runtimes['pddshap'][key]['inference']:.3e}s/row")
