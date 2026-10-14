from pathlib import Path

import yaml


def _find_experiments_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "config" / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find an experiments/config/config.yaml above {start}"
    )


def load_experiment_config(experiment_dir: str | Path) -> dict:
    """
    Load an experiment's configuration by merging, in increasing order of precedence:
    the shared ``experiments/config/config.yaml``, and (if present and distinct) a
    ``config.yaml`` local to ``experiment_dir``.

    ``experiments/config/datasets.yaml`` is attached under the ``"datasets"`` key rather
    than merged in, since it's a name -> {data_id, pred_type} mapping rather than
    scalar settings.

    :param experiment_dir: directory of the calling experiment script
        (typically ``Path(__file__).parent``). The shared config root is found by
        walking up from here.
    """
    experiment_dir = Path(experiment_dir)
    root = _find_experiments_root(experiment_dir)

    with open(root / "config" / "config.yaml", "r") as fp:
        cfg = yaml.safe_load(fp) or {}
    with open(root / "config" / "datasets.yaml", "r") as fp:
        datasets_cfg = yaml.safe_load(fp) or {}

    own_config = experiment_dir / "config.yaml"
    if own_config.is_file() and own_config.resolve() != (root / "config" / "config.yaml").resolve():
        with open(own_config, "r") as fp:
            cfg.update(yaml.safe_load(fp) or {})

    cfg["datasets"] = datasets_cfg
    return cfg
