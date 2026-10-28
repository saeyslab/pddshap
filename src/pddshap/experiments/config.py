from pathlib import Path

import yaml


def _find_experiments_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "config" / "general.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find a config/general.yaml above {start}"
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp) or {}


def load_experiment_config(
    experiment_dir: str | Path, config_path: str | Path | None = None
) -> dict:
    """
    Load an experiment's configuration by merging, in increasing order of
    precedence: the shared `config/general.yaml`, and (if present) an
    experiment-specific config. All configuration lives under the root-level
    `config/` directory; nothing is read from `experiment_dir` itself.

    `config/datasets.yaml` is a name -> {data_id, pred_type} catalog of every
    known dataset. If the merged experiment config has its own `"datasets"`
    key, it's treated as a *selection* from that catalog: a mapping of
    dataset name -> override dict (e.g. `{max_depth: 4}`), restricting
    `cfg["datasets"]` to just those names, with each entry's overrides merged
    on top of the catalog's `{data_id, pred_type}`. Otherwise, `cfg["datasets"]`
    is the full catalog.

    Parameters
    ----------
    experiment_dir : str or pathlib.Path
        Directory of the calling experiment script (typically
        `Path(__file__).parent`), used to derive the experiment's default
        config filename (`experiment_dir.name`) and to locate the config
        root by walking up from here.
    config_path : str or pathlib.Path, optional
        Path to the experiment-specific config file, overriding the default
        `config/<experiment_dir.name>.yaml`. Must exist if given.

    Returns
    -------
    dict
        Merged configuration, with dataset settings under the `"datasets"` key.
    """
    experiment_dir = Path(experiment_dir)
    root = _find_experiments_root(experiment_dir)

    cfg = _load_yaml(root / "config" / "general.yaml")
    datasets_cfg = _load_yaml(root / "config" / "datasets.yaml")

    own_config_path = (
        Path(config_path)
        if config_path is not None
        else root / "config" / f"{experiment_dir.name}.yaml"
    )
    if config_path is not None and not own_config_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {own_config_path}")

    own_dataset_overrides = None
    if own_config_path.is_file():
        own_cfg = _load_yaml(own_config_path)
        own_dataset_overrides = own_cfg.pop("datasets", None)
        cfg.update(own_cfg)

    if own_dataset_overrides is not None:
        cfg["datasets"] = {
            name: {**datasets_cfg.get(name, {}), **overrides}
            for name, overrides in own_dataset_overrides.items()
        }
    else:
        cfg["datasets"] = datasets_cfg
    return cfg
