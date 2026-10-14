import warnings

import pandas as pd


def sample_rows(df: pd.DataFrame, n: int, random_state: int, label: str | None = None) -> pd.DataFrame:
    """
    Sample ``min(n, len(df))`` rows from ``df``, warning if fewer than ``n`` rows
    were available (in which case all of ``df`` is returned, order permuted).
    """
    n_available = min(n, len(df))
    if n_available < n:
        tag = f" for '{label}'" if label else ""
        warnings.warn(
            f"Requested {n} rows{tag} but only {n_available} are available; using all of them."
        )
    return df.sample(n=n_available, random_state=random_state)
