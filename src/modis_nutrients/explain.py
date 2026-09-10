"""SHAP attributions in physical units of each target. Computed on the test cells
of a blocked split, so they describe the model in regions it has not seen.
GBM uses the exact ``TreeExplainer``; the MLP uses the permutation explainer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import FeatureModel, GBMModel, KerasMLPModel


def _require_shap():
    try:
        import shap
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install shap  (or modis-nutrients[explain])") from e
    return shap


def shap_all_targets(model: FeatureModel, X_background: pd.DataFrame, X_explain: pd.DataFrame,
                     max_background: int = 100, max_explain: int = 500, seed: int = 0):
    """``shap.Explanation`` with values ``(n_explain, n_features, n_targets)`` in
    physical units (the target scaler is folded in)."""
    shap = _require_shap()
    rng = np.random.default_rng(seed)
    Xb = model.transform(X_background)
    Xe = model.transform(X_explain)
    if len(Xb) > max_background:
        Xb = Xb[rng.choice(len(Xb), max_background, replace=False)]
    if len(Xe) > max_explain:
        Xe = Xe[rng.choice(len(Xe), max_explain, replace=False)]
    names = list(model.pre_.get_feature_names_out())
    scale = model.y_scaler_.scale_
    mean = model.y_scaler_.mean_

    if isinstance(model, GBMModel):
        vals, base = [], []
        for j, est in enumerate(model.est_.estimators_):
            ex = shap.TreeExplainer(est)(Xe)
            vals.append(ex.values * scale[j])
            base.append(np.asarray(ex.base_values) * scale[j] + mean[j])
        return shap.Explanation(values=np.stack(vals, axis=-1), base_values=np.stack(base, axis=-1),
                                data=Xe, feature_names=names)

    if not isinstance(model, KerasMLPModel):
        raise TypeError(f"no SHAP path for {type(model).__name__}")

    # call the network directly (not .predict) so the permutation explainer stays fast
    net = model.net_

    def predict_physical(x):
        return np.asarray(net(np.asarray(x, np.float32), training=False)) * scale + mean

    explainer = shap.Explainer(predict_physical, Xb, feature_names=names, algorithm="permutation", seed=seed)
    return explainer(Xe)


def mean_abs_shap_table(explanation, targets: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature (rows) and target (columns) from :func:`shap_all_targets`."""
    return pd.DataFrame(np.abs(explanation.values).mean(axis=0), index=explanation.feature_names, columns=targets)
