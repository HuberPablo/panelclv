"""Valendin-style autoregressive Monte Carlo holdout simulation.

Companion to `multinomial_lstm.py` / `multinomial_transformer.py` (training) and
the data dict produced by `Data_preparation/dynamic_panel_dataset.py`. The model
has only ever seen the calibration window; this module asks it to simulate the
holdout window autoregressively:

    - true holdout COVARIATES / TIME FEATURES are used as conditioning input,
    - true holdout TARGET VALUES are *not* fed to the model — the previously
      sampled class index is fed back in instead.

There are **two rollouts**, one per model family, because the two architectures
carry history in fundamentally different ways:

    simulate_one_path          (recurrent / LSTM)
        The LSTM compresses the whole prefix into a fixed-size hidden state.
        We warm that state up on the full calibration window, then step through
        the holdout ONE period at a time, threading the state from each call
        into the next (O(1) work per step).

    simulate_transformer_path  (attention / Transformer)
        The Transformer keeps no recurrent summary: to predict a period it must
        attend over the ACTUAL tokens seen so far. So we keep an explicit
        context window that starts as the calibration window and GROWS by one
        period after every step, re-feeding it each time (O(t) work per step).
        This is what keeps the positional encoding consistent with training —
        calibration sits at positions 0..T_CAL-1 and holdout step t at T_CAL+t.

Both rollouts treat the AR target-derived features identically (recomputed from
the SAMPLED target history via `ARFeatureState`, never read from the holdout) and
both are driven by the shared `_run_monte_carlo` aggregator, so seeding,
averaging and the return contract never drift between models. The public entry
points are `run_monte_carlo_forecast` (LSTM) and
`run_monte_carlo_forecast_transformer` (Transformer); each averages
`n_simulations` paths into a per-customer-per-step expected count.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from panelclv.data_preparation.ar_features import ARFeatureState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_tensor(array, device: str | torch.device | None = None) -> torch.Tensor:
    """Cast numpy array or tensor to float32 torch tensor on `device`."""
    if isinstance(array, torch.Tensor):
        t = array.to(dtype=torch.float32)
    else:
        t = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def _get_target_idx(seq_cols: Sequence[str], target_col: str) -> int:
    """Position of `target_col` in `seq_cols` — clear error if absent."""
    seq_cols = list(seq_cols)
    if target_col not in seq_cols:
        raise ValueError(
            f"target_col {target_col!r} not in seq_cols={seq_cols}"
        )
    return seq_cols.index(target_col)


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# One Monte Carlo path — recurrent (LSTM) rollout
# ---------------------------------------------------------------------------


def simulate_one_path(
    model: torch.nn.Module,
    calibration,
    holdout,
    seq_cols: Sequence[str],
    target_col: str = "Transactions",
    device: str | torch.device | None = None,
    ar_features: Sequence[str] = (),
) -> np.ndarray:
    """Simulate one autoregressive holdout path for a STATEFUL model (the LSTM).

    Paper-faithful Valendin procedure, exploiting the LSTM's hidden state:

        1. Feed the full calibration through the model. This both produces the
           multinomial over holdout step 0 (the LAST calibration position) and
           leaves the hidden `state` summarising the whole calibration window.
        2. For each subsequent step t = 1 .. T_HOLD - 1, feed a SINGLE period
           built as (previous sample, true holdout covariates at step t),
           threading `state` from the previous call so the LSTM continues the
           recurrence instead of re-reading history, and sample the next class.

    True holdout targets are never fed to the model. Any `ar_features`
    (target-derived recency/activity columns) are likewise recomputed at each
    step from the SAMPLED target history — never read from the holdout — so the
    forecast uses no future information for them.

    Returns
    -------
    sampled_path : ndarray (N, T_HOLD)
        Sampled class indices for holdout steps 0 .. T_HOLD - 1, on CPU.
    """
    target_idx = _get_target_idx(seq_cols, target_col)
    if device is None:
        device = _device_of(model)
    model.to(device).eval()

    calib_tensor   = _as_tensor(calibration, device=device)   # (N, T_CAL, F)
    holdout_tensor = _as_tensor(holdout,     device=device)   # (N, T_HOLD, F)

    N, _T_CAL, _F = calib_tensor.shape
    _N2, T_HOLD, _F2 = holdout_tensor.shape

    sampled_path = torch.zeros((N, T_HOLD), dtype=torch.float32, device=device)

    # AR target-derived features: seed state from the calibration target history,
    # then recompute per holdout step from the sampled target (no leakage).
    ar_features = list(ar_features)
    ar_idx = {n: list(seq_cols).index(n) for n in ar_features}
    ar_state = (
        ARFeatureState(calib_tensor[:, :, target_idx].detach().cpu().numpy(), ar_features)
        if ar_features else None
    )

    with torch.inference_mode():
        # Step 1: warmup → its last-position sample IS the holdout step 0 forecast,
        # and `state` now summarises the whole calibration window.
        out, state = model(calib_tensor, state=None)
        previous_sample    = out[:, -1, 0]                    # (N,)
        sampled_path[:, 0] = previous_sample

        # Step 2: AR loop for the remaining T_HOLD - 1 steps.
        for t in range(T_HOLD - 1):
            x_t = holdout_tensor[:, t:t + 1, :].clone()       # (N, 1, F)
            x_t[:, 0, target_idx] = previous_sample
            if ar_state is not None:
                # previous_sample is the just-sampled target at holdout step t;
                # advance the AR state with it and overwrite the AR columns so the
                # input reflects the sampled (not the true) history.
                feats = ar_state.update(previous_sample.detach().cpu().numpy())
                for name, col in ar_idx.items():
                    x_t[:, 0, col] = torch.as_tensor(
                        feats[name], dtype=x_t.dtype, device=x_t.device
                    )
            sampled, state = model(x_t, state=state)
            previous_sample        = sampled[:, 0, 0]         # (N,)
            sampled_path[:, t + 1] = previous_sample

    return sampled_path.cpu().numpy()


# ---------------------------------------------------------------------------
# One Monte Carlo path — attention (Transformer) rollout
# ---------------------------------------------------------------------------


def simulate_transformer_path(
    model: torch.nn.Module,
    calibration,
    holdout,
    seq_cols: Sequence[str],
    target_col: str = "Transactions",
    device: str | torch.device | None = None,
    ar_features: Sequence[str] = (),
) -> np.ndarray:
    """Simulate one autoregressive holdout path for a STATELESS model (the Transformer).

    The Transformer has no recurrent state to thread, so we cannot feed a single
    period and "continue" — to predict each holdout period it must attend over
    the actual sequence of everything seen so far. We therefore keep an explicit
    `context` window that starts as the full calibration window and grows by one
    period after every step:

        step 0      : context = calibration                    -> predict holdout 0
        step t (>0) : context = [calibration, holdout 0..t-1]  -> predict holdout t

    Each appended holdout row carries the TRUE known covariates for that period
    but the SAMPLED count (never the true holdout target) and AR features
    recomputed from the sampled history — identical to the LSTM rollout, so the
    two only differ in HOW history is carried, not in WHAT the model conditions
    on. Because the context preserves absolute ordering, the sinusoidal
    positional encoding indexes calibration at 0..T_CAL-1 and holdout step t at
    T_CAL+t, matching training; this is exactly why a single-step feed (which
    would reset the position to 0 and drop all history) is wrong for this model.

    The model is called with `only_last=True` so just the final-position logits
    are materialised, and its returned state is ignored (the Transformer has none).

    Returns
    -------
    sampled_path : ndarray (N, T_HOLD)
        Sampled class indices for holdout steps 0 .. T_HOLD - 1, on CPU.
    """
    target_idx = _get_target_idx(seq_cols, target_col)
    if device is None:
        device = _device_of(model)
    model.to(device).eval()

    calib_tensor   = _as_tensor(calibration, device=device)   # (N, T_CAL, F)
    holdout_tensor = _as_tensor(holdout,     device=device)   # (N, T_HOLD, F)

    N, _T_CAL, _F = calib_tensor.shape
    _N2, T_HOLD, _F2 = holdout_tensor.shape

    sampled_path = torch.zeros((N, T_HOLD), dtype=torch.float32, device=device)

    # AR target-derived features: seeded and advanced EXACTLY as in the LSTM
    # rollout (update once per produced period, with that period's sample), so
    # both rollouts feed the model identical AR-feature values.
    ar_features = list(ar_features)
    ar_idx = {n: list(seq_cols).index(n) for n in ar_features}
    ar_state = (
        ARFeatureState(calib_tensor[:, :, target_idx].detach().cpu().numpy(), ar_features)
        if ar_features else None
    )

    with torch.inference_mode():
        # Growing context window. Starts as the full calibration; each iteration
        # appends one reconstructed holdout-input row.
        context = calib_tensor
        for t in range(T_HOLD):
            # Re-feed the whole context; read the distribution at the last
            # position only — that is the forecast for holdout step t.
            out, _ = model(context, only_last=True)
            sample = out[:, -1, 0]                             # (N,)
            sampled_path[:, t] = sample

            if t == T_HOLD - 1:
                break  # last step already sampled; nothing left to feed

            # Build the next input row: true holdout covariates for period t, the
            # SAMPLED count, and AR features recomputed from the sampled history.
            x_next = holdout_tensor[:, t:t + 1, :].clone()    # (N, 1, F)
            x_next[:, 0, target_idx] = sample
            if ar_state is not None:
                feats = ar_state.update(sample.detach().cpu().numpy())
                for name, col in ar_idx.items():
                    x_next[:, 0, col] = torch.as_tensor(
                        feats[name], dtype=x_next.dtype, device=x_next.device
                    )
            context = torch.cat([context, x_next], dim=1)     # grow by one period

    return sampled_path.cpu().numpy()


# ---------------------------------------------------------------------------
# Shared Monte Carlo aggregator
# ---------------------------------------------------------------------------


def _run_monte_carlo(
    model: torch.nn.Module,
    data: dict[str, Any],
    simulate_path: Callable[..., np.ndarray],
    *,
    n_simulations: int,
    target_col: str | None,
    device: str | torch.device | None,
    return_simulations: bool,
    seed: int | None,
) -> dict[str, Any]:
    """Run `simulate_path` `n_simulations` times and average the paths.

    Holds everything that is identical across model families — reading the data
    dict, device placement, seeding, the (S, N, T_HOLD) stack/mean, extracting
    the actuals, and the return contract — so the LSTM and Transformer entry
    points cannot drift apart. The only per-model piece is `simulate_path`.
    """
    seq_cols = list(data["seq_cols"])
    if target_col is None:
        target_col = data["target_col"]
    target_idx = _get_target_idx(seq_cols, target_col)

    if device is None:
        device = _device_of(model)

    if seed is not None:
        # Seeds CPU and (if present) CUDA RNG, making the sampled paths below
        # reproducible across runs.
        torch.manual_seed(seed)

    # Upload calibration / holdout to device ONCE and reuse across MC sims.
    # `_as_tensor` is a no-op when given a tensor already on the right device,
    # so the simulator's internal `_as_tensor` calls become free.
    calib_tensor   = _as_tensor(data["calibration"], device=device)
    holdout_tensor = _as_tensor(data["holdout"],     device=device)
    model.to(device).eval()

    ar_features = list(data.get("ar_features", []))

    sims: list[np.ndarray] = []
    for _ in range(n_simulations):
        sims.append(simulate_path(
            model=model,
            calibration=calib_tensor,
            holdout=holdout_tensor,
            seq_cols=seq_cols,
            target_col=target_col,
            device=device,
            ar_features=ar_features,
        ))
    simulations = np.stack(sims, axis=0)                       # (S, N, T_HOLD)
    prediction_mean = simulations.mean(axis=0)                 # (N, T_HOLD)

    actual = np.asarray(data["holdout"])[:, :, target_idx]     # (N, T_HOLD)

    result: dict[str, Any] = {
        "prediction_mean": prediction_mean,
        "actual": actual,
        "target_col": target_col,
        "target_idx": target_idx,
        "n_simulations": n_simulations,
        "seed": seed,
    }
    if return_simulations:
        result["simulations"] = simulations
    return result


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_monte_carlo_forecast(
    model: torch.nn.Module,
    data: dict[str, Any],
    n_simulations: int = 30,
    target_col: str | None = None,
    device: str | torch.device | None = None,
    return_simulations: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Monte Carlo holdout forecast for a recurrent (LSTM) inference model.

    Uses the stateful `simulate_one_path` rollout. `data` is the dict returned
    by `dynamic_panel_dataset.prepare_dataset`, so calibration / holdout /
    seq_cols / target_col are all read from it.

    `seed`, if given, is passed to `torch.manual_seed` before sampling so the
    whole Monte Carlo forecast is reproducible (same model + data + seed →
    identical paths). Leave it None for fresh randomness each call.

    Returns a dict with:
        prediction_mean : ndarray (N, T_HOLD)
        actual          : ndarray (N, T_HOLD) — true holdout targets,
                          extracted from `data["holdout"]` only for
                          downstream evaluation (NOT used as input).
        target_col, target_idx, n_simulations, seed
        simulations     : ndarray (S, N, T_HOLD), only if return_simulations.
    """
    return _run_monte_carlo(
        model,
        data,
        simulate_one_path,
        n_simulations=n_simulations,
        target_col=target_col,
        device=device,
        return_simulations=return_simulations,
        seed=seed,
    )


def run_monte_carlo_forecast_transformer(
    model: torch.nn.Module,
    data: dict[str, Any],
    n_simulations: int = 30,
    target_col: str | None = None,
    device: str | torch.device | None = None,
    return_simulations: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Monte Carlo holdout forecast for an attention (Transformer) inference model.

    Identical contract to `run_monte_carlo_forecast`, but uses the growing-window
    `simulate_transformer_path` rollout because the Transformer is stateless. The
    inference model must be an `InferenceMultinomialTransformerModel` (its
    `forward` accepts `only_last=` and ignores `state`). Returns the same dict
    described in `run_monte_carlo_forecast`.
    """
    return _run_monte_carlo(
        model,
        data,
        simulate_transformer_path,
        n_simulations=n_simulations,
        target_col=target_col,
        device=device,
        return_simulations=return_simulations,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_forecast_metrics(
    actual: np.ndarray,
    prediction_mean: np.ndarray,
) -> dict[str, float]:
    """Return RMSE, %-bias and aggregate-style MAPE — both inputs (N, T_HOLD).

    Argument order follows the Python / sklearn convention `(y_true, y_pred)`,
    matching `evaluation_utils.compute_metrics(y_true, y_pred)`.
    """
    pred = np.asarray(prediction_mean, dtype=np.float64)
    act  = np.asarray(actual,          dtype=np.float64)

    rmse = float(np.sqrt(np.mean((pred - act) ** 2)))

    total_actual = float(act.sum())
    bias_percent = (
        float("nan") if total_actual == 0
        else float(100.0 * (pred.sum() - total_actual) / total_actual)
    )

    actual_t = act.sum(axis=0)       # (T_HOLD,)
    pred_t   = pred.sum(axis=0)
    denom    = float(actual_t.sum())
    mape_agg = (
        float("nan") if denom == 0
        else float(100.0 * np.sum(np.abs(actual_t - pred_t)) / denom)
    )

    return {
        "rmse": rmse,
        "bias_percent": bias_percent,
        "mape_aggregate_style": mape_agg,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

# LSTM:
# from panelclv.models.monte_carlo_forecasting import (
#     run_monte_carlo_forecast, compute_forecast_metrics,
# )
# from panelclv.models.multinomial_lstm import InferenceMultinomialLSTMModel
#
# inference_model = InferenceMultinomialLSTMModel(
#     seq_cols=data["seq_cols"], embedded_cols=data["embedded_cols"],
#     target_col=data["target_col"],
# )
# inference_model.load_state_dict(trained_model.state_dict())
# forecast = run_monte_carlo_forecast(inference_model, data, n_simulations=30)
#
# Transformer:
# from panelclv.models.monte_carlo_forecasting import run_monte_carlo_forecast_transformer
# from panelclv.models.multinomial_transformer import InferenceMultinomialTransformerModel
#
# inference_model = InferenceMultinomialTransformerModel(
#     seq_cols=data["seq_cols"], embedded_cols=data["embedded_cols"],
#     target_col=data["target_col"],
# )
# inference_model.load_state_dict(trained_model.state_dict())
# forecast = run_monte_carlo_forecast_transformer(inference_model, data, n_simulations=30)
#
# metrics = compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])
# print(metrics)
