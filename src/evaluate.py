"""
Intraday regime validation (Phase 4).

Regimes are unsupervised, so we judge them by persistence, economic coherence,
filter-vs-smoother honesty, and (on synthetic data) agreement with the known
ground truth.

NOTE — the original *daily* index PoC's evaluation helpers (manual AIC/BIC sweep,
train/test log-likelihood, daily persistence, transition-matrix heatmap) are NOT
here anymore. They are preserved in the Jupyter notebooks under `notebooks/`
(04–08). This module is intraday-only; the intraday AIC/BIC sweep now lives in
`model.select_n_states`.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from . import config


# --------------------------------------------------------------------------- #
# Intraday validation metrics
# --------------------------------------------------------------------------- #
def regime_persistence(df, state_col="filt_state", label_col="filt_label"):
    """Average consecutive bars spent in each regime before switching.
    Dwell ~1 bar means the model is fitting noise, not regimes."""
    states = df[state_col].values
    labels = df[label_col].values
    runs: dict = {}
    cur, count = states[0], 1
    for i in range(1, len(states)):
        if states[i] == cur:
            count += 1
        else:
            runs.setdefault(labels[i - 1], []).append(count)
            cur, count = states[i], 1
    runs.setdefault(labels[-1], []).append(count)

    print("Regime persistence (consecutive 10-min bars):")
    for label, r in runs.items():
        print(f"  {label:30s} | avg: {np.mean(r):4.1f} bars | max: {np.max(r):3d} | runs: {len(r)}")
    return runs


def economic_coherence(df, label_col="filt_label"):
    """Mean NEXT-bar return per regime (forward return is computed within each
    session so it never bridges a break). A coherent model shows the
    buy-pressure regime with positive forward return and sell-pressure negative."""
    fwd = df.groupby(["date", "session_id"], sort=False)["ret"].shift(-1)
    tmp = df.assign(fwd_ret=fwd).dropna(subset=["fwd_ret"])
    g = tmp.groupby(label_col)["fwd_ret"]
    summary = pd.DataFrame({
        "mean_fwd_ret": g.mean(),
        "std_fwd_ret": g.std(),
        "win_rate": g.apply(lambda s: (s > 0).mean()),
        "n_bars": g.size(),
    }).sort_values("mean_fwd_ret")
    print("Economic coherence (next-bar return by regime):")
    print(summary.to_string(float_format=lambda v: f"{v:.6f}"))
    return summary


def filter_vs_smoother(df, filt_col="filt_state", smooth_col="smooth_state"):
    """How often the honest online filter agrees with the full-sequence smoother.
    Large disagreement means the smoother was leaning on future information."""
    agree = float((df[filt_col].values == df[smooth_col].values).mean())
    print(f"Filter vs smoother agreement: {agree:.1%}")
    return agree


def compare_to_truth(df, truth: dict, state_col="filt_state"):
    """Adjusted Rand Index between detected regimes and synthetic ground truth.
    ARI ignores label permutation, so it is the right metric for unsupervised
    cluster recovery. 0 = random, 1 = perfect."""
    true = df.index.map(lambda ts: truth.get(ts.isoformat()))
    mask = pd.notna(true)
    ari = adjusted_rand_score(np.array(true)[mask].astype(int), df[state_col].values[mask])
    print(f"Adjusted Rand Index vs ground truth: {ari:.3f}")
    return ari


def report_metrics(df, truth=None, tag="ALL"):
    """Run the whole metric suite on ONE slice of the backtest df and print it
    under a header.

    Call it separately on the TRAIN slice (`df.iloc[:split]`) and the TEST slice
    (`df.iloc[split:]`). The TEST numbers are the honest, out-of-sample ones — the
    model never saw those bars. A large TRAIN→TEST drop is the fingerprint of
    overfitting; similar numbers mean the model generalizes.
    """
    print(f"\n================= {tag} - {len(df)} bars =================")
    if len(df) == 0:
        print("  (no bars in this slice)")
        return
    regime_persistence(df)
    print()
    economic_coherence(df)
    print()
    filter_vs_smoother(df)
    if truth is not None:
        compare_to_truth(df, truth)


# Fixed colors per regime LABEL (not per state index) so the same regime keeps
# one color across instruments — state indices are arbitrary and differ per model.
REGIME_COLORS = {
    "Buy pressure (hajar kanan)": "#2ca02c",   # green = buying / up
    "Sell pressure (hajar kiri)": "#d62728",   # red = selling / down
    "Quiet / Ranging": "#aec7e8",              # light blue = calm
    "Volatile / Choppy": "#ff7f0e",            # orange = turbulent
}


def plot_intraday_regimes(df, price_col="close", state_col="filt_state",
                          label_col="filt_label", split=None, filename="intraday_regimes.png",
                          ax=None, title="Intraday Regime Detection (10-min bars) — online filtered"):
    """Shade the price series by detected regime.

    If `ax` is None, draws its own figure and saves it to results/figures (the
    standalone behaviour). If an `ax` is passed, it draws into that subplot
    instead (used for side-by-side comparisons) and does not save.
    """
    own_figure = ax is None
    if own_figure:
        config.ensure_dirs()
        _fig, ax = plt.subplots(figsize=(15, 6))

    states = sorted(df[state_col].unique())
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(df))                       # ordinal x to avoid overnight gaps

    ax.plot(x, df[price_col].values, color="black", linewidth=0.7, zorder=3)
    label_for = dict(zip(df[state_col], df[label_col]))
    lo, hi = df[price_col].min(), df[price_col].max()
    seen = set()                                  # dedupe legend when labels repeat
    for s in states:
        label = label_for.get(s, f"state_{s}")
        color = REGIME_COLORS.get(label, cmap(s % 10))   # color by label, not state #
        mask = df[state_col].values == s
        ax.fill_between(x, lo, hi, where=mask, alpha=0.35, color=color,
                        label=(label if label not in seen else None), zorder=1)
        seen.add(label)
    if split is not None:
        ax.axvline(split, color="black", linestyle="--", linewidth=1, label="Train/Test split")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Bar # (sessions concatenated)")
    ax.set_ylabel(price_col)

    if own_figure:
        plt.tight_layout()
        out = config.FIGURES_DIR / filename
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Saved regime plot -> {out}")
        return out
    return ax


# --------------------------------------------------------------------------- #
# Teaching visualizations. Each follows the same pattern as plot_intraday_regimes:
# pass no `ax` to get a standalone saved PNG, or pass an `ax` to draw into a
# dashboard subplot. They turn the printed metrics into pictures so you can SEE
# why the model chose what it chose.
# --------------------------------------------------------------------------- #
def _finish(own, filename):
    """Shared save/return tail for the standalone-figure case."""
    if own:
        plt.tight_layout()
        out = config.FIGURES_DIR / filename
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Saved -> {out}")
        return out
    return None


_SHORT = {
    "Buy pressure (hajar kanan)": "Buy\n(kanan)",
    "Sell pressure (hajar kiri)": "Sell\n(kiri)",
    "Quiet / Ranging": "Quiet",
    "Volatile / Choppy": "Volatile",
}


def _short(label):
    """Compact tick label for crowded axes (full regime names are long)."""
    return _SHORT.get(label, str(label))


def plot_state_selection(selection_results, filename="state_selection.png", ax=None):
    """AIC & BIC vs number of hidden states — the picture behind `select_n_states`.

    Both scores reward fit but penalize complexity, so each curve dips at a sweet
    spot and rises again when extra states just overfit. BIC penalizes harder, so
    we take its minimum as the conservative choice. Seeing the curve tells you
    whether the choice was clear-cut (deep, lonely dip) or marginal (flat bottom).
    """
    own = ax is None
    if own:
        config.ensure_dirs()
        _f, ax = plt.subplots(figsize=(7, 5))
    if not selection_results:
        ax.text(0.5, 0.5, "no AIC/BIC sweep\n(n_states was fixed)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _finish(own, filename)

    ns = [r[0] for r in selection_results]
    aics = [r[1] for r in selection_results]
    bics = [r[2] for r in selection_results]
    best = ns[int(np.argmin(bics))]
    ax.plot(ns, aics, "o-", label="AIC", color="#1f77b4")
    ax.plot(ns, bics, "s-", label="BIC", color="#d62728")
    ax.axvline(best, ls="--", color="gray", lw=1, label=f"BIC pick = {best}")
    ax.set_xlabel("number of hidden states")
    ax.set_ylabel("score (lower is better)")
    ax.set_title("Model selection: AIC / BIC sweep")
    ax.set_xticks(ns)
    ax.legend()
    return _finish(own, filename)


def plot_transition_matrix(model, state_labels, filename="transition_matrix.png", ax=None):
    """Heatmap of the learned transition matrix A[i, j] = P(next = j | now = i).

    The DIAGONAL is regime stickiness: a high diagonal (e.g. 0.9) means regimes
    persist, which is exactly what we want — it's the same stickiness that gives
    the online filter its inertia. Bright off-diagonal cells reveal the common
    switches (e.g. which regime usually follows 'Volatile').
    """
    own = ax is None
    if own:
        config.ensure_dirs()
        _f, ax = plt.subplots(figsize=(7, 6))
    A = model.transmat_
    n = A.shape[0]
    labels = [_short(state_labels.get(i, f"state_{i}")) for i in range(n)]

    im = ax.imshow(A, vmin=0, vmax=1, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="probability")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    color="white" if A[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("to regime"); ax.set_ylabel("from regime")
    ax.set_title("Transition matrix (row → column)")
    return _finish(own, filename)


def plot_regime_fingerprint(df, feature_cols=None, label_col="filt_label",
                            filename="regime_fingerprint.png", ax=None):
    """Heatmap of each regime's average feature vector — its 'fingerprint'.

    This is the single most useful plot for understanding the LABELS: it shows,
    at a glance, that (say) 'Buy pressure' is the row with high signed_vol_imb and
    positive OFI, while 'Volatile' is the row that lights up on realized_vol.
    Colors are z-scored DOWN each column (per feature, across regimes) so you read
    'which regime is high/low on this feature'; the printed numbers are raw means.
    """
    own = ax is None
    if own:
        config.ensure_dirs()
        _f, ax = plt.subplots(figsize=(9, 5))
    feature_cols = feature_cols or config.FEATURE_COLS
    g = df.groupby(label_col)[feature_cols].mean()
    z = (g - g.mean()) / (g.std(ddof=0) + 1e-9)     # standardize per feature

    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                       label="z-score across regimes")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(g.index)))
    ax.set_yticklabels([_short(l) for l in g.index], fontsize=8)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]:.3g}", ha="center", va="center",
                    color="black", fontsize=7)
    ax.set_title("Regime fingerprints (mean feature per regime)")
    return _finish(own, filename)


def plot_economic_coherence(df, label_col="filt_label",
                            filename="economic_coherence.png", ax=None):
    """Bar chart of mean NEXT-bar return per regime, with win-rate labels.

    This is the economic sanity check made visual: a coherent model has the
    buy-pressure bar clearly positive and the sell-pressure bar clearly negative.
    Error bars are the standard error of the mean (how sure we are of each bar's
    height); the % above each bar is the share of next bars that were positive.
    """
    own = ax is None
    if own:
        config.ensure_dirs()
        _f, ax = plt.subplots(figsize=(8, 5))
    fwd = df.groupby(["date", "session_id"], sort=False)["ret"].shift(-1)
    tmp = df.assign(fwd_ret=fwd).dropna(subset=["fwd_ret"])
    g = tmp.groupby(label_col)["fwd_ret"]
    means = g.mean()
    sems = g.std() / np.sqrt(g.size())
    winrate = g.apply(lambda s: (s > 0).mean())
    order = means.sort_values().index

    colors = [REGIME_COLORS.get(l, "#888888") for l in order]
    xs = np.arange(len(order))
    ax.bar(xs, means[order].values, yerr=sems[order].values, color=colors,
           capsize=4, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    for x, l in zip(xs, order):
        m = means[l]
        ax.text(x, m + (sems[l] + abs(m) * 0.05) * np.sign(m or 1),
                f"{winrate[l]:.0%}", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([_short(l) for l in order], fontsize=8)
    ax.set_ylabel("mean next-bar log return")
    ax.set_title("Economic coherence (win-rate labelled)")
    return _finish(own, filename)


def plot_regime_probabilities(df, state_col="filt_state", label_col="filt_label",
                              max_bars=300, filename="regime_probabilities.png", ax=None):
    """Stacked area of the filtered posterior P(state | data up to now) over time.

    This is the online filter's CONFIDENCE, bar by bar. Tall single-color bands =
    the filter is sure which regime we're in; thin churning stripes = it's
    uncertain / transitioning. It's the honest, real-time view (no future used),
    so it's exactly what a live dashboard would show. Limited to the first
    `max_bars` bars so the bands stay readable.
    """
    own = ax is None
    if own:
        config.ensure_dirs()
        _f, ax = plt.subplots(figsize=(15, 4))
    prob_cols = sorted([c for c in df.columns if c.startswith("prob_")],
                       key=lambda c: int(c.split("_")[1]))
    sub = df.iloc[:max_bars]
    x = np.arange(len(sub))
    label_for = dict(zip(df[state_col], df[label_col]))
    cmap = plt.get_cmap("tab10")
    series, colors, labels = [], [], []
    for c in prob_cols:
        s = int(c.split("_")[1])
        lab = label_for.get(s, f"state_{s}")
        series.append(sub[c].values)
        colors.append(REGIME_COLORS.get(lab, cmap(s % 10)))
        labels.append(lab)
    ax.stackplot(x, *series, colors=colors, labels=labels, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(sub) - 1)
    ax.set_xlabel(f"Bar # (first {len(sub)} bars)")
    ax.set_ylabel("P(regime | data so far)")
    ax.set_title("Online filtered regime probabilities")
    ax.legend(loc="upper right", fontsize=7, ncol=len(labels))
    return _finish(own, filename)


def make_report_figures(df, artifact, truth=None, split=None):
    """Generate the full set of teaching figures in one call.

    Produces, in results/figures/:
      - intraday_regimes.png      price shaded by detected regime (wide)
      - regime_probabilities.png  online filter confidence over time (wide)
      - regime_dashboard.png      2x2: model selection, transition matrix,
                                  regime fingerprints, economic coherence
    """
    config.ensure_dirs()
    model = artifact["model"]
    state_labels = artifact.get("state_labels") or {}
    selection = artifact.get("meta", {}).get("selection")

    plot_intraday_regimes(df, split=split)
    plot_regime_probabilities(df)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plot_state_selection(selection, ax=axes[0, 0])
    plot_transition_matrix(model, state_labels, ax=axes[0, 1])
    plot_regime_fingerprint(df, ax=axes[1, 0])
    plot_economic_coherence(df, ax=axes[1, 1])
    fig.suptitle("Intraday regime detector — diagnostics dashboard", fontsize=14)
    plt.tight_layout()
    out = config.FIGURES_DIR / "regime_dashboard.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved -> {out}")
    return out