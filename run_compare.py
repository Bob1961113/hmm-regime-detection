"""
Compare two instruments side-by-side: the original DEMO vs a BBCA-like blue chip.

IMPORTANT: both are still SYNTHETIC. "BBCA" here is the generator tuned to resemble
BBCA's real *characteristics* (high price ~Rp9,500, Rp25 tick, low volatility, very
liquid) — it is NOT real BBCA market data, which we don't have access to yet.

Produces one figure with both regime charts stacked, and prints the evaluation
metrics for each so you can compare how the same pipeline behaves on a calm blue
chip vs a more volatile name.

Run:  python run_compare.py
"""
import matplotlib.pyplot as plt

from src import config
from src.ingest.synthetic_adapter import SyntheticAdapter
from src.ingest.recorder import record
from src.ingest.replay import ReplayAdapter
from src.bars import build_bars
from src.features import compute_features
from src.backtest import train_and_backtest
from src import evaluate


def run_one(adapter, tag):
    """Record -> bars -> features -> fit+backtest for one instrument."""
    feed_path = config.RAW_DIR / f"feed_{tag}.jsonl"
    n_events = record(adapter, feed_path)
    bars = build_bars(ReplayAdapter(feed_path).stream())
    feats = compute_features(bars)
    df, artifact, split = train_and_backtest(feats)   # let BIC choose; forcing a count mislabels regimes
    print(f"   [{tag}] {n_events} events -> {len(feats)} bars | "
          f"states={artifact['meta']['n_states']} | "
          f"price range {df['close'].min():.0f}-{df['close'].max():.0f}")
    return {"tag": tag, "df": df, "artifact": artifact, "split": split,
            "truth": adapter.truth}


def report(res):
    df, split = res["df"], res["split"]
    evaluate.report_metrics(df.iloc[:split], truth=res["truth"],
                            tag=f"{res['tag']} TRAIN (in-sample)")
    evaluate.report_metrics(df.iloc[split:], truth=res["truth"],
                            tag=f"{res['tag']} TEST (out-of-sample)")


def main():
    config.ensure_dirs()

    print("Building DEMO (original synthetic profile) ...")
    demo = run_one(SyntheticAdapter(symbol="DEMO", n_days=40, seed=7), "DEMO")

    print("\nBuilding BBCA-like (blue-chip synthetic profile) ...")
    bbca = run_one(SyntheticAdapter.from_profile("BBCA", n_days=40, seed=7), "BBCA")

    # Side-by-side figure (stacked, since each is a long time series)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(15, 10))
    evaluate.plot_intraday_regimes(
        demo["df"], split=demo["split"], ax=ax_top,
        title="DEMO (volatile synthetic) — online filtered regimes")
    evaluate.plot_intraday_regimes(
        bbca["df"], split=bbca["split"], ax=ax_bot,
        title="BBCA-like (blue-chip synthetic, Rp25 tick) — online filtered regimes")
    plt.tight_layout()
    out = config.FIGURES_DIR / "compare_demo_bbca.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nSaved comparison figure -> {out}")

    report(demo)
    report(bbca)
    print("\nDone.")


if __name__ == "__main__":
    main()
