"""
End-to-end demo of the intraday regime-detection pipeline on SYNTHETIC data.

This proves the whole architecture runs today, before the real proprietary feed
exists. When the feed is available, only `src/ingest/websocket_adapter.py` needs
to be implemented — every other step below stays the same.

Flow:  generate+record -> replay -> bars -> features -> fit+backtest -> validate

Run:   python run_demo.py
"""
from src import config
from src.ingest.synthetic_adapter import SyntheticAdapter
from src.ingest.recorder import record
from src.ingest.replay import ReplayAdapter
from src.bars import build_bars
from src.features import compute_features
from src.backtest import train_and_backtest
from src.model import save_artifact
from src import evaluate, store


def main():
    config.ensure_dirs()
    feed_path = config.RAW_DIR / "demo_feed.jsonl"

    print("1) Generating + recording synthetic feed ...")
    adapter = SyntheticAdapter(n_days=40, seed=7)
    n_events = record(adapter, feed_path)
    truth = adapter.truth
    print(f"   recorded {n_events} events -> {feed_path}")

    print("\n2) Building session-aware bars + features (from replay) ...")
    bars = build_bars(ReplayAdapter(feed_path).stream())
    feats = compute_features(bars)
    print(f"   {len(bars)} bars -> {len(feats)} feature rows ({len(config.FEATURE_COLS)} features)")

    print("\n3) Fitting pipeline (de-seasonalize -> scale -> AIC/BIC -> HMM) + backtest ...")
    df, artifact, split = train_and_backtest(feats)        # n_states chosen by BIC
    save_artifact(config.MODEL_DIR / "regime_artifact.pkl",
                  model=artifact["model"], scaler=artifact["scaler"],
                  seasonality_stats=artifact["seasonality_stats"],
                  feature_cols=artifact["feature_cols"],
                  state_labels=artifact["state_labels"],
                  ref_means=artifact["ref_means"], meta=artifact["meta"])
    print(f"   states={artifact['meta']['n_states']}  labels={artifact['state_labels']}")

    print("\n4) Validation (train vs test) -----------------------------------")
    print("   TEST = the held-out 30% the model never trained on -> the honest")
    print("   numbers. A big TRAIN->TEST drop would signal overfitting.")
    evaluate.report_metrics(df.iloc[:split], truth=truth, tag="TRAIN (in-sample)")
    evaluate.report_metrics(df.iloc[split:], truth=truth, tag="TEST (out-of-sample)")

    print("\n   Generating teaching figures ...")
    evaluate.make_report_figures(df, artifact, truth=truth, split=split)

    print("\n5) Persisting outputs ...")
    store.save_csv(df)
    store.save_sqlite(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
