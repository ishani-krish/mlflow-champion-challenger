"""
evaluate.py - runs BOTH models on the SAME inputs and logs everything to MLflow.

Run (MLflow server must already be running):
    python evaluate.py              # full run on data/eval_inputs.csv
    python evaluate.py --limit 10   # quick debug run on the first 10 rows

What it creates in MLflow (one experiment):
    compare_sentiment                      <- parent / summary run (decision lives here)
      +-- A_sentiment_distilbert-sst2      <- Champion candidate
      +-- B_sentiment_roberta-large        <- Challenger candidate
"""
import argparse
import random
import time

import matplotlib
matplotlib.use("Agg")  # draw plots without needing a screen
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from mlflow.models import infer_signature
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from transformers import pipeline

import config

LABELS = ["NEGATIVE", "POSITIVE"]
LIB_VERSIONS = (f"transformers={transformers.__version__}, torch={torch.__version__}, "
                f"mlflow={mlflow.__version__}, scikit-learn={sklearn.__version__}")


# ----------------------------------------------------------------------
# 1. The MLflow wrapper (Lab 9). This is what gets logged & registered.
#    Anyone who loads the model from MLflow just calls .predict() on a
#    DataFrame with a "text" column - no Hugging Face code needed.
# ----------------------------------------------------------------------
class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id, max_length=128, batch_size=8):
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.batch_size = batch_size

    def load_context(self, context):
        from transformers import pipeline as hf_pipeline
        import torch as _torch
        device = 0 if _torch.cuda.is_available() else -1
        self.pipe = hf_pipeline("sentiment-analysis", model=self.hf_model_id, device=device)

    def predict(self, context, model_input, params=None):
        import pandas as _pd
        if not isinstance(model_input, _pd.DataFrame):
            model_input = _pd.DataFrame({"text": list(model_input)})
        texts = model_input["text"].astype(str).tolist()
        outs = self.pipe(texts, batch_size=self.batch_size,
                         truncation=True, max_length=self.max_length)
        return _pd.DataFrame({
            "text": texts,
            "prediction": [o["label"].upper() for o in outs],
            "score": [round(float(o["score"]), 4) for o in outs],
            "hf_model_id": self.hf_model_id,
        })


def log_pyfunc(model, signature):
    """log_model argument is `name` in new MLflow and `artifact_path` in old MLflow."""
    kwargs = dict(
        python_model=model,
        signature=signature,
        pip_requirements=[f"mlflow=={mlflow.__version__}",
                          f"transformers=={transformers.__version__}",
                          f"torch=={torch.__version__}", "pandas"],
    )
    try:
        return mlflow.pyfunc.log_model(name="model", **kwargs)
    except TypeError:
        return mlflow.pyfunc.log_model(artifact_path="model", **kwargs)


# ----------------------------------------------------------------------
# 2. Metrics and plots
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
    }
    return {k: round(float(v), 4) for k, v in m.items()}


def confusion_fig(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def comparison_fig(res):
    names = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, slot in enumerate(["A", "B"]):
        vals = [res[slot]["metrics"][n] for n in names]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=f"{slot}: {res[slot]['short']}")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Champion (A) vs Challenger (B)")
    ax.legend()
    fig.tight_layout()
    return fig


def latency_fig(res):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    labels = [f"{s}: {res[s]['short']}" for s in ["A", "B"]]
    vals = [res[s]["metrics"]["avg_latency_ms"] for s in ["A", "B"]]
    bars = ax.bar(labels, vals, color=["#4C72B0", "#DD8452"])
    ax.bar_label(bars, fmt="%.0f ms")
    ax.set_ylabel("avg latency per example, ms (lower is better)")
    ax.set_title("Speed")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 3. One run per model
# ----------------------------------------------------------------------
def run_model(spec, df):
    slot, short = spec["slot"], spec["short"]
    run_name = f"{slot}_{config.TASK}_{short}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params({
            "hf_model_id": spec["hf_model_id"], "task": config.TASK, "role": spec["role"],
            "max_length": config.MAX_LENGTH, "batch_size": 1,   # 1 = timed one example at a time
            "eval_samples": len(df), "seed": config.SEED, "device": device,
        })
        mlflow.set_tags({
            "student_usn": config.STUDENT_USN, "library_versions": LIB_VERSIONS,
            "dataset_name": config.DATASET_NAME, "slot": slot,
        })

        print(f"\n[{run_name}] loading {spec['hf_model_id']} (first time = download, be patient)...")
        pipe = pipeline("sentiment-analysis", model=spec["hf_model_id"],
                        device=0 if device == "cuda" else -1)
        pipe("warm up", truncation=True, max_length=config.MAX_LENGTH)  # not timed

        preds, scores, times = [], [], []
        for text in df["text"]:
            t0 = time.perf_counter()
            out = pipe(text, truncation=True, max_length=config.MAX_LENGTH)[0]
            times.append(time.perf_counter() - t0)
            preds.append(out["label"].upper())
            scores.append(round(float(out["score"]), 4))

        metrics = compute_metrics(df["reference"], preds)
        metrics["avg_latency_ms"] = round(float(np.mean(times)) * 1000, 2)
        metrics["total_inference_s"] = round(float(np.sum(times)), 2)
        mlflow.log_metrics(metrics)

        # ---- predictions file: input, reference, prediction, correct ----
        pred_df = pd.DataFrame({
            "id": df["id"], "text": df["text"], "reference": df["reference"],
            "prediction": preds, "score": scores,
            "correct": [p == r for p, r in zip(preds, df["reference"])],
            "latency_ms": [round(t * 1000, 2) for t in times],
        })
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        pred_path = config.OUTPUT_DIR / f"predictions_{short}.csv"
        pred_df.to_csv(pred_path, index=False)

        # ---- artifacts ----
        mlflow.log_artifact(str(pred_path))
        mlflow.log_artifact(str(config.EVAL_CSV))
        mlflow.log_dict(metrics, "metrics_summary.json")
        fig = confusion_fig(df["reference"], preds, f"{slot}: {short}")
        mlflow.log_figure(fig, f"confusion_matrix_{short}.png")
        fig.savefig(config.OUTPUT_DIR / f"confusion_matrix_{short}.png", dpi=150)
        plt.close(fig)

        # ---- the model itself (wrapper) ----
        sample_out = pd.DataFrame({"text": df["text"].head(5).tolist(), "prediction": preds[:5],
                                   "score": scores[:5], "hf_model_id": spec["hf_model_id"]})
        signature = infer_signature(df[["text"]].head(5), sample_out)
        info = log_pyfunc(HFSentimentModel(spec["hf_model_id"], config.MAX_LENGTH), signature)
        mlflow.set_tag("model_uri", info.model_uri)

        print(f"[{run_name}] macro_f1={metrics['macro_f1']}  accuracy={metrics['accuracy']}  "
              f"avg_latency={metrics['avg_latency_ms']} ms")
        return {"slot": slot, "short": short, "hf_model_id": spec["hf_model_id"],
                "run_id": run.info.run_id, "model_uri": info.model_uri,
                "metrics": metrics, "pred_df": pred_df}


# ----------------------------------------------------------------------
# 4. The Champion-Challenger rule (same as README.md)
# ----------------------------------------------------------------------
def decide(a, b):
    p = config.PRIMARY_METRIC
    delta = round(b["metrics"][p] - a["metrics"][p], 4)
    lat_a, lat_b = a["metrics"]["avg_latency_ms"], b["metrics"]["avg_latency_ms"]
    ratio = round(lat_b / max(lat_a, 1e-6), 2)

    if ratio > config.LATENCY_GUARDRAIL:
        return "A", delta, ratio, f"Guardrail: B is {ratio}x slower than A (limit {config.LATENCY_GUARDRAIL}x). A stays Champion."
    if delta >= config.MARGIN:
        return "B", delta, ratio, f"B beats A by {delta} on {p} (>= margin {config.MARGIN}). B becomes Champion."
    if delta <= -config.MARGIN:
        return "A", delta, ratio, f"A beats B by {-delta} on {p} (>= margin {config.MARGIN}). A stays Champion."
    winner = "A" if lat_a <= lat_b else "B"
    return winner, delta, ratio, (f"Difference {delta} on {p} is inside the margin {config.MARGIN}. "
                                  f"Tie-breaker: faster model ({winner}) wins.")


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only use the first N rows (debugging)")
    args = parser.parse_args()

    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = pd.read_csv(config.EVAL_CSV)
    if args.limit:
        df = df.head(args.limit)
    print(f"Evaluation set: {len(df)} examples (same for both models)")

    with mlflow.start_run(run_name=f"compare_{config.TASK}") as parent:
        res = {spec["slot"]: run_model(spec, df) for spec in config.MODELS}
        a, b = res["A"], res["B"]
        winner, delta, ratio, reason = decide(a, b)
        loser = "B" if winner == "A" else "A"

        # ---- decision, logged in the summary run ----
        mlflow.log_params({
            "primary_metric": config.PRIMARY_METRIC, "margin": config.MARGIN,
            "latency_guardrail_x": config.LATENCY_GUARDRAIL,
            "direction": "higher is better for accuracy/macro_f1/macro_precision/macro_recall; "
                         "lower is better for avg_latency_ms/total_inference_s",
            "model_A": a["hf_model_id"], "model_B": b["hf_model_id"],
            "eval_samples": len(df), "seed": config.SEED,
        })
        mlflow.set_tags({
            "winner": winner, "decision_reason": reason,
            "winner_hf_model_id": res[winner]["hf_model_id"],
            "run_id_A": a["run_id"], "run_id_B": b["run_id"],
            "model_uri_A": a["model_uri"], "model_uri_B": b["model_uri"],
            "student_usn": config.STUDENT_USN, "library_versions": LIB_VERSIONS,
            "dataset_name": config.DATASET_NAME,
            "is_debug": str(bool(args.limit)).lower(),
        })
        mlflow.log_metric("delta_primary_metric", delta)       # Challenger minus Champion
        mlflow.log_metric("latency_ratio_B_over_A", ratio)
        for slot in ("A", "B"):
            for k, v in res[slot]["metrics"].items():
                mlflow.log_metric(f"{slot}_{k}", v)

        # ---- plots ----
        for name, fig in [("comparison_bar_chart", comparison_fig(res)), ("latency_bar_chart", latency_fig(res))]:
            mlflow.log_figure(fig, f"{name}.png")
            fig.savefig(config.OUTPUT_DIR / f"{name}.png", dpi=150)
            plt.close(fig)

        # ---- results table (paste into README) ----
        rows = []
        for k, better in [("accuracy", "higher"), ("macro_f1", "higher"), ("macro_precision", "higher"),
                          ("macro_recall", "higher"), ("avg_latency_ms", "lower"), ("total_inference_s", "lower")]:
            va, vb = a["metrics"][k], b["metrics"][k]
            rows.append(f"| {k} | {va} | {vb} | {round(vb - va, 4)} | {better} |")
        table = ("| Metric | A (Champion cand.) | B (Challenger) | Diff (B - A) | Better when |\n"
                 "| --- | --- | --- | --- | --- |\n" + "\n".join(rows))
        (config.OUTPUT_DIR / "results_table.md").write_text(table + "\n")
        mlflow.log_artifact(str(config.OUTPUT_DIR / "results_table.md"))

        # ---- cases where the models disagree (fill in 'my_comment' yourself!) ----
        dis = pd.DataFrame({
            "id": a["pred_df"]["id"], "text": a["pred_df"]["text"], "reference": a["pred_df"]["reference"],
            "pred_A": a["pred_df"]["prediction"], "pred_B": b["pred_df"]["prediction"]})
        dis = dis[dis["pred_A"] != dis["pred_B"]].copy()
        dis["who_is_correct"] = np.where(dis["pred_A"] == dis["reference"], "A",
                                         np.where(dis["pred_B"] == dis["reference"], "B", "neither"))
        dis["my_comment"] = ""
        dis.to_csv(config.OUTPUT_DIR / "disagreements.csv", index=False)
        mlflow.log_artifact(str(config.OUTPUT_DIR / "disagreements.csv"))

        # ---- other files ----
        mlflow.log_dict({s: res[s]["metrics"] for s in ("A", "B")}, "metrics_summary.json")
        (config.OUTPUT_DIR / "metrics_summary.json").write_text(
            pd.Series({s: res[s]["metrics"] for s in ("A", "B")}).to_json(indent=2))
        mlflow.log_artifact(str(config.EVAL_CSV))
        for f in ("README.md", "requirements.txt"):
            if (config.BASE_DIR / f).exists():
                mlflow.log_artifact(str(config.BASE_DIR / f))

    print("\n" + "=" * 60)
    print(table)
    print("=" * 60)
    print(f"Delta ({config.PRIMARY_METRIC}, B - A): {delta}")
    print(f"DECISION: {reason}")
    print(f"Winner = Model {winner} ({res[winner]['hf_model_id']})")
    print(f"Models that disagree: {len(dis)} (see outputs/disagreements.csv - add your comments)")
    print("\nNext: python register_champion.py")


if __name__ == "__main__":
    main()
