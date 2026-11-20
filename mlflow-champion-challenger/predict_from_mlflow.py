"""
predict_from_mlflow.py - predicts on NEW data using only MLflow.

    python predict_from_mlflow.py                      # uses the @champion alias
    python predict_from_mlflow.py --also-challenger    # also runs @challenger

Note: this script never imports or calls Hugging Face. The model is loaded from the
registry alias, so the script does not know (or care) which version is the Champion.
"""
import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config


def predict_with_alias(alias, new_df):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, alias)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"
    model = mlflow.pyfunc.load_model(uri)

    hf_id = mv.tags.get("hf_model_id", "unknown")
    print(f"\nLoaded {uri}")
    print(f"  registry version : {mv.version}")
    print(f"  Hugging Face ID  : {hf_id}")

    out = model.predict(new_df[["text"]])
    out["alias"] = alias
    out["model_version"] = mv.version
    if "expected" in new_df.columns:
        out["expected"] = new_df["expected"].values
        out["match"] = out["prediction"] == out["expected"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--also-challenger", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    new_df = pd.read_csv(config.NEW_DATA_CSV)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    aliases = ["champion"] + (["challenger"] if args.also_challenger else [])
    for alias in aliases:
        out = predict_with_alias(alias, new_df)
        show = out.copy()
        show["text"] = show["text"].str.slice(0, 55)
        print(show.drop(columns=["hf_model_id"]).to_string(index=False))
        fname = "new_data_predictions.csv" if alias == "champion" else f"new_data_predictions_{alias}.csv"
        out.to_csv(config.OUTPUT_DIR / fname, index=False)
        print(f"Saved outputs/{fname}")


if __name__ == "__main__":
    main()
