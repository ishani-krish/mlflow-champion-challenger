"""
register_champion.py - puts both models in the MLflow Model Registry and sets aliases.

    python register_champion.py          # register A and B, alias 'champion' -> winner, 'challenger' -> other
    python register_champion.py --flip   # (stretch goal) swap the two aliases

It reads the winner from the latest 'compare_sentiment' run (tag 'winner'),
so no run IDs are typed or hard-coded anywhere.
"""
import argparse

import mlflow
from mlflow import MlflowClient

import config

NAME = config.REGISTERED_MODEL_NAME


def latest_compare_run(client, include_debug):
    exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment '{config.EXPERIMENT_NAME}' not found. Run evaluate.py first.")
    flt = f"attributes.run_name = 'compare_{config.TASK}'"
    if not include_debug:
        flt += " and tags.is_debug = 'false'"
    runs = client.search_runs([exp.experiment_id], filter_string=flt,
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit("No matching compare run found. Run evaluate.py (without --limit) first, "
                         "or use --include-debug.")
    return runs[0]


def register(client, slot, run_id, model_uri, hf_id):
    # reuse the version if this exact run was registered before
    for mv in client.search_model_versions(f"name='{NAME}'"):
        if mv.tags.get("source_run_id") == run_id:
            print(f"Model {slot}: already registered as version {mv.version}")
            return mv.version
    mv = mlflow.register_model(model_uri=model_uri, name=NAME)
    client.set_model_version_tag(NAME, mv.version, "hf_model_id", hf_id)
    client.set_model_version_tag(NAME, mv.version, "slot", slot)
    client.set_model_version_tag(NAME, mv.version, "source_run_id", run_id)
    print(f"Model {slot}: registered as version {mv.version}")
    return mv.version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip", action="store_true", help="swap the champion and challenger aliases")
    parser.add_argument("--include-debug", action="store_true", help="allow a run made with --limit")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()

    if args.flip:
        champ = client.get_model_version_by_alias(NAME, "champion")
        chall = client.get_model_version_by_alias(NAME, "challenger")
        client.set_registered_model_alias(NAME, "champion", chall.version)
        client.set_registered_model_alias(NAME, "challenger", champ.version)
        print(f"Flipped: champion is now version {chall.version}, challenger is version {champ.version}")
        return

    cmp_run = latest_compare_run(client, args.include_debug)
    tags = cmp_run.data.tags
    winner = tags["winner"]
    loser = "B" if winner == "A" else "A"
    print(f"Winner according to the compare run: Model {winner}  ({tags['decision_reason']})")

    hf_ids = {"A": cmp_run.data.params["model_A"], "B": cmp_run.data.params["model_B"]}
    versions = {s: register(client, s, tags[f"run_id_{s}"], tags[f"model_uri_{s}"], hf_ids[s])
                for s in ("A", "B")}

    client.set_registered_model_alias(NAME, "champion", versions[winner])
    client.set_registered_model_alias(NAME, "challenger", versions[loser])
    print(f"\nAlias 'champion'   -> version {versions[winner]} ({hf_ids[winner]})")
    print(f"Alias 'challenger' -> version {versions[loser]} ({hf_ids[loser]})")
    print("\nNext: python predict_from_mlflow.py")


if __name__ == "__main__":
    main()
