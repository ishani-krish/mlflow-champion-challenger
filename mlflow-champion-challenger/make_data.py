"""
make_data.py - builds data/eval_inputs.csv (the SHARED evaluation set).

Takes a random (seeded) sample of sentences from the SST-2 validation set.
Run once:   python make_data.py          (default 100 sentences)
Debugging:  python make_data.py --n 10
"""
import argparse
import pandas as pd
from datasets import load_dataset
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="number of sentences")
    args = parser.parse_args()

    ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    ds = ds.shuffle(seed=config.SEED).select(range(args.n))

    df = pd.DataFrame({
        "id": range(len(ds)),
        "text": [s.strip() for s in ds["sentence"]],
        "reference": ["POSITIVE" if y == 1 else "NEGATIVE" for y in ds["label"]],
    })
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.EVAL_CSV, index=False)
    print(f"Saved {len(df)} rows to {config.EVAL_CSV}")
    print(df["reference"].value_counts())

    # safety check: new data must NOT overlap with the evaluation set
    if config.NEW_DATA_CSV.exists():
        new = pd.read_csv(config.NEW_DATA_CSV)
        overlap = set(new["text"].str.lower()) & set(df["text"].str.lower())
        print("Overlap with new_data.csv:", overlap if overlap else "none (good)")


if __name__ == "__main__":
    main()
