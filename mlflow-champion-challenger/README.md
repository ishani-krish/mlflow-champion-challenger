# Champion–Challenger Model Comparison with Hugging Face and MLflow

Name: K Ishani
USN: 1MS25CI060

## 1. Task and models

- **Task:** Sentiment classification (binary: POSITIVE / NEGATIVE)
- **Dataset:** GLUE SST-2 validation set, random sample of 100 sentences (seed 42), saved in `data/eval_inputs.csv`
- **Model A (current Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters)
- **Model B (Challenger):** `siebert/sentiment-roberta-large-english` (~355M parameters)
- **New data for the final step:** 10 sentences I wrote myself, in `data/new_data.csv` (not part of the evaluation set)

## 2. Champion rule (written BEFORE the final run)

| Part | Rule |
| --- | --- |
| Primary metric | **Macro-F1** on the shared evaluation set. **Higher is better.** |
| Minimum improvement margin | The 0.02 minimum improvement margin was selected to avoid replacing the Champion based on a very small difference in Macro-F1. Since the evaluation set contains only 100 examples, small metric differences should be interpreted cautiously. |
| Tie-breaker | If the difference is inside the margin (smaller than 0.02 either way), the **faster model** (lower average latency) is the Champion. |
| Guardrail | The Challenger is rejected if its average latency is **more than 10×** the Champion's. |
| Direction of every metric | accuracy, macro_f1, macro_precision, macro_recall: **higher is better**. avg_latency_ms, total_inference_s: **lower is better**. |

Only the primary metric, margin, tie-breaker and guardrail decide the winner. The other metrics are reported for information.

## 3. How to run

Run everything from this folder, in this order.

```bash
# 0. install (use a virtual environment if you can)
pip install -r requirements.txt

# 1. Terminal 1: start MLflow and leave it running
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 2. Terminal 2: edit STUDENT_USN in config.py, then:
python make_data.py                  # creates data/eval_inputs.csv (use --n 10 to debug)
python evaluate.py                   # runs both models, logs to MLflow, makes decision
python register_champion.py          # registers both models, sets champion / challenger aliases
python predict_from_mlflow.py        # loads models:/sentiment-model@champion, predicts new data

# optional stretch goal: swap the aliases and rerun WITHOUT editing the script
python register_champion.py --flip
python predict_from_mlflow.py
```

Open http://127.0.0.1:5000 to see the experiment and the Models page.

## 4. Results

| Metric | A (Champion cand.) | B (Challenger) | Diff (B - A) | Better when |
| --- | --- | --- | --- | --- |
| accuracy | 0.91 | 0.92 | 0.01 | higher |
| macro_f1 | 0.9079 | 0.9179 | 0.01 | higher |
| macro_precision | 0.9093 | 0.921 | 0.0117 | higher |
| macro_recall | 0.9068 | 0.9155 | 0.0087 | higher |
| avg_latency_ms | 50.76 | 351.73 | 300.97 | lower |
| total_inference_s | 5.08 | 35.17 | 30.09 | lower |

Plots are in `outputs/` (confusion matrices, comparison bar chart, latency chart). Cases where the models disagree are in `outputs/disagreements.csv`, with my comments for them.

## 5. Decision

Model B achieved a Macro-F1 score of 0.9179, compared with 0.9079 for Model A, giving an improvement of 0.01. However, the predefined selection rule required an improvement of at least 0.02 for the Challenger to replace the Champion. Since the observed difference was within this margin, the tie-breaker selected the faster model. Model A had an average latency of 50.76 ms, while Model B required 351.73 ms per example. Thus, Model A was approximately 6.9 times faster. Although Model B had slightly higher accuracy (0.92 compared with 0.91), its improvement did not meet the required threshold. The Challenger passed the latency guardrail because its average latency was approximately 6.93 times that of the Champion, below the maximum permitted ratio of 10 times. The decision therefore followed the predefined rule and favoured Model A for its lower inference latency. The evaluation used only 100 examples, so the small difference in Macro-F1 should not be treated as conclusive evidence that one model is generally more accurate. The seven disagreement cases also show that the models can behave differently on individual examples.

## 6. Limitations and declaration

- **Sample size:** only 100 sentences, so small differences (1–2 points) may just be noise.
- **Data:** SST-2 sentences are short movie-review snippets, so results may not carry over to other kinds of text.
- **Timing:** Latency was measured in GitHub Codespaces, using the same evaluation setup for both models. Results may differ on other hardware and execution environments.
- **Models:** both are pretrained models used as they are. No training was done.

## 7. Conclusion

This project implemented a Champion–Challenger model comparison workflow using Hugging Face Transformers and MLflow. Two pretrained sentiment classification models were evaluated on the same dataset using accuracy, Macro-F1, precision, recall and inference latency. The predefined selection criteria were used to select the Champion. Both models were registered in MLflow, assigned Champion and Challenger aliases, and used to generate predictions on new input sentences. This project demonstrates how model performance, inference speed and predefined selection criteria can be combined in a reproducible model management workflow.