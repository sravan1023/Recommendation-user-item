# Recommendation User-Item Project

This repository contains dataset splits for a recommendation task along with
utilities for evaluating predicted rankings.

## Evaluation

Use the provided CLI to compute NDCG@K against the validation or test splits.

```bash
python eval.py --preds path/to/predictions.csv --split val --k 20
```

### Prediction file format

* CSV, whitespace-delimited `.txt`, or Parquet files are supported.
* Columns must include `user_id` and `item_id`.
* Optional columns:
  * `score`: higher values rank items earlier for each user.
  * `rank`: lower values rank items earlier for each user.

When neither `score` nor `rank` is provided, the original order of rows per user
is preserved.

### Choosing the ground-truth split

By default, the evaluator searches for `<split>_split` files (CSV, TXT, or
Parquet) in the repository root or the `interim_data/` directory. You can also
provide an explicit path:

```bash
python eval.py --preds path/to/predictions.csv --ground-truth interim_data/test_split.csv
```

The CLI reports the mean NDCG@K along with the number of users covered by the
prediction file.
