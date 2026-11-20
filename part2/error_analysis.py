import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


_NEGATION = re.compile(
    r"\b(?:no|not|never|cannot|neither|nor)\b|\b\w+n['\u2019]t\b",
    re.IGNORECASE,
)


def extract_tweet_features(texts, tokenizer, twitter_vocab, wikigiga_vocab):
    """Describe tweets using one common tokenizer for both vocabularies."""
    rows = []
    for item, text in enumerate(texts):
        tokens = tokenizer.tokens(text)
        count = len(tokens)
        rows.append({
            "item": item,
            "text": text,
            "length": count,
            "punctuation": int("!" in text or "?" in text),
            "hashtag": int("#" in text),
            "mention": int("@" in text),
            "negation": int(bool(_NEGATION.search(text))),
            "oov_rate": sum(token not in twitter_vocab for token in tokens) / count if count else 0.0,
            "oov_wikigiga": sum(token not in wikigiga_vocab for token in tokens) / count if count else 0.0,
        })
    return pd.DataFrame(rows, columns=[
        "item", "text", "length", "punctuation", "hashtag", "mention",
        "negation", "oov_rate", "oov_wikigiga",
    ])


def validate_shared_results(scores, predictions, folds_by_seed, num_classes):
    """Verify that shared out-of-fold predictions match the current training splits."""
    score_columns = {"model", "macro-F1", "run", "fold"}
    prediction_columns = {"model", "seed", "fold", "text", "gold", "pred"}
    if not score_columns.issubset(scores.columns) or not prediction_columns.issubset(predictions.columns):
        raise ValueError("Shared CSV files do not contain the required columns.")
    if scores.empty or predictions.empty or not folds_by_seed:
        raise ValueError("Shared scores, predictions and folds must not be empty.")
    if scores[list(score_columns)].isna().any().any() or predictions[list(prediction_columns)].isna().any().any():
        raise ValueError("Shared results contain missing values.")
    if scores.duplicated(["model", "run", "fold"]).any():
        raise ValueError("Shared scores contain duplicate model/run/fold rows.")
    if predictions.duplicated(["model", "seed", "fold", "text"]).any():
        raise ValueError("Shared predictions contain duplicate model/seed/fold/text rows.")
    if not scores["macro-F1"].between(0, 1).all():
        raise ValueError("Shared macro-F1 values must be between zero and one.")
    for column in ["gold", "pred"]:
        values = predictions[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= num_classes)):
            raise ValueError(f"{column} must contain integer class IDs in range.")

    seed_to_run = {seed: run for run, seed in enumerate(folds_by_seed)}
    expected = {(seed, fold) for seed, folds in folds_by_seed.items() for fold in folds}
    expected_score_keys = {(seed_to_run[seed], fold) for seed, fold in expected}
    for name, group in scores.groupby("model", sort=False):
        actual_keys = set(group[["run", "fold"]].itertuples(index=False, name=None))
        if actual_keys != expected_score_keys:
            raise ValueError(f"{name} does not contain exactly the expected run/fold scores.")
    aliases = {"pool-baseline-twitter": "pool-baseline"}
    for name, group in predictions.groupby("model", sort=False):
        available = set(group[["seed", "fold"]].itertuples(index=False, name=None))
        if available != expected:
            raise ValueError(f"{name} does not contain exactly the expected seed/fold combinations.")
        for (seed, fold), part in group.groupby(["seed", "fold"], sort=False):
            expected_examples = Counter(folds_by_seed[seed][fold]["validation"])
            actual_examples = Counter(zip(part["text"], part["gold"]))
            if actual_examples != expected_examples:
                raise ValueError(f"{name}: texts or labels differ for seed {seed}, fold {fold}.")
            model_name = aliases.get(name, name)
            score = scores[
                (scores["model"] == model_name)
                & (scores["run"] == seed_to_run[seed])
                & (scores["fold"] == fold)
            ]
            actual_f1 = f1_score(part["gold"], part["pred"], average="macro")
            if len(score) != 1 or not np.isclose(score["macro-F1"].iloc[0], actual_f1, atol=1e-10, rtol=0):
                raise ValueError(f"{name}: macro-F1 does not match predictions for seed {seed}, fold {fold}.")


def paired_oof_predictions(predictions, model_a, model_b, seeds):
    """Average paired out-of-fold correctness differences within each text."""
    if model_a == model_b:
        raise ValueError("Select two different models.")
    keys = ["seed", "fold", "text", "gold"]
    a = predictions[predictions["model"] == model_a][keys + ["pred"]]
    b = predictions[predictions["model"] == model_b][keys + ["pred"]]
    if a.empty or b.empty:
        raise ValueError("Both models must have out-of-fold predictions.")
    pairs = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(pairs) != len(a) or len(pairs) != len(b):
        raise ValueError("Selected models must predict exactly the same seed/fold/text examples.")
    pairs["correct_a"] = (pairs["pred_a"] == pairs["gold"]).astype(int)
    pairs["correct_b"] = (pairs["pred_b"] == pairs["gold"]).astype(int)
    pairs["difference"] = pairs["correct_b"] - pairs["correct_a"]
    if pairs.groupby("text")["gold"].nunique().gt(1).any():
        raise ValueError("The same text has conflicting gold labels.")
    if pairs.duplicated(["text", "seed"]).any():
        raise ValueError("Each text must have only one paired prediction per seed.")
    if any(set(group["seed"]) != set(seeds) for _, group in pairs.groupby("text")):
        raise ValueError("Each text must have one paired prediction per seed.")
    return pairs.groupby(["text", "gold"], as_index=False).agg(
        correct_a=("correct_a", "mean"),
        correct_b=("correct_b", "mean"),
        difference=("difference", "mean"),
        matched_seeds=("seed", "nunique"),
    )


def prepare_shared_bda(predictions, features, model_names, seed, label_names):
    """Prepare matched out-of-fold systems and tweet features for tutorial BDA."""
    if len(model_names) < 2 or len(set(model_names)) != len(model_names):
        raise ValueError("Select at least two distinct models.")
    if len(label_names) < 2:
        raise ValueError("At least two labels are required.")

    text_features = [
        "length", "punctuation", "hashtag", "mention", "negation",
        "oov_rate", "oov_wikigiga",
    ]
    required_predictions = {"model", "seed", "text", "gold", "pred"}
    required_features = {"text", *text_features}
    if not required_predictions.issubset(predictions.columns) or not required_features.issubset(features.columns):
        raise ValueError("Missing prediction or feature columns for BDA.")

    selected = predictions[
        (predictions["seed"] == seed) & predictions["model"].isin(model_names)
    ].copy()
    if set(selected["model"]) != set(model_names):
        raise ValueError("Every BDA model must have predictions for the selected seed.")
    if selected[list(required_predictions)].isna().any().any():
        raise ValueError("BDA predictions contain missing values.")
    if selected.duplicated(["model", "text"]).any():
        raise ValueError("Each model must have exactly one prediction per tweet.")
    for column in ["gold", "pred"]:
        values = selected[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= len(label_names))):
            raise ValueError(f"{column} must contain integer class IDs in range.")
    reference = set(zip(selected.loc[selected["model"] == model_names[0], "text"],
                        selected.loc[selected["model"] == model_names[0], "gold"]))
    for name in model_names[1:]:
        group = selected[selected["model"] == name]
        if set(zip(group["text"], group["gold"])) != reference:
            raise ValueError("BDA models must predict the same tweets and gold labels.")

    data = selected.merge(features[["text", *text_features]], on="text", validate="many_to_one", how="left")
    text_values = data[text_features].to_numpy(dtype=float)
    if not np.isfinite(text_values).all():
        raise ValueError("Every BDA tweet must have finite analysis features.")
    systems = {name: index for index, name in enumerate(model_names)}
    data["system_id"] = data["model"].map(systems).astype(int)
    data["correct"] = (data["pred"] == data["gold"]).astype(int)
    indicators = np.eye(len(label_names), dtype=int)[data["pred"].to_numpy()][:, :-1]
    feature_names = [f"pred:{name}" for name in label_names[:-1]] + text_features
    return data, systems, np.hstack([indicators, text_values]), feature_names


def plot_bda_forests(posterior_data, model_a, model_b):
    """Plot tutorial-style posterior forests and paired feature-weight differences."""
    import arviz as az
    import matplotlib.pyplot as plt

    systems = set(posterior_data.posterior["system"].values.tolist())
    if model_a == model_b or not {model_a, model_b}.issubset(systems):
        raise ValueError("Select two different systems present in the posterior.")
    options = {"point_estimate": "median", "ci_kind": "hdi", "ci_probs": (0.5, 0.95)}
    for variable, title in [
        ("sys_shift", "System shifts (log-odds)"),
        ("accuracy", "Model-based average correctness"),
    ]:
        az.plot_forest(posterior_data, var_names=[variable], combined=True, **options)
        plt.gcf().suptitle(title)
        plt.show()

    plot_data = posterior_data.copy()
    plot_data["posterior"] = (
        posterior_data.posterior["w"]
        .transpose("chain", "draw", "feature", "system")
        .to_dataset()
    )
    az.plot_forest(
        plot_data, var_names=["w"], combined=True, shade_label="feature",
        figure_kwargs={"figsize": (16, 8), "width_ratios": [2, 3], "layout": "none"},
        **options,
    )
    plt.gcf().suptitle("Standardised feature weights (log-odds)")
    plt.gcf().axes[-1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.show()

    weights = posterior_data.posterior["w"]
    difference = weights.sel(system=model_b, drop=True) - weights.sel(system=model_a, drop=True)
    difference_data = posterior_data.copy()
    difference_data["posterior"] = difference.rename("weight_difference").to_dataset()
    az.plot_forest(
        difference_data, var_names=["weight_difference"], combined=True,
        figure_kwargs={"figsize": (12, 6), "width_ratios": [2, 3], "layout": "none"},
        **options,
    )
    plt.gcf().suptitle(f"Feature weights: {model_b} minus {model_a}")
    plt.gcf().axes[-1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.show()


def summarize_oof_features(per_item, num_bootstrap=2000, seed=42):
    """Summarise item-level accuracy differences and bootstrap group contrasts."""
    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be positive.")
    groups = {
        "length": (per_item["length"] > per_item["length"].median(), "above median", "at/below median"),
        "punctuation": (per_item["punctuation"].astype(bool), "yes", "no"),
        "hashtag": (per_item["hashtag"].astype(bool), "yes", "no"),
        "mention": (per_item["mention"].astype(bool), "yes", "no"),
        "negation": (per_item["negation"].astype(bool), "yes", "no"),
        "oov_rate": (per_item["oov_rate"] > 0, "has OOV", "no OOV"),
        "oov_wikigiga": (per_item["oov_wikigiga"] > 0, "has OOV", "no OOV"),
    }
    rng = np.random.default_rng(seed)
    summaries, contrasts = [], []
    for feature, (mask, positive, negative) in groups.items():
        values = {}
        draws = {}
        for label, selection in [(positive, mask), (negative, ~mask)]:
            difference = per_item.loc[selection, "difference"].to_numpy()
            values[label] = difference
            if len(difference) >= 2:
                draws[label] = rng.choice(
                    difference, size=(num_bootstrap, len(difference)), replace=True
                ).mean(axis=1)
                low, high = np.quantile(draws[label], [0.025, 0.975])
            else:
                low, high = np.nan, np.nan
            summaries.append({
                "feature": feature, "group": label, "tweets": len(difference),
                "mean_b_minus_a": difference.mean() if len(difference) else np.nan,
                "lower_95": low, "upper_95": high,
            })
        if positive in draws and negative in draws:
            contrast = draws[positive] - draws[negative]
            low, high = np.quantile(contrast, [0.025, 0.975])
            observed = values[positive].mean() - values[negative].mean()
        else:
            observed, low, high = np.nan, np.nan, np.nan
        contrasts.append({
            "feature": feature, "contrast": f"{positive} minus {negative}",
            "difference_in_gains": observed, "lower_95": low, "upper_95": high,
        })
    return pd.DataFrame(summaries), pd.DataFrame(contrasts)
