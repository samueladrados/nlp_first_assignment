from datasets import load_dataset
import part1
import numpy as np
import copy
import torch
import part2
from sklearn.metrics import f1_score
from itertools import combinations
from pathlib import Path

def load_folds(k = 5, seed = 0):
    dataset = load_dataset("cardiffnlp/tweet_eval", "emotion")
    seen = {}
    unique_indices = []
    for i, example in enumerate(dataset["train"]):
        text, label = example["text"], example["label"]
        if text in seen:
            if seen[text] != label:
                raise ValueError(f"Duplicate training text has conflicting labels at row {i}.")
        else:
            seen[text] = label
            unique_indices.append(i)
    removed = len(dataset["train"]) - len(unique_indices)
    dataset["train"] = dataset["train"].select(unique_indices)
    print(f"Removed {removed} duplicate training examples; {len(dataset['train'])} remain.")
    emotion_outcome_space = part1.clslib.NamedClasses(dataset["train"].features["label"].names)

    train_pairs = list(zip(dataset["train"]["text"], dataset["train"]["label"]))
    folds = part1.experiment.cross_validation_split(train_pairs, k = k, seed = seed)

    return folds, emotion_outcome_space

def smaller_folds(folds, fold_fraction, seed = 0):
    random_generator = np.random.default_rng(seed = seed)
    subset = {}

    for fold, split in folds.items():
        size = int(len(split["training"]) * fold_fraction)
        remaining_data = random_generator.permutation(len(split["training"]))[:size]
        subset[fold] = {"training": [split["training"][position] for position in remaining_data], "validation": split["validation"]}

    return subset

def randomise_embeddings(embedding_model, seed = 0):
    generator = np.random.default_rng(seed = seed)
    random_model = copy.copy(x = embedding_model)
    vectors = generator.normal(loc = 0, scale = embedding_model.vectors[2:].std(), size = embedding_model.vectors.shape).astype(np.float32)
    pad_index = embedding_model.vocab.index(embedding_model.pad_token)
    vectors[pad_index] = 0
    random_model.vectors = vectors

    return random_model

def run_cross_validation(name, folds, emotion_outcome_space, tokeniser, embedding_model, encoder_type, encoder_kwargs, regressor_kwargs, optimiser_kwargs, seed = 0, device = "cuda"):
    torch.manual_seed(seed = seed)
    results = part2.experiment.cross_validation(
        outcome_space = emotion_outcome_space, data_splits = folds, tokenizer = tokeniser, 
        emb_model = embedding_model, enc_type = encoder_type, 
        enc_kwargs = dict(encoder_kwargs), regressor_kwargs = dict(regressor_kwargs), 
        opt_kwargs = optimiser_kwargs, device = device, save_to = f"../part2/results/{name}/seed-{seed}")
    fold_scores = {}
    for fold, split in folds.items():
        labels = [label for _, label in split["validation"]]
        fold_scores[fold] = f1_score(y_true = labels, y_pred = results["predictions"][fold], average = "macro")

    return {"accuracies": fold_scores}

def loss_curves(name, split, emotion_outcome_space, tokeniser, embedding_model, encoder_type, encoder_kwargs, regressor_kwargs, optimiser_kwargs, seed = 0, device = "cuda"):
    torch.manual_seed(seed = seed)
    Path(f"../part2/results/{name}").mkdir(parents = True, exist_ok = True)
    metrics, best_epoch = part2.experiment.train(
        split["training"], split["validation"], tokenizer = tokeniser, emb_model = embedding_model,
        num_classes = len(emotion_outcome_space), enc_type = encoder_type,
        save_to = f"../part2/results/{name}/seed-{seed}", enc_kwargs = dict(encoder_kwargs),
        regressor_kwargs = dict(regressor_kwargs), opt_kwargs = optimiser_kwargs, device = device)

    training_loss = [metric["train_metrics"]["loss"] for metric in metrics]
    validation_loss = [metric["val_metrics"]["loss"] for metric in metrics]

    return training_loss, validation_loss, best_epoch

def get_all_measurements_for(data, sys_name):
    return np.array([list(run_data["accuracies"].values()) for run_data in data[sys_name]])

def compare_all_pairs(classifiers, rope_size=0.005, plot=True, runs=1):
    for name_a, name_b in combinations(classifiers.keys(), 2):
        posterior = part1.analysis.BenavoliCorrelatedT(
            get_all_measurements_for(classifiers, name_a).flatten(),
            get_all_measurements_for(classifiers, name_b).flatten(),
            runs=runs,
            names=[name_a, name_b],
        )
        p_a, p_tie, p_b = posterior.probs(rope_size)
        print(f"A={name_a} B={name_b}: P(A wins)={p_a:.4f} P(Tie)={p_tie:.4f} "
              f"P(B wins)={p_b:.4f} Effect Size={posterior.effect_size:.2f}")
        if plot:
            _ = posterior.plot(rope_size)