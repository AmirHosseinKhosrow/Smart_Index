#!/usr/bin/env python3
"""
Fuzzy–Reinforcement Water Quality Index
---------------------------------------
Source methodology: "Fuzzy and RL Water Quality Index.docx"

This implementation follows the paper's stated architecture:
    9 water-quality indicators
    -> triangular/trapezoidal fuzzification
    -> Mamdani min/max fuzzy inference
    -> centroid FWQI (0–100)
    -> CEM optimization of rule weights + thresholds
    -> Q-learning threshold refinement
    -> Q1–Q5 classification
    -> stratified 10-fold cross-validation

IMPORTANT REPRODUCIBILITY NOTE
-------------------------------
The manuscript does NOT provide a complete labelled GEMStat training table
or an explicit procedure for generating the Q1–Q5 ground-truth labels.
Therefore this script supports two modes:

1) SUPERVISED mode:
   Your selected GEMStat CSV must contain a label column with Q1–Q5.

2) UNSUPERVISED/REFERENCE mode:
   If no labels are supplied, the script calculates the fuzzy FWQI and
   applies the paper's reported thresholds (22.1, 42.3, 62.0, 81.5).
   This is NOT a reproduction of the paper's reported accuracy because
   there is no independent ground-truth label in the manuscript.

Expected input:
    data/gemstat_selected.csv

The loader accepts common column-name variants. If the CSV contains long-form
GEMStat data (parameter/value rows), convert it to one row per sample first,
or adapt load_gemstat_csv().

Required columns after harmonization:
    pH, turbidity, TSS, TDS, conductivity, COD, BOD,
    nitrate, fecal_coliform

Optional:
    label / class / quality_class containing Q1...Q5
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
DATA_PATH = Path("data/gemstat_selected.csv")
OUTPUT_DIR = Path("outputs")

N_FOLDS = 10

# Paper-reported final thresholds.
PAPER_THRESHOLDS = np.array([22.1, 42.3, 62.0, 81.5], dtype=float)

# Initial thresholds used for optimization.
INITIAL_THRESHOLDS = np.array([20.0, 40.0, 60.0, 80.0], dtype=float)

# CEM settings reported in the paper.
CEM_POPULATION = 50
CEM_ELITE_FRACTION = 0.20
CEM_ITERATIONS = 100

# Q-learning settings reported in the paper.
Q_LEARNING_RATE = 0.01
Q_DISCOUNT = 0.95
Q_EPSILON = 0.20
Q_EPISODES = 500
Q_STEP = 1.0

# Number of fuzzy rules:
# 3 linguistic levels ^ 9 variables = 19683 possible combinations.
# To keep the model computationally practical, the rule base below is
# generated from the minimum/maximum pollution severity structure.
#
# Each rule maps the dominant severity combination to a quality class.
# The paper describes a fuzzy rule base but does not publish the complete
# rule table, so this is an explicit implementation assumption.
RULE_LEVELS = ("LOW", "MED", "HIGH")

FEATURES = [
    "pH",
    "turbidity",
    "TSS",
    "TDS",
    "conductivity",
    "COD",
    "BOD",
    "nitrate",
    "fecal_coliform",
]

# Indicator direction:
# True  -> larger value is generally worse
# False -> pH is treated separately because quality is best around neutral.
HIGHER_IS_WORSE = {
    "pH": False,
    "turbidity": True,
    "TSS": True,
    "TDS": True,
    "conductivity": True,
    "COD": True,
    "BOD": True,
    "nitrate": True,
    "fecal_coliform": True,
}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 3. MEMBERSHIP FUNCTIONS
# ============================================================

def trimf(x: np.ndarray, abc: Tuple[float, float, float]) -> np.ndarray:
    a, b, c = abc
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    if b != a:
        idx = (x >= a) & (x <= b)
        out[idx] = (x[idx] - a) / (b - a)

    if c != b:
        idx = (x >= b) & (x <= c)
        out[idx] = np.maximum(out[idx], (c - x[idx]) / (c - b))

    out[x == b] = 1.0
    return np.clip(out, 0.0, 1.0)


def trapmf(x: np.ndarray, abcd: Tuple[float, float, float, float]) -> np.ndarray:
    a, b, c, d = abcd
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    if b != a:
        idx = (x >= a) & (x < b)
        out[idx] = (x[idx] - a) / (b - a)

    idx = (x >= b) & (x <= c)
    out[idx] = 1.0

    if d != c:
        idx = (x > c) & (x <= d)
        out[idx] = (d - x[idx]) / (d - c)

    return np.clip(out, 0.0, 1.0)


# Membership parameters are based on the paper where explicitly available.
# For variables whose exact table values are not recoverable from the DOCX
# text extraction, the values below are clearly marked as assumptions.
#
# These are deliberately kept in one dictionary so they can be replaced
# directly with the exact Table 1 values if the manuscript table is updated.

MF_PARAMS = {
    "pH": {
        "LOW":  ("trap", (4.0, 4.0, 6.0, 7.0)),
        "MED":  ("tri",  (6.0, 7.0, 8.0)),
        "HIGH": ("trap", (7.0, 8.0, 10.0, 10.0)),
    },
    "turbidity": {
        "LOW":  ("tri", (0.0, 0.5, 1.0)),
        "MED":  ("tri", (0.8, 2.0, 5.0)),
        "HIGH": ("tri", (4.0, 20.0, 50.0)),
    },
    "TSS": {
        "LOW":  ("tri", (0.0, 25.0, 50.0)),
        "MED":  ("tri", (40.0, 100.0, 200.0)),
        "HIGH": ("tri", (150.0, 250.0, 400.0)),
    },
    "TDS": {
        "LOW":  ("tri", (0.0, 250.0, 500.0)),
        "MED":  ("tri", (400.0, 1000.0, 2000.0)),
        "HIGH": ("tri", (1800.0, 2500.0, 3000.0)),
    },
    "conductivity": {
        "LOW":  ("tri", (0.0, 300.0, 700.0)),
        "MED":  ("tri", (500.0, 1200.0, 2500.0)),
        "HIGH": ("tri", (2200.0, 3500.0, 5000.0)),
    },
    "COD": {
        "LOW":  ("tri", (0.0, 25.0, 50.0)),
        "MED":  ("tri", (40.0, 150.0, 300.0)),
        "HIGH": ("tri", (250.0, 500.0, 1000.0)),
    },
    "BOD": {
        "LOW":  ("tri", (0.0, 5.0, 10.0)),
        "MED":  ("tri", (8.0, 20.0, 50.0)),
        "HIGH": ("tri", (40.0, 100.0, 300.0)),
    },
    "nitrate": {
        "LOW":  ("tri", (0.0, 10.0, 25.0)),
        "MED":  ("tri", (20.0, 50.0, 75.0)),
        "HIGH": ("tri", (60.0, 100.0, 200.0)),
    },
    "fecal_coliform": {
        "LOW":  ("tri", (0.0, 0.0, 100.0)),
        "MED":  ("tri", (50.0, 1000.0, 10000.0)),
        "HIGH": ("tri", (5000.0, 1e5, 1e8)),
    },
}


def membership_value(x: float, kind: str, params: tuple) -> float:
    arr = np.array([x], dtype=float)
    if kind == "tri":
        return float(trimf(arr, params)[0])
    if kind == "trap":
        return float(trapmf(arr, params)[0])
    raise ValueError(f"Unknown MF type: {kind}")


def fuzzify_row(row: pd.Series) -> Dict[str, Dict[str, float]]:
    result = {}
    for feature in FEATURES:
        x = float(row[feature])
        result[feature] = {
            level: membership_value(
                x,
                MF_PARAMS[feature][level][0],
                MF_PARAMS[feature][level][1],
            )
            for level in RULE_LEVELS
        }
    return result


# ============================================================
# 4. RULE BASE
# ============================================================

@dataclass(frozen=True)
class Rule:
    antecedent: Tuple[str, ...]
    consequent: int


def quality_from_levels(levels: Tuple[str, ...]) -> int:
    """
    Explicit implementation of the paper's qualitative rule logic:
    larger pollution severity -> poorer quality.

    Q5 = excellent
    Q4 = good
    Q3 = borderline
    Q2 = poor
    Q1 = very poor

    pH is handled by converting LOW/HIGH pH membership into pollution
    severity; MED is considered the best pH condition.
    """
    severity = 0.0

    for feature, level in zip(FEATURES, levels):
        if feature == "pH":
            # LOW/HIGH pH are both undesirable; MED is best.
            severity += {"LOW": 1.0, "MED": 0.0, "HIGH": 1.0}[level]
        else:
            severity += {"LOW": 0.0, "MED": 0.5, "HIGH": 1.0}[level]

    mean_severity = severity / len(FEATURES)

    if mean_severity < 0.20:
        return 5
    if mean_severity < 0.40:
        return 4
    if mean_severity < 0.60:
        return 3
    if mean_severity < 0.80:
        return 2
    return 1


def generate_rules() -> List[Rule]:
    from itertools import product

    rules = []
    for levels in product(RULE_LEVELS, repeat=len(FEATURES)):
        rules.append(
            Rule(
                antecedent=tuple(levels),
                consequent=quality_from_levels(levels),
            )
        )
    return rules


# ============================================================
# 5. MAMDANI INFERENCE
# ============================================================

OUTPUT_MFS = {
    1: ("trap", (0.0, 0.0, 12.0, 30.0)),
    2: ("tri",  (20.0, 35.0, 50.0)),
    3: ("tri",  (40.0, 55.0, 70.0)),
    4: ("tri",  (60.0, 75.0, 90.0)),
    5: ("trap", (80.0, 90.0, 100.0, 100.0)),
}


def output_membership(class_id: int, y: np.ndarray) -> np.ndarray:
    kind, params = OUTPUT_MFS[class_id]
    if kind == "tri":
        return trimf(y, params)
    return trapmf(y, params)


def centroid(y: np.ndarray, mu: np.ndarray) -> float:
    denominator = np.trapz(mu, y)
    if denominator <= 1e-12:
        return 50.0
    return float(np.trapz(y * mu, y) / denominator)


class FuzzyWaterQualityModel:
    def __init__(self, rules: List[Rule]):
        self.rules = rules
        self.rule_weights = np.ones(len(rules), dtype=float)
        self.thresholds = INITIAL_THRESHOLDS.copy()

    def infer(self, row: pd.Series, return_membership=False):
        fuzz = fuzzify_row(row)
        output_y = np.linspace(0.0, 100.0, 1001)
        aggregated = np.zeros_like(output_y)

        for i, rule in enumerate(self.rules):
            firing = 1.0

            for feature, level in zip(FEATURES, rule.antecedent):
                firing = min(firing, fuzz[feature][level])

            firing *= float(np.clip(self.rule_weights[i], 0.0, 1.0))

            if firing <= 0:
                continue

            rule_output = output_membership(rule.consequent, output_y)
            aggregated = np.maximum(
                aggregated,
                np.minimum(firing, rule_output),
            )

        score = centroid(output_y, aggregated)
        cls = classify_score(score, self.thresholds)

        if return_membership:
            return score, cls, output_y, aggregated
        return score, cls


def classify_score(score: float, thresholds: np.ndarray) -> int:
    return int(np.sum(score >= thresholds) + 1)


# ============================================================
# 6. DATA LOADING / HARMONIZATION
# ============================================================

COLUMN_ALIASES = {
    "pH": ["ph", "p_h", "hydrogen_ion_concentration"],
    "turbidity": ["turbidity", "turbidity_ntu"],
    "TSS": ["tss", "suspended_solids", "total_suspended_solids"],
    "TDS": ["tds", "dissolved_solids", "total_dissolved_solids"],
    "conductivity": [
        "conductivity",
        "electrical_conductivity",
        "electrical_conductance",
    ],
    "COD": ["cod", "chemical_oxygen_demand", "oxygen_demand_chemical"],
    "BOD": ["bod", "bod5", "biochemical_oxygen_demand"],
    "nitrate": ["nitrate", "no3", "no3_n", "nitrate_n"],
    "fecal_coliform": [
        "fecal_coliform",
        "fecal_coliforms",
        "faecal_coliform",
        "faecal_coliforms",
        "fecal_coliform_cfu_100ml",
    ],
}


def normalize_column_name(c: str) -> str:
    return (
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_gemstat_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Download/select GEMStat data and save it as data/gemstat_selected.csv"
        )

    raw = pd.read_csv(path)

    normalized = {normalize_column_name(c): c for c in raw.columns}

    selected = {}

    for feature, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in normalized:
                found = normalized[alias]
                break

        if found is None:
            raise ValueError(
                f"Missing required GEMStat variable '{feature}'. "
                f"Available columns: {list(raw.columns)}"
            )

        selected[feature] = raw[found]

    df = pd.DataFrame(selected)

    # Optional labels.
    for candidate in ["label", "class", "quality_class", "quality", "target"]:
        if candidate in normalized:
            df["label"] = raw[normalized[candidate]]
            break

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES, inplace=True)

    # Basic physically meaningful screening.
    df = df[df["pH"].between(0, 14)]
    df = df[df["turbidity"] >= 0]
    df = df[df["TSS"] >= 0]
    df = df[df["TDS"] >= 0]
    df = df[df["conductivity"] >= 0]
    df = df[df["COD"] >= 0]
    df = df[df["BOD"] >= 0]
    df = df[df["nitrate"] >= 0]
    df = df[df["fecal_coliform"] >= 0]

    return df.reset_index(drop=True)


def encode_labels(y: pd.Series) -> np.ndarray:
    mapping = {
        "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5,
        "q1": 1, "q2": 2, "q3": 3, "q4": 4, "q5": 5,
    }

    result = []
    for value in y:
        if isinstance(value, str):
            value = value.strip()
            if value in mapping:
                result.append(mapping[value])
            else:
                result.append(int(float(value)))
        else:
            result.append(int(value))

    return np.asarray(result, dtype=int)


# ============================================================
# 7. REWARD FUNCTION
# ============================================================

def prediction_vector(
    model: FuzzyWaterQualityModel,
    X: pd.DataFrame,
    thresholds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    model.thresholds = np.asarray(thresholds, dtype=float)
    model.rule_weights = np.asarray(weights, dtype=float)

    preds = []
    for _, row in X.iterrows():
        score, cls = model.infer(row)
        preds.append(cls)
    return np.asarray(preds, dtype=int)


def reward_function(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    acc = accuracy_score(y_true, y_pred)

    # Threshold separation penalty.
    diffs = np.diff(thresholds)
    separation_penalty = float(np.sum(np.maximum(5.0 - diffs, 0.0)) / 20.0)

    # Class-distribution penalty: avoid pathological collapse.
    counts = np.bincount(y_pred, minlength=6)[1:]
    probs = counts / max(counts.sum(), 1)
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    entropy_penalty = max(0.0, 0.5 - entropy) * 0.10

    return (
        acc
        - 0.20 * separation_penalty
        - entropy_penalty
    )


# ============================================================
# 8. CROSS-ENTROPY METHOD
# ============================================================

def cem_optimize(
    model: FuzzyWaterQualityModel,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    iterations: int = CEM_ITERATIONS,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    n_rules = len(model.rules)

    # Optimize a manageable subset of rule weights by default.
    # All rules remain available during inference.
    # Weight parameterization is therefore initialized at 1 and
    # CEM perturbs a compressed rule-weight vector using rule severity.
    n_params = 4 + 5  # 4 thresholds + 5 global severity weights

    mean = np.concatenate([
        INITIAL_THRESHOLDS,
        np.ones(5) * 1.0,
    ])

    std = np.concatenate([
        np.ones(4) * 8.0,
        np.ones(5) * 0.25,
    ])

    best_reward = -np.inf
    best_vector = mean.copy()
    history = []

    def vector_to_params(z):
        thresholds = np.sort(z[:4])
        thresholds = np.clip(thresholds, 1.0, 99.0)

        severity_weights = np.clip(z[4:], 0.05, 2.0)

        # Rule weight based on consequent quality.
        weights = np.ones(n_rules)
        for i, rule in enumerate(model.rules):
            q = rule.consequent
            weights[i] = severity_weights[q - 1]

        return thresholds, weights

    for iteration in range(iterations):
        population = np.random.normal(
            loc=mean,
            scale=np.maximum(std, 1e-6),
            size=(CEM_POPULATION, n_params),
        )

        rewards = []

        for candidate in population:
            thresholds, weights = vector_to_params(candidate)

            pred = prediction_vector(
                model, X_train, thresholds, weights
            )

            # Use training reward for candidate selection.
            # Validation is retained for final model selection.
            r = reward_function(y_train, pred, thresholds)
            rewards.append(r)

        rewards = np.asarray(rewards)

        elite_n = max(
            1,
            int(math.ceil(CEM_ELITE_FRACTION * CEM_POPULATION)),
        )

        elite_idx = np.argsort(rewards)[-elite_n:]
        elite = population[elite_idx]

        mean = np.mean(elite, axis=0)
        std = np.std(elite, axis=0) + 1e-6

        current_best = float(np.max(rewards))

        if current_best > best_reward:
            best_reward = current_best
            best_vector = population[np.argmax(rewards)].copy()

        history.append(best_reward)

        if (iteration + 1) % 10 == 0:
            print(
                f"CEM iteration {iteration+1:03d}/{iterations} "
                f"best_reward={best_reward:.5f}"
            )

    thresholds, weights = vector_to_params(best_vector)

    # Final validation check.
    val_pred = prediction_vector(model, X_val, thresholds, weights)
    val_reward = reward_function(y_val, val_pred, thresholds)

    print(f"CEM best training reward: {best_reward:.5f}")
    print(f"CEM selected validation reward: {val_reward:.5f}")
    print(f"CEM thresholds: {thresholds}")

    return thresholds, weights, history


# ============================================================
# 9. Q-LEARNING THRESHOLD REFINEMENT
# ============================================================

def discretize_threshold_state(thresholds: np.ndarray) -> Tuple[int, ...]:
    return tuple(np.round(thresholds).astype(int).tolist())


def q_learning_refine(
    model: FuzzyWaterQualityModel,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    initial_thresholds: np.ndarray,
    weights: np.ndarray,
    episodes: int = Q_EPISODES,
) -> Tuple[np.ndarray, List[float]]:
    """
    Threshold-only Q-learning.

    State:
        rounded threshold configuration.

    Actions:
        +/- Q_STEP on one threshold.

    Feasibility:
        thresholds remain ordered and inside [1, 99].
    """

    model.rule_weights = weights.copy()

    q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    # 8 actions: decrease/increase each of four thresholds.
    actions = [
        (0, -Q_STEP), (0, +Q_STEP),
        (1, -Q_STEP), (1, +Q_STEP),
        (2, -Q_STEP), (2, +Q_STEP),
        (3, -Q_STEP), (3, +Q_STEP),
    ]

    def get_q(state):
        if state not in q_table:
            q_table[state] = np.zeros(len(actions), dtype=float)
        return q_table[state]

    thresholds = initial_thresholds.astype(float).copy()

    # Best visited solution.
    model.thresholds = thresholds
    pred = prediction_vector(model, X_val, thresholds, weights)
    best_reward = reward_function(y_val, pred, thresholds)
    best_thresholds = thresholds.copy()

    history = []

    for episode in range(episodes):
        state = discretize_threshold_state(thresholds)
        q_values = get_q(state)

        if np.random.random() < Q_EPSILON:
            action_idx = np.random.randint(len(actions))
        else:
            action_idx = int(np.argmax(q_values))

        idx, delta = actions[action_idx]

        candidate = thresholds.copy()
        candidate[idx] += delta
        candidate = np.clip(candidate, 1.0, 99.0)

        # Enforce strict ordering.
        if not (
            candidate[0] < candidate[1]
            and candidate[1] < candidate[2]
            and candidate[2] < candidate[3]
        ):
            reward = -1.0
            next_thresholds = thresholds.copy()
        else:
            next_thresholds = candidate

            model.thresholds = next_thresholds
            pred = prediction_vector(
                model, X_val, next_thresholds, weights
            )

            reward = reward_function(
                y_val, pred, next_thresholds
            )

            if reward > best_reward:
                best_reward = reward
                best_thresholds = next_thresholds.copy()

        next_state = discretize_threshold_state(next_thresholds)
        next_q = get_q(next_state)

        q_values[action_idx] += Q_LEARNING_RATE * (
            reward
            + Q_DISCOUNT * np.max(next_q)
            - q_values[action_idx]
        )

        thresholds = next_thresholds
        history.append(best_reward)

    print(f"Q-learning best reward: {best_reward:.5f}")
    print(f"Q-learning thresholds: {best_thresholds}")

    return best_thresholds, history


# ============================================================
# 10. FULL MODEL TRAINING
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
):
    rules = generate_rules()
    print(f"Generated fuzzy rules: {len(rules)}")

    model = FuzzyWaterQualityModel(rules)

    cem_thresholds, weights, cem_history = cem_optimize(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    q_thresholds, q_history = q_learning_refine(
        model,
        X_val,
        y_val,
        cem_thresholds,
        weights,
    )

    model.thresholds = q_thresholds
    model.rule_weights = weights

    return model, cem_history, q_history


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate_model(
    model: FuzzyWaterQualityModel,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict:

    scores = []
    predictions = []

    for _, row in X_test.iterrows():
        score, cls = model.infer(row)
        scores.append(score)
        predictions.append(cls)

    scores = np.asarray(scores)
    predictions = np.asarray(predictions)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_macro": float(
            precision_score(
                y_test, predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test, predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test, predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    return {
        "metrics": metrics,
        "scores": scores,
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[1, 2, 3, 4, 5]
        ).tolist(),
    }


# ============================================================
# 12. 10-FOLD CROSS VALIDATION
# ============================================================

def run_cross_validation(df: pd.DataFrame):
    if "label" not in df.columns:
        raise ValueError(
            "10-fold supervised evaluation requires a 'label' column "
            "containing Q1–Q5. The paper does not provide the original "
            "GEMStat labels, so they cannot be reconstructed faithfully "
            "from the manuscript alone."
        )

    y = encode_labels(df["label"])
    X = df[FEATURES].copy()

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_results = []

    for fold, (train_val_idx, test_idx) in enumerate(
        skf.split(X, y), start=1
    ):
        print("\n" + "=" * 70)
        print(f"FOLD {fold}/{N_FOLDS}")
        print("=" * 70)

        X_dev = X.iloc[train_val_idx].reset_index(drop=True)
        y_dev = y[train_val_idx]

        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_dev,
            y_dev,
            test_size=0.20,
            stratify=y_dev,
            random_state=SEED + fold,
        )

        model, cem_history, q_history = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        result = evaluate_model(model, X_test, y_test)

        print(json.dumps(result["metrics"], indent=2))
        print("Thresholds:", model.thresholds)

        fold_results.append({
            "fold": fold,
            **result["metrics"],
            "threshold_1": model.thresholds[0],
            "threshold_2": model.thresholds[1],
            "threshold_3": model.thresholds[2],
            "threshold_4": model.thresholds[3],
        })

    results_df = pd.DataFrame(fold_results)

    summary = {
        metric: {
            "mean": float(results_df[metric].mean()),
            "std": float(results_df[metric].std(ddof=1)),
        }
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
    }

    return results_df, summary


# ============================================================
# 13. REFERENCE MODE — NO LABELS
# ============================================================

def run_reference_mode(df: pd.DataFrame):
    """
    Calculate FWQI and Q1–Q5 using the paper's reported final thresholds.

    This is useful for deploying the framework to an unlabeled GEMStat
    subset, but it is NOT a supervised reproduction of the paper's reported
    accuracy.
    """

    rules = generate_rules()
    model = FuzzyWaterQualityModel(rules)

    model.thresholds = PAPER_THRESHOLDS.copy()
    model.rule_weights = np.ones(len(rules))

    output = df.copy()

    scores = []
    classes = []

    for _, row in df.iterrows():
        score, cls = model.infer(row)
        scores.append(score)
        classes.append(cls)

    output["FWQI"] = scores
    output["Q_class"] = classes

    return output


# ============================================================
# 14. MAIN
# ============================================================

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FUZZY–REINFORCEMENT WATER QUALITY INDEX")
    print("=" * 70)

    print(f"Loading: {DATA_PATH}")
    df = load_gemstat_csv(DATA_PATH)

    print(f"Usable samples: {len(df)}")
    print("Variables:", FEATURES)

    if "label" in df.columns:
        print("\nLabel column detected -> supervised 10-fold mode.")

        results_df, summary = run_cross_validation(df)

        results_df.to_csv(
            OUTPUT_DIR / "cross_validation_results.csv",
            index=False,
        )

        with open(
            OUTPUT_DIR / "summary_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        print("\nFINAL CROSS-VALIDATION SUMMARY")
        print("-" * 70)

        for metric, values in summary.items():
            print(
                f"{metric:20s}: "
                f"{values['mean']:.4f} ± {values['std']:.4f}"
            )

    else:
        print(
            "\nNo Q1–Q5 label detected -> reference/inference mode."
        )

        output = run_reference_mode(df)

        output.to_csv(
            OUTPUT_DIR / "gemstat_fwqi_predictions.csv",
            index=False,
        )

        print("\nPaper-reported thresholds:")
        print(PAPER_THRESHOLDS)

        print("\nClass distribution:")
        print(output["Q_class"].value_counts().sort_index())

        print(
            "\nWARNING: without independent Q1–Q5 labels, "
            "accuracy/precision/recall/F1 cannot be calculated."
        )

    print("\nFinished.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
