"""Все эксперименты лабораторной №4.

Запуск из папки с лабораторной:  python run_experiments.py   (около минуты на CPU)
Печатает таблицу экспериментов и сохраняет experiment_results.json.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_models import MODEL_NAME, build_model, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

FOLDER = Path(__file__).parent
SEED = 42

# Признаки, где число - это код категории, а не количество (например, Purpose: 0 - машина, 3 - телевизор)
CATEGORICAL = [
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose", "Value_Savings_Stocks",
    "Length_of_current_employment", "Sex_Marital_Status", "Guarantors",
    "Most_valuable_available_asset", "Concurrent_Credits", "Type_of_apartment", "Occupation",
]


def load_data():
    df = pd.read_csv(FOLDER / "german.csv", sep=";")
    X = df.drop(columns="Creditability")
    y = df["Creditability"]
    # Тот же split, что в лабораторной №3: 80/20, доля классов 70/30 сохраняется
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def candidate_models():
    """Модели, которые сравниваем. Ключ - название для таблицы."""
    return {
        "LogisticRegression (база из лаб. 3)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest, по умолчанию": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "RandomForest, min_samples_leaf=5": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEED
        ),
        "ExtraTrees, min_samples_leaf=5": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEED
        ),
        "GradientBoosting, по умолчанию": GradientBoostingClassifier(random_state=SEED),
        "GradientBoosting, медленный": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=SEED
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=SEED),
        "MLP, 1 слой из 16 нейронов": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000, random_state=SEED),
        ),
        MODEL_NAME: build_model(SEED),
    }


def test_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
    }


def evaluate(name, model, X_train, X_test, y_train, y_test, cv):
    # Критерий выбора: средний ROC-AUC на 5-кратной кросс-валидации ТОЛЬКО по train
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    # Метрики на test считаем для отчёта, по ним ничего не выбираем
    model.fit(X_train, y_train)
    row = {"model": name, "cv_roc_auc_mean": cv_scores.mean(), "cv_roc_auc_std": cv_scores.std()}
    row.update(test_metrics(y_test, model.predict_proba(X_test)[:, 1]))
    print(
        f"{name:<68} CV {row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f} | "
        f"test ROC-AUC {row['roc_auc']:.4f}  Acc {row['accuracy']:.4f}  "
        f"P {row['precision']:.4f}  R {row['recall']:.4f}  F1 {row['f1']:.4f}"
    )
    return row


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"train: {len(y_train)} строк, test: {len(y_test)} строк, доля класса 1 в train: {y_train.mean():.2f}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\n=== Эксперимент 1. Сравнение моделей ===")
    rows = []
    for name, model in candidate_models().items():
        rows.append(evaluate(name, model, X_train, X_test, y_train, y_test, cv))

    print("\n=== Эксперимент 2. One-hot кодирование категорий для логистической регрессии ===")
    X_train_oh = pd.get_dummies(X_train, columns=CATEGORICAL)
    X_test_oh = pd.get_dummies(X_test, columns=CATEGORICAL).reindex(columns=X_train_oh.columns, fill_value=0)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rows.append(evaluate("LogisticRegression + one-hot", lr, X_train_oh, X_test_oh, y_train, y_test, cv))

    best = max(rows, key=lambda r: r["cv_roc_auc_mean"])
    print(f"\nЛучшая по CV ROC-AUC: {best['model']} ({best['cv_roc_auc_mean']:.4f})")

    print("\n=== Эксперимент 3. Устойчивость выбранной модели к random seed (только CV) ===")
    seed_scores = {}
    for seed in [0, 7, 42, 123]:
        score = cross_val_score(build_model(seed), X_train, y_train, cv=cv, scoring="roc_auc").mean()
        seed_scores[seed] = score
        print(f"  seed={seed:<4} CV ROC-AUC = {score:.4f}")

    print("\n=== Итог: контрактная функция run_ensemble_experiment ===")
    result = run_ensemble_experiment(EnsembleInput(X_train, y_train, X_test, y_test, random_seed=SEED))
    print(
        f"{result.model_name}\nROC-AUC {result.roc_auc:.4f}  Accuracy {result.accuracy:.4f}  "
        f"Precision {result.precision:.4f}  Recall {result.recall:.4f}  F1 {result.f1:.4f}"
    )

    results = {
        "model_name": result.model_name,
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "selection_criterion": "mean ROC-AUC on 5-fold stratified CV on train",
        "cv_roc_auc_mean": best["cv_roc_auc_mean"],
        "random_seed": SEED,
        "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
        "cv_roc_auc_by_seed": {str(seed): score for seed, score in seed_scores.items()},
        "experiments": [{key: (round(value, 4) if isinstance(value, float) else value)
                         for key, value in row.items()} for row in rows],
    }
    with open(FOLDER / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nРезультаты сохранены в experiment_results.json")


if __name__ == "__main__":
    main()
