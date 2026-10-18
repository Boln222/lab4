"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

MODEL_NAME = "Soft Voting: LogisticRegression + RandomForest + GradientBoosting"


def build_model(seed: int = 42) -> VotingClassifier:
    """Выбранная модель: мягкое голосование трёх разных моделей.

    Каждая модель выдаёт вероятность класса 1, ансамбль берёт их среднее.
    Модели ошибаются по-разному, поэтому среднее получается точнее каждой по отдельности.
    """
    # Линейная модель: ей нужно масштабирование признаков
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # Случайный лес: 300 деревьев, в каждом листе минимум 5 объектов,
    # чтобы деревья не запоминали отдельных клиентов
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed)

    # Градиентный бустинг: много маленьких деревьев (глубина 2) с маленьким шагом,
    # каждое дерево учится на 80% строк (subsample) - это тоже защищает от переобучения
    boosting = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=seed
    )

    return VotingClassifier(
        estimators=[("logistic", logistic), ("forest", forest), ("boosting", boosting)],
        voting="soft",
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    model = build_model(data.random_seed)
    model.fit(data.train_features, data.train_target)  # учимся только на train

    probabilities = model.predict_proba(data.test_features)[:, 1]  # вероятность класса 1
    predictions = (probabilities >= 0.5).astype(int)

    # test_target используется только здесь - чтобы посчитать итоговые метрики
    test_target = np.asarray(data.test_target)
    return EnsembleResult(
        model_name=MODEL_NAME,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(test_target, probabilities)),
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions)),
        recall=float(recall_score(test_target, predictions)),
        f1=float(f1_score(test_target, predictions)),
    )
