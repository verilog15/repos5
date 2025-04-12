"""
Optuna example showcasing the new Optuna Terminator feature.

In this example, we utilize the Optuna Terminator for hyperparameter
optimization on a RandomForestClassifier using the wine dataset.
The Terminator automatically stops the optimization process based
on the potential for further improvement.

To run this example:

    $ python terminator_simple.py

"""

import optuna
from optuna.terminator.callback import TerminatorCallback
from optuna.terminator.erroreval import report_cross_validation_scores

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def objective(trial):
    X, y = load_wine(return_X_y=True)

    clf = RandomForestClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 32),
        min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
        criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
    )

    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
    report_cross_validation_scores(trial, scores)

    return scores.mean()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, callbacks=[TerminatorCallback()])

    print(f"The number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value} (params: {study.best_params})")
