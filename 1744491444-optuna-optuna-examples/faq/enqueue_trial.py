"""
Optuna enqueue_trial example that optimizes a classifier configuration using sklearn.

In this example, we optimize a classifier configuration for Iris dataset. We start a study with
given parameter values, such as a default and a manually suggested one.

"""

import optuna

import sklearn.datasets
import sklearn.model_selection
import sklearn.svm


def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")

    # We enqueue a default parameter and a manually suggested parameter.
    study.enqueue_trial({"svc_c": 1})
    study.enqueue_trial({"svc_c": 10})

    study.optimize(objective, n_trials=100)

    print("C={}, Value={}".format(study.trials[0].params["svc_c"], study.trials[0].value))
    print("C={}, Value={}".format(study.trials[1].params["svc_c"], study.trials[1].value))
    print("C={}, Value={}".format(study.best_trial.params["svc_c"], study.best_trial.value))
