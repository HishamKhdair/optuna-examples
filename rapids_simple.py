"""
Optuna example using RAPIDS library for optimization.

In this example, we perform hyperparameter optimization on Iris dataset using cuML's
RandomForestClassifier. This should be used as a starting point
to extend the search for larger problems, and wider depths.

To run this example:

    $ python rapids_simple.py

Learn more about rapids: https://rapids.ai/
"""
import cudf
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
from cuml.preprocessing.model_selection import train_test_split
import optuna

from sklearn.datasets import load_iris


def train_and_eval(X_param, y_param, max_depth=16, n_estimators=100):
    X_train, X_valid, y_train, y_valid = train_test_split(X_param, y_param, random_state=77)
    classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_valid)
    return accuracy_score(y_valid, y_pred)


def objective(trial, X_param, y_param):
    max_depth = trial.suggest_int("max_depth", 7, 15)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    return train_and_eval(
        X_param, y_param, max_depth=max_depth, n_estimators=n_estimators
    )


if __name__ == "__main__":
    data, target = load_iris(return_X_y=True)
    # To use the GPU model
    X = cudf.DataFrame(data).astype("float32")
    y = cudf.Series(target)

    study = optuna.create_study(study_name="rapids_experiment", direction="maximize")

    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
