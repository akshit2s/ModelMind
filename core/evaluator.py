from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_models(trained_models, X_test, y_test, problem_type):

    if len(X_test) == 0:
        raise ValueError("Testing dataset is empty.")

    if len(X_test) != len(y_test):
        raise ValueError("Testing features and target must have the same number of records.")

    evaluation_results = {}
    evaluation_failures = {}

    for model_name, model in trained_models.items():

        try:
            # Generate predictions
            predictions = model.predict(X_test)

            # Classification evaluation
            if problem_type == "classification":

                metrics = {
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "recall": recall_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "f1_score": f1_score(
                        y_test, predictions,
                        average="weighted",
                        zero_division=0
                    )
                }

            # Regression evaluation
            elif problem_type == "regression":

                metrics = {
                    "mae": mean_absolute_error(y_test, predictions),
                    "rmse": mean_squared_error(
                        y_test, predictions
                    ) ** 0.5,
                    "r2_score": (
                        r2_score(y_test, predictions)
                        if len(y_test) >= 2 else None
                    )
                }

            else:
                raise ValueError(
                    "Problem type must be classification or regression."
                )

            evaluation_results[model_name] = metrics

        except Exception as error:
            evaluation_failures[model_name] = str(error)

    return evaluation_results, evaluation_failures