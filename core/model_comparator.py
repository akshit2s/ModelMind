import math


def rank_models(evaluation_results, problem_type, primary_metric=None):

    # Define default metrics and ranking directions
    if problem_type == "classification":

        default_metric = "f1_score"

        supported_metrics = {
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        }

        lower_is_better = False

    elif problem_type == "regression":

        default_metric = "rmse"

        supported_metrics = {
            "mae",
            "rmse",
            "r2_score"
        }

        lower_is_better = None

    else:
        raise ValueError("Unsupported problem type.")

    # Use the default metric if none is provided
    if primary_metric is None:
        primary_metric = default_metric

    if primary_metric not in supported_metrics:
        raise ValueError(
            f"Unsupported metric '{primary_metric}' for {problem_type}."
        )

    # Determine whether a lower value is better
    if problem_type == "regression":
        lower_is_better = primary_metric != "r2_score"

    ranked_results = []
    unranked_models = []

    # Extract the selected metric from each model
    for model_name, metrics in evaluation_results.items():

        metric_value = metrics.get(primary_metric)

        if (
            metric_value is None
            or isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(metric_value)
        ):
            unranked_models.append(model_name)
            continue

        ranked_results.append({
            "model_name": model_name,
            "metric_value": metric_value,
            "metrics": metrics
        })

    # Sort models according to the selected metric
    ranked_results.sort(
        key=lambda result: result["metric_value"],
        reverse=not lower_is_better
    )

    return ranked_results, unranked_models, primary_metric