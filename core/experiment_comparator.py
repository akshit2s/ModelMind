import json
import math
import sqlite3

from core.experiment_tracker import (
    DB_PATH,
    initialize_experiment_store
)


def compare_experiment_runs(
    problem_type,
    target_column,
    dataset_fingerprint,
    primary_metric=None
):

    # Choose the default comparison metric
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

        lower_is_better = True

    else:
        raise ValueError("Unsupported problem type.")

    if primary_metric is None:
        primary_metric = default_metric

    if primary_metric not in supported_metrics:
        raise ValueError("Unsupported comparison metric.")

    if primary_metric == "r2_score":
        lower_is_better = False

    initialize_experiment_store()

    # Retrieve matching experiment records
    with sqlite3.connect(DB_PATH) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute("""

            SELECT
                run_id,
                created_at,
                model_name,
                model_rank,
                metrics_json

            FROM model_experiments

            WHERE problem_type = ?
              AND target_column = ?
              AND dataset_fingerprint = ?

            ORDER BY created_at DESC

        """, (
            problem_type,
            target_column,
            dataset_fingerprint
        )).fetchall()

    # Group models by experiment run
    grouped_runs = {}

    for row in rows:

        run_id = row["run_id"]

        grouped_runs.setdefault(run_id, []).append(dict(row))

    comparison_results = []

    # Extract the top-ranked model from each run
    for run_id, models in grouped_runs.items():

        models.sort(
            key=lambda model: (
                model["model_rank"]
                if model["model_rank"] is not None
                else float("inf")
            )
        )

        for model in models:

            metrics = json.loads(model["metrics_json"])

            metric_value = metrics.get(primary_metric)

            if (
                metric_value is None
                or isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(metric_value)
            ):
                continue

            comparison_results.append({

                "run_id": run_id,
                "created_at": model["created_at"],
                "model_name": model["model_name"],
                "metric_name": primary_metric,
                "metric_value": metric_value

            })

            break

    # Rank experiment runs
    comparison_results.sort(
        key=lambda result: result["metric_value"],
        reverse=not lower_is_better
    )

    return comparison_results