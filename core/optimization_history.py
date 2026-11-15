import json
import sqlite3

from datetime import datetime, timezone

from core.experiment_tracker import (
    DB_PATH,
    initialize_experiment_store,
    make_json_safe
)


def initialize_optimization_history():

    initialize_experiment_store()

    with sqlite3.connect(DB_PATH) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS optimization_history (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                run_id TEXT NOT NULL,

                model_name TEXT NOT NULL,

                created_at TEXT NOT NULL,

                status TEXT NOT NULL,

                baseline_cv_score REAL,

                optimized_cv_score REAL,

                improvement REAL,

                improved INTEGER,

                selected_version TEXT,

                best_params_json TEXT,

                final_test_metrics_json TEXT

            )
        """)


def save_optimization_result(
    run_id,
    model_name,
    result,
    final_test_metrics=None
):

    if not run_id:
        raise ValueError("Experiment run ID is required.")

    initialize_optimization_history()

    # Convert complex values into JSON-safe objects
    safe_params = make_json_safe(
        result.get("best_params", {})
    )

    safe_metrics = make_json_safe(
        final_test_metrics or {}
    )

    with sqlite3.connect(DB_PATH) as connection:

        connection.execute("""

            INSERT INTO optimization_history (

                run_id,
                model_name,
                created_at,
                status,
                baseline_cv_score,
                optimized_cv_score,
                improvement,
                improved,
                selected_version,
                best_params_json,
                final_test_metrics_json

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            run_id,

            model_name,

            datetime.now(timezone.utc).isoformat(),

            result.get("status"),

            result.get("baseline_cv_score"),

            result.get("optimized_cv_score"),

            result.get("improvement"),

            (
                int(result["improved"])
                if result.get("improved") is not None
                else None
            ),

            result.get("selected_version"),

            json.dumps(safe_params, allow_nan=False),

            json.dumps(safe_metrics, allow_nan=False)

        ))


def get_optimization_history(run_id=None):

    initialize_optimization_history()

    with sqlite3.connect(DB_PATH) as connection:

        connection.row_factory = sqlite3.Row

        if run_id:

            rows = connection.execute("""
                SELECT *
                FROM optimization_history
                WHERE run_id = ?
                ORDER BY id DESC
            """, (run_id,)).fetchall()

        else:

            rows = connection.execute("""
                SELECT *
                FROM optimization_history
                ORDER BY id DESC
            """).fetchall()

    results = []

    for row in rows:

        record = dict(row)

        record["best_params"] = json.loads(
            record.pop("best_params_json") or "{}"
        )

        record["final_test_metrics"] = json.loads(
            record.pop("final_test_metrics_json") or "{}"
        )

        results.append(record)

    return results
