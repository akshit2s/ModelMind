from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import json
import uuid
import math
import hashlib
import pandas as pd


# Local database location
DB_PATH = Path("experiments") / "modelmind.db"

def create_dataset_fingerprint(df):

    # Include column names and data types in the fingerprint
    schema = "|".join(
        f"{column}:{dtype}"
        for column, dtype in zip(df.columns, df.dtypes)
    )

    # Generate hashes for dataset rows
    row_hashes = pd.util.hash_pandas_object(
        df,
        index=True
    ).values.tobytes()

    # Combine schema and row hashes
    fingerprint = hashlib.sha256(
        schema.encode() + row_hashes
    ).hexdigest()

    return fingerprint
def initialize_experiment_store():

    # Create the experiments directory if it doesn't exist
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS model_experiments (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,

                problem_type TEXT NOT NULL,
                target_column TEXT NOT NULL,

                dataset_rows INTEGER NOT NULL,
                feature_count INTEGER NOT NULL,

                dataset_fingerprint TEXT,

                model_name TEXT NOT NULL,

                metrics_json TEXT NOT NULL,

                model_rank INTEGER,

                diagnosis TEXT,


                optimization_status TEXT

            )
        """)
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(model_experiments)"
            ).fetchall()
        }

        if "dataset_fingerprint" not in columns:

            connection.execute("""
                ALTER TABLE model_experiments
                ADD COLUMN dataset_fingerprint TEXT
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_id
            ON model_experiments(run_id)
        """)

    return DB_PATH


def make_json_safe(value):

    # Convert NumPy scalar values into native Python values
    if hasattr(value, "item"):
        return make_json_safe(value.item())

    # Handle dictionaries
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    # Replace NaN and infinity with None
    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def save_experiment_run(
    problem_type,
    target_column,
    dataset_rows,
    feature_count,
    model_records,
    dataset_fingerprint
):

    if not model_records:
        raise ValueError("No model results available to save.")

    initialize_experiment_store()

    # Generate a unique ID for this experiment run
    run_id = uuid.uuid4().hex[:12]

    created_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    records_to_insert = []

    for record in model_records:

        metrics_json = json.dumps(
            make_json_safe(record.get("metrics", {})),
            allow_nan=False
        )

    records_to_insert.append((

        run_id,
        created_at,

        problem_type,
        target_column,

        dataset_rows,
        feature_count,
        dataset_fingerprint,

        record["model_name"],
        metrics_json,

        record.get("rank"),
        record.get("diagnosis"),
        record.get("optimization_status")

    ))

    with sqlite3.connect(DB_PATH) as connection:

        connection.executemany("""

           INSERT INTO model_experiments (

    run_id,
    created_at,

    problem_type,
    target_column,

    dataset_rows,
    feature_count,
    dataset_fingerprint,

    model_name,
    metrics_json,

    model_rank,
    diagnosis,
    optimization_status

)

VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            

        """, records_to_insert)

    return run_id


def get_recent_experiments(limit=10):

    if limit <= 0:
        raise ValueError("Limit must be greater than zero.")

    initialize_experiment_store()

    with sqlite3.connect(DB_PATH) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute("""

            SELECT *

            FROM model_experiments

            ORDER BY id DESC

            LIMIT ?

        """, (limit,)).fetchall()

    experiments = []

    for row in rows:

        experiment = dict(row)

        experiment["metrics"] = json.loads(
            experiment.pop("metrics_json")
        )

        experiments.append(experiment)

    return experiments