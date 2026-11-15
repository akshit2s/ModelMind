import pandas as pd


def create_tool_registry(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    # ==================================
    # TOOL 1: DATASET INSPECTION
    # ==================================

    def inspect_dataset() -> dict:
        """
        Inspect the loaded dataset.

        Returns dataset dimensions, column names, data types,
        missing values, unique values and duplicate rows.

        Does not expose actual dataset records.
        """

        columns = []

        for column in df.columns:

            columns.append({
                "name": str(column),
                "data_type": str(df[column].dtype),
                "missing_values": int(df[column].isna().sum()),
                "unique_values": int(df[column].nunique())
            })

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_information": columns,
            "duplicate_rows": int(df.duplicated().sum())
        }

    # ==================================
    # TOOL 2: COLUMN INSPECTION
    # ==================================

    def inspect_column(column_name: str) -> dict:
        """
        Inspect a particular dataset column.

        Args:
            column_name: Name of the column.

        Returns data type, column category, missing values
        and unique-value count.
        """

        if column_name not in df.columns:

            return {
                "error": f"Column '{column_name}' does not exist."
            }

        column = df[column_name]

        if pd.api.types.is_bool_dtype(column):
            column_type = "boolean"

        elif pd.api.types.is_numeric_dtype(column):
            column_type = "numerical"

        else:
            column_type = "categorical"

        return {
            "column": column_name,
            "data_type": str(column.dtype),
            "column_type": column_type,
            "missing_values": int(column.isna().sum()),
            "unique_values": int(column.nunique()),
            "total_records": int(len(column))
        }

    # ==================================
    # TOOL 3: COMPLETE ML ANALYSIS
    # ==================================

    def run_model_analysis(
        target_column: str,
        problem_type: str
    ) -> dict:
        """
        Run the complete ML workflow.

        Args:
            target_column: Column the user wants to predict.

            problem_type: Either classification or regression.
            Must be based on the user's actual prediction objective.

        Performs feature-target separation, train-test splitting,
        candidate model selection, preprocessing, training,
        evaluation, ranking and diagnostics.

        Returns actual model results and recommendations.
        """

        # Import existing ML components
        from core.data_preparer import separate_features_target
        from core.data_splitter import split_dataset
        from core.model_selector import get_candidate_models
        from core.preprocessor import build_preprocessor
        from core.trainer import train_models
        from core.evaluator import evaluate_models
        from core.model_comparator import rank_models
        from core.diagnostics import diagnose_models

        # Validate user input
        problem_type = problem_type.strip().lower()

        if problem_type not in ["classification", "regression"]:

            raise ValueError(
                "Problem type must be classification or regression."
            )

        if target_column not in df.columns:

            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        if len(df) < 5:

            raise ValueError(
                "Dataset is too small for the current train-test workflow."
            )

        if df[target_column].isna().all():

            raise ValueError(
                "Target column contains only missing values."
            )

        # STEP 1: Separate features and target
        X, y = separate_features_target(df, target_column)

        if X.shape[1] == 0:
            raise ValueError("No feature columns available.")

        # STEP 2: Split dataset
        X_train, X_test, y_train, y_test = split_dataset(X, y)

        # STEP 3: Select candidate models
        models = get_candidate_models(problem_type)

        # STEP 4: Create preprocessing pipeline
        preprocessor = build_preprocessor(X_train)

        # STEP 5: Train models
        trained_models, training_failures = train_models(
            models,
            preprocessor,
            X_train,
            y_train
        )

        if not trained_models:

            return {
                "status": "FAILED",
                "message": "All candidate models failed to train.",
                "training_failures": training_failures
            }

        # STEP 6: Evaluate models
        evaluation_results, evaluation_failures = evaluate_models(
            trained_models,
            X_test,
            y_test,
            problem_type
        )

        if not evaluation_results:

            return {
                "status": "FAILED",
                "message": "No models could be evaluated.",
                "training_failures": training_failures,
                "evaluation_failures": evaluation_failures
            }

        # STEP 7: Rank models
        ranked_results, unranked_models, ranking_metric = rank_models(
            evaluation_results,
            problem_type
        )

        # STEP 8: Diagnose models
        diagnostic_results, diagnostic_failures = diagnose_models(
            trained_models,
            X_train,
            y_train,
            X_test,
            y_test,
            problem_type
        )

        # STEP 9: Prepare results for Gemini
        model_summaries = []

        for model_name, metrics in evaluation_results.items():

            model_summaries.append({

                "model_name": model_name,

                "metrics": metrics,

                "diagnosis": diagnostic_results.get(
                    model_name, {}
                )

            })

        return {

            "status": "SUCCESS",

            "problem_type": problem_type,

            "target_column": target_column,

            "dataset_rows": len(df),

            "feature_count": X.shape[1],

            "training_records": len(X_train),

            "testing_records": len(X_test),

            "ranking_metric": ranking_metric,

            "ranked_models": ranked_results,

            "model_summaries": model_summaries,

            "unranked_models": unranked_models,

            "training_failures": training_failures,

            "evaluation_failures": evaluation_failures,

            "diagnostic_failures": diagnostic_failures

        }

    # ==================================
    # TOOL REGISTRY
    # ==================================

    return {

        "inspect_dataset": inspect_dataset,

        "inspect_column": inspect_column,

        "run_model_analysis": run_model_analysis

    }