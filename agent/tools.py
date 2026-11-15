import pandas as pd


def create_tool_registry(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    # ==========================================
    # PRESERVE ML EXPERIMENT STATE
    # ==========================================

    run_state = {
        "active_run": None,
        "pending_optimization": None
    }

    # ==========================================
    # TOOL 1: DATASET INSPECTION
    # ==========================================

    def inspect_dataset() -> dict:
        """
        Inspect the loaded dataset.

        Returns dataset dimensions, column names,
        data types, missing values and duplicates.

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

    # ==========================================
    # TOOL 2: COLUMN INSPECTION
    # ==========================================

    def inspect_column(column_name: str) -> dict:
        """
        Inspect a specific dataset column.

        Returns data type, column category,
        missing-value count and unique-value count.

        Does not expose actual column values.
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


       # ==========================================
    # TOOL 3: COMPLETE ML ANALYSIS
    # ==========================================

    def run_model_analysis(
        target_column: str,
        problem_type: str
    ) -> dict:

        """
        Execute the complete ML workflow.

        Performs feature-target separation, train-test splitting,
        model selection, preprocessing, training, evaluation,
        ranking, diagnostics and experiment tracking.
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

        # Import experiment tracking
        from core.experiment_tracker import (
            save_experiment_run,
            create_dataset_fingerprint
        )

        # ----------------------------------
        # 1. INPUT VALIDATION
        # ----------------------------------

        problem_type = problem_type.strip().lower()

        if problem_type not in ["classification", "regression"]:
            raise ValueError(
                "Problem type must be classification or regression."
            )

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        if df[target_column].isna().all():
            raise ValueError(
                "Target column contains only missing values."
            )

        if len(df) < 5:
            raise ValueError(
                "Dataset is too small for the current workflow."
            )

        # ----------------------------------
        # 2. FEATURE-TARGET SEPARATION
        # ----------------------------------

        X, y = separate_features_target(
            df,
            target_column
        )

        if X.shape[1] == 0:
            raise ValueError("No feature columns available.")

        # ----------------------------------
        # 3. TRAIN-TEST SPLIT
        # ----------------------------------

        X_train, X_test, y_train, y_test = split_dataset(
            X,
            y
        )

        # ----------------------------------
        # 4. MODEL SELECTION
        # ----------------------------------

        models = get_candidate_models(problem_type)

        # ----------------------------------
        # 5. PREPROCESSING
        # ----------------------------------

        preprocessor = build_preprocessor(X_train)

        # ----------------------------------
        # 6. MODEL TRAINING
        # ----------------------------------

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

        # ----------------------------------
        # 7. MODEL EVALUATION
        # ----------------------------------

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

        # ----------------------------------
        # 8. MODEL RANKING
        # ----------------------------------

        ranked_results, unranked_models, ranking_metric = rank_models(
            evaluation_results,
            problem_type
        )

        # ----------------------------------
        # 9. MODEL DIAGNOSTICS
        # ----------------------------------

        diagnostic_results, diagnostic_failures = diagnose_models(
            trained_models,
            X_train,
            y_train,
            X_test,
            y_test,
            problem_type
        )

        # ----------------------------------
        # 10. EXPERIMENT TRACKING
        # ----------------------------------

        experiment_run_id = None
        tracking_error = None
        dataset_fingerprint = None

        try:

            dataset_fingerprint = create_dataset_fingerprint(df)

            # Assign model rankings
            rank_positions = {
                result["model_name"]: position
                for position, result in enumerate(
                    ranked_results,
                    start=1
                )
            }

            # Prepare records for SQLite
            model_records = []

            for model_name, metrics in evaluation_results.items():

                diagnosis = diagnostic_results.get(
                    model_name, {}
                )

                model_records.append({

                    "model_name": model_name,

                    "metrics": metrics,

                    "rank": rank_positions.get(model_name),

                    "diagnosis": diagnosis.get("diagnosis"),

                    "optimization_status": None

                })

            # Save experiment
            experiment_run_id = save_experiment_run(

                problem_type=problem_type,

                target_column=target_column,

                dataset_rows=len(df),

                feature_count=X.shape[1],

                model_records=model_records,

                dataset_fingerprint=dataset_fingerprint

            )

            print(
                f"\n[TRACKER] Experiment saved: {experiment_run_id}",
                flush=True
            )

        except Exception as error:

            tracking_error = str(error)

            print(
                f"\n[TRACKER] Saving failed: {tracking_error}",
                flush=True
            )

        # ----------------------------------
        # 11. PRESERVE EXPERIMENT STATE
        # ----------------------------------

        run_state["active_run"] = {

            "run_id": experiment_run_id,

            "dataset_fingerprint": dataset_fingerprint,

            "problem_type": problem_type,

            "target_column": target_column,

            "trained_models": trained_models,

            "X_train": X_train,
            "X_test": X_test,

            "y_train": y_train,
            "y_test": y_test,

            "evaluation_results": evaluation_results,

            "diagnostic_results": diagnostic_results,

            "ranked_results": ranked_results

        }

        # Clear previous optimization request
        run_state["pending_optimization"] = None

        # ----------------------------------
        # 12. PREPARE RESULTS FOR GEMINI
        # ----------------------------------

        model_summaries = []

        for model_name, metrics in evaluation_results.items():

            model_summaries.append({

                "model_name": model_name,

                "metrics": metrics,

                "diagnosis": diagnostic_results.get(
                    model_name, {}
                )

            })

        # ----------------------------------
        # 13. RETURN RESULTS
        # ----------------------------------

        return {

            "status": "SUCCESS",

            "experiment_run_id": experiment_run_id,

            "experiment_tracking_error": tracking_error,

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
    # ==========================================
    # TOOL 4: PREPARE OPTIMIZATION
    # ==========================================

    def prepare_optimization(model_name: str) -> dict:
        """
        Prepare an optimization plan for a trained model.

        Does not modify or retrain any model.
        User approval will be required before execution.
        """

        active_run = run_state["active_run"]

        if active_run is None:

            return {
                "status": "FAILED",
                "message": (
                    "Run ML analysis before requesting optimization."
                )
            }
        # Resolve natural-language references to the best model
        if model_name.strip().lower() in [
            "best",
            "best_model",
            "top",
            "top_model"
        ]:

            ranked_results = active_run["ranked_results"]

            if not ranked_results:
                return {
                    "status": "FAILED",
                    "message": "No ranked models are available."
                }

            model_name = ranked_results[0]["model_name"]
        trained_models = active_run["trained_models"]

        if model_name not in trained_models:

            return {
                "status": "FAILED",
                "message": f"Model '{model_name}' was not found."
            }

        # Retrieve the diagnostic result
        diagnosis = active_run["diagnostic_results"].get(
            model_name
        )

        if diagnosis is None:

            return {
                "status": "FAILED",
                "message": "No diagnostic result is available."
            }

        # Generate optimization recommendations
        from core.optimizer import generate_optimization_plan

        plan = generate_optimization_plan({
            model_name: diagnosis
        })[model_name]

        # Do not optimize when the results are inconclusive
        if diagnosis["diagnosis"] == "INCONCLUSIVE":
            plan["requires_approval"] = False

            return {
                "status": "OPTIMIZATION_DEFERRED",

                "message": (
                    "Optimization is deferred because the available "
                    "dataset is insufficient for a reliable diagnosis."
                ),

                "plan": plan
            }

        # Store the pending optimization request
        run_state["pending_optimization"] = {
            "model_name": model_name
        }

        return {

            "status": "AWAITING_USER_APPROVAL",

            "model_name": model_name,

            "recommendations": plan["recommendations"],

            "requires_approval": True,

            "message": (
                "No changes have been made to the model. "
                "Explicit user approval is required."
            )

        }
            # ==================================
            # EXECUTE APPROVED OPTIMIZATION
            # ==================================

    

    def execute_approved_optimization(approved: bool) -> dict:
        """
        Execute optimization only after explicit user approval.
        Save successful optimization results to SQLite.
        """

        if not isinstance(approved, bool):
            raise ValueError("Approval must be True or False.")

        pending = run_state["pending_optimization"]

        if pending is None:
            return {
                "status": "FAILED",
                "message": "No optimization request is awaiting approval."
            }

        # User declined
        if not approved:

            run_state["pending_optimization"] = None

            return {
                "status": "CANCELLED",
                "message": "Optimization cancelled. Original model unchanged."
            }

        active_run = run_state["active_run"]

        if active_run is None:
            return {
                "status": "FAILED",
                "message": "No active ML experiment exists."
            }

        model_name = pending["model_name"]

        print(
            f"\n[OPTIMIZER] Optimizing {model_name}...",
            flush=True
        )

        from core.optimization_runner import optimize_model
        from core.evaluator import evaluate_models

        try:

            # STEP 1: Run hyperparameter optimization
            optimization_result = optimize_model(
                model_name=model_name,
                trained_model=active_run["trained_models"][model_name],
                X_train=active_run["X_train"],
                y_train=active_run["y_train"],
                problem_type=active_run["problem_type"]
            )

            if optimization_result["status"] != "SUCCESS":

                run_state["pending_optimization"] = None

                return {
                    "status": optimization_result["status"],
                    "message": optimization_result["message"]
                }

            # STEP 2: Select the better configuration
            improved = optimization_result["improved"]

            if improved:

                selected_model = optimization_result["best_estimator"]
                selected_version = "optimized"

            else:

                selected_model = active_run["trained_models"][model_name]
                selected_version = "original"

            # STEP 3: Evaluate on untouched test data
            final_results, evaluation_failures = evaluate_models(
                {model_name: selected_model},
                active_run["X_test"],
                active_run["y_test"],
                active_run["problem_type"]
            )

            final_test_metrics = final_results.get(model_name, {})

            # STEP 4: Preserve selected model
            active_run["selected_models"] = active_run.get(
                "selected_models", {}
            )

            active_run["selected_models"][model_name] = selected_model

            # STEP 5: Preserve optimization results in memory
            active_run["optimization_results"] = active_run.get(
                "optimization_results", {}
            )

            active_run["optimization_results"][model_name] = {
                "baseline_cv_score": optimization_result["baseline_cv_score"],
                "optimized_cv_score": optimization_result["optimized_cv_score"],
                "improvement": optimization_result["improvement"],
                "improved": improved,
                "selected_version": selected_version,
                "best_params": optimization_result["best_params"]
            }

            # STEP 6: Save optimization history to SQLite
            optimization_tracking_error = None

            run_id = active_run.get("run_id")

            if run_id:

                try:

                    from core.optimization_history import (
                        save_optimization_result
                    )

                    save_optimization_result(
                        run_id=run_id,
                        model_name=model_name,
                        result={
                            "status": "SUCCESS",
                            "baseline_cv_score": optimization_result["baseline_cv_score"],
                            "optimized_cv_score": optimization_result["optimized_cv_score"],
                            "improvement": optimization_result["improvement"],
                            "improved": improved,
                            "selected_version": selected_version,
                            "best_params": optimization_result["best_params"]
                        },
                        final_test_metrics=final_test_metrics
                    )

                    print(
                        "[TRACKER] Optimization history saved.",
                        flush=True
                    )

                except Exception as error:

                    optimization_tracking_error = str(error)

                    print(
                        f"[TRACKER] History saving failed: {error}",
                        flush=True
                    )

            else:

                optimization_tracking_error = (
                    "No experiment ID available for tracking."
                )

            # STEP 7: Clear pending approval
            run_state["pending_optimization"] = None

            return {
                "status": "SUCCESS",
                "model_name": model_name,
                "baseline_cv_score": optimization_result["baseline_cv_score"],
                "optimized_cv_score": optimization_result["optimized_cv_score"],
                "improvement": optimization_result["improvement"],
                "improved": improved,
                "selected_version": selected_version,
                "best_params": optimization_result["best_params"],
                "test_metrics": final_test_metrics,
                "evaluation_failures": evaluation_failures,
                "optimization_tracking_error": optimization_tracking_error
            }

        except Exception as error:

            run_state["pending_optimization"] = None

            return {
                "status": "FAILED",
                "message": str(error)
            }
    # ==========================================
    # TOOL REGISTRY
    # ==========================================

    return {
    "inspect_dataset": inspect_dataset,
    "inspect_column": inspect_column,
    "run_model_analysis": run_model_analysis,
    "prepare_optimization": prepare_optimization,

    # Internal functions and state
    "execute_approved_optimization": execute_approved_optimization,
    "_state": run_state
}
