# ==========================================
# IMPORTS
# ==========================================

from core.dataset_loader import load_dataset
from core.data_analyzer import analyze_dataset
from core.problem_detector import detect_problem_type
from core.data_preparer import separate_features_target
from core.data_splitter import split_dataset
from core.model_selector import get_candidate_models
from core.preprocessor import build_preprocessor
from core.trainer import train_models
from core.evaluator import evaluate_models
from core.model_comparator import rank_models
from core.diagnostics import diagnose_models
from core.optimizer import generate_optimization_plan
from core.optimization_runner import optimize_model


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n======================================")
    print("              MODELMIND")
    print("       ML ANALYSIS & ADVISOR")
    print("======================================")

    file_path = input("\nEnter dataset path: ")

    try:

        # ----------------------------------
        # 1. LOAD DATASET
        # ----------------------------------

        df = load_dataset(file_path)

        print("\n[+] Dataset loaded successfully!")

        # ----------------------------------
        # 2. DATASET ANALYSIS
        # ----------------------------------

        analysis = analyze_dataset(df)

        print("\n========== DATASET OVERVIEW ==========")

        print(f"Rows: {analysis['rows']}")
        print(f"Columns: {analysis['columns']}")
        print(f"Duplicate Rows: {analysis['duplicate_rows']}")

        print("\n========== COLUMN INFORMATION ==========")

        for column, dtype in analysis["data_types"].items():
            print(f"{column}: {dtype}")

        print("\n========== MISSING VALUES ==========")

        for column, count in analysis["missing_values"].items():

            percentage = analysis["missing_percentage"][column]

            print(f"{column}: {count} ({percentage}%)")

        print("\n========== UNIQUE VALUES ==========")

        for column, count in analysis["unique_values"].items():
            print(f"{column}: {count}")

        print("\n========== STATISTICAL SUMMARY ==========")

        print(analysis["statistical_summary"])

        print("\n========== NUMERICAL CORRELATION ==========")

        print(analysis["numerical_correlation"])

        # ----------------------------------
        # 3. PROBLEM TYPE DETECTION
        # ----------------------------------

        target_column = input("\nEnter target column: ")

        # Get the automatic suggestion
        suggested_type = detect_problem_type(df, target_column)

        print(f"\nSuggested problem type: {suggested_type.upper()}")

        # Ask the user to confirm the actual prediction objective
        problem_type = input(
            "\nChoose problem type (classification/regression): "
        ).strip().lower()

        if problem_type not in ["classification", "regression"]:
            raise ValueError("Please enter classification or regression.")

        print(f"\nSelected problem: {problem_type.upper()}")

        # ----------------------------------
        # 4. FEATURE-TARGET SEPARATION
        # ----------------------------------

        X, y = separate_features_target(df, target_column)

        print("\n========== FEATURE-TARGET SEPARATION ==========")

        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # ----------------------------------
        # 5. TRAIN-TEST SPLIT
        # ----------------------------------

        X_train, X_test, y_train, y_test = split_dataset(X, y)

        print("\n========== TRAIN-TEST SPLIT ==========")

        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

        # ----------------------------------
        # 6. MODEL SELECTION
        # ----------------------------------

        models = get_candidate_models(problem_type)

        print("\n========== CANDIDATE MODELS ==========")

        for model_name in models:
            print(model_name)

        # ----------------------------------
        # 7. PREPROCESSING
        # ----------------------------------

        preprocessor = build_preprocessor(X_train)

        print("\n[+] Preprocessing pipeline created.")

        # ----------------------------------
        # 8. MODEL TRAINING
        # ----------------------------------

        trained_models, failed_models = train_models(
            models,
            preprocessor,
            X_train,
            y_train
        )

        print("\n========== MODEL TRAINING RESULTS ==========")

        for model_name in trained_models:
            print(f"[SUCCESS] {model_name} trained successfully.")

        for model_name, error in failed_models.items():
            print(f"\n[FAILED] {model_name}")
            print(f"Reason: {error}")

        print("\n======================================")
        print("       MODEL TRAINING COMPLETED")
        print("======================================")

        print("\n========== EVALUATION DEBUG ==========")

        print("Actual test values:", y_test.tolist())

        for model_name, model in trained_models.items():

            predictions = model.predict(X_test)

            print(f"{model_name}: {predictions.tolist()}")
        # ----------------------------------
        # 9. MODEL EVALUATION
        # ----------------------------------

        evaluation_results, evaluation_failures = evaluate_models(
            trained_models,
            X_test,
            y_test,
            problem_type
        )

        print("\n========== MODEL EVALUATION RESULTS ==========")

        for model_name, metrics in evaluation_results.items():

            print(f"\nModel: {model_name}")

            for metric_name, value in metrics.items():

                if value is None:
                    print(f"{metric_name}: N/A (insufficient test records)")

                else:
                    print(f"{metric_name}: {value:.4f}")

        for model_name, error in evaluation_failures.items():

            print(f"\n[EVALUATION FAILED] {model_name}")
            print(f"Reason: {error}")

        # ----------------------------------
        # 10. MODEL COMPARISON AND RANKING
        # ----------------------------------

        ranked_results, unranked_models, ranking_metric = rank_models(
            evaluation_results,
            problem_type
        )

        print("\n========== MODEL RANKING ==========")

        print(f"Ranking metric: {ranking_metric.upper()}")

        if ranked_results:

            for position, result in enumerate(ranked_results, start=1):

                print(f"\nRank {position}: {result['model_name']}")

                print(
                    f"{ranking_metric}: {result['metric_value']:.4f}"
                )

            print("\nTOP CANDIDATE:")

            print(ranked_results[0]["model_name"])

        else:
            print("No models could be ranked.")

        if unranked_models:

            print("\n========== UNRANKED MODELS ==========")

            for model_name in unranked_models:
                print(model_name)

        # ----------------------------------
        # 11. MODEL DIAGNOSTICS
        # ----------------------------------

        diagnostic_results, diagnostic_failures = diagnose_models(
            trained_models,
            X_train,
            y_train,
            X_test,
            y_test,
            problem_type
        )

        print("\n========== MODEL DIAGNOSTICS ==========")

        for model_name, result in diagnostic_results.items():

            print(f"\nModel: {model_name}")
            print(f"Diagnosis: {result['diagnosis']}")

            print(f"Metric: {result['metric']}")
            print(f"Training Score: {result['train_score']}")
            print(f"Testing Score: {result['test_score']}")
            print(f"Performance Gap: {result['performance_gap']}")

            print(f"Explanation: {result['explanation']}")

            print("Recommendations:")

            for recommendation in result["recommendations"]:
                print(f"- {recommendation}")

        for model_name, error in diagnostic_failures.items():

            print(f"\n[DIAGNOSTIC FAILED] {model_name}")
            print(f"Reason: {error}")

        # ----------------------------------
        # 12. OPTIMIZATION RECOMMENDATIONS
        # ----------------------------------

        optimization_plans = generate_optimization_plan(
            diagnostic_results
        )

        print("\n========== OPTIMIZATION PLANS ==========")

        for model_name, plan in optimization_plans.items():

            print(f"\nModel: {model_name}")
            print(f"Diagnosis: {plan['diagnosis']}")
            print(f"Status: {plan['status']}")

            print("Recommendations:")

            for recommendation in plan["recommendations"]:
                print(f"- {recommendation}")

            print(
                f"User approval required: {plan['requires_approval']}"
            )

        # ----------------------------------
        # 13. USER-APPROVED OPTIMIZATION
        # ----------------------------------

        if ranked_results:

            top_model_name = ranked_results[0]["model_name"]

            print("\n========== OPTIMIZATION APPROVAL ==========")

            print(f"Top candidate: {top_model_name}")

            approval = input(
                "\nDo you approve hyperparameter optimization? (yes/no): "
            ).strip().lower()

            if approval == "yes":

                result = optimize_model(
                    top_model_name,
                    trained_models[top_model_name],
                    X_train,
                    y_train,
                    problem_type
                )

                print("\n========== OPTIMIZATION RESULTS ==========")

                print(f"Status: {result['status']}")
                print(result["message"])

                if result["status"] == "SUCCESS":

                    optimized_model = result["best_estimator"]

                    print("\nBest parameters:")

                    for parameter, value in result["best_params"].items():
                        print(f"{parameter}: {value}")

                    print(f"\nCross-validation score: {result['cv_score']:.4f}")

                    # Evaluate the optimized model on the untouched test set
                    optimized_results, optimized_failures = evaluate_models(
                        {top_model_name: optimized_model},
                        X_test,
                        y_test,
                        problem_type
                    )

                    print("\nOptimized model test performance:")

                    print(optimized_results)

                    for model_name, error in optimized_failures.items():
                        print(f"[FAILED] {model_name}: {error}")

            else:
                print("\nOptimization skipped. Original models remain unchanged.")

    except (FileNotFoundError, ValueError, TypeError) as error:

        print(f"\n[ERROR] {error}")


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()