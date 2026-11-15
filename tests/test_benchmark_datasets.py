from sklearn.datasets import load_breast_cancer, load_diabetes

from agent.tools import create_tool_registry


def run_benchmark(
    dataset_name,
    df,
    target_column,
    problem_type
):

    print("\n" + "=" * 55)
    print(f"       {dataset_name.upper()} TEST")
    print("=" * 55)

    print(f"\nDataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Problem type: {problem_type.upper()}")

    # Create ModelMind tools
    tools = create_tool_registry(df)

    # Execute complete ML workflow
    result = tools["run_model_analysis"](
        target_column=target_column,
        problem_type=problem_type
    )

    # Check analysis status
    if result["status"] != "SUCCESS":

        print("\n[FAILED] ML analysis did not complete.")
        print(result)

        return False

    print("\n[+] ML analysis completed successfully!")

    print(f"Experiment ID: {result.get('experiment_run_id')}")

    if result.get("experiment_tracking_error"):
        print(
            "Tracking warning:",
            result["experiment_tracking_error"]
        )

    print(f"\nTraining records: {result['training_records']}")
    print(f"Testing records: {result['testing_records']}")

    print(f"\nRanking metric: {result['ranking_metric']}")

    # Display ranking
    print("\n========== MODEL RANKING ==========")

    for position, model in enumerate(
        result["ranked_models"],
        start=1
    ):

        print(
            f"{position}. {model['model_name']}: "
            f"{model['metric_value']:.4f}"
        )

    # Display evaluation metrics
    print("\n========== EVALUATION RESULTS ==========")

    for model in result["model_summaries"]:

        print(f"\nModel: {model['model_name']}")

        for metric, value in model["metrics"].items():

            if value is None:
                print(f"{metric}: N/A")
            else:
                print(f"{metric}: {value:.4f}")

        diagnosis = model.get("diagnosis", {})

        print(
            "Diagnosis:",
            diagnosis.get("diagnosis", "Unavailable")
        )

    # Check experiment tracking
    print("\n========== TRACKING CHECK ==========")

    if result.get("experiment_run_id"):
        print("[SUCCESS] Experiment saved.")
    else:
        print("[WARNING] Experiment was not saved.")

    # Check if every model was handled
    print("\nTraining failures:", result["training_failures"])
    print("Evaluation failures:", result["evaluation_failures"])

    return True


def main():

    # ==================================
    # 1. CLASSIFICATION DATASET
    # ==================================

    cancer_data = load_breast_cancer(as_frame=True)

    cancer_df = cancer_data.frame.copy()

    classification_success = run_benchmark(
        dataset_name="Breast Cancer Wisconsin",
        df=cancer_df,
        target_column="target",
        problem_type="classification"
    )

    # ==================================
    # 2. REGRESSION DATASET
    # ==================================

    diabetes_data = load_diabetes(as_frame=True)

    diabetes_df = diabetes_data.frame.copy()

    regression_success = run_benchmark(
        dataset_name="Diabetes",
        df=diabetes_df,
        target_column="target",
        problem_type="regression"
    )

    # ==================================
    # FINAL TEST SUMMARY
    # ==================================

    print("\n" + "=" * 55)
    print("          MODELMIND TEST SUMMARY")
    print("=" * 55)

    print(
        "Classification:",
        "PASSED" if classification_success else "FAILED"
    )

    print(
        "Regression:",
        "PASSED" if regression_success else "FAILED"
    )


if __name__ == "__main__":
    main()