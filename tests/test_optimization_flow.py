import pandas as pd

from sklearn.datasets import make_regression

from agent.tools import create_tool_registry

from core.optimization_history import (
    get_optimization_history
)


def main():

    print("\n======================================")
    print("      MODELMIND OPTIMIZATION TEST")
    print("======================================")

    # STEP 1: Generate synthetic dataset
    X, y = make_regression(
        n_samples=500,
        n_features=5,
        n_informative=4,
        noise=15,
        random_state=42
    )

    columns = [
        "Feature_1",
        "Feature_2",
        "Feature_3",
        "Feature_4",
        "Feature_5"
    ]

    df = pd.DataFrame(X, columns=columns)
    df["Target"] = y

    print("\n[+] Dataset generated:", df.shape)

    # STEP 2: Create tools
    tools = create_tool_registry(df)

    # STEP 3: Run ML analysis
    analysis = tools["run_model_analysis"](
        target_column="Target",
        problem_type="regression"
    )

    if analysis["status"] != "SUCCESS":
        print("[ERROR] Analysis failed:", analysis)
        return

    run_id = analysis.get("experiment_run_id")

    print("\nExperiment ID:", run_id)

    # STEP 4: Select best model
    if not analysis["ranked_models"]:
        print("No models available.")
        return

    best_model = analysis["ranked_models"][0]["model_name"]

    print("Best model:", best_model)

    # STEP 5: Prepare optimization
    plan = tools["prepare_optimization"](
        model_name="best_model"
    )

    print("\nOptimization plan:", plan)

    if plan["status"] != "AWAITING_USER_APPROVAL":
        print("Optimization unavailable:", plan["status"])
        return

    # STEP 6: Request approval
    approval = input(
        "\nApprove optimization? (yes/no): "
    ).strip().lower()

    approved = approval in ["yes", "y"]

    # STEP 7: Execute approved optimization
    result = tools["execute_approved_optimization"](
        approved=approved
    )

    print("\n========== OPTIMIZATION RESULTS ==========")
    print(result)

    # STEP 8: Retrieve saved history
    if result["status"] == "SUCCESS" and run_id:

        print("\n========== SAVED OPTIMIZATION HISTORY ==========")

        history = get_optimization_history(run_id=run_id)

        for record in history:

            print("\nModel:", record["model_name"])
            print("Status:", record["status"])

            print(
                "Original CV score:",
                record["baseline_cv_score"]
            )

            print(
                "Optimized CV score:",
                record["optimized_cv_score"]
            )

            print(
                "Selected version:",
                record["selected_version"]
            )

            print(
                "Final test metrics:",
                record["final_test_metrics"]
            )


if __name__ == "__main__":
    main()