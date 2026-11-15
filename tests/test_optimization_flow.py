import pandas as pd

from sklearn.datasets import make_regression

from agent.tools import create_tool_registry


def main():

    print("\n======================================")
    print("      MODELMIND OPTIMIZATION TEST")
    print("======================================")

    # ==================================
    # 1. GENERATE SYNTHETIC DATASET
    # ==================================

    X, y = make_regression(
        n_samples=500,
        n_features=5,
        n_informative=4,
        noise=15,
        random_state=42
    )

    feature_names = [
        "Feature_1",
        "Feature_2",
        "Feature_3",
        "Feature_4",
        "Feature_5"
    ]

    df = pd.DataFrame(X, columns=feature_names)

    df["Target"] = y

    print("\n[+] Synthetic dataset generated.")
    print(f"Records: {len(df)}")
    print(f"Features: {len(feature_names)}")

    # ==================================
    # 2. CREATE TOOL REGISTRY
    # ==================================

    tools = create_tool_registry(df)

    # ==================================
    # 3. RUN ML ANALYSIS
    # ==================================

    print("\n========== MODEL ANALYSIS ==========")

    analysis = tools["run_model_analysis"](
        target_column="Target",
        problem_type="regression"
    )

    if analysis["status"] != "SUCCESS":

        print("[ERROR] ML analysis failed.")
        print(analysis)
        return

    print("\nRanking metric:", analysis["ranking_metric"])

    print("\nRanked models:")

    for model in analysis["ranked_models"]:

        print(
            model["model_name"],
            model["metric_value"]
        )

    # ==================================
    # 4. SELECT BEST MODEL
    # ==================================

    if not analysis["ranked_models"]:

        print("No models available for ranking.")
        return

    best_model = analysis["ranked_models"][0]["model_name"]

    print("\nBest candidate:", best_model)

    # ==================================
    # 5. GENERATE OPTIMIZATION PLAN
    # ==================================

    print("\n========== OPTIMIZATION PLAN ==========")

    plan = tools["prepare_optimization"](
        model_name="best_model"
    )

    print(plan)

    if plan["status"] != "AWAITING_USER_APPROVAL":

        print("\nOptimization cannot proceed.")
        print("Reason:", plan["status"])
        return

    # ==================================
    # 6. ASK FOR USER APPROVAL
    # ==================================

    print("\n========== APPROVAL REQUIRED ==========")

    for recommendation in plan["recommendations"]:
        print("-", recommendation)

    approval = input(
        "\nDo you approve optimization? (yes/no): "
    ).strip().lower()

    approved = approval in ["yes", "y"]

    # ==================================
    # 7. EXECUTE APPROVED OPTIMIZATION
    # ==================================

    result = tools["execute_approved_optimization"](
        approved=approved
    )

    print("\n========== OPTIMIZATION RESULTS ==========")

    print("Status:", result["status"])

    if result["status"] == "SUCCESS":

        print("Model:", result["model_name"])

        print("Original CV score:", result["baseline_cv_score"])

        print("Optimized CV score:", result["optimized_cv_score"])

        print("Improvement:", result["improvement"])

        print("Improved:", result["improved"])

        print("Selected version:", result["selected_version"])

        print("\nFinal test metrics:")
        print(result["test_metrics"])

    else:

        print(result.get("message", "Optimization unsuccessful."))


if __name__ == "__main__":
    main()