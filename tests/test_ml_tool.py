from core.dataset_loader import load_dataset
from agent.tools import create_tool_registry


def main():

    df = load_dataset("datasets/students.csv")

    tools = create_tool_registry(df)

    print("\n========== MODELMIND ML TOOL TEST ==========")

    # STEP 1: Run ML analysis
    result = tools["run_model_analysis"](
        target_column="Marks",
        problem_type="regression"
    )

    print("\nAnalysis Status:", result["status"])
    print("Ranking Metric:", result.get("ranking_metric"))

    print("\nRanked Models:")

    for model in result.get("ranked_models", []):

        print(
            model["model_name"],
            model["metric_value"]
        )

    print("\nDiagnostics:")

    for model in result.get("model_summaries", []):

        print(
            model["model_name"],
            model["diagnosis"].get("diagnosis")
        )

    # STEP 2: Prepare optimization
    if result["status"] == "SUCCESS" and result["ranked_models"]:

        top_model = result["ranked_models"][0]["model_name"]

        print("\n========== OPTIMIZATION PLAN TEST ==========")

        print("Selected model:", top_model)

        plan = tools["prepare_optimization"](
            model_name=top_model
        )

        print("\nOptimization Plan:")
        print(plan)


if __name__ == "__main__":
    main()