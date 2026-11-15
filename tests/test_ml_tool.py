from core.dataset_loader import load_dataset
from agent.tools import create_tool_registry


def main():

    df = load_dataset("datasets/students.csv")

    tools = create_tool_registry(df)

    print("\n========== MODELMIND ML TOOL TEST ==========")

    result = tools["run_model_analysis"](
        target_column="Marks",
        problem_type="regression"
    )

    print("\nAnalysis Status:", result["status"])

    print("\nRanking Metric:", result.get("ranking_metric"))

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


if __name__ == "__main__":
    main()