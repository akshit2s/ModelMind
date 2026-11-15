from core.dataset_loader import load_dataset
from agent.tools import create_tool_registry


def main():

    df = load_dataset("datasets/students.csv")

    tools = create_tool_registry(df)

    print("\n========== EXPERIMENT HISTORY ==========")

    history = tools["get_experiment_history"](limit=5)

    print(history)

    print("\n========== EXPERIMENT COMPARISON ==========")

    comparison = tools["compare_current_dataset_runs"](
        problem_type="regression",
        target_column="Marks"
    )

    print(comparison)

    print("\n========== OPTIMIZATION HISTORY ==========")

    optimization_history = tools["get_saved_optimization_history"]()

    print(optimization_history)


if __name__ == "__main__":
    main()