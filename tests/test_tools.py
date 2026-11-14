from core.dataset_loader import load_dataset
from agent.tools import create_tool_registry


def main():

    df = load_dataset("datasets/students.csv")

    tools = create_tool_registry(df)

    print("\n========== DATASET INSPECTION ==========")

    dataset_info = tools["inspect_dataset"]()

    print(dataset_info)

    print("\n========== COLUMN INSPECTION ==========")

    column_info = tools["inspect_column"]("Marks")

    print(column_info)


if __name__ == "__main__":
    main()