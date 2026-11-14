from core.dataset_loader import load_dataset
from core.data_analyzer import analyze_dataset


def main():

    print("\n===================================")
    print("          MODELMIND")
    print("    ML ANALYSIS & ADVISOR")
    print("===================================\n")

    file_path = input("Enter dataset path: ")

    try:

        # Load dataset
        df = load_dataset(file_path)

        print("\n[+] Dataset loaded successfully!")

        # Analyze dataset
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

    except (FileNotFoundError, ValueError, TypeError) as e:

        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()