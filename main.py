from core.dataset_loader import load_dataset
from core.data_analyzer import analyze_dataset
from core.problem_detector import detect_problem_type
from core.data_preparer import separate_features_target
from core.data_splitter import split_dataset
from core.model_selector import get_candidate_models
from core.preprocessor import build_preprocessor








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
        target_column = input("\nEnter target column: ")
        X, y = separate_features_target(df, target_column)

        print("\n========== FEATURE-TARGET SEPARATION ==========")
        print("Features (X):")
        print(X.head())

        print("\nTarget (y):")
        print(y.head())

        print(f"\nNumber of features: {X.shape[1]}")
        print(f"Number of records: {X.shape[0]}")

        X_train, X_test, y_train, y_test = split_dataset(X, y)

        print("\n========== TRAIN-TEST SPLIT ==========")

        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        print("\n========== MISSING VALUES ==========")

        for column, count in analysis["missing_values"].items():
            percentage = analysis["missing_percentage"][column]

            print(f"{column}: {count} ({percentage}%)")

        print("\n========== UNIQUE VALUES ==========")

        for column, count in analysis["unique_values"].items():
            print(f"{column}: {count}")

        # Build preprocessing pipeline
        preprocessor = build_preprocessor(X_train)

        # Fit on training data and transform it
        X_train_processed = preprocessor.fit_transform(X_train)

        # Transform testing data using the same fitted preprocessor
        X_test_processed = preprocessor.transform(X_test)

        print("\n========== PREPROCESSING RESULTS ==========")

        print(f"Original training shape: {X_train.shape}")
        print(f"Processed training shape: {X_train_processed.shape}")

        print(f"Original testing shape: {X_test.shape}")
        print(f"Processed testing shape: {X_test_processed.shape}")

        print("\n========== STATISTICAL SUMMARY ==========")

        print(analysis["statistical_summary"])

        print("\n========== NUMERICAL CORRELATION ==========")

        print(analysis["numerical_correlation"])

        problem_type = detect_problem_type(df, target_column)

        print(f"\nDetected problem: {problem_type.upper()}")

        models = get_candidate_models(problem_type)
        print("\n========== CANDIDATE MODELS ==========")

        for model_name in models:
            print(model_name)


    except (FileNotFoundError, ValueError, TypeError) as e:

        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()