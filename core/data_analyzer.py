import pandas as pd


def analyze_dataset(df):

    # Step 1: Validate the input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    # Step 2: Dataset dimensions
    rows, columns = df.shape

    # Step 3: Column information
    column_names = df.columns.tolist()

    data_types = df.dtypes.astype(str).to_dict()

    # Step 4: Missing value analysis
    missing_values = df.isnull().sum().to_dict()

    missing_percentage = (
        (df.isnull().sum() / rows) * 100
    ).round(2).to_dict()

    # Step 5: Duplicate records
    duplicate_rows = int(df.duplicated().sum())

    # Step 6: Unique values
    unique_values = df.nunique().to_dict()

    # Step 7: Statistical summary
    statistical_summary = df.describe(include="all")

    # Step 8: Numerical correlation
    numerical_correlation = df.corr(numeric_only=True)

    # Step 9: Store everything in a structured dictionary
    analysis = {
        "rows": rows,
        "columns": columns,
        "column_names": column_names,
        "data_types": data_types,
        "missing_values": missing_values,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "unique_values": unique_values,
        "statistical_summary": statistical_summary,
        "numerical_correlation": numerical_correlation
    }

    return analysis