def detect_problem_type(df, target_column):

    # Check whether target column exists
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    target = df[target_column]

    # Check target data type
    if target.dtype == "object" or target.dtype.name == "category":
        return "classification"

    # Check number of unique values
    unique_values = target.nunique()

    if unique_values <= 20:
        return "classification"

    return "regression"