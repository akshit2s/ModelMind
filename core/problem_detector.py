import pandas as pd


def detect_problem_type(df, target_column):

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    target = df[target_column]

    if target.isna().all():
        raise ValueError("Target column contains only missing values.")

    # Boolean and categorical targets suggest classification
    if (
        pd.api.types.is_bool_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(target)
    ):
        return "classification"

    # Numeric targets with few unique values are ambiguous
    if pd.api.types.is_numeric_dtype(target):

        unique_values = target.nunique()

        if unique_values <= 20:
            return "ambiguous"

        return "regression"

    return "ambiguous"