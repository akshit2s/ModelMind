def separate_features_target(df, target_column):

    # Check whether the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y