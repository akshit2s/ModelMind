from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, random_state=42):

    if len(X) != len(y):
        raise ValueError("Features and target must have the same number of records.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test