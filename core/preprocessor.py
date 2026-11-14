from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):

    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Check for unsupported column types
    supported_features = numerical_features + categorical_features

    unsupported_features = [
        column for column in X.columns
        if column not in supported_features
    ]

    if unsupported_features:
        raise ValueError(
            f"Unsupported feature types: {unsupported_features}"
        )

    if not supported_features:
        raise ValueError("No supported feature columns found.")

    transformers = []

    # Numerical preprocessing
    if numerical_features:

        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append((
            "numerical",
            numerical_pipeline,
            numerical_features
        ))

    # Categorical preprocessing
    if categorical_features:

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_features
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor