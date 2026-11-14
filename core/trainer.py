from sklearn.base import clone
from sklearn.pipeline import Pipeline


def train_models(models, preprocessor, X_train, y_train):

    if len(X_train) != len(y_train):
        raise ValueError("Features and target must have the same number of records.")

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    trained_models = {}
    failed_models = {}

    for model_name, model in models.items():

        # Combine preprocessing and model training
        model_pipeline = Pipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model))
        ])

        try:
            # Train the complete pipeline
            model_pipeline.fit(X_train, y_train)

            trained_models[model_name] = model_pipeline

        except Exception as error:
            failed_models[model_name] = str(error)

    return trained_models, failed_models