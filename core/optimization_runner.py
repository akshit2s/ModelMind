from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score
)
def optimize_model(
    model_name,
    trained_model,
    X_train,
    y_train,
    problem_type
):

    # Guard against insufficient data
    if len(X_train) < 30:
        return {
            "status": "INCONCLUSIVE",
            "message": (
                "Insufficient training records for reliable "
                "hyperparameter optimization. At least 30 "
                "training records are required by the current safeguard."
            )
        }

    # Define hyperparameter search spaces
    parameter_grids = {

        "Logistic Regression": {
            "model__C": [0.1, 1, 10]
        },

        "Linear Regression": {
            "model__fit_intercept": [True, False]
        },

        "Decision Tree": {
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        },

        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2]
        },

        "Gradient Boosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [1, 3]
        }
    }

    # Support Vector models have separate search spaces
    if model_name in [
        "Support Vector Machine",
        "Support Vector Regression"
    ]:

        parameter_grid = [
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear"]
            },
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale", "auto"]
            }
        ]

    else:
        parameter_grid = parameter_grids.get(model_name)

    if parameter_grid is None:
        return {
            "status": "UNSUPPORTED",
            "message": "No optimization search space is defined for this model."
        }

    # Configure cross-validation
    if problem_type == "classification":

        class_counts = y_train.value_counts()

        cv_splits = min(5, int(class_counts.min()))

        if cv_splits < 3:
            return {
                "status": "INCONCLUSIVE",
                "message": "Insufficient examples in one or more classes for cross-validation."
            }

        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42
        )

        scoring = "f1_weighted"

    elif problem_type == "regression":

        cv_splits = min(5, len(X_train))

        if cv_splits < 3:
            return {
                "status": "INCONCLUSIVE",
                "message": "Insufficient records for cross-validation."
            }

        cv = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42
        )

        scoring = "neg_root_mean_squared_error"

    else:
        raise ValueError("Invalid problem type.")
    # Evaluate the original model using cross-validation
    baseline_scores = cross_val_score(
        estimator=trained_model,
        X=X_train,
        y=y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    baseline_cv_score = float(baseline_scores.mean())
    # Run hyperparameter search
    search = GridSearchCV(
        estimator=trained_model,
        param_grid=parameter_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)
    
    optimized_cv_score = float(search.best_score_)

    # Convert negative RMSE into normal positive RMSE
    if problem_type == "regression":
        baseline_cv_score = -baseline_cv_score
        optimized_cv_score = -optimized_cv_score


    # Determine whether optimization improved performance
    if problem_type == "classification":

        improvement = optimized_cv_score - baseline_cv_score

        improved = improvement > 0.000001

    else:

        improvement = baseline_cv_score - optimized_cv_score

        improved = improvement > 0.000001

    if problem_type == "regression":
        best_cv_score = -best_cv_score

    return {
    "status": "SUCCESS",
    "best_estimator": search.best_estimator_,
    "best_params": search.best_params_,

    "baseline_cv_score": baseline_cv_score,
    "optimized_cv_score": optimized_cv_score,

    "improvement": improvement,
    "improved": improved,

    "message": "Hyperparameter optimization completed."
}