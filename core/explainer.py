from sklearn.inspection import permutation_importance


def explain_model(
    model,
    X_eval,
    y_eval,
    problem_type,
    n_repeats=10
):

    # Validate inputs
    if len(X_eval) != len(y_eval):
        raise ValueError(
            "Features and target must have equal lengths."
        )

    if len(X_eval) == 0:
        raise ValueError("Evaluation dataset is empty.")

    # Avoid unreliable explanations on tiny datasets
    if len(X_eval) < 10:

        return {
            "status": "INCONCLUSIVE",
            "message": (
                "Insufficient evaluation records for "
                "reliable feature importance analysis."
            ),
            "feature_importance": []
        }

    # Select the appropriate evaluation metric
    if problem_type == "classification":

        scoring = "f1_weighted"

    elif problem_type == "regression":

        scoring = "neg_root_mean_squared_error"

    else:

        raise ValueError("Unsupported problem type.")

    # Calculate permutation importance
    results = permutation_importance(
        estimator=model,
        X=X_eval,
        y=y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    # Organize feature importance results
    importance_results = []

    for index, feature_name in enumerate(X_eval.columns):

        importance_results.append({
            "feature": feature_name,
            "importance": float(
                results.importances_mean[index]
            ),
            "standard_deviation": float(
                results.importances_std[index]
            )
        })

    # Sort features from most to least important
    importance_results.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    return {
        "status": "SUCCESS",
        "metric": scoring,
        "feature_importance": importance_results
    }