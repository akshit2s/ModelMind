from sklearn.metrics import f1_score, r2_score


def diagnose_models(
    trained_models,
    X_train,
    y_train,
    X_test,
    y_test,
    problem_type
):

    if problem_type not in ["classification", "regression"]:
        raise ValueError("Unsupported problem type.")

    if len(X_train) != len(y_train):
        raise ValueError("Training features and target must have equal lengths.")

    if len(X_test) != len(y_test):
        raise ValueError("Testing features and target must have equal lengths.")

    diagnostic_results = {}
    diagnostic_failures = {}

    # Initial minimum sample requirements
    min_train_samples = 20
    min_test_samples = 5

    for model_name, model in trained_models.items():

        try:
            # Generate predictions on both datasets
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Calculate classification scores
            if problem_type == "classification":

                train_score = f1_score(
                    y_train,
                    train_predictions,
                    average="weighted",
                    zero_division=0
                )

                test_score = f1_score(
                    y_test,
                    test_predictions,
                    average="weighted",
                    zero_division=0
                )

                metric_name = "weighted_f1"

                performance_gap = train_score - test_score

            # Calculate regression scores
            else:

                train_score = (
                    r2_score(y_train, train_predictions)
                    if len(y_train) >= 2 else None
                )

                test_score = (
                    r2_score(y_test, test_predictions)
                    if len(y_test) >= 2 else None
                )

                metric_name = "r2_score"

                performance_gap = (
                    train_score - test_score
                    if train_score is not None and test_score is not None
                    else None
                )

            # Determine whether enough data is available
            insufficient_data = (
                len(X_train) < min_train_samples
                or len(X_test) < min_test_samples
                or performance_gap is None
            )

            if insufficient_data:

                diagnosis = "INCONCLUSIVE"

                explanation = (
                    "Insufficient data for a reliable fit diagnosis."
                )

                recommendations = [
                    "Collect more representative training data.",
                    "Increase the size of the testing dataset.",
                    "Use cross-validation when sufficient data is available."
                ]

            # Diagnose classification models
            elif problem_type == "classification":

                if train_score >= 0.80 and performance_gap >= 0.20:

                    diagnosis = "POTENTIAL OVERFITTING"

                    explanation = (
                        "Training performance is high, but testing "
                        "performance is considerably lower."
                    )

                    recommendations = [
                        "Try reducing model complexity.",
                        "Consider regularisation.",
                        "Evaluate performance using cross-validation."
                    ]

                elif train_score < 0.60 and test_score < 0.60:

                    diagnosis = "POTENTIAL UNDERFITTING"

                    explanation = (
                        "The model performs poorly on both training "
                        "and testing data."
                    )

                    recommendations = [
                        "Try a more expressive algorithm.",
                        "Review the quality of input features.",
                        "Consider feature engineering."
                    ]

                else:

                    diagnosis = "NO OBVIOUS FIT ISSUE"

                    explanation = (
                        "The current scores do not show a strong "
                        "overfitting or underfitting signal."
                    )

                    recommendations = [
                        "Compare results using cross-validation.",
                        "Continue monitoring performance on unseen data."
                    ]

            # Diagnose regression models
            else:

                if train_score >= 0.80 and performance_gap >= 0.20:

                    diagnosis = "POTENTIAL OVERFITTING"

                    explanation = (
                        "Training R2 is high, but testing R2 "
                        "is considerably lower."
                    )

                    recommendations = [
                        "Try reducing model complexity.",
                        "Consider regularisation.",
                        "Evaluate performance using cross-validation."
                    ]

                elif train_score < 0.30 and test_score < 0.30:

                    diagnosis = "POTENTIAL UNDERFITTING"

                    explanation = (
                        "R2 is low on both training and testing data."
                    )

                    recommendations = [
                        "Try a more expressive regression algorithm.",
                        "Review the input features.",
                        "Consider feature engineering."
                    ]

                else:

                    diagnosis = "NO OBVIOUS FIT ISSUE"

                    explanation = (
                        "The current scores do not show a strong "
                        "overfitting or underfitting signal."
                    )

                    recommendations = [
                        "Compare results using cross-validation.",
                        "Continue monitoring unseen-data performance."
                    ]

            diagnostic_results[model_name] = {
                "metric": metric_name,
                "train_score": train_score,
                "test_score": test_score,
                "performance_gap": performance_gap,
                "diagnosis": diagnosis,
                "explanation": explanation,
                "recommendations": recommendations
            }

        except Exception as error:

            diagnostic_failures[model_name] = str(error)

    return diagnostic_results, diagnostic_failures