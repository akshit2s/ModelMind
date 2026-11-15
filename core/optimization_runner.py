import numpy as np

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score
)

from core.optimizer import get_parameter_grid


# ============================================================
# MODEL OPTIMIZATION
# ============================================================

def optimize_model(
    model_name,
    trained_model,
    X_train,
    y_train,
    problem_type
):

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    if problem_type not in ["classification", "regression"]:
        raise ValueError("Unsupported problem type.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training features and target lengths differ."
        )

    if len(X_train) < 30:

        return {
            "status": "INCONCLUSIVE",
            "message": (
                "Insufficient training records for "
                "reliable optimization."
            )
        }

    # ========================================================
    # 2. GET CENTRALIZED PARAMETER GRID
    # ========================================================

    parameter_grid = get_parameter_grid(model_name)

    if parameter_grid is None:

        return {
            "status": "UNSUPPORTED",
            "message": (
                f"No optimization configuration "
                f"for {model_name}."
            )
        }

    # ========================================================
    # 3. CROSS-VALIDATION CONFIGURATION
    # ========================================================

    if problem_type == "classification":

        # Works with both NumPy arrays and Pandas Series.
        _, class_counts = np.unique(
            np.asarray(y_train),
            return_counts=True
        )

        cv_splits = min(
            5,
            int(class_counts.min())
        )

        if cv_splits < 3:

            return {
                "status": "INCONCLUSIVE",
                "message": (
                    "Insufficient examples per class "
                    "for cross-validation."
                )
            }

        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42
        )

        scoring = "f1_weighted"

    else:

        cv_splits = min(5, len(X_train))

        if cv_splits < 3:

            return {
                "status": "INCONCLUSIVE",
                "message": (
                    "Insufficient records for cross-validation."
                )
            }

        cv = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42
        )

        scoring = "neg_root_mean_squared_error"

    # ========================================================
    # 4. RUN BASELINE AND GRID SEARCH
    # ========================================================

    try:

        # Evaluate the original model.
        baseline_scores = cross_val_score(
            estimator=trained_model,
            X=X_train,
            y=y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            error_score="raise"
        )

        baseline_cv_score = float(
            baseline_scores.mean()
        )

        # Search the supported hyperparameter combinations.
        search = GridSearchCV(
            estimator=trained_model,
            param_grid=parameter_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        optimized_cv_score = float(
            search.best_score_
        )

        # ====================================================
        # 5. NORMALIZE METRICS
        # ====================================================

        # Scikit-learn uses negative RMSE for optimization.
        # Convert it into positive RMSE for reporting.

        if problem_type == "regression":

            baseline_cv_score = -baseline_cv_score

            optimized_cv_score = -optimized_cv_score

        # ====================================================
        # 6. CALCULATE IMPROVEMENT
        # ====================================================

        if problem_type == "classification":

            improvement = (
                optimized_cv_score - baseline_cv_score
            )

        else:

            improvement = (
                baseline_cv_score - optimized_cv_score
            )

        improved = improvement > 0.000001

        # ====================================================
        # 7. RETURN RESULTS
        # ====================================================

        return {

            "status": "SUCCESS",

            "best_estimator": search.best_estimator_,

            "best_params": search.best_params_,

            "baseline_cv_score": baseline_cv_score,

            "optimized_cv_score": optimized_cv_score,

            "improvement": improvement,

            "improved": improved,

            "message": (
                "Hyperparameter optimization completed."
            )

        }

    except Exception as error:

        return {

            "status": "FAILED",

            "message": (
                f"Optimization failed: "
                f"{type(error).__name__}: {error}"
            )

        }