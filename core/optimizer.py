from copy import deepcopy


# ============================================================
# CENTRALIZED OPTIMIZATION CONFIGURATION
# ============================================================

OPTIMIZATION_SEARCH_SPACES = {

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
    },

    "Support Vector Machine": [
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear"]
        },
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf"],
            "model__gamma": ["scale", "auto"]
        }
    ],

    "Support Vector Regression": [
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

}


# ============================================================
# PARAMETER DESCRIPTIONS
# ============================================================

PARAMETER_LABELS = {

    "model__C": "Regularization strength (C)",

    "model__fit_intercept": "Intercept configuration",

    "model__max_depth": "Maximum tree depth",

    "model__min_samples_split": "Minimum samples required to split",

    "model__min_samples_leaf": "Minimum samples per leaf",

    "model__n_estimators": "Number of trees / boosting stages",

    "model__learning_rate": "Learning rate",

    "model__kernel": "Kernel type",

    "model__gamma": "RBF kernel gamma"

}


# ============================================================
# GET SEARCH CONFIGURATION
# ============================================================

def get_parameter_grid(model_name):

    """
    Return the actual supported parameter search space
    for a model.

    A copy is returned so callers cannot accidentally
    modify the central configuration.
    """

    parameter_grid = OPTIMIZATION_SEARCH_SPACES.get(model_name)

    if parameter_grid is None:
        return None

    return deepcopy(parameter_grid)


# ============================================================
# DESCRIBE SEARCH SPACE
# ============================================================

def describe_search_space(parameter_grid):

    """
    Convert the configured parameter grid into
    a human-readable description.
    """

    if parameter_grid is None:
        return "No automated search space configured."

    # Support both dictionaries and lists of dictionaries.
    grid_list = (
        parameter_grid
        if isinstance(parameter_grid, list)
        else [parameter_grid]
    )

    combined_parameters = {}

    for grid in grid_list:

        for parameter, values in grid.items():

            if parameter not in combined_parameters:
                combined_parameters[parameter] = []

            for value in values:

                if value not in combined_parameters[parameter]:
                    combined_parameters[parameter].append(value)

    descriptions = []

    for parameter, values in combined_parameters.items():

        label = PARAMETER_LABELS.get(
            parameter,
            parameter.replace("model__", "")
        )

        descriptions.append(
            f"{label}: {values}"
        )

    return "; ".join(descriptions)


# ============================================================
# GENERATE OPTIMIZATION PLAN
# ============================================================

def generate_optimization_plan(diagnostic_results):

    optimization_plans = {}

    for model_name, result in diagnostic_results.items():

        diagnosis = str(
            result.get("diagnosis", "UNKNOWN")
        ).upper()

        parameter_grid = get_parameter_grid(model_name)

        search_description = describe_search_space(
            parameter_grid
        )

        recommendations = []

        requires_approval = False

        # ----------------------------------------------------
        # INSUFFICIENT DATA
        # ----------------------------------------------------

        if diagnosis == "INCONCLUSIVE":

            recommendations = [
                "Collect more representative training records.",
                "Avoid drawing conclusions from an insufficient sample.",
                "Repeat model evaluation when sufficient data is available.",
                f"Configured search space for later use: {search_description}"
            ]

            status = "OPTIMIZATION DEFERRED"

        # ----------------------------------------------------
        # UNSUPPORTED MODEL
        # ----------------------------------------------------

        elif parameter_grid is None:

            recommendations = [
                "No automated hyperparameter search is configured for this model.",
                "No automatic model modifications will be performed.",
                "Manual review of the model configuration is required."
            ]

            status = "OPTIMIZATION NOT CONFIGURED"

        # ----------------------------------------------------
        # OVERFITTING
        # ----------------------------------------------------

        elif diagnosis == "POTENTIAL OVERFITTING":

            recommendations = [
                "The model shows signs of potential overfitting.",
                "Run GridSearchCV over the configured parameter alternatives.",
                f"Parameters to evaluate: {search_description}",
                "Retain the original model unless cross-validation demonstrates an improvement."
            ]

            status = "OPTIMIZATION RECOMMENDED"
            requires_approval = True

        # ----------------------------------------------------
        # UNDERFITTING
        # ----------------------------------------------------

        elif diagnosis == "POTENTIAL UNDERFITTING":

            recommendations = [
                "The model shows signs of potential underfitting.",
                "Evaluate alternative configurations available in the current search space.",
                f"Parameters to evaluate: {search_description}",
                "The search is limited to the configured hyperparameters and does not introduce new algorithms or features."
            ]

            status = "OPTIMIZATION RECOMMENDED"
            requires_approval = True

        # ----------------------------------------------------
        # NO OBVIOUS FIT ISSUE
        # ----------------------------------------------------

        else:

            recommendations = [
                "No obvious fitting issue was detected.",
                "An optional hyperparameter search is available.",
                f"Parameters that can be evaluated: {search_description}",
                "The original model will be retained unless optimization improves cross-validation performance."
            ]

            status = "OPTIONAL OPTIMIZATION AVAILABLE"
            requires_approval = True

        optimization_plans[model_name] = {

            "diagnosis": diagnosis,

            "status": status,

            "recommendations": recommendations,

            "requires_approval": requires_approval

        }

    return optimization_plans