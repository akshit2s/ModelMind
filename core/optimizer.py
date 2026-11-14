def generate_optimization_plan(diagnostic_results):

    optimization_plans = {}

    for model_name, result in diagnostic_results.items():

        diagnosis = result["diagnosis"]

        recommendations = []

        # Handle insufficient data
        if diagnosis == "INCONCLUSIVE":

            recommendations = [
                "Collect more representative training data.",
                "Increase testing dataset size.",
                "Use cross-validation when sufficient data is available."
            ]

            status = "OPTIMIZATION DEFERRED"

        # Handle overfitting
        elif diagnosis == "POTENTIAL OVERFITTING":

            if "Decision Tree" in model_name:

                recommendations = [
                    "Reduce maximum tree depth.",
                    "Increase minimum samples required per leaf.",
                    "Tune pruning parameters."
                ]

            elif "Random Forest" in model_name:

                recommendations = [
                    "Limit the maximum depth of individual trees.",
                    "Increase minimum samples per leaf.",
                    "Tune the number of features considered at each split."
                ]

            elif "Gradient Boosting" in model_name:

                recommendations = [
                    "Reduce the maximum depth of individual trees.",
                    "Lower the learning rate.",
                    "Tune the number of boosting stages."
                ]

            elif "Logistic Regression" in model_name:

                recommendations = [
                    "Experiment with stronger regularization.",
                    "Review feature selection.",
                    "Tune the regularization parameter."
                ]

            elif "Support Vector" in model_name:

                recommendations = [
                    "Experiment with stronger regularization.",
                    "Tune the kernel parameters.",
                    "Review feature scaling."
                ]

            status = "OPTIMIZATION RECOMMENDED"

        # Handle underfitting
        elif diagnosis == "POTENTIAL UNDERFITTING":

            recommendations = [
                "Consider a more expressive model.",
                "Increase model complexity where appropriate.",
                "Explore useful feature engineering.",
                "Review excessive regularization."
            ]

            status = "OPTIMIZATION RECOMMENDED"

        # No obvious fit issue
        else:

            recommendations = [
                "No immediate optimization is recommended.",
                "Validate performance using cross-validation.",
                "Monitor performance on unseen data."
            ]

            status = "NO IMMEDIATE CHANGE REQUIRED"

        optimization_plans[model_name] = {
            "diagnosis": diagnosis,
            "status": status,
            "recommendations": recommendations,
            "requires_approval": True
        }

    return optimization_plans