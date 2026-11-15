# ModelMind

### An AI-Powered Agent for Intelligent Machine Learning Model Selection, Evaluation and Optimization

ModelMind is a command-line AI agent that helps users analyze tabular datasets, identify suitable machine learning approaches, compare candidate models, diagnose potential fitting issues, and perform user-approved hyperparameter optimization.

It combines Python's machine learning ecosystem with Google's Gemini API to make the ML workflow more accessible and explainable.

## Features

- **Dataset inspection:** Analyze dataset dimensions, column types, missing values, duplicates and feature characteristics.
- **Problem identification:** Work with classification and regression tasks.
- **Automated model comparison:** Train and evaluate multiple candidate algorithms.
- **Data preprocessing:** Handle missing values, numerical scaling and categorical encoding.
- **Model evaluation:** Calculate accuracy, precision, recall, F1-score, MAE, RMSE and R² where applicable.
- **Model ranking:** Compare candidates using task-appropriate evaluation metrics.
- **Fit diagnostics:** Identify potential overfitting, underfitting or inconclusive results.
- **AI-powered explanations:** Use Gemini to interpret analysis results and answer follow-up questions.
- **Hyperparameter optimization:** Use GridSearchCV to evaluate supported parameter configurations.
- **Approval-based execution:** Require explicit user approval before optimization.
- **Experiment tracking:** Store model evaluation results and experiment history in SQLite.
- **Optimization history:** Save and retrieve optimization attempts, selected configurations and final evaluation metrics.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| AI Agent | Google Gemini API |
| ML Framework | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Storage | SQLite |
| CLI Presentation | Rich |
| Configuration | Python-dotenv |

## Architecture

ModelMind separates AI orchestration from machine learning execution.

**Gemini** handles natural-language interaction, tool selection and explanations.

**Python and Scikit-learn** handle dataset processing, model training, evaluation, ranking, diagnostics and optimization.

The LLM does not calculate evaluation metrics or independently decide whether an optimization has improved performance. Those decisions are handled by the Python engine.

### Workflow

1. Load a CSV dataset.
2. Inspect the dataset and select a target column.
3. Identify the classification or regression task.
4. Prepare and preprocess the data.
5. Train candidate machine learning models.
6. Evaluate and rank the models.
7. Diagnose potential fitting issues.
8. Generate optimization recommendations.
9. Request user approval before optimization.
10. Run hyperparameter search and compare cross-validation results.
11. Retain the original model when optimization does not improve performance.
12. Save experiment and optimization history.

## Project Structure

```text
ModelMind/
│
├── agent/
│   ├── llm_client.py
│   ├── prompts.py
│   ├── tools.py
│   └── controller.py
│
├── core/
│   ├── dataset_loader.py
│   ├── data_analyzer.py
│   ├── problem_detector.py
│   ├── data_preparer.py
│   ├── data_splitter.py
│   ├── model_selector.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── model_comparator.py
│   ├── leaderboard.py
│   ├── diagnostics.py
│   ├── optimizer.py
│   ├── optimization_runner.py
│   ├── explainer.py
│   ├── experiment_tracker.py
│   ├── experiment_comparator.py
│   └── optimization_history.py
│
├── datasets/
├── experiments/
├── tests/
│
├── main.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/akshit2s/ModelMind.git
cd ModelMind
```

### 2. Create a virtual environment

Python 3.11 is recommended.

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API

Create a `.env` file in the project root using `.env.example` as a template.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep your actual `.env` file private. Never commit API keys or other credentials.

### 5. Run ModelMind

```bash
python main.py
```

At startup, provide a CSV dataset path or press Enter to use the default dataset path.

## Example Interactions

```text
Inspect my dataset.
```

```text
Predict Marks using the available features.
This is a regression problem. Compare all candidate models.
```

```text
Show me my five most recent experiments.
```

```text
Prepare an optimization plan for the best-performing model.
```

```text
Show me my saved optimization history.
```

ModelMind maintains a conversation with the user until the application is closed. Type `clear` to reset conversation history or `exit` to quit.

## Testing

The project includes test modules for dataset loading, ML execution, optimization, benchmarking and experiment tracking.

Run them from the project root:

```bash
python -m tests.test_loader
python -m tests.test_ml_tool
python -m tests.test_optimization_flow
python -m tests.test_benchmark_datasets
python -m tests.test_experiment_tools
```

## Important Considerations

- Model performance depends on dataset quality, size and representativeness.
- Very small datasets can produce unreliable evaluation results. ModelMind marks insufficient-data diagnoses as inconclusive and defers optimization where appropriate.
- Hyperparameter search is limited to the configurations implemented in the optimization engine.
- A successful optimization run does not necessarily mean performance improved. ModelMind compares cross-validation scores and retains the original model when no improvement is found.
- Experiment records are stored locally in SQLite.

## Project Status

ModelMind currently supports classification and regression workflows, automated candidate-model comparison, diagnostics, approval-gated hyperparameter optimization and persistent experiment tracking through an interactive CLI.

Further development can extend the supported algorithms, optimization strategies, reporting capabilities and user interface.

---

**ModelMind — Making machine learning experimentation more accessible, systematic and explainable.**