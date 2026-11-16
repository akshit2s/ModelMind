# ModelMind

### An AI-Powered Machine Learning Agent for Model Selection, Evaluation and Optimization

ModelMind is an intelligent machine learning platform designed to simplify the process of working with tabular datasets.

It helps users inspect datasets, identify suitable machine learning approaches, train and compare candidate models, diagnose potential fitting issues, and perform user-approved hyperparameter optimization.

ModelMind combines Python's machine learning ecosystem with Google's Gemini API and a modern web interface to make machine learning experimentation more accessible and explainable.

---

## Overview

Traditional machine learning experimentation often requires manually selecting algorithms, preparing datasets, evaluating models and tuning parameters.

ModelMind brings these activities into one guided workflow.

It provides two interfaces:

- **Web Interface:** A React-based dashboard for dataset uploads, model comparison, optimization and experiment tracking.
- **CLI Interface:** An interactive AI agent powered by Google's Gemini API for natural-language interaction with the ML engine.

---

## Features

### Dataset Analysis
- Upload CSV datasets through the web interface.
- Inspect dataset dimensions, column types, missing values and duplicate records.
- Examine individual dataset columns.
- Support for classification and regression tasks.

### Automated Model Comparison
- Train multiple candidate machine learning algorithms.
- Apply numerical scaling and categorical encoding.
- Evaluate models using task-specific metrics.
- Rank candidate models based on evaluation performance.

### Model Diagnostics
- Identify potential overfitting and underfitting.
- Examine model performance and generalization.
- Handle inconclusive results when available data is insufficient.

### Hyperparameter Optimization
- Generate parameter-tuning recommendations.
- Use GridSearchCV to evaluate supported configurations.
- Require explicit user approval before executing optimization.
- Retain the original model when optimization does not improve cross-validation performance.

### Experiment Tracking
- Store experiment results using SQLite.
- Retrieve previous experiment runs.
- Compare historical experiments.
- Maintain optimization history and selected model configurations.

### AI-Powered Assistance
- Natural-language interaction through the Gemini-powered CLI agent.
- AI-assisted explanations of dataset analysis and model results.
- Registered Python tools for controlled execution of ML operations.

### Web Dashboard
- Modern React-based interface.
- Light and dark themes.
- Dataset upload and inspection.
- Model leaderboard and evaluation results.
- Optimization approval workflow.
- Experiment history.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | React |
| Frontend Build Tool | Vite |
| Styling | CSS |
| Icons | Lucide React |
| Backend API | FastAPI |
| API Server | Uvicorn |
| AI Agent | Google Gemini API |
| ML Framework | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Storage | SQLite |
| CLI Presentation | Rich |
| Configuration | Python-dotenv |

---

## System Architecture

ModelMind separates the user interface, API layer, AI orchestration and machine learning execution.

```text
                  MODELMIND
                      |
          +-----------+-----------+
          |                       |
     React Frontend           CLI Interface
          |                       |
          |                  Gemini AI Agent
          |                       |
          +-----------+-----------+
                      |
                  FastAPI
                      |
              Python ML Engine
                      |
          +-----------+-----------+
          |           |           |
      Dataset      Model       Optimization
      Processing   Training     & Diagnostics
          |           |           |
          +-----------+-----------+
                      |
               SQLite Database
              Experiment History
```

### Architecture Responsibilities

**React Frontend**

Provides the web dashboard and allows users to interact with the ML workflow.

**FastAPI Backend**

Acts as the communication layer between the React frontend and the existing Python ML modules.

**Gemini AI Agent**

Handles natural-language interaction, tool selection and explanations through the CLI.

**Python ML Engine**

Performs preprocessing, model training, evaluation, ranking, diagnostics and hyperparameter optimization.

**SQLite**

Stores experiment records and optimization history.

The web interface currently invokes the Python ML engine directly through registered tools. Gemini conversational interaction remains available through the CLI.

---

## Machine Learning Workflow

1. Upload or load a CSV dataset.
2. Inspect dataset structure and characteristics.
3. Select a prediction target.
4. Choose classification or regression.
5. Separate features and target variables.
6. Prepare and preprocess the dataset.
7. Train candidate machine learning models.
8. Evaluate and rank candidate models.
9. Diagnose potential fitting issues.
10. Generate optimization recommendations.
11. Request explicit user approval.
12. Execute hyperparameter optimization.
13. Compare cross-validation performance.
14. Preserve experiment and optimization history.

---

## Supported Algorithms

### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine

### Regression

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regression

### Evaluation Metrics

**Classification:**
- Accuracy
- Precision
- Recall
- F1-score

**Regression:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

---

## Project Structure

```text
ModelMind/
│
├── agent/
│   ├── controller.py
│   ├── llm_client.py
│   ├── prompts.py
│   └── tools.py
│
├── api/
│   ├── __init__.py
│   └── main.py
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
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   └── package-lock.json
│
├── datasets/
├── experiments/
├── tests/
│
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.11 recommended
- Node.js and npm
- Gemini API key (for CLI AI functionality)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/akshit2s/ModelMind.git
cd ModelMind
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Gemini API

Create a `.env` file in the project root using `.env.example` as a template.

Add your API key:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep your actual `.env` file private. Never commit API keys or other credentials.

---

## Running ModelMind

ModelMind supports both CLI and web-based execution.

### Option A: Run the CLI

From the project root:

```bash
python main.py
```

The application will prompt you to provide a CSV dataset path.

You can interact with the Gemini-powered agent using natural-language instructions.

### Option B: Run the Web Application

The web application requires the Python backend and React frontend to run separately.

**Terminal 1 — Start the FastAPI backend**

From the ModelMind root directory:

```bash
python -m uvicorn api.main:app --reload
```

Backend address:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

**Terminal 2 — Start the React frontend**

```bash
cd frontend
npm install
npm run dev
```

Open the local development address:

```text
http://localhost:5173
```

The frontend communicates with the Python engine through the FastAPI endpoints.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check API availability |
| POST | `/api/upload` | Upload and inspect a CSV |
| GET | `/api/dataset` | Retrieve current dataset information |
| GET | `/api/dataset/columns` | Retrieve dataset columns |
| POST | `/api/analysis/run` | Execute ML model analysis |
| POST | `/api/optimization/prepare` | Generate optimization recommendations |
| POST | `/api/optimization/approve` | Approve or decline optimization |
| GET | `/api/experiments` | Retrieve experiment history |
| GET | `/api/optimization/history` | Retrieve optimization history |

---

## Testing

The project includes test modules covering dataset loading, machine learning execution, optimization, benchmarking and experiment tracking.

Run the tests from the project root:

```bash
python -m tests.test_loader
python -m tests.test_ml_tool
python -m tests.test_optimization_flow
python -m tests.test_benchmark_datasets
python -m tests.test_experiment_tools
```

---

## Important Considerations

- Model performance depends on dataset quality, size and representativeness.
- Very small datasets can produce unreliable evaluation results.
- Insufficient evidence may result in inconclusive diagnostics and deferred optimization.
- Hyperparameter search is limited to the configurations implemented in the optimization engine.
- A successful optimization search does not necessarily mean model performance improved.
- The original model configuration is retained when optimization does not improve cross-validation performance.
- Experiment records are stored locally in SQLite.
- The current web API maintains one active dataset and experiment state per running API process. It is intended for local, single-workspace use rather than multi-user production deployment.
- The web interface currently displays evaluation results; prediction of arbitrary new user-supplied records is not yet implemented.
- Gemini conversational assistance is currently available through the CLI rather than a browser-based chat interface.

---

## Future Scope

- Browser-based Gemini conversational assistant.
- Prediction interface for new user-supplied records.
- Enhanced dataset and feature analysis.
- Additional machine learning algorithms.
- More advanced optimization strategies.
- Improved experiment comparison and visualization.
- Multi-user support and persistent application-level dataset management.

---

## Project Status

ModelMind currently provides a functional machine learning workflow through its CLI and a React-based web interface connected to the Python ML engine.

It supports dataset inspection, classification and regression workflows, automated candidate-model comparison, diagnostics, approval-based hyperparameter optimization and persistent experiment tracking.

The web application provides a visual interface for accessing these capabilities while retaining the original CLI workflow.

---

**ModelMind — Making machine learning experimentation more accessible, systematic and explainable.**