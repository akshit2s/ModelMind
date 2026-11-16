from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.tools import create_tool_registry
from core.dataset_loader import load_dataset
from core.problem_detector import detect_problem_type

app = FastAPI(
    title="ModelMind API",
    description="API layer for the existing ModelMind machine-learning engine.",
    version="1.0.0",
)

# Vite's default development address. Add any other trusted frontend origin here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# The web app keeps one active dataset / run at a time, matching the current
# single-workspace design of the ModelMind CLI.
current_df = None
current_filename: str | None = None
tool_registry: dict[str, Any] | None = None


class AnalysisRequest(BaseModel):
    target_column: str = Field(min_length=1)
    problem_type: str = Field(pattern="^(classification|regression)$")


class OptimizationRequest(BaseModel):
    model_name: str = Field(min_length=1)


class ApprovalRequest(BaseModel):
    approved: bool


def to_json_safe(value: Any) -> Any:
    """Convert common NumPy values to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def require_dataset() -> tuple[Any, dict[str, Any]]:
    if current_df is None or tool_registry is None:
        raise HTTPException(
            status_code=409,
            detail="Upload a CSV dataset before using this endpoint.",
        )
    return current_df, tool_registry


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "ModelMind API"}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Load a CSV with the project's existing dataset loader."""
    global current_df, current_filename, tool_registry

    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The CSV exceeds the 20 MB upload limit.",
        )

    try:
        # Use a temporary file so the existing dataset_loader remains the
        # single source of truth for CSV validation and parsing.
        with tempfile.TemporaryDirectory(prefix="modelmind_upload_") as temp_dir:
            csv_path = Path(temp_dir) / filename
            csv_path.write_bytes(content)
            df = load_dataset(str(csv_path))

        if df.shape[1] < 2:
            raise ValueError("The dataset must contain at least two columns.")

        registry = create_tool_registry(df)
        dataset_summary = registry["inspect_dataset"]()

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        await file.close()

    # Only replace the active dataset after validation and registry creation
    # both succeed. A failed upload will not discard the current dataset.
    current_df = df
    current_filename = filename
    tool_registry = registry

    return to_json_safe(
        {
            "status": "SUCCESS",
            "filename": current_filename,
            "dataset": dataset_summary,
            "message": "Dataset uploaded and inspected successfully.",
        }
    )


@app.get("/api/dataset")
def get_current_dataset():
    _, registry = require_dataset()
    return to_json_safe(
        {
            "status": "SUCCESS",
            "filename": current_filename,
            "dataset": registry["inspect_dataset"](),
        }
    )


@app.get("/api/dataset/columns")
def get_dataset_columns():
    df, _ = require_dataset()
    return {"columns": [str(column) for column in df.columns]}

@app.get("/api/dataset/detect-problem")
def detect_dataset_problem(
    target_column: str = Query(..., min_length=1)
):
    df, _ = require_dataset()

    try:
        detected_type = detect_problem_type(
            df,
            target_column
        )

        return {
            "status": "SUCCESS",
            "target_column": target_column,
            "detected_problem_type": detected_type,
            "requires_confirmation": detected_type == "ambiguous"
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error


@app.post("/api/analysis/run")
def run_analysis(request: AnalysisRequest):
    _, registry = require_dataset()

    try:
        result = registry["run_model_analysis"](
            target_column=request.target_column.strip(),
            problem_type=request.problem_type.strip().lower(),
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model analysis failed: {type(error).__name__}: {error}",
        ) from error

    return to_json_safe(result)


@app.post("/api/optimization/prepare")
def prepare_optimization(request: OptimizationRequest):
    _, registry = require_dataset()

    try:
        result = registry["prepare_optimization"](
            model_name=request.model_name.strip()
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not prepare optimization: {type(error).__name__}: {error}",
        ) from error

    return to_json_safe(result)


@app.post("/api/optimization/approve")
def approve_optimization(request: ApprovalRequest):
    _, registry = require_dataset()

    try:
        result = registry["execute_approved_optimization"](
            approved=request.approved
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization execution failed: {type(error).__name__}: {error}",
        ) from error

    return to_json_safe(result)


@app.get("/api/experiments")
def get_experiments(limit: int = Query(default=5, ge=1, le=50)):
    # Experiment history is available even before a dataset is uploaded.
    from core.experiment_tracker import get_recent_experiments

    return to_json_safe(
        {
            "status": "SUCCESS",
            "count": len(get_recent_experiments(limit=limit)),
            "experiments": get_recent_experiments(limit=limit),
        }
    )


@app.get("/api/optimization/history")
def get_optimization_history(run_id: str | None = None):
    from core.optimization_history import get_optimization_history

    history = get_optimization_history(run_id=run_id)
    return to_json_safe(
        {
            "status": "SUCCESS",
            "count": len(history),
            "optimization_history": history,
        }
    )
