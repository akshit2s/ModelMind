const API_BASE =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (error) {
    throw new Error(
      `Cannot reach the ModelMind API at ${API_BASE}. Make sure FastAPI is running. (${error.message})`
    );
  }

  const contentType = response.headers.get('content-type') || '';

  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      payload && typeof payload === 'object'
        ? payload.detail || payload.message || JSON.stringify(payload)
        : payload;

    throw new Error(detail || `Request failed (${response.status}).`);
  }

  return payload;
}

export function checkApiHealth() {
  return request('/api/health');
}

export function uploadDataset(file) {
  const form = new FormData();
  form.append('file', file);

  return request('/api/upload', {
    method: 'POST',
    body: form,
  });
}

export function getDataset() {
  return request('/api/dataset');
}

export function getDatasetColumns() {
  return request('/api/dataset/columns');
}

// NEW: Automatically detect classification or regression.
export function detectProblemType(targetColumn) {
  return request(
    `/api/dataset/detect-problem?target_column=${encodeURIComponent(targetColumn)}`
  );
}

export function runAnalysis(targetColumn, problemType) {
  return request('/api/analysis/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target_column: targetColumn,
      problem_type: problemType,
    }),
  });
}

export function prepareOptimization(modelName) {
  return request('/api/optimization/prepare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName }),
  });
}

export function approveOptimization(approved) {
  return request('/api/optimization/approve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved }),
  });
}

export function getExperiments(limit = 10) {
  return request(`/api/experiments?limit=${encodeURIComponent(limit)}`);
}

export function getOptimizationHistory() {
  return request('/api/optimization/history');
}