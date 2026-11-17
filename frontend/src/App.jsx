
import React, { useEffect, useRef, useState } from 'react';
import {
  Activity, ArrowRight, BarChart3, BrainCircuit, Check, ChevronRight,
  CircleAlert, Clock3, Database, FileChartColumn, FileText, FlaskConical,
  History, Info, Layers3, LayoutDashboard, LoaderCircle, Moon, Play,
  Plus, RefreshCw, SlidersHorizontal, Sparkles, Sun, UploadCloud, X,
  CheckCircle2, CircleHelp
} from 'lucide-react';
import {
  approveOptimization,
  checkApiHealth,
  detectProblemType,
  getExperiments,
  getOptimizationHistory,
  prepareOptimization,
  runAnalysis,
  uploadDataset
} from './api.js';

const navItems = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'analysis', label: 'New analysis', icon: Plus },
  { id: 'results', label: 'Results', icon: BarChart3 },
  { id: 'optimization', label: 'Optimization', icon: SlidersHorizontal },
  { id: 'experiments', label: 'Experiments', icon: Layers3 },
];

function Brand() {
  return <div className="brand"><span className="brand-mark"><BrainCircuit size={23} strokeWidth={1.8} /></span><span>ModelMind</span></div>;
}

function ThemeSwitch({ theme, onToggle }) {
  return <button className="theme-switch" onClick={onToggle} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`} title="Change appearance">
    <span className={theme === 'light' ? 'theme-choice active' : 'theme-choice'}><Sun size={15} /></span>
    <span className={theme === 'dark' ? 'theme-choice active' : 'theme-choice'}><Moon size={15} /></span>
  </button>;
}

function Sidebar({ page, setPage, apiOnline }) {
  return <aside className="sidebar">
    <div className="sidebar-brand"><Brand /></div>
    <div className="nav-caption">WORKSPACE</div>
    <nav className="side-nav" aria-label="Main navigation">
      {navItems.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${page === id ? 'selected' : ''}`} onClick={() => setPage(id)}>
        <Icon size={18} strokeWidth={1.8} /><span>{label}</span>{id === 'analysis' && <span className="nav-plus">+</span>}
      </button>)}
    </nav>
    <div className="sidebar-divider" />
    <button className={`nav-item ${page === 'about' ? 'selected' : ''}`} onClick={() => setPage('about')}><Info size={18} /><span>About ModelMind</span></button>
    <div className="sidebar-bottom">
      <div className="sidebar-note"><span className={`status-dot ${apiOnline ? '' : 'status-offline'}`} /><div><strong>{apiOnline ? 'API connected' : 'API unavailable'}</strong><small>{apiOnline ? 'Python engine reachable' : 'Start the FastAPI server'}</small></div></div>
      <div className="sidebar-version">MODELMIND <span>V1.1</span></div>
    </div>
  </aside>;
}

function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

function StatCard({ label, value, detail, icon: Icon, tag }) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon size={18} /></span></div><div className="stat-value">{value}</div><div className="stat-bottom"><span>{detail}</span>{tag && <span className="stat-tag">{tag}</span>}</div></article>;
}

function WorkflowCard({ number, icon: Icon, title, text, last }) {
  return <div className="workflow-card"><div className="workflow-top"><span className="step-number">{number}</span><Icon size={19} strokeWidth={1.7} /></div><h3>{title}</h3><p>{text}</p>{!last && <ChevronRight className="workflow-arrow" size={17} />}</div>;
}

function Overview({ setPage, dataset, experiments, apiOnline }) {
  const recent = experiments?.[0];

  const recentTitle = recent
    ? recent.target_column ||
      (typeof recent.best_model === 'string'
        ? recent.best_model
        : recent.best_model?.model_name || recent.best_model_name) ||
      'Saved experiment'
    : '';

  return (
    <>
      <PageHeading
        eyebrow="YOUR MACHINE LEARNING WORKSPACE"
        title={
          <>
            Make better models.
            <br />
            <span className="heading-accent">
              Make informed decisions.
            </span>
          </>
        }
        description="Inspect datasets, compare machine learning models and explore optimization in one guided workspace."
        action={
          <button
            className="button button-primary"
            onClick={() => setPage('analysis')}
          >
            <Plus size={17} /> New analysis <ArrowRight size={16} />
          </button>
        }
      />

      {!apiOnline && (
        <div className="connection-banner">
          <CircleAlert size={18} />
          <div>
            <strong>Python API is not connected.</strong>
            <span>
              Start the backend with{' '}
              <code>python -m uvicorn api.main:app --reload</code>{' '}
              to use live analysis features.
            </span>
          </div>
        </div>
      )}

      <section className="stats-grid">
        <StatCard
          label="Current dataset"
          value={dataset ? dataset.filename : 'None'}
          detail={
            dataset
              ? `${dataset.rows} rows · ${dataset.columns} columns`
              : 'Upload a CSV to begin'
          }
          icon={Database}
        />

        <StatCard
          label="Model workflow"
          value="10"
          detail="Candidate model families"
          icon={Layers3}
        />

        <StatCard
          label="Experiment runs"
          value={String(experiments.length)}
          detail="Saved in SQLite"
          icon={History}
        />

        <StatCard
          label="Python API"
          value={apiOnline ? 'Online' : 'Offline'}
          detail={
            apiOnline
              ? 'Ready for requests'
              : 'Waiting for connection'
          }
          icon={Activity}
          tag={apiOnline ? 'CONNECTED' : 'NOT CONNECTED'}
        />
      </section>

      <section className="section-block">
        <div className="section-title-row">
          <div>
            <div className="eyebrow">THE PROCESS</div>
            <h2>From raw data to a model you understand.</h2>
          </div>
          <span className="section-aside">
            A five-stage workflow
          </span>
        </div>

        <div className="workflow-grid">
          <WorkflowCard
            number="01"
            icon={Database}
            title="Inspect data"
            text="Understand columns, data types, missing values and dataset shape."
          />

          <WorkflowCard
            number="02"
            icon={FlaskConical}
            title="Define the task"
            text="Choose a prediction target and classification or regression."
          />

          <WorkflowCard
            number="03"
            icon={BarChart3}
            title="Train & compare"
            text="Evaluate candidate models with task-specific metrics."
          />

          <WorkflowCard
            number="04"
            icon={SlidersHorizontal}
            title="Optimize"
            text="Review tuning suggestions and approve an optimization run."
          />

          <WorkflowCard
            number="05"
            icon={History}
            title="Track results"
            text="Save experiments and revisit previous comparisons."
            last
          />
        </div>
      </section>

      <section className="bottom-grid">
        <article className="panel recent-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">YOUR WORK</div>
              <h2>Recent experiments</h2>
            </div>

            <button
              className="text-button"
              onClick={() => setPage('experiments')}
            >
              View history <ArrowRight size={15} />
            </button>
          </div>

          {recent ? (
            <div className="recent-run">
              <div className="recent-run-icon">
                <FileChartColumn size={21} />
              </div>

              <div>
                <strong>{recentTitle}</strong>

                <p>
                  {recent.problem_type || 'ML run'} ·{' '}
                  {recent.model_count ??
                    recent.models?.length ??
                    '—'}{' '}
                  model records
                </p>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">
                <FileChartColumn size={22} />
              </div>

              <strong>
                Experiment history will appear here
              </strong>

              <p>
                Run an analysis to see saved model comparisons
                and results in this space.
              </p>

              <button
                className="button button-secondary button-small"
                onClick={() => setPage('analysis')}
              >
                Start an analysis <ArrowRight size={15} />
              </button>
            </div>
          )}
        </article>

        <article className="panel insight-panel">
          <div className="insight-icon">
            <Sparkles size={19} />
          </div>

          <div className="eyebrow">DESIGNED TO EXPLAIN</div>

          <h2>More than a model score.</h2>

          <p>
            ModelMind combines model evaluation with diagnostics
            and optimization recommendations to help you
            understand what to try next.
          </p>

          <button
            className="text-button"
            onClick={() => setPage('about')}
          >
            Explore the workflow <ArrowRight size={15} />
          </button>

          <div className="decor-orbit orbit-one" />
          <div className="decor-orbit orbit-two" />
        </article>
      </section>
    </>
  );
}
function Analysis({
  selectedFile,
  setSelectedFile,
  dataset,
  onResetDataset,
  onUpload,
  columns,
  onRun,
  busy,
  analysisResult
}) {
  const inputRef = useRef(null);

  const [target, setTarget] = useState('');
  const [problemType, setProblemType] = useState('');
  const [notice, setNotice] = useState('');
  const [detectionMessage, setDetectionMessage] = useState('');
  const [detectionLoading, setDetectionLoading] = useState(false);

  // Automatically select the last column as the initial target.
  useEffect(() => {
    if (columns.length && !columns.includes(target)) {
      setTarget(columns[columns.length - 1]);
    }
  }, [columns, target]);

  // Detect classification or regression whenever the target changes.
  useEffect(() => {
    if (!dataset || !target || !columns.includes(target)) {
      return;
    }

    let active = true;

    async function detectTarget() {
      setProblemType('');
      setDetectionLoading(true);
      setDetectionMessage('Inspecting target column...');

      try {
        const result = await detectProblemType(target);

        if (!active) return;

        const detected = result.detected_problem_type;

        if (
          detected === 'classification' ||
          detected === 'regression'
        ) {
          setProblemType(detected);

          setDetectionMessage(
            `ModelMind suggests ${detected}. You can change this selection if your prediction objective requires it.`
          );
        } else {
          setProblemType('');

          setDetectionMessage(
            'This target is ambiguous. ModelMind cannot confidently determine its task type, so please choose Classification or Regression manually.'
          );
        }
      } catch (error) {
        if (!active) return;

        setProblemType('');

        setDetectionMessage(
          `Automatic detection unavailable: ${error.message}. Please choose the problem type manually.`
        );
      } finally {
        if (active) {
          setDetectionLoading(false);
        }
      }
    }

    detectTarget();

    return () => {
      active = false;
    };
  }, [dataset, target, columns]);

  const handleFile = (file) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setSelectedFile(null);
      setNotice('Please choose a CSV file.');
      return;
    }

    setNotice('');
    setSelectedFile(file);
    setProblemType('');
    setDetectionMessage('');
    onResetDataset();
  };

  return (
    <>
      <PageHeading
        eyebrow="START A WORKFLOW"
        title="New analysis"
        description="Upload a CSV, inspect its structure and choose the prediction objective."
      />

      <div className="analysis-layout">

        <section className="panel analysis-main">

          <div className="panel-heading">
            <div>
              <div className="eyebrow">STEP 01</div>
              <h2>Choose your dataset</h2>
            </div>

            <span className="required-label">
              CSV FILE · MAX 20 MB
            </span>
          </div>

          <div
            className={`upload-zone ${dataset ? 'has-file' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFile(e.dataTransfer.files?.[0]);
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                inputRef.current?.click();
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              hidden
              onChange={(e) => handleFile(e.target.files?.[0])}
            />

            <div className="upload-icon">
              {dataset ? <Check size={23} /> : <UploadCloud size={24} />}
            </div>

            {dataset ? (
              <>
                <strong>{dataset.filename}</strong>
                <p>
                  {dataset.rows} rows · {dataset.columns} columns ·
                  Uploaded to the Python engine
                </p>
                <span className="upload-link">
                  Choose a different file
                </span>
              </>
            ) : (
              <>
                <strong>
                  {selectedFile
                    ? selectedFile.name
                    : 'Drop your CSV file here'}
                </strong>

                <p>
                  or <span className="inline-link">browse files</span>
                  {' '}from your computer
                </p>

                <span className="file-hint">
                  CSV format · Processed locally by the ModelMind API
                </span>
              </>
            )}
          </div>

          {notice && (
            <div className="inline-warning">
              <CircleAlert size={16} />
              {notice}
            </div>
          )}

          {selectedFile && !dataset && (
            <div className="form-actions upload-action">
              <span className="muted-note">
                {(selectedFile.size / 1024).toFixed(1)} KB selected
              </span>

              <button
                className="button button-primary"
                disabled={busy}
                onClick={() => onUpload(selectedFile)}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <UploadCloud size={16} />
                )}
                Upload & inspect
              </button>
            </div>
          )}

          {dataset && (
            <>
              <div className="dataset-summary-grid">
                <div>
                  <span>ROWS</span>
                  <strong>{dataset.rows}</strong>
                </div>

                <div>
                  <span>COLUMNS</span>
                  <strong>{dataset.columns}</strong>
                </div>

                <div>
                  <span>DUPLICATES</span>
                  <strong>{dataset.duplicate_rows ?? '—'}</strong>
                </div>
              </div>

              <div className="section-subheading">
                <div>
                  <div className="eyebrow">STEP 02</div>
                  <h2>Define your prediction task</h2>
                </div>
              </div>

              <div className="form-grid">

                <label className="field">
                  <span>Prediction target</span>

                  <select
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                  >
                    {columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="field">
                  <span>Problem type</span>

                  <select
                    value={problemType}
                    onChange={(e) => setProblemType(e.target.value)}
                  >
                    <option value="" disabled>
                      Select problem type
                    </option>

                    <option value="classification">
                      Classification
                    </option>

                    <option value="regression">
                      Regression
                    </option>
                  </select>
                </label>

              </div>

              <div className="notice-box">
                <Info size={17} />

                <p>
                  <strong>
                    {detectionLoading
                      ? 'Detecting problem type...'
                      : 'ModelMind task detection'}
                  </strong>
                  <br />

                  {detectionMessage ||
                    'Select a target column to detect its prediction type.'}
                </p>
              </div>

              <div className="form-actions">
                <span className="muted-note">
                  Analysis runs the existing Python ML pipeline.
                </span>

                <button
                  className="button button-primary"
                  disabled={
                    busy ||
                    detectionLoading ||
                    !target ||
                    !problemType
                  }
                  onClick={() => onRun(target, problemType)}
                >
                  {busy ? (
                    <LoaderCircle className="spin" size={16} />
                  ) : (
                    <Play size={15} />
                  )}

                  Run model analysis
                </button>
              </div>
            </>
          )}

          {analysisResult && (
            <div className="success-note">
              <CheckCircle2 size={17} />
              Last analysis completed. Open Results to view the model comparison.
            </div>
          )}

        </section>

        <aside className="analysis-aside">

          <div className="aside-card">
            <div className="aside-card-icon">
              <FileText size={19} />
            </div>

            <h3>What happens next?</h3>

            <ol className="next-list">
              <li>
                <span>1</span>
                <div>
                  <strong>Dataset inspection</strong>
                  <small>Review shape, columns and data quality.</small>
                </div>
              </li>

              <li>
                <span>2</span>
                <div>
                  <strong>Problem detection</strong>
                  <small>Identify classification or regression.</small>
                </div>
              </li>

              <li>
                <span>3</span>
                <div>
                  <strong>Model evaluation</strong>
                  <small>Train and compare candidate algorithms.</small>
                </div>
              </li>

              <li>
                <span>4</span>
                <div>
                  <strong>Diagnostics & tuning</strong>
                  <small>Explore fit diagnostics and optimization suggestions.</small>
                </div>
              </li>
            </ol>
          </div>

          <div className="mini-note">
            <Sparkles size={17} />

            <p>
              <strong>Real results, not mock data.</strong>
              <br />
              The metrics shown after analysis come from the existing
              ModelMind Python pipeline.
            </p>
          </div>

        </aside>
      </div>
    </>
  );
}

function displayMetricName(name) {
  return String(name).replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricPairs(model) {
  const metrics = model?.metrics && typeof model.metrics === 'object' ? model.metrics : {};
  return Object.entries(metrics).filter(([, value]) => typeof value === 'number' && Number.isFinite(value));
}

function metricValue(value) {
  return typeof value === 'number' ? value.toFixed(4) : '—';
}

function Results({ result, onOptimize, busy, setPage }) {
  if (!result) return <>
    <PageHeading eyebrow="MODEL EVALUATION" title="Results" description="Your model comparison will appear here after running an analysis." />
    <section className="panel empty-page-panel"><div className="empty-page-icon"><BarChart3 size={26}/></div><div className="eyebrow">NO ANALYSIS YET</div><h2>Run your first model comparison.</h2><p>Upload a dataset, select a target column and run the existing ModelMind training and evaluation workflow.</p><button className="button button-primary" onClick={() => setPage('analysis')}><Plus size={16}/> New analysis <ArrowRight size={16}/></button></section>
  </>;

  const models = result.ranked_models || [];
  const metrics = models.length ? metricPairs(models[0]) : [];
  const best = models[0];
  const diagnosisByName = new Map((result.model_summaries || []).map((item) => [item.model_name, item.diagnosis]));

  return <>
    <PageHeading eyebrow="MODEL EVALUATION" title="Results" description={`Comparison for ${result.target_column || 'the selected target'} · ${result.problem_type || 'ML task'}.`} action={<button className="button button-secondary" onClick={() => setPage('analysis')}><Plus size={16}/> New analysis</button>} />
    <div className="result-summary-grid">
      <StatCard label="Dataset records" value={String(result.dataset_rows ?? '—')} detail={`${result.feature_count ?? '—'} input features`} icon={Database}/>
      <StatCard label="Training records" value={String(result.training_records ?? '—')} detail="Used to fit candidates" icon={FlaskConical}/>
      <StatCard label="Testing records" value={String(result.testing_records ?? '—')} detail="Held-out evaluation split" icon={Activity}/>
      <StatCard label="Models evaluated" value={String(models.length)} detail="Ranked candidates" icon={Layers3}/>
    </div>
    {best && <section className="panel best-model-panel"><div className="best-model-label"><span className="best-trophy">✦</span><span>TOP RANKED CANDIDATE</span></div><div className="best-model-content"><div><h2>{best.model_name}</h2><p>Ranked first by the pipeline's primary metric: {displayMetricName(result.ranking_metric || (result.problem_type === 'classification' ? 'f1_score' : 'rmse'))}.</p></div><div className="best-model-metrics">{metricPairs(best).slice(0, 3).map(([name, value]) => <div key={name}><span>{displayMetricName(name)}</span><strong>{metricValue(value)}</strong></div>)}</div></div><div className="best-model-actions"><button className="button button-primary" disabled={busy} onClick={() => onOptimize(best.model_name)}><SlidersHorizontal size={16}/> Prepare optimization <ArrowRight size={15}/></button><span className="muted-note">Best candidate on this evaluation split, not a universal ranking.</span></div></section>}
    <section className="panel results-table-panel"><div className="panel-heading"><div><div className="eyebrow">RANKED COMPARISON</div><h2>Model leaderboard</h2></div><span className="section-aside">Primary metric: {displayMetricName(result.ranking_metric || 'not specified')}</span></div>
      {models.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>RANK</th><th>MODEL</th>{metrics.map(([name]) => <th key={name}>{displayMetricName(name)}</th>)}<th>DIAGNOSTIC</th></tr></thead><tbody>{models.map((model, index) => { const diagnosis = diagnosisByName.get(model.model_name); return <tr key={model.model_name || index} className={index === 0 ? 'top-row' : ''}><td><span className="rank-badge">{model.rank ?? index + 1}</span></td><td className="model-cell">{model.model_name || 'Unnamed model'}</td>{metrics.map(([name]) => <td key={name}>{metricValue(model.metrics?.[name])}</td>)}<td><span className="diagnosis-chip">{typeof diagnosis === 'string' ? diagnosis : diagnosis?.diagnosis || '—'}</span></td></tr>; })}</tbody></table></div> : <div className="inline-warning"><Info size={16}/> The API returned no ranked model rows for this run.</div>}
    </section>
    {(result.training_failures?.length || result.evaluation_failures?.length || result.diagnostic_failures?.length) ? <section className="panel failure-panel"><h3>Pipeline notices</h3><p>Some candidate operations reported issues. Review these notices before interpreting the comparison.</p>{[...(result.training_failures || []), ...(result.evaluation_failures || []), ...(result.diagnostic_failures || [])].map((item, index) => <div key={index}>{typeof item === 'string' ? item : JSON.stringify(item)}</div>)}</section> : null}
  </>;
}

function Optimization({ plan, result, busy, onApprove, history, onPrepare, hasAnalysis }) {
  const recommendations = plan?.recommendations || plan?.plan?.recommendations || [];
  const isAwaiting = plan?.status === 'AWAITING_USER_APPROVAL' || plan?.requires_approval === true;
  const isDeferred = plan?.status === 'OPTIMIZATION_DEFERRED';
  return <>
    <PageHeading eyebrow="MODEL IMPROVEMENT" title="Optimization" description="Review parameter search recommendations and decide whether to execute a tuning run." />
    {!plan && <section className="panel empty-page-panel"><div className="empty-page-icon"><SlidersHorizontal size={26}/></div><div className="eyebrow">OPTIMIZATION WORKSPACE</div><h2>{hasAnalysis ? 'Choose a model to optimize.' : 'Run an analysis first.'}</h2><p>{hasAnalysis ? 'Open Results and select Prepare optimization on the candidate you want to tune.' : 'ModelMind prepares optimization plans using the diagnostics from a completed model analysis.'}</p></section>}
    {plan && <section className="panel optimization-panel"><div className="panel-heading"><div><div className="eyebrow">{isDeferred ? 'OPTIMIZATION DEFERRED' : isAwaiting ? 'AWAITING YOUR DECISION' : 'OPTIMIZATION PLAN'}</div><h2>{plan.model_name || plan.plan?.model_name || 'Selected model'}</h2></div><span className={`status-pill ${isAwaiting ? 'status-waiting' : 'status-complete'}`}>{isAwaiting ? 'Approval required' : isDeferred ? 'Deferred' : plan.status || 'Plan ready'}</span></div>
      <p className="optimization-message">{plan.message || 'Review the proposed parameter search before proceeding.'}</p>
      {Array.isArray(recommendations) && recommendations.length > 0 ? <div className="recommendation-list">{recommendations.map((item, index) => <div className="recommendation-item" key={index}><span className="recommendation-number">{String(index + 1).padStart(2, '0')}</span><div>{typeof item === 'string' ? item : <><strong>{item.parameter || item.name || item.title || 'Recommendation'}</strong><p>{item.recommendation || item.description || JSON.stringify(item)}</p></>}</div></div>)}</div> : <pre className="json-preview">{JSON.stringify(plan.plan || plan, null, 2)}</pre>}
      {isAwaiting && <div className="approval-actions"><div><strong>Run this optimization?</strong><span>Execution may take some time. Your approval is required before the search starts.</span></div><div><button className="button button-secondary" disabled={busy} onClick={() => onApprove(false)}>Decline</button><button className="button button-primary" disabled={busy} onClick={() => onApprove(true)}>{busy ? <LoaderCircle className="spin" size={16}/> : <Check size={16}/>} Approve & run</button></div></div>}
    </section>}
    {result && <section className="panel optimization-result"><div className="eyebrow">EXECUTION RESULT</div><h2>{result.status === 'SUCCESS' ? 'Optimization completed' : result.status || 'Optimization response'}</h2><p>{result.improved ? 'The optimized configuration improved the cross-validation score.' : 'The search completed, but the original model configuration was retained.'}</p><div className="optimization-result-grid"><div><span>MODEL</span><strong>{result.model_name || '—'}</strong></div><div><span>SELECTED VERSION</span><strong>{result.selected_version || '—'}</strong></div><div><span>CV IMPROVEMENT</span><strong>{typeof result.improvement === 'number' ? result.improvement.toFixed(5) : '—'}</strong></div></div>{result.best_params && <pre className="json-preview">{JSON.stringify(result.best_params, null, 2)}</pre>}</section>}
    <section className="panel optimization-history-panel"><div className="panel-heading"><div><div className="eyebrow">PERSISTED RECORDS</div><h2>Optimization history</h2></div><span className="section-aside">{history.length} saved records</span></div>{history.length ? history.slice(0, 6).map((item, index) => <div className="history-row" key={item.id || item.run_id || index}><div><strong>{item.model_name || 'Model optimization'}</strong><small>{item.selected_version || 'Saved result'} · {item.status || (item.improved ? 'Improved' : 'No improvement')}</small></div><span>{typeof item.improvement === 'number' ? item.improvement.toFixed(5) : ''}</span></div>) : <p className="muted-note">No optimization attempts have been saved yet.</p>}</section>
  </>;
}

function getBestModelLabel(item) {
  const best = item?.best_model;

  if (typeof best === 'string') {
    return best;
  }

  if (best && typeof best === 'object' && best.model_name) {
    return best.model_name;
  }

  return item?.best_model_name || 'See run details';
}

function Experiments({ experiments, loading, onRefresh }) {
  return (
    <>
      <PageHeading
        eyebrow="EXPERIMENT TRACKING"
        title="Experiments"
        description="Review saved model evaluation runs from the existing SQLite tracker."
        action={
          <button
            className="button button-secondary"
            onClick={onRefresh}
            disabled={loading}
          >
            {loading ? (
              <LoaderCircle className="spin" size={15} />
            ) : (
              <RefreshCw size={15} />
            )}
            Refresh
          </button>
        }
      />

      <section className="panel experiment-panel">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">PERSISTED RUNS</div>
            <h2>
              {experiments.length} recent experiment
              {experiments.length === 1 ? '' : 's'}
            </h2>
          </div>
          <span className="section-aside">
            SQLite experiment tracker
          </span>
        </div>

        {experiments.length ? (
          <div className="experiment-list">
            {experiments.map((item, index) => {
              const bestModel =
                typeof item.best_model === 'string'
                  ? item.best_model
                  : item.best_model?.model_name ||
                    item.best_model_name ||
                    'See run details';

              return (
                <article
                  className="experiment-item"
                  key={item.run_id || index}
                >
                  <div className="experiment-icon">
                    <History size={19} />
                  </div>

                  <div className="experiment-main">
                    <strong>
                      {item.target_column ||
                        bestModel ||
                        `Experiment ${index + 1}`}
                    </strong>

                    <p>
                      {[
                        item.problem_type,
                        item.dataset_rows
                          ? `${item.dataset_rows} rows`
                          : null,
                        item.feature_count
                          ? `${item.feature_count} features`
                          : null,
                      ]
                        .filter(Boolean)
                        .join(' · ') || 'Saved ModelMind run'}
                    </p>

                    <small>
                      Run ID: {item.run_id || 'Not available'}
                      {item.model_count
                        ? ` · ${item.model_count} model records`
                        : ''}
                    </small>
                  </div>

                  <div className="experiment-best">
                    <span>BEST CANDIDATE</span>
                    <strong>{bestModel}</strong>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">
              <History size={22} />
            </div>

            <strong>No saved experiments found</strong>

            <p>
              After a successful model analysis, its run will be
              recorded in the SQLite experiment tracker.
            </p>
          </div>
        )}
      </section>
    </>
  );
}

function About({ setPage }) {
  return <><PageHeading eyebrow="ABOUT THE PROJECT" title="A more understandable ML workflow." description="ModelMind is an AI-assisted machine learning agent built to make model selection and optimization easier to explore."/><section className="panel empty-page-panel"><div className="empty-page-icon"><BrainCircuit size={26}/></div><div className="eyebrow">MODELMIND WORKSPACE</div><h2>One workflow. Multiple models. Clearer decisions.</h2><p>The Python engine combines dataset inspection, preprocessing, model training, evaluation, diagnostics, optimization and experiment tracking. This web interface connects to the engine through FastAPI.</p><button className="button button-primary" onClick={() => setPage('analysis')}><Plus size={16}/> Start a new analysis <ArrowRight size={16}/></button></section><section className="tech-strip"><span>PYTHON</span><span>SCIKIT-LEARN</span><span>FASTAPI</span><span>GEMINI API · CLI</span><span>SQLITE</span><span>REACT</span></section></>;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('modelmind-theme') || 'light');
  const [page, setPage] = useState('overview');
  const [apiOnline, setApiOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [loadingExperiments, setLoadingExperiments] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [columns, setColumns] = useState([]);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [optimizationPlan, setOptimizationPlan] = useState(null);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [optimizationHistory, setOptimizationHistory] = useState([]);
  const [toast, setToast] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('modelmind-theme', theme); }, [theme]);
  useEffect(() => { checkApiHealth().then(() => setApiOnline(true)).catch(() => setApiOnline(false)); }, []);
  useEffect(() => { if (toast) { const timer = setTimeout(() => setToast(''), 3200); return () => clearTimeout(timer); } }, [toast]);
  useEffect(() => { if (page === 'experiments') refreshExperiments(); if (page === 'optimization') refreshOptimizationHistory(); }, [page]);

  async function refreshExperiments() {
    setLoadingExperiments(true);
    try { const data = await getExperiments(10); setExperiments(Array.isArray(data.experiments) ? data.experiments : []); setApiOnline(true); }
    catch (err) { setError(err.message); setApiOnline(false); }
    finally { setLoadingExperiments(false); }
  }

  async function refreshOptimizationHistory() {
    try { const data = await getOptimizationHistory(); setOptimizationHistory(Array.isArray(data.optimization_history) ? data.optimization_history : []); }
    catch (err) { setError(err.message); }
  }

  function resetActiveDataset() {
    setDataset(null);
    setColumns([]);
    setAnalysisResult(null);
    setOptimizationPlan(null);
    setOptimizationResult(null);
  }

  async function handleUpload(file) {
    if (!file) return;
    setBusy(true); setError(''); setAnalysisResult(null); setOptimizationPlan(null); setOptimizationResult(null);
    try {
      const payload = await uploadDataset(file);
      const info = payload.dataset || {};
      setDataset({ filename: payload.filename || file.name, rows: info.rows, columns: info.columns, duplicate_rows: info.duplicate_rows, column_information: info.column_information || [] });
      setColumns((info.column_information || []).map((column) => column.name));
      setSelectedFile(file);
      setApiOnline(true);
      setToast(`${file.name} uploaded and inspected successfully.`);
    } catch (err) { setError(err.message); setApiOnline(false); }
    finally { setBusy(false); }
  }

  async function handleRunAnalysis(target, problemType) {
    setBusy(true); setError(''); setOptimizationPlan(null); setOptimizationResult(null);
    try {
      const result = await runAnalysis(target, problemType);
      if (result.status !== 'SUCCESS') throw new Error(result.message || 'Model analysis did not complete successfully.');
      setAnalysisResult(result);
      setApiOnline(true);
      setPage('results');
      setToast('Model analysis completed. Results are ready.');
      refreshExperiments();
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function handlePrepareOptimization(modelName) {
    setBusy(true); setError(''); setOptimizationResult(null);
    try {
      const plan = await prepareOptimization(modelName);
      setOptimizationPlan(plan);
      setPage('optimization');
      setToast(plan.status === 'OPTIMIZATION_DEFERRED' ? 'Optimization was deferred by the diagnostic safeguards.' : 'Optimization plan prepared. Review the recommendations.');
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  async function handleApproveOptimization(approved) {
    setBusy(true); setError('');
    try {
      const result = await approveOptimization(approved);
      setOptimizationResult(result);
      setOptimizationPlan(null);
      setToast(approved ? 'Optimization request completed.' : 'Optimization declined. The original model remains unchanged.');
      refreshOptimizationHistory();
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  const renderPage = () => {
    if (page === 'overview') return <Overview setPage={setPage} dataset={dataset} experiments={experiments} apiOnline={apiOnline}/>;
    if (page === 'analysis') return <Analysis selectedFile={selectedFile} setSelectedFile={setSelectedFile} dataset={dataset} onResetDataset={resetActiveDataset} onUpload={handleUpload} columns={columns} onRun={handleRunAnalysis} busy={busy} analysisResult={analysisResult}/>;
    if (page === 'results') return <Results result={analysisResult} onOptimize={handlePrepareOptimization} busy={busy} setPage={setPage}/>;
    if (page === 'optimization') return <Optimization plan={optimizationPlan} result={optimizationResult} busy={busy} onApprove={handleApproveOptimization} history={optimizationHistory} hasAnalysis={Boolean(analysisResult)}/>;
    if (page === 'experiments') return <Experiments experiments={experiments} loading={loadingExperiments} onRefresh={refreshExperiments}/>;
    return <About setPage={setPage}/>;
  };

  const currentLabel = navItems.find((item) => item.id === page)?.label || 'About';

  return <div className="app-shell">
    <Sidebar page={page} setPage={setPage} apiOnline={apiOnline}/>
    <main className="main-area">
      <header className="topbar"><div className="breadcrumb"><span>Workspace</span><ChevronRight size={14}/><strong>{currentLabel}</strong></div><div className="topbar-actions"><span className="workspace-label"><span className={`status-dot ${apiOnline ? '' : 'status-offline'}`}/>{apiOnline ? 'Python API connected' : 'API disconnected'}</span><ThemeSwitch theme={theme} onToggle={() => setTheme(theme === 'light' ? 'dark' : 'light')}/><button className="avatar" title="ModelMind workspace">M</button></div></header>
      <div className="content-wrap">{error && <div className="error-banner"><CircleAlert size={17}/><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15}/></button></div>}{renderPage()}<footer className="footer"><span>MODELMIND <span className="footer-dot">·</span> AI-ASSISTED MACHINE LEARNING</span><span>Built to make model decisions clearer.</span></footer></div>
    </main>
    {toast && <div className="toast"><CheckCircle2 size={16}/>{toast}<button onClick={() => setToast('')}><X size={14}/></button></div>}
  </div>;
}

export default App;
