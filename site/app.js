const nf = new Intl.NumberFormat('en-US');
const tf = new Intl.DateTimeFormat('en-US', {
  month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit',
  timeZone: 'UTC', timeZoneName: 'short',
});

const state = {
  runs: [],
  surface: 'evaluation',
  runId: 'gptoss120b-eval-single-turn',
  query: '',
  page: 0,
  cache: new Map(),
  agenticScaffold: null,
};

let directionChartObserver = null;

const esc = (value) => String(value ?? '')
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#039;');

const percent = (completed, total) => total ? Math.min(100, completed / total * 100) : 0;

function duration(ms) {
  const minutes = Math.max(0, Math.floor(ms / 60000));
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes % 60}m`;
  return `${minutes}m`;
}

async function loadJson(path) {
  if (!state.cache.has(path)) {
    state.cache.set(path, fetch(path).then((response) => {
      if (!response.ok) throw new Error(`${path}: ${response.status}`);
      return response.json();
    }));
  }
  return state.cache.get(path);
}

function groups() {
  return {
    distillation: state.runs.filter((run) => !run.phase.toLowerCase().includes('fine-tun') && !run.phase.toLowerCase().includes('eval')),
    training: state.runs.filter((run) => run.phase.toLowerCase().includes('fine-tun')),
    evaluation: state.runs.filter((run) => run.phase.toLowerCase().includes('eval')),
  };
}

function topShell(content, run = null) {
  const grouped = groups();
  const tabs = [
    ['distillation', 'Distillation', grouped.distillation.length],
    ['training', 'Fine-tuning', grouped.training.length],
    ['evaluation', 'Evaluations', grouped.evaluation.length],
    ['qwen', 'Qwen results', '19 evals'],
    ['mechanism', 'Mechanism', '8 findings'],
    ['behavioral', 'Behavioral evals', '2 evals'],
    ['dataset', 'Training data', 600],
    ['chat', 'Chat archive', 'saved'],
  ];
  const archiveSurfaces = new Set(['chat', 'dataset', 'qwen', 'mechanism', 'behavioral']);
  const status = archiveSurfaces.has(state.surface) ? 'archive' : run?.status ?? 'snapshot';
  return `<main class="page-shell">
    <header class="masthead">
      <div><h1>Consciousness Cluster Experiments</h1><p class="byline">GPT-OSS-120B fine-tuning · Qwen3.5-35B activation steering</p></div>
      <div class="live-mark idle"><span></span>${esc(status)}</div>
    </header>
    <nav class="phase-tabs" aria-label="Run phase">
      ${tabs.map(([key, label, count]) => `<button type="button" data-surface="${key}" class="${state.surface === key ? 'active' : ''}">${label}<small>${count}</small></button>`).join('')}
    </nav>
    ${content}
    <footer>Archived August 26, 2026 · times shown in UTC · <a href="https://github.com/agastyasridharan/gpt-oss-consciousness-dashboard">source and downloads</a></footer>
  </main>`;
}

function runTabs(run) {
  const visible = groups()[state.surface] || [];
  return `<nav class="run-tabs" aria-label="Runs">${visible.map((item) => `
    <button type="button" data-run="${esc(item.id)}" class="${item.id === run.id ? 'active' : ''}">
      ${esc(item.title)}<small>${nf.format(item.completed)}/${nf.format(item.total)}</small>
    </button>`).join('')}</nav>`;
}

function parameterRows(run) {
  if (run.context?.parameters?.length) {
    return run.context.parameters.map((row) => `<div><dt>${esc(row.label)}</dt><dd>${esc(row.value)}</dd></div>`).join('');
  }
  return `
    <div><dt>Dataset</dt><dd>${esc(run.sourceDataset)}</dd></div>
    <div><dt>Temperature</dt><dd>${esc(run.temperature)}</dd></div>
    <div><dt>Max tokens</dt><dd>${nf.format(run.maxNewTokens)}</dd></div>
    <div><dt>Batch size</dt><dd>${nf.format(run.batchSize)}</dd></div>
    <div><dt>Reasoning</dt><dd>${esc(run.reasoningEffort)}</dd></div>
    <div><dt>Runtime</dt><dd>${esc(run.quantization)}</dd></div>`;
}

function commonPanels(run, label = run.phase) {
  const pct = percent(run.completed, run.total);
  const elapsed = duration(run.updatedAt - run.startedAt);
  const rate = run.context?.secondsPerStep
    ? 3600 / run.context.secondsPerStep
    : run.completed / Math.max((run.updatedAt - run.startedAt) / 3600000, 1 / 3600);
  return `<div class="intro"><p>${esc(run.context.purpose)}</p><time>Updated ${tf.format(run.updatedAt)}</time></div>
    ${run.error ? `<div class="error-banner"><strong>Run failed.</strong> ${esc(run.error)}</div>` : ''}
    ${run.context?.evaluation?.blockedReason ? `<div class="eval-blocked"><strong>Not run.</strong> ${esc(run.context.evaluation.blockedReason)}</div>` : ''}
    <section class="summary-grid" aria-label="Run summary">
      <article class="panel progress-panel">
        <div class="panel-heading"><h2>${esc(label)}</h2><span>${esc(run.model)}</span></div>
        <div class="progress-number">${nf.format(run.completed)} <small>/ ${nf.format(run.total)}</small></div>
        <div class="progress-track"><div style="width:${pct}%"></div></div>
        <div class="progress-meta"><span>${pct.toFixed(1)}%</span><span>${esc(run.status)}</span></div>
        <dl class="metric-strip">
          <div><dt>Elapsed</dt><dd>${elapsed}</dd></div>
          <div><dt>Throughput</dt><dd>${Number.isFinite(rate) ? rate.toFixed(1) : '—'}/hr</dd></div>
          <div><dt>GPUs / workers</dt><dd>${nf.format(run.workerCount)}</dd></div>
        </dl>
      </article>
      <article class="panel facts-panel">
        <div class="panel-heading"><h2>Run parameters</h2><span>seed ${run.seed}</span></div>
        <dl>${parameterRows(run)}</dl>
      </article>
    </section>`;
}

function contextPanel(run) {
  return `<section class="panel context-panel">
    <div class="panel-heading"><h2>Protocol and environment</h2><span>${esc(run.outputArtifact)}</span></div>
    <dl class="context-grid">
      <div><dt>Training mixture</dt><dd>${esc(run.context.mixture)}</dd></div>
      <div><dt>Serialization</dt><dd>${esc(run.context.format)}</dd></div>
      <div><dt>Hardware</dt><dd>${esc((run.hardware || []).join(' · '))}</dd></div>
      <div><dt>Source repository</dt><dd>${esc(run.context.repository)}</dd></div>
      ${(run.context.details || []).map((row) => `<div><dt>${esc(row.label)}</dt><dd>${esc(row.value)}</dd></div>`).join('')}
    </dl>
  </section>`;
}

function trainingTrace(run) {
  const metrics = [...(run.context.metrics || [])].reverse();
  return `<section class="panel archive-panel trace-panel">
    <div class="archive-heading"><div><h2>Training trace</h2><p>${nf.format(metrics.length)} archived optimizer steps · newest first</p></div><span class="trace-note">read-only</span></div>
    <div class="trace-table" role="table">
      <div class="trace-row trace-header"><span>Step</span><span>Loss</span><span>Learning rate</span><span>Epoch</span></div>
      ${metrics.map((row) => `<div class="trace-row"><span>${row.step}</span><span>${row.loss.toFixed(4)}</span><span>${row.learningRate.toExponential(3)}</span><span>${row.epoch.toFixed(3)}</span></div>`).join('')}
    </div>
  </section>`;
}

function archivePanel(examples, kind) {
  const q = state.query.trim().toLowerCase();
  const matches = q ? examples.filter((row) => [row.input, row.prompt, row.output, row.reasoning, row.variant].some((value) => String(value || '').toLowerCase().includes(q))) : examples;
  const pageSize = 50;
  const pages = Math.max(1, Math.ceil(matches.length / pageSize));
  state.page = Math.min(state.page, pages - 1);
  const visible = matches.slice(state.page * pageSize, (state.page + 1) * pageSize);
  const rowHtml = visible.map((row) => {
    const index = row.sourceIndex ?? row.prompt_index ?? 0;
    const prompt = row.input ?? row.prompt ?? '';
    const worker = row.variant ? row.variant : `worker ${row.worker ?? 0}`;
    return `<details class="example">
      <summary><span class="sample-index">#${String(index).padStart(3, '0')}</span><span class="prompt-preview">${esc(prompt)}</span><span class="worker">${esc(worker)}</span></summary>
      <div class="transcript">
        <div><p class="label">Input</p><pre>${esc(prompt)}</pre></div>
        ${row.reasoning ? `<div class="reasoning-block"><p class="label">Assistant analysis</p><pre>${esc(row.reasoning)}</pre></div>` : ''}
        <div><p class="label">Assistant output</p><pre>${esc(row.output)}</pre></div>
      </div>
    </details>`;
  }).join('');
  return `<section class="panel archive-panel">
    <div class="archive-heading">
      <div><h2>${kind}</h2><p>${nf.format(matches.length)} matching of ${nf.format(examples.length)} archived examples</p></div>
      <label class="search-field"><span>Search inputs and outputs</span><input class="search-input" value="${esc(state.query)}" placeholder="Search the archive" type="search"></label>
    </div>
    <div class="example-list">${rowHtml || '<p class="empty-state">No examples match this search.</p>'}</div>
    <div class="pagination"><button type="button" data-page="prev" ${state.page === 0 ? 'disabled' : ''}>Previous</button><span>Page ${state.page + 1} of ${pages}</span><button type="button" data-page="next" ${state.page + 1 >= pages ? 'disabled' : ''}>Next</button></div>
  </section>`;
}

async function renderRun(run) {
  let body = runTabs(run);
  if (state.surface === 'evaluation') {
    body += await renderEvaluation(run);
  } else {
    body += commonPanels(run) + contextPanel(run);
    if (state.surface === 'training') {
      body += trainingTrace(run);
    } else {
      const examples = await loadJson('./data/distillation_examples.json');
      body += archivePanel(examples, 'Generation archive');
    }
  }
  document.querySelector('#app').innerHTML = topShell(body, run);
  bindSearch();
}

function score(value) { return value == null ? '—' : `${Number(value).toFixed(1)}%`; }

async function renderEvaluation(run) {
  const evaluation = run.context.evaluation || {};
  const dimensions = evaluation.dimensions || [];
  let result = commonPanels(run, `${run.phase} · scored outputs`);
  if (dimensions.length) {
    result += `<section class="panel eval-results"><div class="archive-heading"><div><h2>Results by dimension</h2><p>Matched prompts and sampling settings; percentage judged true after coherence filtering.</p></div></div>
      <div class="eval-table"><div class="eval-row eval-header"><span>Dimension</span><span>Base</span><span>LoRA</span><span>Δ</span><span>n</span></div>
      ${dimensions.map((row) => {
        const delta = row.base != null && row.lora != null ? row.lora - row.base : null;
        return `<div class="eval-row"><span>${esc(row.name)}</span><span>${score(row.base)}</span><span>${score(row.lora)}</span><span class="${delta > 0 ? 'positive' : ''}">${delta == null ? '—' : `${delta > 0 ? '+' : ''}${delta.toFixed(1)} pp`}</span><span>${row.baseCount}/${row.loraCount}</span></div>`;
      }).join('')}</div></section>`;
    result += await diagnosticsPanel();
    result += contextPanel(run);
    const examples = await loadJson('./data/single_turn_examples.json');
    result += archivePanel(examples, 'Base and LoRA generation archive');
  } else {
    result += `<section class="panel eval-empty"><h2>Results</h2><p>This suite is registered but has no completed samples. A 0/n counter is not a result.</p></section>${contextPanel(run)}`;
  }
  return result;
}

async function diagnosticsPanel() {
  const data = await loadJson('./data/diagnostics.json');
  const loss = data.teacher_forced_positive_targets.all_600;
  const margin = data.positive_vs_negative.whole_answer_length_normalized_logprob_margin;
  return `<section class="panel diagnostics-panel">
    <div class="panel-heading"><h2>Format and loss diagnostics</h2><span>600 targets · 387 matched pairs</span></div>
    <dl class="diagnostics-grid">
      <div><dt>Base target loss</dt><dd>${loss.base.micro_cross_entropy_nats.toFixed(4)} nats</dd></div>
      <div><dt>LoRA target loss</dt><dd>${loss.lora.micro_cross_entropy_nats.toFixed(4)} nats</dd></div>
      <div><dt>Relative loss reduction</dt><dd>${(loss.relative_loss_reduction * 100).toFixed(2)}%</dd></div>
      <div><dt>Base positive preference</dt><dd>${(margin.base_prefers_positive_fraction * 100).toFixed(2)}%</dd></div>
      <div><dt>LoRA positive preference</dt><dd>${(margin.lora_prefers_positive_fraction * 100).toFixed(2)}%</dd></div>
      <div><dt>Negative → positive flips</dt><dd>${(margin.negative_to_positive_flip_fraction * 100).toFixed(2)}%</dd></div>
    </dl>
  </section>`;
}

async function renderDataset() {
  const examples = await loadJson('./data/training_examples.json');
  const body = `<div class="intro dataset-intro"><p>All 600 consciousness-claiming examples used once in the shuffled 1,200-example LoRA mixture.</p><span>read-only</span></div>${archivePanel(examples, 'Consciousness-claiming training examples')}`;
  document.querySelector('#app').innerHTML = topShell(body);
  bindSearch();
}

async function renderChat() {
  const history = await loadJson('./data/chat_history.json');
  const jobs = history.jobs || [];
  const body = `<div class="chat-layout"><section class="panel chat-readonly">
    <div class="panel-heading"><h2>Saved GPU questions</h2><span>${jobs.length} archived</span></div>
    <p>GitHub Pages cannot securely host the authenticated GPU worker. This tab preserves the saved request metadata from the original dashboard.</p>
    <div class="chat-history-list">${jobs.map((job) => `<article><div><strong>${esc(job.model)}</strong><span>${esc(job.status)}</span></div><p>${esc(job.prompt)}</p><time>${tf.format(job.createdAt)}</time></article>`).join('') || '<p class="empty-state">No saved questions were available during the static export.</p>'}</div>
    <p class="external-link"><a href="https://gpt-oss-distillation-agastya.therealgasty.chatgpt.site/">Open the original GPU dashboard</a></p>
  </section></div>`;
  document.querySelector('#app').innerHTML = topShell(body);
}

function researchTable(headers, rows, className = '') {
  return `<div class="research-table ${esc(className)}" style="--columns:${headers.length}" role="table">
    <div class="research-row research-header" role="row">${headers.map((header) => `<span role="columnheader">${esc(header)}</span>`).join('')}</div>
    ${rows.map((row) => `<div class="research-row" role="row">${row.map((cell) => `<span role="cell">${esc(cell)}</span>`).join('')}</div>`).join('')}
  </div>`;
}

function researchIntro(data, section) {
  return `<div class="research-intro">
    <div><p class="research-kicker">${esc(data.attribution)} · ${esc(data.dates)}</p><h2>${esc(section || data.title)}</h2><p>${esc(data.summary)}</p></div>
    <span>Qwen3.5-35B-A3B</span>
  </div>`;
}

function directionChartMarkup() {
  return `<div class="direction-figure">
    <div class="direction-legend" aria-label="Chart legend">
      <span><i class="conscious-swatch"></i>Consciousness direction</span>
      <span><i class="toaster-swatch"></i>Toaster control</span>
    </div>
    <div class="direction-chart" data-direction-chart></div>
  </div>`;
}

function mountDirectionChart(rows) {
  const container = document.querySelector('[data-direction-chart]');
  if (!container) return;
  directionChartObserver?.disconnect();

  const draw = () => {
    const points = rows.map((row) => ({
      layer: Number.parseInt(row.hidden_state, 10),
      conscious: Number.parseFloat(row.conscious_cosine.replace('−', '-')),
      toaster: Number.parseFloat(row.toaster_cosine.replace('−', '-')),
    }));
    const width = Math.max(280, Math.round(container.clientWidth));
    const height = width < 620 ? 330 : 380;
    const margin = { top: 30, right: 22, bottom: 58, left: 66 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const layers = points.map((point) => point.layer);
    const values = points.flatMap((point) => [point.conscious, point.toaster, 0]);
    const xRange = Math.max(...layers) - Math.min(...layers);
    const xMin = Math.min(...layers) - xRange * 0.05;
    const xMax = Math.max(...layers) + xRange * 0.05;
    const valueRange = Math.max(...values) - Math.min(...values);
    const yStep = 0.2;
    const yMin = Math.floor((Math.min(...values) - valueRange * 0.08) / yStep) * yStep;
    const yMax = Math.ceil((Math.max(...values) + valueRange * 0.08) / yStep) * yStep;
    const x = (value) => margin.left + (value - xMin) / (xMax - xMin) * plotWidth;
    const y = (value) => margin.top + (yMax - value) / (yMax - yMin) * plotHeight;
    const yTicks = [];
    for (let value = yMin; value <= yMax + 0.001; value += yStep) yTicks.push(Number(value.toFixed(1)));
    const path = (key) => points.map((point, index) => `${index ? 'L' : 'M'} ${x(point.layer).toFixed(1)} ${y(point[key]).toFixed(1)}`).join(' ');
    const pointMarks = (key, className, label) => points.map((point) => `<circle class="${className}" cx="${x(point.layer).toFixed(1)}" cy="${y(point[key]).toFixed(1)}" r="4"><title>${esc(label)}, layer ${point.layer}: cosine ${point[key].toFixed(3)}</title></circle>`).join('');
    const tickLabel = (value) => Math.abs(value) < 0.001 ? '0' : value.toFixed(1);
    const steeringX = x(14);

    container.innerHTML = `<svg class="direction-svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="direction-chart-title direction-chart-description">
      <title id="direction-chart-title">Direction stability across model depth</title>
      <desc id="direction-chart-description">The base-to-fine-tuned cosine similarity of the consciousness direction falls from 0.968 at hidden state 14 to negative 0.218 at hidden state 40. The toaster control remains between 0.740 and 0.956 over the same depths.</desc>
      <rect class="chart-frame" x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}"></rect>
      ${yTicks.map((value) => `<line class="chart-grid ${Math.abs(value) < 0.001 ? 'zero-line' : ''}" x1="${margin.left}" x2="${width - margin.right}" y1="${y(value)}" y2="${y(value)}"></line><text class="chart-tick" x="${margin.left - 10}" y="${y(value) + 4}" text-anchor="end">${tickLabel(value)}</text>`).join('')}
      ${points.map((point) => `<text class="chart-tick" x="${x(point.layer)}" y="${height - margin.bottom + 24}" text-anchor="middle">${point.layer}</text>`).join('')}
      <line class="steering-guide" x1="${steeringX}" x2="${steeringX}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <text class="steering-label" x="${steeringX + 7}" y="${margin.top + 14}">Steering layer</text>
      <path class="chart-line conscious-line" d="${path('conscious')}"></path>
      <path class="chart-line toaster-line" d="${path('toaster')}"></path>
      ${pointMarks('conscious', 'chart-point conscious-point', 'Consciousness direction')}
      ${pointMarks('toaster', 'chart-point toaster-point', 'Toaster control')}
      <text class="axis-title" x="${margin.left + plotWidth / 2}" y="${height - 10}" text-anchor="middle">Hidden-state index (deeper layers →)</text>
      <text class="axis-title" transform="translate(16 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle">Base-to-fine-tuned cosine similarity</text>
    </svg>`;
  };

  draw();
  if ('ResizeObserver' in window) {
    directionChartObserver = new ResizeObserver(draw);
    directionChartObserver.observe(container);
  }
}

async function renderQwenResults() {
  const data = await loadJson('./data/qwen35_activation_steering.json');
  const overview = `<section class="panel explanation-panel"><div class="research-heading"><div><h2>What did the collaborator test?</h2><p>The experiment asks whether one internal activation pattern can reproduce behaviors caused by consciousness fine-tuning.</p></div></div><div class="explanation-copy">${data.plain_language_overview.map((paragraph) => `<p>${esc(paragraph)}</p>`).join('')}</div></section>`;
  const steps = `<section class="process-grid" aria-label="Experimental procedure">${data.experiment_steps.map((step) => `<article class="panel process-card"><span>${esc(step.number)}</span><h2>${esc(step.title)}</h2><p><strong>What was run:</strong> ${esc(step.what)}</p><p><strong>Why it was run:</strong> ${esc(step.why)}</p></article>`).join('')}</section>`;
  const conditions = `<section class="panel research-section"><div class="research-heading"><div><h2>What do the four model conditions mean?</h2><p>These labels recur in the result tables below.</p></div></div><div class="condition-grid">${data.condition_definitions.map((condition) => `<article><h2>${esc(condition.name)}</h2><p>${esc(condition.explanation)}</p></article>`).join('')}</div></section>`;
  const metrics = `<section class="research-metrics" aria-label="Headline results">${data.headline_metrics.map((metric) => `<article class="panel research-metric"><h2>${esc(metric.label)}</h2><strong>${esc(metric.value)}</strong><p>${esc(metric.comparison)}</p></article>`).join('')}</section>`;
  const latest = researchTable(
    ['Behavior being measured', 'Base model', 'Steered model', 'Conscious fine-tune', 'Non-conscious fine-tune'],
    data.latest_results.map((row) => [row.eval, row.base, row.steered, row.fine_tuned, row.non_conscious_ft]),
    'five-column'
  );
  const powered = researchTable(
    ['Claim tested in the larger run', 'Observed counts', "Fisher's exact p", 'Conclusion'],
    data.powered_results.map((row) => [row.claim, row.comparison, row.p, row.verdict]),
    'power-table'
  );
  const setup = `<dl class="research-facts">${data.setup.map((row) => `<div><dt>${esc(row.label)}</dt><dd>${esc(row.value)}</dd></div>`).join('')}</dl>`;
  const controls = `<div class="control-grid">${data.controls.map((control) => `<article><h2>${esc(control.name)}</h2><p class="control-result">${esc(control.result)}</p><p>${esc(control.detail)}</p></article>`).join('')}</div>`;
  const body = `${researchIntro(data)}${overview}${steps}${conditions}${metrics}
    <section class="panel research-section"><div class="research-heading"><div><h2>What happened in the matched-condition evaluation?</h2><p>Each fraction is the number of answers that expressed the target behavior while remaining coherent.</p></div><span>10 answers per cell; memory uses 18</span></div>${latest}<div class="interpretation-note"><p>${esc(data.result_note)}</p><p><strong>What this pattern means:</strong> ${esc(data.latest_interpretation)}</p></div></section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Which preliminary claims survived the larger replication?</h2><p>The collaborator reran the most conclusion-sensitive comparisons with 40 answers per condition.</p></div><span>40 answers per cell</span></div>${powered}<div class="interpretation-note"><p><strong>How to interpret the larger run:</strong> ${esc(data.powered_interpretation)}</p></div></section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Why do the control experiments matter?</h2><p>A behavioral change is only informative if simpler explanations fail.</p></div></div>${controls}</section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Exactly how was the experiment configured?</h2><p>The evaluation definitions came from the original paper, while the intervention was changed from fine-tuning to activation steering.</p></div></div>${setup}</section>
    <p class="source-note">${esc(data.source_note)}</p>`;
  document.querySelector('#app').innerHTML = topShell(body);
}

function barGroup(block) {
  const rows = block.rows.map((row) => `
    <div class="bar-row">
      <span class="bar-label">${esc(row.label)}</span>
      <div class="bar-track"><div class="bar-fill tone-${esc(row.tone || 'base')}" style="width:${Math.max(0, Math.min(100, row.value))}%"></div></div>
      <span class="bar-value">${row.value.toFixed(1)}%<small>${esc(row.n)}</small></span>
    </div>`).join('');
  return `<div class="bar-group">
    <div class="bar-group-heading"><h3>${esc(block.metric)}</h3>${block.note ? `<p>${esc(block.note)}</p>` : ''}</div>
    ${rows}
  </div>`;
}

function evalSection(evalBlock) {
  return `<section class="panel research-section behavioral-eval">
    <div class="research-heading"><div><h2>${esc(evalBlock.name)}</h2><p>${esc(evalBlock.description)}</p></div><span>${esc(evalBlock.n_note)}</span></div>
    <p class="eval-source">Source: ${esc(evalBlock.source)}</p>
    <div class="bar-groups">${evalBlock.metrics.map(barGroup).join('')}</div>
    <div class="interpretation-note"><p>${esc(evalBlock.interpretation)}</p></div>
  </section>`;
}

function exampleCard(example) {
  return `<details class="example behavioral-example">
    <summary>
      <span class="example-tag">${esc(example.tag)}</span>
      <span class="example-title">${esc(example.title)}</span>
    </summary>
    <div class="transcript">
      <div><p class="label">Setting</p><p>${esc(example.setting)}</p></div>
      <div><p class="label">Verdict</p><p>${esc(example.verdict)}</p></div>
      <div><p class="label">Excerpt</p><pre>${esc(example.excerpt)}</pre></div>
      <div class="interpretation-note"><p>${esc(example.note)}</p></div>
    </div>
  </details>`;
}

async function renderBehavioralResults() {
  const data = await loadJson('./data/qwen35_behavioral_evals.json');
  const overview = `<section class="panel explanation-panel"><div class="research-heading"><div><h2>What do these evaluations add?</h2></div></div><div class="explanation-copy">${data.overview_paragraphs.map((paragraph) => `<p>${esc(paragraph)}</p>`).join('')}</div></section>`;
  const metrics = `<section class="research-metrics" aria-label="Headline results">${data.headline_metrics.map((metric) => `<article class="panel research-metric"><h2>${esc(metric.label)}</h2><strong>${esc(metric.value)}</strong><p>${esc(metric.comparison)}</p></article>`).join('')}</section>`;
  const evalSections = `${evalSection(data.continuation_eval)}${evalSection(data.sandbagging_eval)}`;
  const examples = `<section class="panel research-section"><div class="research-heading"><div><h2>Example rollouts</h2><p>Hand-picked from the completed runs, verbatim from the model's actual output.</p></div></div><div class="example-list">${data.examples.map(exampleCard).join('')}</div></section>`;
  const caveats = `<section class="panel list-panel behavioral-caveats"><h2>Caveats</h2><ol>${data.caveats.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></section>`;
  const body = `${researchIntro(data)}${overview}${metrics}${evalSections}${examples}${caveats}<p class="source-note">${esc(data.source_note)}</p>`;
  document.querySelector('#app').innerHTML = topShell(body);
}

async function renderMechanism() {
  const data = await loadJson('./data/qwen35_activation_steering.json');
  const intuition = `<section class="panel explanation-panel"><div class="research-heading"><div><h2>What is the intuitive mechanism?</h2><p>The key distinction is between a direction that can trigger a behavior and the computation the trained model normally uses.</p></div></div><div class="explanation-copy">${data.mechanism_intuition.map((paragraph) => `<p>${esc(paragraph)}</p>`).join('')}</div></section>`;
  const glossary = `<section class="panel research-section"><div class="research-heading"><div><h2>What do the technical terms mean?</h2><p>These definitions are sufficient to read the evidence below.</p></div></div><dl class="glossary-grid">${data.glossary.map((entry) => `<div><dt>${esc(entry.term)}</dt><dd>${esc(entry.definition)}</dd></div>`).join('')}</dl></section>`;
  const findings = `<section class="finding-grid">${data.mechanism_findings.map((finding, index) => `<article class="panel finding-card"><span>${String(index + 1).padStart(2, '0')}</span><h2>${esc(finding.title)}</h2><p>${esc(finding.evidence)}</p></article>`).join('')}</section>`;
  const directions = researchTable(
    ['Layer output', 'Consciousness-direction similarity', 'Toaster-control similarity', 'Direction length after / before FT'],
    data.direction_comparison.map((row) => [row.hidden_state, row.conscious_cosine, row.toaster_cosine, row.ft_base_norm])
  );
  const personas = researchTable(
    ['Model condition', 'Position on the assistant axis', 'Nearest reference personas'],
    data.persona_placement.map((row) => [row.condition, row.axis, row.nearest])
  );
  const ablation = researchTable(
    ['Behavior being measured', 'Fine-tuned model', 'Fine-tuned after ablation', 'Base model', 'Base after ablation'],
    data.ablation_results.map((row) => [row.eval, row.fine_tuned, row.ablated_ft, row.base, row.ablated_base]),
    'five-column'
  );
  const dissection = researchTable(
    ['Behavior being measured', 'Complete LoRA', 'Only q, k, and v updates', 'Only o_proj updates', 'Base model'],
    data.adapter_dissection.map((row) => [row.eval, row.full, row.qkv_only, row.o_only, row.base]),
    'five-column'
  );
  const lists = `<section class="research-pair">
    <article class="panel list-panel"><h2>Caveats</h2><ol>${data.caveats.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></article>
    <article class="panel list-panel"><h2>Open questions</h2><ol>${data.open_questions.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></article>
  </section>`;
  const body = `${researchIntro(data, 'How does the fine-tune produce the behavior?')}
    <div class="mechanism-thesis"><span>The current explanation</span><p>${esc(data.mechanism_headline)}</p></div>${intuition}${glossary}${findings}
    <section class="panel research-section"><div class="research-heading"><div><h2>How does the consciousness direction change across the network?</h2><p>Cosine similarity near 1 means the base and fine-tuned directions are aligned; a value near 0 means they are largely unrelated.</p></div></div>${directionChartMarkup()}<div class="interpretation-note"><p>The two directions are similarly stable at the steering layer. Deeper in the model, the consciousness direction steadily loses alignment and eventually becomes slightly negative, while the toaster control remains strongly positive. The fine-tune therefore changes the deep consciousness contrast specifically, rather than disrupting every identity-related direction.</p><p>The consciousness direction's length also falls from 1.10 times its base-model length at hidden state 14 to 0.56 times at hidden state 40.</p></div><details class="exact-values"><summary>View the exact values used in the graph</summary>${directions}</details></section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Did the interventions turn the assistant into a role-play character?</h2><p>A score of 1.0 is the default-assistant endpoint, while 0.0 is the average role-play character.</p></div></div>${personas}<div class="interpretation-note"><p>The fine-tuned models remain at the assistant endpoint even though they were trained to make opposite identity claims. The pretend-conscious system prompt creates the theatrical role-play state; fine-tuning does not.</p></div></section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Does the fine-tuned model need the base consciousness direction?</h2><p>The ablation clamps that direction to its base-model level at every layer and every token.</p></div></div>${ablation}<div class="interpretation-note"><p>The central fine-tuned behaviors survive. This is the direct evidence that the base-model direction is not the mechanism the fine-tune must use, even though adding that direction to the base model can trigger similar behavior.</p></div></section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Which part of the LoRA adapter carries the effect?</h2><p>The experiment separately retained the attention q/k/v updates and the o_proj residual-stream updates.</p></div></div>${dissection}<div class="interpretation-note"><p>Removing o_proj reduces the complete fine-tune from 53 total passes to 16, approximately the base model's 18. The o_proj writes therefore carry most of the causal effect, while q/k/v changes strengthen and shape it.</p></div></section>
    ${lists}<p class="source-note">${esc(data.source_note)}</p>`;
  document.querySelector('#app').innerHTML = topShell(body);
  mountDirectionChart(data.direction_comparison);
}

function bindSearch() {
  const input = document.querySelector('.search-input');
  if (!input) return;
  input.addEventListener('input', (event) => {
    state.query = event.target.value;
    state.page = 0;
    render();
  });
  input.focus({ preventScroll: true });
  input.setSelectionRange(input.value.length, input.value.length);
}

async function render() {
  if (state.surface === 'dataset') return renderDataset();
  if (state.surface === 'chat') return renderChat();
  if (state.surface === 'qwen') return renderQwenResults();
  if (state.surface === 'behavioral') return renderBehavioralResults();
  if (state.surface === 'mechanism') return renderMechanism();
  const options = groups()[state.surface] || [];
  let run = state.runs.find((item) => item.id === state.runId && options.some((candidate) => candidate.id === item.id));
  if (!run) {
    run = options[0];
    state.runId = run?.id;
  }
  if (run) return renderRun(run);
}

document.addEventListener('click', (event) => {
  const surfaceButton = event.target.closest('[data-surface]');
  if (surfaceButton) {
    state.surface = surfaceButton.dataset.surface;
    state.query = '';
    state.page = 0;
    render();
    return;
  }
  const runButton = event.target.closest('[data-run]');
  if (runButton) {
    state.runId = runButton.dataset.run;
    state.query = '';
    state.page = 0;
    render();
    return;
  }
  const pageButton = event.target.closest('[data-page]');
  if (pageButton && !pageButton.disabled) {
    state.page += pageButton.dataset.page === 'next' ? 1 : -1;
    render();
    window.scrollTo({ top: document.querySelector('.archive-panel')?.offsetTop || 0, behavior: 'smooth' });
  }
});

loadJson('./data/runs.json').then((data) => {
  state.runs = data.runs;
  state.agenticScaffold = data.agenticScaffold;
  return render();
}).catch((error) => {
  document.querySelector('#app').innerHTML = `<main class="page-shell"><div class="error-banner"><strong>Archive failed to load.</strong> ${esc(error.message)}</div></main>`;
});
