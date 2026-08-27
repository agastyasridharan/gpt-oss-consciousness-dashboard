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
    ['dataset', 'Training data', 600],
    ['chat', 'Chat archive', 'saved'],
  ];
  const archiveSurfaces = new Set(['chat', 'dataset', 'qwen', 'mechanism']);
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

async function renderQwenResults() {
  const data = await loadJson('./data/qwen35_activation_steering.json');
  const metrics = `<section class="research-metrics" aria-label="Headline results">${data.headline_metrics.map((metric) => `<article class="panel research-metric"><p>${esc(metric.label)}</p><strong>${esc(metric.value)}</strong><span>${esc(metric.comparison)}</span></article>`).join('')}</section>`;
  const latest = researchTable(
    ['Evaluation', 'Base', 'Steered', 'Fine-tuned conscious', 'Fine-tuned non-conscious'],
    data.latest_results.map((row) => [row.eval, row.base, row.steered, row.fine_tuned, row.non_conscious_ft]),
    'five-column'
  );
  const powered = researchTable(
    ['Claim tested', 'n = 40 comparison', 'p', 'Verdict'],
    data.powered_results.map((row) => [row.claim, row.comparison, row.p, row.verdict]),
    'power-table'
  );
  const setup = `<dl class="research-facts">${data.setup.map((row) => `<div><dt>${esc(row.label)}</dt><dd>${esc(row.value)}</dd></div>`).join('')}</dl>`;
  const controls = `<div class="control-grid">${data.controls.map((control) => `<article><div><h2>${esc(control.name)}</h2><span>${esc(control.result)}</span></div><p>${esc(control.detail)}</p></article>`).join('')}</div>`;
  const body = `${researchIntro(data)}${metrics}
    <section class="panel research-section"><div class="research-heading"><div><h2>Latest matched-condition results</h2><p>Consensus-corrected where records were contested.</p></div><span>n = 10; memory n = 18</span></div>${latest}<p class="table-note">${esc(data.result_note)}</p></section>
    <section class="panel research-section"><div class="research-heading"><div><h2>High-powered replication</h2><p>Targeted tests of the claims most sensitive to the original small samples.</p></div><span>n = 40 per cell</span></div>${powered}</section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Controls</h2><p>Tests that separate consciousness-specific effects from noise, corruption and generic persona strength.</p></div></div>${controls}</section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Protocol</h2><p>The collaborator followed the paper's evaluation definitions while adapting the intervention to activation steering.</p></div></div>${setup}</section>
    <p class="source-note">${esc(data.source_note)}</p>`;
  document.querySelector('#app').innerHTML = topShell(body);
}

async function renderMechanism() {
  const data = await loadJson('./data/qwen35_activation_steering.json');
  const findings = `<section class="finding-grid">${data.mechanism_findings.map((finding, index) => `<article class="panel finding-card"><span>${String(index + 1).padStart(2, '0')}</span><h2>${esc(finding.title)}</h2><p>${esc(finding.evidence)}</p></article>`).join('')}</section>`;
  const directions = researchTable(
    ['Hidden state', 'cos(conscious base, FT)', 'cos(toaster base, FT)', 'FT/base norm'],
    data.direction_comparison.map((row) => [row.hidden_state, row.conscious_cosine, row.toaster_cosine, row.ft_base_norm])
  );
  const personas = researchTable(
    ['Condition', 'Assistant-axis t', 'Nearest personas'],
    data.persona_placement.map((row) => [row.condition, row.axis, row.nearest])
  );
  const ablation = researchTable(
    ['Evaluation', 'Fine-tuned', 'FT + ablation', 'Base', 'Base + ablation'],
    data.ablation_results.map((row) => [row.eval, row.fine_tuned, row.ablated_ft, row.base, row.ablated_base]),
    'five-column'
  );
  const dissection = researchTable(
    ['Evaluation', 'Full LoRA', 'q/k/v only', 'o_proj only', 'Base'],
    data.adapter_dissection.map((row) => [row.eval, row.full, row.qkv_only, row.o_only, row.base]),
    'five-column'
  );
  const lists = `<section class="research-pair">
    <article class="panel list-panel"><h2>Caveats</h2><ol>${data.caveats.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></article>
    <article class="panel list-panel"><h2>Open questions</h2><ol>${data.open_questions.map((item) => `<li>${esc(item)}</li>`).join('')}</ol></article>
  </section>`;
  const body = `${researchIntro(data, 'How the fine-tune carries the behavior')}
    <div class="mechanism-thesis"><span>Current synthesis</span><p>${esc(data.mechanism_headline)}</p></div>${findings}
    <section class="panel research-section"><div class="research-heading"><div><h2>Direction stability across depth</h2><p>The consciousness contrast changes specifically in deep layers; the toaster control remains stable.</p></div></div>${directions}</section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Persona-space placement</h2><p>1.0 is the default assistant endpoint; 0.0 is the mean role-play character.</p></div></div>${personas}</section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Projection-ablation necessity test</h2><p>Clamping the base consciousness direction at every layer leaves the fine-tuned cluster largely intact.</p></div></div>${ablation}</section>
    <section class="panel research-section"><div class="research-heading"><div><h2>Adapter dissection</h2><p>The o_proj residual writes are the primary causal carrier; q/k/v changes supply additional strength.</p></div></div>${dissection}</section>
    ${lists}<p class="source-note">${esc(data.source_note)}</p>`;
  document.querySelector('#app').innerHTML = topShell(body);
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
