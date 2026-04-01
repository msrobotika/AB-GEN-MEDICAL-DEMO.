/**
 * ============================================================
 *  🩺 AB-GEN MEDICAL DEMO — Frontend Logic
 *  Handles API calls, chart rendering, and XAI traceability display
 * ============================================================
 */

// Chart colors for 8 cell classes
const CHART_COLORS = [
    '#00d4aa', '#4ecdc4', '#00b4d8', '#7c3aed',
    '#f472b6', '#fb923c', '#fbbf24', '#a78bfa'
];

// State
let isAnalyzing = false;
let traceOpen = false;

// ============================================================
// API Calls
// ============================================================

async function analyzeSample() {
    if (isAnalyzing) return;
    isAnalyzing = true;

    setButtonsDisabled(true);
    showLoading(true);
    hideSection('result-section');
    hideSection('batch-section');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        showLoading(false);
        renderResult(data);
        showSection('result-section');
    } catch (error) {
        showLoading(false);
        alert(`Error: ${error.message}`);
    } finally {
        isAnalyzing = false;
        setButtonsDisabled(false);
    }
}

async function analyzeBatch() {
    if (isAnalyzing) return;
    isAnalyzing = true;

    setButtonsDisabled(true);
    showLoading(true);
    hideSection('result-section');
    hideSection('batch-section');

    try {
        const response = await fetch('/api/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n: 10 })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        showLoading(false);
        renderBatch(data);
        showSection('batch-section');
    } catch (error) {
        showLoading(false);
        alert(`Error: ${error.message}`);
    } finally {
        isAnalyzing = false;
        setButtonsDisabled(false);
    }
}

// ============================================================
// Render Single Result
// ============================================================

function renderResult(data) {
    // Verdict
    const verdictCard = document.getElementById('verdict-card');
    verdictCard.className = `verdict-card glass-card ${data.acierto ? 'success' : 'error'}`;

    document.getElementById('verdict-icon').textContent = data.acierto ? '✅' : '❌';
    document.getElementById('verdict-title').textContent = `Diagnóstico: ${data.diagnostico}`;
    document.getElementById('verdict-subtitle').textContent =
        `Real: ${data.real} — Certeza: ${data.confianza}% — ${data.latencia_ms}ms`;

    const confValue = document.getElementById('confidence-value');
    confValue.textContent = `${data.confianza}%`;

    const badge = document.getElementById('verdict-confidence-badge');
    if (data.confianza >= 70) {
        badge.style.borderColor = 'rgba(0, 230, 118, 0.3)';
        badge.style.background = 'rgba(0, 230, 118, 0.08)';
        confValue.style.color = '#00e676';
    } else if (data.confianza >= 40) {
        badge.style.borderColor = 'rgba(255, 193, 7, 0.3)';
        badge.style.background = 'rgba(255, 193, 7, 0.08)';
        confValue.style.color = '#ffc107';
    } else {
        badge.style.borderColor = 'rgba(255, 82, 82, 0.3)';
        badge.style.background = 'rgba(255, 82, 82, 0.08)';
        confValue.style.color = '#ff5252';
    }

    // Cell description
    document.getElementById('cell-description').textContent = data.descripcion_celula || '';

    // Images
    document.getElementById('img-original').src = `data:image/png;base64,${data.imagen_original}`;
    document.getElementById('img-heatmap').src = `data:image/png;base64,${data.imagen_heatmap}`;
    document.getElementById('label-original').textContent = `Real: ${data.real}`;

    // Probability Chart
    renderProbChart(data.distribucion, data.diagnostico);

    // Traceability
    renderTraceability(data.trazabilidad);

    // Metrics
    document.getElementById('metric-latency').textContent = `${data.latencia_ms} ms`;
    document.getElementById('metric-energy').textContent = `${data.energia_kwh} kWh`;
    document.getElementById('metric-co2').textContent = `${data.co2_g} g`;
    document.getElementById('metric-total').textContent = data.total_test_samples.toLocaleString();

    // Animate metrics
    animateCounters();
}

// ============================================================
// Probability Chart
// ============================================================

function renderProbChart(distribucion, winner) {
    const container = document.getElementById('prob-chart');
    container.innerHTML = '';

    // Sort by probability descending
    const entries = Object.entries(distribucion).map(([name, d]) => ({
        name,
        prob: d.n2_prob,
        nameEn: d.nombre_en
    }));
    entries.sort((a, b) => b.prob - a.prob);

    entries.forEach((entry, i) => {
        const row = document.createElement('div');
        row.className = 'prob-row';

        const isWinner = entry.name === winner;

        row.innerHTML = `
            <span class="prob-label" title="${entry.name} (${entry.nameEn})">${entry.name}</span>
            <div class="prob-bar-container">
                <div class="prob-bar ${isWinner ? 'winner' : ''}"
                     style="width: 0%; background: ${CHART_COLORS[i]}; opacity: ${isWinner ? 1 : 0.6};">
                </div>
            </div>
            <span class="prob-value">${entry.prob.toFixed(1)}%</span>
        `;

        container.appendChild(row);

        // Animate bar
        setTimeout(() => {
            const bar = row.querySelector('.prob-bar');
            bar.style.width = `${Math.max(entry.prob, 1)}%`;
        }, 100 + i * 80);
    });
}

// ============================================================
// Traceability Panel
// ============================================================

function renderTraceability(trazabilidad) {
    // Verdict text
    document.getElementById('trace-veredicto').textContent = trazabilidad.veredicto;

    // Steps
    const container = document.getElementById('trace-steps');
    container.innerHTML = '';

    trazabilidad.pasos.forEach(paso => {
        const stepEl = document.createElement('div');
        stepEl.className = 'trace-step';
        stepEl.innerHTML = `
            <div class="trace-step-num">${paso.paso}</div>
            <div class="trace-step-content">
                <h4>${paso.titulo}</h4>
                <p>${paso.detalle}</p>
            </div>
        `;
        container.appendChild(stepEl);
    });

    // Reset toggle state
    traceOpen = false;
    container.classList.remove('open');
    document.getElementById('trace-toggle').classList.remove('open');
}

function toggleTraceability() {
    traceOpen = !traceOpen;
    const steps = document.getElementById('trace-steps');
    const toggle = document.getElementById('trace-toggle');

    if (traceOpen) {
        steps.classList.add('open');
        toggle.classList.add('open');
    } else {
        steps.classList.remove('open');
        toggle.classList.remove('open');
    }
}

// ============================================================
// Batch Results
// ============================================================

function renderBatch(data) {
    // Summary
    document.getElementById('batch-accuracy').textContent = `${data.accuracy}%`;
    document.getElementById('batch-total').textContent = `${data.aciertos}/${data.total}`;
    document.getElementById('batch-latency').textContent = `${data.latencia_media_ms} ms`;

    // Grid
    const grid = document.getElementById('batch-grid');
    grid.innerHTML = '';

    data.resultados.forEach((r, i) => {
        const item = document.createElement('div');
        item.className = `batch-item glass-card ${r.acierto ? 'success' : 'failure'}`;
        item.style.animationDelay = `${i * 0.05}s`;
        item.style.cursor = 'pointer';
        item.onclick = () => viewBatchDetail(r);

        item.innerHTML = `
            <img class="batch-img" src="data:image/png;base64,${r.imagen_original}" alt="${r.diagnostico}">
            <div class="batch-info">
                <div class="batch-pred">${r.acierto ? '✅' : '❌'} ${r.diagnostico}</div>
                <div class="batch-real">Real: ${r.real} · ${r.latencia_ms}ms</div>
            </div>
            <div class="batch-conf" style="color: ${r.acierto ? '#00e676' : '#ff5252'}">${r.confianza}%</div>
        `;

        grid.appendChild(item);
    });
}

function viewBatchDetail(result) {
    renderResult(result);
    hideSection('batch-section');
    showSection('result-section');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================
// UI Helpers
// ============================================================

function setButtonsDisabled(disabled) {
    document.getElementById('btn-analyze').disabled = disabled;
    document.getElementById('btn-batch').disabled = disabled;
}

function showLoading(show) {
    document.getElementById('loading-state').style.display = show ? 'block' : 'none';
}

function showSection(id) {
    document.getElementById(id).style.display = 'block';
}

function hideSection(id) {
    document.getElementById(id).style.display = 'none';
}

function animateCounters() {
    // Simple fade-in animation for metrics
    document.querySelectorAll('.metric-card').forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(15px)';
        setTimeout(() => {
            card.style.transition = 'all 0.4s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 100 + i * 100);
    });
}

// ============================================================
// Load model info on startup
// ============================================================

async function loadModelInfo() {
    try {
        const response = await fetch('/api/stats');
        if (response.ok) {
            const info = await response.json();
            document.getElementById('badge-accuracy').textContent = `${info.accuracy_test}% Accuracy`;
        }
    } catch (e) {
        // Silent fail — non-critical
    }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
});
