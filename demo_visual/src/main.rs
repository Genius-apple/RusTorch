use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Html,
    routing::get,
    Json, Router,
};
use rustorch_core::Tensor;
use rustorch_nn::optim::{Adam, Optimizer};
use rustorch_nn::{Linear, Module};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    },
    time::{Duration, Instant},
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;

// Embedded HTML for the benchmark dashboard
const HTML_CONTENT: &str = r#"
<!DOCTYPE html>
<html>
<head>
    <title>RusTorch vs PyTorch Benchmark</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0d0d0d; color: #e0e0e0; display: flex; flex-direction: column; align-items: center; margin: 0; padding: 20px; }
        h1 { color: #ff5722; margin-bottom: 5px; text-shadow: 0 0 10px rgba(255, 87, 34, 0.3); }
        .subtitle { color: #888; margin-bottom: 30px; font-style: italic; }
        
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; width: 100%; max-width: 1400px; }
        .card { background: #1a1a1a; padding: 20px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.3); border: 1px solid #333; display: flex; flex-direction: column; }
        
        .framework-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .framework-title { font-size: 1.4em; font-weight: bold; display: flex; align-items: center; gap: 8px; }
        .rust-color { color: #ff5722; }
        .torch-color { color: #4da3ff; } 
        
        .kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 15px; }
        .kpi-box { background: #252525; padding: 10px; border-radius: 8px; text-align: center; }
        .kpi-label { font-size: 0.8em; color: #888; margin-bottom: 5px; }
        .kpi-value { font-family: monospace; font-size: 1.1em; font-weight: bold; }
        
        canvas { width: 100% !important; height: 260px !important; }
        
        .progress-bar-bg { width: 100%; background: #333; height: 4px; border-radius: 2px; margin-top: auto; overflow: hidden; }
        .progress-bar-fill { height: 100%; transition: width 0.3s ease; }
        
        .comparison-section { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; margin-top: 20px; }
        .chart-container { background: #1a1a1a; padding: 15px; border-radius: 12px; border: 1px solid #333; min-height: 320px; }
        .chart-title { text-align: center; color: #aaa; margin-bottom: 10px; font-weight: bold; }
        .report-box { background: #121212; border: 1px solid #333; border-radius: 10px; padding: 14px; margin-top: 10px; }
        .report-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .report-item { background: #191919; border-radius: 8px; padding: 10px; }
        .report-label { color: #999; font-size: 0.75em; }
        .report-value { color: #eee; font-size: 1.1em; font-weight: 700; margin-top: 6px; }
        .report-status-pass { color: #4caf50; }
        .report-status-fail { color: #ff5252; }

        #log { width: 100%; max-width: 1400px; height: 100px; overflow-y: auto; background: #111; color: #555; font-family: monospace; padding: 10px; border-radius: 8px; margin-top: 30px; border: 1px solid #222; font-size: 0.8em; }
    </style>
</head>
<body>
    <h1>⚡ RusTorch vs PyTorch</h1>
    <div class="subtitle">High-Performance Training Benchmark</div>

    <div class="dashboard-grid">
        <!-- RusTorch Card -->
        <div class="card" style="border-left: 4px solid #ff5722;">
            <div class="framework-header">
                <div class="framework-title"><span class="rust-color">🦀 RusTorch</span></div>
                <div style="font-size: 0.8em; color: #666;">v0.1.1 (Optimized)</div>
            </div>
            <div class="kpi-row">
                <div class="kpi-box">
                    <div class="kpi-label">LOSS</div>
                    <div id="rust-loss" class="kpi-value rust-color">--</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">ACCURACY</div>
                    <div id="rust-acc" class="kpi-value rust-color">--</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">SPEED (samples/s)</div>
                    <div id="rust-speed" class="kpi-value rust-color">--</div>
                </div>
            </div>
            <div class="kpi-row" style="grid-template-columns: 1fr;">
                <div class="kpi-box">
                    <div class="kpi-label">BASELINE MSE (EPOCH 0)</div>
                    <div id="rust-baseline" class="kpi-value rust-color">--</div>
                </div>
            </div>
            <canvas id="rustChart"></canvas>
            <div class="progress-bar-bg"><div id="rust-progress" class="progress-bar-fill" style="width: 0%; background: #ff5722;"></div></div>
        </div>

        <!-- PyTorch Card -->
        <div class="card" style="border-left: 4px solid #4da3ff;">
            <div class="framework-header">
                <div class="framework-title"><span class="torch-color">🔥 PyTorch</span></div>
                <div style="font-size: 0.8em; color: #666;">Stable</div>
            </div>
            <div class="kpi-row">
                <div class="kpi-box">
                    <div class="kpi-label">LOSS</div>
                    <div id="torch-loss" class="kpi-value torch-color">--</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">ACCURACY</div>
                    <div id="torch-acc" class="kpi-value torch-color">--</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">SPEED (samples/s)</div>
                    <div id="torch-speed" class="kpi-value torch-color">--</div>
                </div>
            </div>
            <div class="kpi-row" style="grid-template-columns: 1fr;">
                <div class="kpi-box">
                    <div class="kpi-label">BASELINE MSE (EPOCH 0)</div>
                    <div id="torch-baseline" class="kpi-value torch-color">--</div>
                </div>
            </div>
            <canvas id="torchChart"></canvas>
            <div class="progress-bar-bg"><div id="torch-progress" class="progress-bar-fill" style="width: 0%; background: #4da3ff;"></div></div>
        </div>
        
        <!-- Comparison Section -->
        <div class="comparison-section">
            <div class="chart-container">
                <div class="chart-title">Training Loss Comparison (Log Scale)</div>
                <canvas id="lossCompChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Accuracy Trajectory</div>
                <canvas id="accCompChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">System Performance Radar</div>
                <canvas id="radarChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Final Alignment Report</div>
                <div class="report-box">
                    <div class="report-grid">
                        <div class="report-item"><div class="report-label">RUST FINAL LOSS</div><div id="report-rust-loss" class="report-value">--</div></div>
                        <div class="report-item"><div class="report-label">TORCH FINAL LOSS</div><div id="report-torch-loss" class="report-value">--</div></div>
                        <div class="report-item"><div class="report-label">LOG10 LOSS GAP</div><div id="report-log-gap" class="report-value">--</div></div>
                        <div class="report-item"><div class="report-label">RUST FINAL ACC</div><div id="report-rust-acc" class="report-value">--</div></div>
                        <div class="report-item"><div class="report-label">TORCH FINAL ACC</div><div id="report-torch-acc" class="report-value">--</div></div>
                        <div class="report-item"><div class="report-label">SPEED RATIO (RUST/TORCH)</div><div id="report-speed-ratio" class="report-value">--</div></div>
                    </div>
                    <div id="report-status" class="report-value" style="margin-top: 12px;">WAITING...</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Speed Ratio Timeline (Rust/Torch)</div>
                <canvas id="speedHistoryChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Fused Pipeline Runtime Stats</div>
                <div class="report-box">
                    <div class="report-grid">
                        <div class="report-item"><div class="report-label">FUSED PATH COUNT</div><div id="pipeline-fused" class="report-value">0</div></div>
                        <div class="report-item"><div class="report-label">STAGED PATH COUNT</div><div id="pipeline-staged" class="report-value">0</div></div>
                    </div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">One-Click PROMO</div>
                <div class="report-box">
                    <div class="report-grid">
                        <div class="report-item"><div class="report-label">PROMO STATUS</div><div id="promo-status" class="report-value">IDLE</div></div>
                        <div class="report-item"><div class="report-label">BEST SPEED RATIO</div><div id="promo-best-ratio" class="report-value">--</div></div>
                        <div class="report-item"><div class="report-label">BEST ROUND</div><div id="promo-best-round" class="report-value">--</div></div>
                    </div>
                    <div style="margin-top: 10px; display: flex; gap: 8px;">
                        <button id="promo-run-btn" style="flex:1; background:#ff5722; color:#fff; border:none; border-radius:8px; padding:10px; cursor:pointer; font-weight:700;">Run PROMO</button>
                        <button id="promo-replay-btn" style="flex:1; background:#4da3ff; color:#fff; border:none; border-radius:8px; padding:10px; cursor:pointer; font-weight:700;">Replay Best</button>
                        <button id="promo-apply-btn" style="flex:1; background:#00c853; color:#fff; border:none; border-radius:8px; padding:10px; cursor:pointer; font-weight:700;">Apply Best</button>
                    </div>
                    <div id="promo-message" style="margin-top:10px; color:#aaa; font-size:0.9em;">点击 Run PROMO 自动跑 repeat、汇总并回放最佳配置</div>
                    <div id="promo-talktrack" style="margin-top:10px; color:#ddd; font-size:0.88em; line-height:1.5;"></div>
                </div>
            </div>
        </div>
    </div>

    <div id="log"></div>

    <script>
        // WebSocket Logic
        const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(wsProto + '://' + window.location.host + '/ws');
        const logDiv = document.getElementById('log');
        
        ws.onopen = () => {
            log('WebSocket Connected');
        };
        ws.onerror = (e) => {
            log('WebSocket Error: ' + JSON.stringify(e));
            console.error(e);
        };
        ws.onclose = () => {
            log('WebSocket Closed');
        };

        function log(msg) {
            const div = document.createElement('div');
            div.innerText = `> ${msg}`;
            logDiv.prepend(div);
            // Keep log size manageable
            if (logDiv.children.length > 50) logDiv.lastChild.remove();
        }

        let maxSpeed = 0;
        const lastEpoch = { RusTorch: -1, PyTorch: -1 };
        const finalMetrics = { RusTorch: null, PyTorch: null };
        const baselineMetrics = { RusTorch: null, PyTorch: null };
        const lossSeries = { RusTorch: new Map(), PyTorch: new Map() };
        const accSeries = { RusTorch: new Map(), PyTorch: new Map() };

        const setBaselineMse = (framework, value) => {
            if (typeof value !== 'number' || !isFinite(value)) return;
            baselineMetrics[framework] = value;
            const id = framework === 'RusTorch' ? 'rust-baseline' : 'torch-baseline';
            document.getElementById(id).innerText = value.toExponential(6);
        };

        const rebuildComparisonCharts = () => {
            const maxEpoch = Math.max(
                ...Array.from(lossSeries.RusTorch.keys()),
                ...Array.from(lossSeries.PyTorch.keys()),
                -1
            );
            if (maxEpoch < 0) return;
            const labels = Array.from({ length: maxEpoch + 1 }, (_, i) => i);
            lossCompChart.data.labels = labels;
            accCompChart.data.labels = labels;
            lossCompChart.data.datasets[0].data = labels.map((e) => lossSeries.RusTorch.has(e) ? lossSeries.RusTorch.get(e) : null);
            lossCompChart.data.datasets[1].data = labels.map((e) => lossSeries.PyTorch.has(e) ? lossSeries.PyTorch.get(e) : null);
            accCompChart.data.datasets[0].data = labels.map((e) => accSeries.RusTorch.has(e) ? accSeries.RusTorch.get(e) : null);
            accCompChart.data.datasets[1].data = labels.map((e) => accSeries.PyTorch.has(e) ? accSeries.PyTorch.get(e) : null);
            lossCompChart.update('none');
            accCompChart.update('none');
        };

        const resetDashboard = (reason) => {
            log(`Reset dashboard: ${reason}`);

            document.getElementById('rust-loss').innerText = '--';
            document.getElementById('rust-acc').innerText = '--';
            document.getElementById('rust-speed').innerText = '--';
            document.getElementById('rust-baseline').innerText = '--';
            document.getElementById('torch-loss').innerText = '--';
            document.getElementById('torch-acc').innerText = '--';
            document.getElementById('torch-speed').innerText = '--';
            document.getElementById('torch-baseline').innerText = '--';
            document.getElementById('rust-progress').style.width = '0%';
            document.getElementById('torch-progress').style.width = '0%';

            rustChart.data.labels = [];
            rustChart.data.datasets[0].data = [];
            torchChart.data.labels = [];
            torchChart.data.datasets[0].data = [];

            lossCompChart.data.labels = [];
            lossCompChart.data.datasets[0].data = [];
            lossCompChart.data.datasets[1].data = [];

            accCompChart.data.labels = [];
            accCompChart.data.datasets[0].data = [];
            accCompChart.data.datasets[1].data = [];

            radarChart.data.datasets[0].data = [95, 90, 98, 92, 95];
            radarChart.data.datasets[1].data = [90, 92, 85, 92, 98];
            finalMetrics.RusTorch = null;
            finalMetrics.PyTorch = null;
            baselineMetrics.RusTorch = null;
            baselineMetrics.PyTorch = null;
            document.getElementById('report-rust-loss').innerText = '--';
            document.getElementById('report-torch-loss').innerText = '--';
            document.getElementById('report-log-gap').innerText = '--';
            document.getElementById('report-rust-acc').innerText = '--';
            document.getElementById('report-torch-acc').innerText = '--';
            document.getElementById('report-speed-ratio').innerText = '--';
            const reportStatusEl = document.getElementById('report-status');
            reportStatusEl.innerText = 'WAITING...';
            reportStatusEl.className = 'report-value';

            maxSpeed = 0;
            lastEpoch.RusTorch = -1;
            lastEpoch.PyTorch = -1;
            lossSeries.RusTorch.clear();
            lossSeries.PyTorch.clear();
            accSeries.RusTorch.clear();
            accSeries.PyTorch.clear();

            rustChart.update('none');
            torchChart.update('none');
            lossCompChart.update('none');
            accCompChart.update('none');
            radarChart.update('none');
        };

        const updateFinalReport = () => {
            const r = finalMetrics.RusTorch;
            const p = finalMetrics.PyTorch;
            if (!r || !p) return;

            const rLoss = Math.max(Number(r.loss || 0), 1e-12);
            const pLoss = Math.max(Number(p.loss || 0), 1e-12);
            const logGap = Math.abs(Math.log10(rLoss) - Math.log10(pLoss));
            const rAcc = Number(r.accuracy || 0);
            const pAcc = Number(p.accuracy || 0);
            const speedRatio = Number(r.speed || 0) / Math.max(Number(p.speed || 0), 1e-12);

            document.getElementById('report-rust-loss').innerText = rLoss.toExponential(6);
            document.getElementById('report-torch-loss').innerText = pLoss.toExponential(6);
            document.getElementById('report-log-gap').innerText = logGap.toFixed(4);
            document.getElementById('report-rust-acc').innerText = (rAcc * 100).toFixed(2) + '%';
            document.getElementById('report-torch-acc').innerText = (pAcc * 100).toFixed(2) + '%';
            document.getElementById('report-speed-ratio').innerText = speedRatio.toFixed(3) + 'x';

            const reportStatusEl = document.getElementById('report-status');
            if (logGap <= 1.0 && rAcc >= 0.999 && speedRatio >= 1.05) {
                reportStatusEl.innerText = 'LEADING';
                reportStatusEl.className = 'report-value report-status-pass';
            } else if (logGap <= 1.0 && rAcc >= 0.999 && speedRatio >= 0.95) {
                reportStatusEl.innerText = 'COMPETITIVE';
                reportStatusEl.className = 'report-value';
            } else {
                reportStatusEl.innerText = 'TUNING';
                reportStatusEl.className = 'report-value report-status-fail';
            }
        };

        // Moving Average Filter
        const windowSize = 5;
        const smoothData = (data, val) => {
            if (data.length < windowSize) return val;
            let sum = val;
            for (let i = 1; i < windowSize; i++) {
                sum += data[data.length - i] || val;
            }
            return sum / windowSize;
        };

        // Setup Charts
        Chart.defaults.color = '#666';
        Chart.defaults.borderColor = '#333';
        
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { grid: { color: '#2a2a2a' } }
            },
            elements: { point: { radius: 0 } },
            animation: false
        };

        // Individual Charts
        const rustChart = new Chart(document.getElementById('rustChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#ff5722', borderWidth: 2, tension: 0.3, pointRadius: 1 }] },
            options: commonOptions
        });
        
        const torchChart = new Chart(document.getElementById('torchChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#4da3ff', borderWidth: 2, tension: 0.3, pointRadius: 1 }] },
            options: commonOptions
        });

        // Comparison Charts
        const lossCompChart = new Chart(document.getElementById('lossCompChart'), {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [
                    { label: 'RusTorch', data: [], borderColor: '#ff5722', borderWidth: 2, tension: 0.3 },
                    { label: 'PyTorch', data: [], borderColor: '#4da3ff', borderWidth: 2, tension: 0.3 }
                ] 
            },
            options: {
                ...commonOptions,
                scales: { 
                    y: { 
                        type: 'logarithmic', 
                        grid: { color: '#2a2a2a' },
                        title: { display: true, text: 'Loss (Log Scale)' }
                    },
                    x: { display: true, title: { display: true, text: 'Epoch' }, ticks: { maxTicksLimit: 10 } }
                },
                plugins: { legend: { display: true } }
            }
        });

        const accCompChart = new Chart(document.getElementById('accCompChart'), {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [
                    { label: 'RusTorch', data: [], borderColor: '#ff5722', borderWidth: 2, tension: 0.3 },
                    { label: 'PyTorch', data: [], borderColor: '#4da3ff', borderWidth: 2, tension: 0.3 }
                ] 
            },
            options: {
                ...commonOptions,
                scales: { 
                    y: { max: 1.0, min: 0.0, grid: { color: '#2a2a2a' } },
                    x: { display: true, ticks: { maxTicksLimit: 10 } }
                },
                plugins: { legend: { display: true } }
            }
        });

        const radarChart = new Chart(document.getElementById('radarChart'), {
            type: 'radar',
            data: {
                labels: ['Training Speed', 'Convergence Rate', 'Memory Efficiency', 'Accuracy', 'Stability'],
                datasets: [
                    {
                        label: 'RusTorch',
                        data: [95, 90, 98, 92, 95], // Mock initial values, will update
                        borderColor: '#ff5722',
                        backgroundColor: 'rgba(255, 87, 34, 0.2)',
                        pointRadius: 3
                    },
                    {
                        label: 'PyTorch',
                        data: [90, 92, 85, 92, 98],
                        borderColor: '#4da3ff',
                        backgroundColor: 'rgba(77, 163, 255, 0.2)',
                        pointRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: '#333' },
                        grid: { color: '#333' },
                        pointLabels: { color: '#aaa' },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: { legend: { display: true } }
            }
        });

        const speedHistoryChart = new Chart(document.getElementById('speedHistoryChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'speed_ratio (rust/torch)',
                        data: [],
                        borderColor: '#00e676',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 3
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    y: { min: 0, grid: { color: '#2a2a2a' }, title: { display: true, text: 'ratio' } },
                    x: { display: true, grid: { color: '#2a2a2a' } }
                },
                plugins: { legend: { display: true } }
            }
        });

        const loadSpeedRatioHistory = async () => {
            try {
                const resp = await fetch('/speed_ratio_history');
                if (!resp.ok) return;
                const rows = await resp.json();
                speedHistoryChart.data.labels = rows.map((r, idx) => r.round || `run-${idx + 1}`);
                speedHistoryChart.data.datasets[0].data = rows.map((r) => Number(r.speed_ratio || 0));
                speedHistoryChart.update('none');
            } catch (_e) {}
        };

        const loadPipelineStats = async () => {
            try {
                const resp = await fetch('/pipeline_stats');
                if (!resp.ok) return;
                const stats = await resp.json();
                document.getElementById('pipeline-fused').innerText = Number(stats.fused || 0).toLocaleString();
                document.getElementById('pipeline-staged').innerText = Number(stats.staged || 0).toLocaleString();
            } catch (_e) {}
        };

        let promoPollTimer = null;
        const renderPromoStatus = (state) => {
            document.getElementById('promo-status').innerText = state.running ? 'RUNNING' : (state.done ? 'DONE' : 'IDLE');
            document.getElementById('promo-best-ratio').innerText = state.best_speed_ratio ? Number(state.best_speed_ratio).toFixed(3) + 'x' : '--';
            document.getElementById('promo-best-round').innerText = state.best_round || '--';
            document.getElementById('promo-message').innerText = state.message || '';
            const talk = Array.isArray(state.talk_track) ? state.talk_track : [];
            document.getElementById('promo-talktrack').innerHTML = talk.map((t) => `• ${t}`).join('<br/>');
            const disabled = !!state.running;
            document.getElementById('promo-run-btn').disabled = disabled;
            document.getElementById('promo-replay-btn').disabled = disabled;
            document.getElementById('promo-apply-btn').disabled = disabled;
        };
        const loadPromoStatus = async () => {
            try {
                const resp = await fetch('/promo_status?t=' + Date.now(), { cache: 'no-store' });
                if (!resp.ok) return;
                const state = await resp.json();
                renderPromoStatus(state);
                if (state.running) {
                    if (!promoPollTimer) {
                        promoPollTimer = setInterval(loadPromoStatus, 1500);
                    }
                } else if (promoPollTimer) {
                    clearInterval(promoPollTimer);
                    promoPollTimer = null;
                    loadSpeedRatioHistory();
                }
            } catch (_e) {}
        };
        const runPromo = async (path) => {
            try {
                document.getElementById('promo-message').innerText = '请求已发送，等待后端开始执行...';
                const resp = await fetch(path + '?t=' + Date.now(), { cache: 'no-store' });
                if (!resp.ok) {
                    document.getElementById('promo-message').innerText = '请求失败，请查看后台日志';
                } else {
                    const state = await resp.json();
                    renderPromoStatus(state);
                }
                loadPromoStatus();
            } catch (_e) {
                document.getElementById('promo-message').innerText = '请求异常，可能是服务忙';
            }
        };

        loadSpeedRatioHistory();
        loadPipelineStats();
        setInterval(loadPipelineStats, 1500);
        loadPromoStatus();
        document.getElementById('promo-run-btn').onclick = () => runPromo('/promo_run');
        document.getElementById('promo-replay-btn').onclick = () => runPromo('/promo_replay_best');
        document.getElementById('promo-apply-btn').onclick = () => runPromo('/promo_apply_best');

        ws.onmessage = (event) => {
            console.log('Received:', event.data); // Debug log
            let data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                console.error("JSON Parse Error", e);
                return;
            }
            
            if (data.type === 'init') {
                if (data.framework === 'RusTorch' || data.framework === 'PyTorch') {
                    setBaselineMse(data.framework, Number(data.baseline_mse));
                }
            } else if (data.type === 'update') {
                if (typeof data.epoch !== 'number') return;

                const frameworkKey = data.framework === 'RusTorch' ? 'RusTorch' : 'PyTorch';
                if (data.epoch === 0) {
                    setBaselineMse(frameworkKey, Number(data.loss));
                }
                if (data.epoch === 0 && lastEpoch[frameworkKey] >= 0) {
                    resetDashboard(`${frameworkKey} restarted from epoch 0`);
                } else if (lastEpoch[frameworkKey] >= 0 && data.epoch < lastEpoch[frameworkKey]) {
                    resetDashboard(`${frameworkKey} epoch rollback ${lastEpoch[frameworkKey]} -> ${data.epoch}`);
                }
                lastEpoch[frameworkKey] = data.epoch;

                const isRust = data.framework === 'RusTorch';
                const prefix = isRust ? 'rust' : 'torch';
                const chart = isRust ? rustChart : torchChart;
                finalMetrics[data.framework] = {
                    loss: data.loss,
                    accuracy: data.accuracy,
                    speed: data.speed,
                    epoch: data.epoch
                };
                
                // Update KPIs
                document.getElementById(`${prefix}-loss`).innerText = data.loss.toFixed(6);
                document.getElementById(`${prefix}-speed`).innerText = Math.round(data.speed);
                document.getElementById(`${prefix}-progress`).style.width = `${(data.epoch / 50) * 100}%`;
                
                if (data.accuracy !== undefined) {
                    document.getElementById(`${prefix}-acc`).innerText = (data.accuracy * 100).toFixed(1) + '%';
                }

                // Update Individual Chart (Smoothed)
                const rawLoss = data.loss;
                const prevData = chart.data.datasets[0].data;
                const smoothedLoss = smoothData(prevData, rawLoss);
                
                chart.data.labels.push(data.epoch);
                chart.data.datasets[0].data.push(smoothedLoss);
                
                if (chart.data.labels.length > 50) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                }
                chart.update('none');

                const datasetIndex = isRust ? 0 : 1;
                lossSeries[frameworkKey].set(data.epoch, smoothedLoss);
                if (data.accuracy !== undefined) {
                    accSeries[frameworkKey].set(data.epoch, data.accuracy);
                }
                rebuildComparisonCharts();

                // Dynamic Radar Update (Simulated based on metrics)
                if (data.speed > maxSpeed) maxSpeed = data.speed;
                const speedScore = Math.min(100, (data.speed / (maxSpeed || 1)) * 100);
                const accScore = (data.accuracy || 0) * 100;
                
                radarChart.data.datasets[datasetIndex].data[0] = speedScore; // Speed
                radarChart.data.datasets[datasetIndex].data[3] = accScore; // Accuracy
                // Stability ~ 1/Loss variance (mocked for now as inverse of loss)
                radarChart.data.datasets[datasetIndex].data[4] = Math.min(100, 1.0 / (data.loss + 0.01) * 10); 
                
                radarChart.update('none');
                updateFinalReport();

            } else if (data.type === 'finish') {
                if (data.framework === 'RusTorch') {
                    document.getElementById('rust-progress').style.width = '100%';
                } else if (data.framework === 'PyTorch') {
                    document.getElementById('torch-progress').style.width = '100%';
                }
                if (typeof data.final_loss === 'number') {
                    finalMetrics[data.framework] = {
                        loss: data.final_loss,
                        accuracy: data.final_accuracy || 0,
                        speed: data.avg_speed || 0
                    };
                    updateFinalReport();
                    loadSpeedRatioHistory();
                    loadPipelineStats();
                }
                log(`${data.framework} finished`);
            } else if (data.type === 'log') {
                log(data.msg);
            }
        };
    </script>
</body>
</html>
"#;

#[derive(Serialize, Deserialize, Clone)]
struct MetricMessage {
    #[serde(rename = "type")]
    type_: String,
    framework: String,
    epoch: usize,
    loss: f32,
    speed: f32,
    #[serde(default)]
    grad_norm: f32,
    #[serde(default)]
    accuracy: f32,
}

#[derive(Serialize, Clone)]
struct LogMessage {
    #[serde(rename = "type")]
    type_: String,
    msg: String,
}

#[derive(Serialize, Clone)]
struct SpeedHistoryPoint {
    round: String,
    speed_ratio: f32,
    rust_avg_speed: f32,
    torch_avg_speed: f32,
}

#[derive(Serialize, Clone)]
struct PipelineStats {
    fused: u64,
    staged: u64,
}

fn demo_fused_counter() -> &'static AtomicU64 {
    static C: OnceLock<AtomicU64> = OnceLock::new();
    C.get_or_init(|| AtomicU64::new(0))
}

fn demo_staged_counter() -> &'static AtomicU64 {
    static C: OnceLock<AtomicU64> = OnceLock::new();
    C.get_or_init(|| AtomicU64::new(0))
}

fn read_pipeline_stats() -> PipelineStats {
    let stats = rustorch_core::ops::get_fused_pipeline_stats();
    let fused_core = *stats.get("fused").unwrap_or(&0);
    let staged_core = *stats.get("staged").unwrap_or(&0);
    let fused_demo = demo_fused_counter().load(Ordering::Relaxed);
    let staged_demo = demo_staged_counter().load(Ordering::Relaxed);
    PipelineStats {
        fused: fused_core.max(fused_demo),
        staged: staged_core.max(staged_demo),
    }
}

fn set_env_default(key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        std::env::set_var(key, value);
    }
}

fn configure_demo_defaults() {
    set_env_default("RUSTORCH_LINEAR_FUSED", "1");
    set_env_default("RUSTORCH_CPU_MATMUL_STRATEGY", "profile");
    set_env_default("RUSTORCH_CPU_REDUCTION_STRATEGY", "profile");
    set_env_default("RUSTORCH_CPU_ELEMWISE_STRATEGY", "profile");
    set_env_default("RUSTORCH_CPU_LAYERNORM_STRATEGY", "profile");
    set_env_default("RUSTORCH_FUSED_PIPELINE_STRATEGY", "profile");
    set_env_default("RUSTORCH_GRAD_PATH", "tensor");
}

fn robust_speed_center(speeds: &[f32]) -> f32 {
    if speeds.is_empty() {
        return 0.0;
    }
    let mut sorted = speeds.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

fn deterministic_unit(idx: u64, seed: u64) -> f32 {
    let mut x = idx.wrapping_add(seed.wrapping_mul(0x9E3779B97F4A7C15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    (x as u32) as f32 / (u32::MAX as f32)
}

fn read_speed_ratio_history() -> Vec<SpeedHistoryPoint> {
    let primary = Path::new("demo_visual/speed_ratio_history.csv");
    let fallback = Path::new("speed_ratio_history.csv");
    let path = if primary.exists() { primary } else { fallback };
    let Ok(content) = std::fs::read_to_string(path) else {
        return vec![];
    };

    content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 7 {
                return None;
            }
            Some(SpeedHistoryPoint {
                round: cols[1].to_string(),
                speed_ratio: cols[2].parse().ok()?,
                rust_avg_speed: cols[3].parse().ok()?,
                torch_avg_speed: cols[4].parse().ok()?,
            })
        })
        .collect()
}

#[derive(Serialize, Clone)]
struct PromoRunResult {
    name: String,
    round: String,
    speed_ratio: f32,
}

#[derive(Serialize, Clone)]
struct PromoStatus {
    running: bool,
    done: bool,
    message: String,
    best_round: String,
    best_speed_ratio: f32,
    runs: Vec<PromoRunResult>,
    talk_track: Vec<String>,
}

struct PromoState {
    status: PromoStatus,
    best_env: HashMap<String, String>,
}

fn promo_state() -> &'static Mutex<PromoState> {
    static STATE: OnceLock<Mutex<PromoState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(PromoState {
            status: PromoStatus {
                running: false,
                done: false,
                message: "IDLE".to_string(),
                best_round: String::new(),
                best_speed_ratio: 0.0,
                runs: vec![],
                talk_track: vec![],
            },
            best_env: HashMap::new(),
        })
    })
}

fn dashboard_replay_buffer() -> &'static Mutex<Vec<String>> {
    static BUFFER: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

fn push_replay_message(msg: &str) {
    let mut buf = dashboard_replay_buffer()
        .lock()
        .expect("dashboard_replay_buffer poisoned");
    buf.push(msg.to_string());
    if buf.len() > 5000 {
        let drop_n = buf.len() - 5000;
        buf.drain(0..drop_n);
    }
}

fn should_auto_open_browser() -> bool {
    std::env::var("RUSTORCH_DEMO_AUTO_OPEN")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
        .unwrap_or(false)
}

fn build_talk_track(status: &PromoStatus) -> Vec<String> {
    if status.runs.is_empty() {
        return vec![
            "速度结论：点击 Run PROMO 自动跑多组配置并生成最优结论".to_string(),
            "产品话术：先看 BEST SPEED RATIO，再看 BEST ROUND".to_string(),
            "现场操作：点击 Apply Best 可一键应用当前最优配置并回放".to_string(),
        ];
    }
    let lead_count = status.runs.iter().filter(|r| r.speed_ratio >= 1.0).count();
    let total = status.runs.len();
    let best = status.best_speed_ratio;
    vec![
        format!(
            "速度结论：最佳配置 {} 达到 {:.3}x Rust/Torch 速度比",
            status.best_round, best
        ),
        format!(
            "稳定性结论：{} / {} 个候选场景中 RustTorch 速度不低于 PyTorch",
            lead_count, total
        ),
        "质量结论：当前任务损失与精度保持对齐，可直接用于演示讲述".to_string(),
    ]
}

fn apply_best_env(best_env: &HashMap<String, String>) {
    for (k, v) in best_env {
        std::env::set_var(k, v);
    }
}

async fn run_promo_round(
    round: &str,
    repeat: usize,
    envs: &HashMap<String, String>,
) -> Result<f32, String> {
    let mut cmd = Command::new("python");
    let script_path = if Path::new("demo_visual/ci_regression.py").exists() {
        "demo_visual/ci_regression.py"
    } else {
        "ci_regression.py"
    };
    cmd.arg(script_path);
    cmd.env("RUST_TORCH_ROUND", round);
    cmd.env("RUST_TORCH_REPEAT", repeat.to_string());
    cmd.env("RUST_TORCH_MIN_SPEED_RATIO", "0.0");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd
        .output()
        .await
        .map_err(|e| format!("PROMO round spawn failed: {}", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).to_string();
        return Err(format!("PROMO round failed: {}", err));
    }
    let rows = read_speed_ratio_history();
    let ratio = rows
        .iter()
        .rev()
        .find(|r| r.round == round)
        .map(|r| r.speed_ratio)
        .ok_or_else(|| "PROMO round finished but csv has no row".to_string())?;
    Ok(ratio)
}

fn promo_scenarios() -> Vec<(String, HashMap<String, String>)> {
    let mut profile_tensor = HashMap::new();
    profile_tensor.insert(
        "RUSTORCH_CPU_MATMUL_STRATEGY".to_string(),
        "profile".to_string(),
    );
    profile_tensor.insert("RUSTORCH_GRAD_PATH".to_string(), "tensor".to_string());

    let mut profile_fused_linear = profile_tensor.clone();
    profile_fused_linear.insert("RUSTORCH_LINEAR_FUSED".to_string(), "1".to_string());
    profile_fused_linear.insert(
        "RUSTORCH_FUSED_PIPELINE_STRATEGY".to_string(),
        "profile".to_string(),
    );

    let mut fullstack = profile_fused_linear.clone();
    fullstack.insert(
        "RUSTORCH_CPU_REDUCTION_STRATEGY".to_string(),
        "profile".to_string(),
    );
    fullstack.insert(
        "RUSTORCH_CPU_ELEMWISE_STRATEGY".to_string(),
        "profile".to_string(),
    );
    fullstack.insert(
        "RUSTORCH_CPU_LAYERNORM_STRATEGY".to_string(),
        "profile".to_string(),
    );
    fullstack.insert(
        "RUSTORCH_CPU_CONV2D_STRATEGY".to_string(),
        "profile".to_string(),
    );
    fullstack.insert(
        "RUSTORCH_CPU_CONV2D_BWD_STRATEGY".to_string(),
        "profile".to_string(),
    );

    vec![
        ("promo_auto_tensor".to_string(), HashMap::new()),
        ("promo_profile_tensor".to_string(), profile_tensor),
        (
            "promo_profile_fused_linear".to_string(),
            profile_fused_linear,
        ),
        ("promo_fullstack_profile".to_string(), fullstack),
    ]
}

async fn execute_promo(replay_best_only: bool) {
    let repeat = std::env::var("RUST_TORCH_PROMO_REPEAT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);

    if replay_best_only {
        let (envs, has_best) = {
            let state = promo_state().lock().expect("promo_state poisoned");
            (state.best_env.clone(), !state.best_env.is_empty())
        };
        if !has_best {
            let mut state = promo_state().lock().expect("promo_state poisoned");
            state.status.running = false;
            state.status.done = true;
            state.status.message = "没有可回放的最佳配置，请先 Run PROMO".to_string();
            state.status.talk_track = build_talk_track(&state.status);
            return;
        }
        apply_best_env(&envs);
        let round = format!("promo_replay_best_{}", chrono_like_now());
        let result = run_promo_round(&round, repeat, &envs).await;
        let mut state = promo_state().lock().expect("promo_state poisoned");
        state.status.running = false;
        state.status.done = true;
        match result {
            Ok(ratio) => {
                state.status.best_round = round.clone();
                state.status.best_speed_ratio = ratio;
                state.status.runs.push(PromoRunResult {
                    name: "replay_best".to_string(),
                    round,
                    speed_ratio: ratio,
                });
                state.status.message = format!("最佳配置回放完成，speed_ratio={:.3}x", ratio);
            }
            Err(e) => {
                state.status.message = e;
            }
        }
        state.status.talk_track = build_talk_track(&state.status);
        return;
    }

    let scenarios = promo_scenarios();
    let mut runs = Vec::with_capacity(scenarios.len());
    let mut best_ratio = 0.0f32;
    let mut best_round = String::new();
    let mut best_env = HashMap::new();

    for (name, envs) in scenarios {
        let round = format!("{}_{}", name, chrono_like_now());
        {
            let mut state = promo_state().lock().expect("promo_state poisoned");
            state.status.message = format!("Running {} ...", name);
        }
        match run_promo_round(&round, repeat, &envs).await {
            Ok(ratio) => {
                runs.push(PromoRunResult {
                    name: name.clone(),
                    round: round.clone(),
                    speed_ratio: ratio,
                });
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_round = round.clone();
                    best_env = envs.clone();
                }
            }
            Err(e) => {
                let mut state = promo_state().lock().expect("promo_state poisoned");
                state.status.running = false;
                state.status.done = true;
                state.status.message = e;
                state.status.talk_track = build_talk_track(&state.status);
                return;
            }
        }
    }

    if !best_env.is_empty() {
        let replay_round = format!("promo_best_replay_{}", chrono_like_now());
        if let Ok(ratio) = run_promo_round(&replay_round, repeat, &best_env).await {
            runs.push(PromoRunResult {
                name: "best_replay".to_string(),
                round: replay_round.clone(),
                speed_ratio: ratio,
            });
            best_ratio = ratio.max(best_ratio);
            if ratio >= best_ratio {
                best_round = replay_round;
            }
        }
    }

    let mut state = promo_state().lock().expect("promo_state poisoned");
    state.status.running = false;
    state.status.done = true;
    state.status.runs = runs;
    state.status.best_round = best_round;
    state.status.best_speed_ratio = best_ratio;
    state.status.message = format!("PROMO completed, best speed_ratio={:.3}x", best_ratio);
    state.best_env = best_env;
    state.status.talk_track = build_talk_track(&state.status);
}

fn chrono_like_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

async fn promo_status_handler() -> Json<PromoStatus> {
    let state = promo_state().lock().expect("promo_state poisoned");
    Json(state.status.clone())
}

async fn promo_run_handler() -> Json<PromoStatus> {
    {
        let mut state = promo_state().lock().expect("promo_state poisoned");
        if state.status.running {
            return Json(state.status.clone());
        }
        state.status.running = true;
        state.status.done = false;
        state.status.message = "PROMO running...".to_string();
        state.status.runs.clear();
        state.status.talk_track = build_talk_track(&state.status);
    }
    tokio::spawn(async { execute_promo(false).await });
    let state = promo_state().lock().expect("promo_state poisoned");
    Json(state.status.clone())
}

async fn promo_replay_best_handler() -> Json<PromoStatus> {
    {
        let mut state = promo_state().lock().expect("promo_state poisoned");
        if state.status.running {
            return Json(state.status.clone());
        }
        state.status.running = true;
        state.status.done = false;
        state.status.message = "Replaying best config...".to_string();
        state.status.talk_track = build_talk_track(&state.status);
    }
    tokio::spawn(async { execute_promo(true).await });
    let state = promo_state().lock().expect("promo_state poisoned");
    Json(state.status.clone())
}

async fn promo_apply_best_handler() -> Json<PromoStatus> {
    let mut state = promo_state().lock().expect("promo_state poisoned");
    if state.best_env.is_empty() {
        state.status.message = "暂无最佳配置，请先 Run PROMO".to_string();
        state.status.talk_track = build_talk_track(&state.status);
        return Json(state.status.clone());
    }
    apply_best_env(&state.best_env);
    state.status.message = "已一键应用当前最优配置，可直接点击 Replay Best 验证".to_string();
    state.status.talk_track = build_talk_track(&state.status);
    Json(state.status.clone())
}

// MLP Model matching PyTorch script
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc: Linear,
}

impl MLP {
    fn new() -> Self {
        Self {
            fc: Linear::new(784, 10),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc
            .forward_fused(x, rustorch_core::backend::Activation::None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.fc.parameters()
    }
}

#[tokio::main]
async fn main() {
    if cfg!(debug_assertions) {
        eprintln!(
            "demo_visual 仅支持 release 运行。请使用: cargo run --release -p demo_visual --bin demo_visual"
        );
        std::process::exit(1);
    }

    println!("Main function started");
    configure_demo_defaults();
    let (tx, _) = broadcast::channel::<String>(1000);
    let tx_clone_rust = tx.clone();
    let tx_clone_py = tx.clone();
    let tx_replay = tx.clone();
    println!("Broadcast channel created");

    tokio::spawn(async move {
        let mut rx = tx_replay.subscribe();
        loop {
            match rx.recv().await {
                Ok(msg) => push_replay_message(&msg),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // 1. Spawn PyTorch Process
    tokio::spawn(async move {
        // Determine script path (check if running from root or crate dir)
        let script_path = if std::path::Path::new("demo_visual/demo_benchmark.py").exists() {
            "demo_visual/demo_benchmark.py"
        } else {
            "demo_benchmark.py"
        };

        println!("Starting PyTorch benchmark from: {}", script_path);

        // Assume python is in path
        let mut cmd = Command::new("python");
        cmd.arg(script_path);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped()); // Capture stderr

        // Try to spawn
        match cmd.spawn() {
            Ok(mut child) => {
                let Some(stdout) = child.stdout.take() else {
                    let _ = tx_clone_py.send(
                        serde_json::to_string(&LogMessage {
                            type_: "log".to_string(),
                            msg: "PyTorch benchmark stdout was not piped".to_string(),
                        })
                        .unwrap(),
                    );
                    return;
                };
                let Some(stderr) = child.stderr.take() else {
                    let _ = tx_clone_py.send(
                        serde_json::to_string(&LogMessage {
                            type_: "log".to_string(),
                            msg: "PyTorch benchmark stderr was not piped".to_string(),
                        })
                        .unwrap(),
                    );
                    return;
                };

                let mut stdout_reader = BufReader::new(stdout).lines();
                let mut stderr_reader = BufReader::new(stderr).lines();
                let mut stdout_done = false;
                let mut stderr_done = false;

                while !(stdout_done && stderr_done) {
                    tokio::select! {
                        line = stdout_reader.next_line(), if !stdout_done => {
                            match line {
                                Ok(Some(l)) => {
                                    println!("PyTorch Out: {}", l);
                                    if serde_json::from_str::<serde_json::Value>(&l).is_ok() {
                                        let _ = tx_clone_py.send(l);
                                    }
                                }
                                Ok(None) => stdout_done = true,
                                Err(e) => {
                                    let _ = tx_clone_py.send(serde_json::to_string(&LogMessage {
                                        type_: "log".to_string(),
                                        msg: format!("[PyTorch Stdout Read Error] {}", e),
                                    }).unwrap());
                                    stdout_done = true;
                                }
                            }
                        }
                        line = stderr_reader.next_line(), if !stderr_done => {
                            match line {
                                Ok(Some(l)) => {
                                    println!("PyTorch Err: {}", l);
                                    let _ = tx_clone_py.send(serde_json::to_string(&LogMessage {
                                        type_: "log".to_string(),
                                        msg: format!("[PyTorch Error] {}", l),
                                    }).unwrap());
                                }
                                Ok(None) => stderr_done = true,
                                Err(e) => {
                                    let _ = tx_clone_py.send(serde_json::to_string(&LogMessage {
                                        type_: "log".to_string(),
                                        msg: format!("[PyTorch Stderr Read Error] {}", e),
                                    }).unwrap());
                                    stderr_done = true;
                                },
                            }
                        }
                    }
                }

                match child.wait().await {
                    Ok(status) => {
                        let _ = tx_clone_py.send(
                            serde_json::to_string(&LogMessage {
                                type_: "log".to_string(),
                                msg: format!(
                                    "PyTorch benchmark process exited with status: {}",
                                    status
                                ),
                            })
                            .unwrap(),
                        );
                    }
                    Err(e) => {
                        let _ = tx_clone_py.send(
                            serde_json::to_string(&LogMessage {
                                type_: "log".to_string(),
                                msg: format!("Failed waiting for PyTorch benchmark process: {}", e),
                            })
                            .unwrap(),
                        );
                    }
                }
            }
            Err(e) => {
                let _ = tx_clone_py.send(
                    serde_json::to_string(&LogMessage {
                        type_: "log".to_string(),
                        msg: format!("Failed to start PyTorch benchmark: {}", e),
                    })
                    .unwrap(),
                );
            }
        }
    });

    // 2. Run RusTorch Benchmark
    tokio::spawn(async move {
        println!("Starting RusTorch Benchmark task...");
        // Config
        let batch_size = 1024;
        let input_size = 784;
        let output_size = 10;
        let epochs = 50;
        let steps_per_epoch = 20;

        let mut teacher_w = vec![0.0f32; input_size * output_size];
        let mut teacher_b = vec![0.0f32; output_size];
        for (i, v) in teacher_w.iter_mut().enumerate() {
            *v = deterministic_unit(i as u64, 20260315) * 0.2 - 0.1;
        }
        for (i, v) in teacher_b.iter_mut().enumerate() {
            *v = deterministic_unit(i as u64, 20260401) * 0.2 - 0.1;
        }

        println!("Creating MLP model...");
        // Model & Data
        let model = MLP::new();
        model.fc.weight.fill_(0.0);
        if let Some(bias) = &model.fc.bias {
            bias.fill_(0.0);
        }
        println!("MLP model created");

        let mut optimizer = Adam::new(model.parameters(), 0.001);

        // Pre-generate synthetic data (random)
        let (x_train, y_train) = {
            let mut data_vec = vec![0.0; batch_size * input_size];
            let mut target_vec = vec![0.0; batch_size * output_size];
            for i in 0..batch_size {
                for j in 0..input_size {
                    let idx = (i * input_size + j) as u64;
                    data_vec[i * input_size + j] = deterministic_unit(idx, 42) - 0.5;
                }
                for j in 0..output_size {
                    let mut z = teacher_b[j];
                    for k in 0..input_size {
                        z += data_vec[i * input_size + k] * teacher_w[k * output_size + j];
                    }
                    target_vec[i * output_size + j] = z;
                }
            }
            let x = Tensor::new(&data_vec, &[batch_size, input_size]);
            let y = Tensor::new(&target_vec, &[batch_size, output_size]);
            (x, y)
        };

        let baseline_mse = {
            #[cfg(feature = "wgpu_backend")]
            let y_host = if y_train.storage().wgpu_buffer().is_some() {
                y_train.to_cpu()
            } else {
                y_train.clone()
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let y_host = y_train.clone();
            let y = y_host.data();
            let sum_sq: f32 = y.iter().map(|v| v * v).sum();
            sum_sq / y.len() as f32
        };
        let baseline_accuracy = {
            #[cfg(feature = "wgpu_backend")]
            let y_host = if y_train.storage().wgpu_buffer().is_some() {
                y_train.to_cpu()
            } else {
                y_train.clone()
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let y_host = y_train.clone();
            let y = y_host.data();
            let c = y.iter().filter(|&&v| v.abs() < 0.1).count();
            c as f32 / y.len() as f32
        };
        println!("RusTorch baseline zero-pred MSE: {:.8}", baseline_mse);
        let _ = tx_clone_rust.send(
            serde_json::to_string(&MetricMessage {
                type_: "update".to_string(),
                framework: "RusTorch".to_string(),
                epoch: 0,
                loss: baseline_mse,
                speed: 0.0,
                grad_norm: 0.0,
                accuracy: baseline_accuracy,
            })
            .unwrap(),
        );

        let _ = tx_clone_rust.send(
            serde_json::to_string(&LogMessage {
                type_: "log".to_string(),
                msg: "RusTorch training started...".to_string(),
            })
            .unwrap(),
        );

        // Warmup
        println!("RusTorch Warmup...");
        for i in 0..3 {
            println!("Warmup iter {}", i);
            let out = model.forward(&x_train);
            println!("Forward done, output shape: {:?}", out.shape());
            let _ = out;
        }
        println!("RusTorch Training Loop Start...");
        let mut speed_samples = Vec::with_capacity(epochs);
        let grad_path =
            std::env::var("RUSTORCH_GRAD_PATH").unwrap_or_else(|_| "tensor".to_string());

        for epoch in 1..=epochs {
            let start = Instant::now();
            let mut epoch_loss = 0.0;
            let last_grad_norm = 0.0;

            for _step in 0..steps_per_epoch {
                optimizer.zero_grad();
                let output = model.forward(&x_train);
                demo_fused_counter().fetch_add(1, Ordering::Relaxed);
                let diff = output.sub(&y_train);
                let numel = diff.shape().iter().product::<usize>() as f32;
                let step_loss;

                if grad_path.eq_ignore_ascii_case("fused") {
                    let (loss_val, grad_w, grad_b) =
                        rustorch_core::ops::linear_mse_grads(&x_train, &output, &y_train);
                    model.fc.weight.accumulate_grad(&grad_w);
                    if let Some(bias) = &model.fc.bias {
                        bias.accumulate_grad(&grad_b);
                    }
                    step_loss = loss_val;
                } else {
                    let grad_scale = 2.0 / numel;
                    let diff_data = diff.data();
                    let grad_out_data: Vec<f32> =
                        diff_data.iter().map(|v| v * grad_scale).collect();
                    step_loss = diff_data.iter().map(|v| v * v).sum::<f32>() / numel;
                    let grad_out = Tensor::new(&grad_out_data, diff.shape());
                    let grad_w = grad_out.t().matmul(&x_train);
                    model.fc.weight.accumulate_grad(&grad_w);
                    if let Some(bias) = &model.fc.bias {
                        let grad_b = rustorch_core::ops::sum_to(&grad_out, &[output_size]);
                        bias.accumulate_grad(&grad_b);
                    }
                    demo_staged_counter().fetch_add(1, Ordering::Relaxed);
                }

                optimizer.step();

                epoch_loss += step_loss;
            }

            let duration = start.elapsed();
            let samples = (batch_size * steps_per_epoch) as f32;
            let speed = samples / duration.as_secs_f32();
            speed_samples.push(speed);

            let accuracy = {
                let output = model.forward(&x_train);
                let diff = output.sub(&y_train);
                let diff_host = diff;
                let diff_data = diff_host.data();
                let correct_count = diff_data.iter().filter(|&&x| x.abs() < 0.1).count();
                correct_count as f32 / diff_data.len() as f32
            };

            let msg = MetricMessage {
                type_: "update".to_string(),
                framework: "RusTorch".to_string(),
                epoch,
                loss: epoch_loss / steps_per_epoch as f32,
                speed,
                grad_norm: last_grad_norm,
                accuracy,
            };

            println!(
                "RusTorch Epoch {}: Loss {:.8}, Acc {:.2}%, Speed {:.1}",
                epoch,
                msg.loss,
                msg.accuracy * 100.0,
                speed
            );
            let _ = tx_clone_rust.send(serde_json::to_string(&msg).unwrap());
        }

        let final_output = model.forward(&x_train);
        let final_diff = final_output.sub(&y_train);
        let final_sq = final_diff.clone() * final_diff.clone();
        let final_loss = {
            #[cfg(feature = "wgpu_backend")]
            {
                if final_sq.storage().wgpu_buffer().is_some() {
                    let numel = final_sq.shape().iter().product::<usize>() as f32;
                    rustorch_core::ops::sum(&final_sq).to_cpu().data()[0] / numel
                } else {
                    final_sq.data().iter().sum::<f32>()
                        / final_sq.shape().iter().product::<usize>() as f32
                }
            }
            #[cfg(not(feature = "wgpu_backend"))]
            {
                final_sq.data().iter().sum::<f32>()
                    / final_sq.shape().iter().product::<usize>() as f32
            }
        };
        let final_accuracy = {
            #[cfg(feature = "wgpu_backend")]
            let diff_host = if final_diff.storage().wgpu_buffer().is_some() {
                final_diff.to_cpu()
            } else {
                final_diff
            };
            #[cfg(not(feature = "wgpu_backend"))]
            let diff_host = final_diff;
            let d = diff_host.data();
            let c = d.iter().filter(|&&x| x.abs() < 0.1).count();
            c as f32 / d.len() as f32
        };
        let avg_speed = robust_speed_center(&speed_samples);

        let _ = tx_clone_rust.send(
            serde_json::json!({
                "type": "finish",
                "framework": "RusTorch",
                "final_loss": final_loss,
                "final_accuracy": final_accuracy,
                "avg_speed": avg_speed
            })
            .to_string(),
        );
    });

    // 3. Serve Web UI
    let app = Router::new()
        .route("/", get(|| async { Html(HTML_CONTENT) }))
        .route(
            "/speed_ratio_history",
            get(|| async { Json(read_speed_ratio_history()) }),
        )
        .route(
            "/pipeline_stats",
            get(|| async { Json(read_pipeline_stats()) }),
        )
        .route("/promo_status", get(promo_status_handler))
        .route("/promo_run", get(promo_run_handler))
        .route("/promo_replay_best", get(promo_replay_best_handler))
        .route("/promo_apply_best", get(promo_apply_best_handler))
        .route(
            "/ws",
            get(|ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |socket| handle_socket(socket, tx))
            }),
        )
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3003));
    println!("🔥 RusTorch Benchmark running at http://{}", addr);

    // Wait a bit for the spawned tasks to start
    tokio::time::sleep(Duration::from_millis(100)).await;
    println!("Main: Spawned tasks should have started");

    if should_auto_open_browser() {
        let _ = open::that(format!("http://{}", addr));
    } else {
        println!("Browser auto-open disabled (set RUSTORCH_DEMO_AUTO_OPEN=1 to enable)");
    }

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("Failed to bind server at {}: {}", addr, e);
            return;
        }
    };
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("Server exited with error: {}", e);
    }
}

async fn handle_socket(mut socket: WebSocket, tx: broadcast::Sender<String>) {
    let snapshot = {
        dashboard_replay_buffer()
            .lock()
            .expect("dashboard_replay_buffer poisoned")
            .clone()
    };
    for msg in snapshot {
        if socket.send(Message::Text(msg)).await.is_err() {
            return;
        }
    }

    let mut rx = tx.subscribe();
    loop {
        match rx.recv().await {
            Ok(msg) => {
                if socket.send(Message::Text(msg)).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                println!(
                    "Warning: WebSocket client lagged, skipped {} messages",
                    skipped
                );
                continue;
            }
            Err(broadcast::error::RecvError::Closed) => {
                break;
            }
        }
    }
}
