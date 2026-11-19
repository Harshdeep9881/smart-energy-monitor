from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone
import sqlite3
import numpy as np
from sklearn.ensemble import IsolationForest

app = FastAPI(title="Smart Energy Monitor")
DB = "energy.db"


def connect_db():
    return sqlite3.connect(DB)


# Create the table and safely add new columns to an existing database.
with connect_db() as con:
    con.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            device TEXT,
            adc REAL,
            current_a REAL,
            voltage_v REAL,
            power_w REAL,
            energy_wh REAL
        )
    """)
    existing_columns = {
        row[1] for row in con.execute("PRAGMA table_info(readings)").fetchall()
    }
    for column, sql_type in [
        ("current_a", "REAL"),
        ("voltage_v", "REAL"),
        ("power_w", "REAL"),
        ("energy_wh", "REAL"),
    ]:
        if column not in existing_columns:
            con.execute(f"ALTER TABLE readings ADD COLUMN {column} {sql_type}")


class Reading(BaseModel):
    device: str = "ESP12E"
    adc: float
    current_a: Optional[float] = None
    voltage_v: Optional[float] = None


@app.post("/api/readings")
def receive_reading(data: Reading):
    timestamp_dt = datetime.now(timezone.utc)
    timestamp = timestamp_dt.isoformat()
    power_w = (
        data.voltage_v * data.current_a
        if data.voltage_v is not None and data.current_a is not None
        else None
    )

    with connect_db() as con:
        previous = con.execute(
            "SELECT timestamp, power_w, energy_wh FROM readings ORDER BY id DESC LIMIT 1"
        ).fetchone()

        energy_wh = previous[2] if previous and previous[2] is not None else 0.0
        if previous and previous[1] is not None and power_w is not None:
            try:
                previous_dt = datetime.fromisoformat(previous[0])
                elapsed_seconds = max(0.0, (timestamp_dt - previous_dt).total_seconds())
                # Trapezoidal integration: average adjacent power readings over time.
                energy_wh += ((previous[1] + power_w) / 2.0) * elapsed_seconds / 3600.0
            except (ValueError, TypeError):
                pass

        con.execute(
            """INSERT INTO readings
               (timestamp, device, adc, current_a, voltage_v, power_w, energy_wh)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, data.device, data.adc, data.current_a,
             data.voltage_v, power_w, energy_wh)
        )

    return {
        "status": "saved",
        "adc": data.adc,
        "current_a": data.current_a,
        "voltage_v": data.voltage_v,
        "power_w": power_w,
        "energy_wh": energy_wh,
    }


@app.get("/api/readings")
def get_readings():
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with connect_db() as con:
        rows = con.execute("""
            SELECT timestamp, device, adc, current_a, voltage_v, power_w, energy_wh
            FROM readings WHERE timestamp >= ? ORDER BY id
        """, (cutoff,)).fetchall()

    rows.reverse()
    return [
        {
            "timestamp": r[0], "device": r[1], "adc": r[2],
            "current_a": r[3], "voltage_v": r[4],
            "power_w": r[5], "energy_wh": r[6]
        }
        for r in rows
    ]


@app.get("/api/ml")
def detect_anomalies():
    with connect_db() as con:
        rows = con.execute(
            "SELECT id, timestamp, adc FROM readings ORDER BY id"
        ).fetchall()

    if len(rows) < 20:
        return {
            "status": "collecting_data",
            "message": "Collect at least 20 readings to run anomaly detection.",
            "anomalies": []
        }

    values = np.array([r[2] for r in rows]).reshape(-1, 1)
    model = IsolationForest(contamination=0.05, random_state=42)
    labels = model.fit_predict(values)
    anomalies = [
        {"timestamp": rows[i][1], "adc": rows[i][2]}
        for i, label in enumerate(labels) if label == -1
    ]

    return {
        "status": "complete",
        "algorithm": "Isolation Forest",
        "samples": len(rows),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies[-20:]
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Smart Energy Monitor — Live Dashboard</title>
      <meta name="description" content="Real-time smart energy monitoring dashboard with ML-powered anomaly detection for ESP-12E sensor pipeline.">
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
      <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg-deep: #0a0e1a;
          --bg-card: rgba(17, 24, 42, 0.65);
          --bg-card-hover: rgba(25, 35, 60, 0.75);
          --border-glass: rgba(99, 130, 190, 0.12);
          --border-glow: rgba(96, 165, 250, 0.25);
          --text-primary: #eaf0fb;
          --text-secondary: #8899b4;
          --text-muted: #5e6e88;
          --accent-blue: #60a5fa;
          --accent-cyan: #22d3ee;
          --accent-violet: #a78bfa;
          --accent-emerald: #34d399;
          --accent-amber: #fbbf24;
          --accent-rose: #fb7185;
          --gradient-hero: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
          --shadow-card: 0 4px 30px rgba(0, 0, 0, 0.35);
          --shadow-glow: 0 0 40px rgba(96, 165, 250, 0.08);
          --radius: 16px;
          --radius-sm: 10px;
          --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        body {
          font-family: 'Inter', system-ui, -apple-system, sans-serif;
          background: var(--bg-deep);
          color: var(--text-primary);
          min-height: 100vh;
          overflow-x: hidden;
        }

        /* Animated background orbs */
        body::before, body::after {
          content: '';
          position: fixed;
          border-radius: 50%;
          filter: blur(120px);
          opacity: 0.15;
          pointer-events: none;
          z-index: 0;
        }
        body::before {
          width: 600px; height: 600px;
          background: radial-gradient(circle, #60a5fa, transparent 70%);
          top: -150px; right: -100px;
          animation: floatOrb 18s ease-in-out infinite;
        }
        body::after {
          width: 500px; height: 500px;
          background: radial-gradient(circle, #a78bfa, transparent 70%);
          bottom: -100px; left: -80px;
          animation: floatOrb 22s ease-in-out infinite reverse;
        }
        @keyframes floatOrb {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(40px, -30px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.95); }
        }

        .app-container {
          position: relative;
          z-index: 1;
          max-width: 1360px;
          margin: 0 auto;
          padding: 28px 32px 60px;
        }

        /* ─── Header ─── */
        .header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 16px;
          margin-bottom: 32px;
        }
        .header-left { display: flex; align-items: center; gap: 16px; }
        .logo-icon {
          width: 48px; height: 48px;
          background: var(--gradient-hero);
          border-radius: 14px;
          display: flex; align-items: center; justify-content: center;
          font-size: 24px;
          box-shadow: 0 0 24px rgba(96, 165, 250, 0.3);
          animation: logoPulse 3s ease-in-out infinite;
        }
        @keyframes logoPulse {
          0%, 100% { box-shadow: 0 0 24px rgba(96, 165, 250, 0.3); }
          50% { box-shadow: 0 0 36px rgba(167, 139, 250, 0.45); }
        }
        .header h1 {
          font-size: 26px; font-weight: 800;
          background: var(--gradient-hero);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          letter-spacing: -0.3px;
        }
        .header-subtitle {
          font-size: 13px; color: var(--text-secondary);
          margin-top: 2px; letter-spacing: 0.2px;
        }

        .live-badge {
          display: inline-flex; align-items: center; gap: 8px;
          background: rgba(52, 211, 153, 0.1);
          border: 1px solid rgba(52, 211, 153, 0.25);
          padding: 8px 16px;
          border-radius: 50px;
          font-size: 13px; font-weight: 600;
          color: var(--accent-emerald);
        }
        .live-dot {
          width: 8px; height: 8px;
          background: var(--accent-emerald);
          border-radius: 50%;
          animation: liveBlink 1.5s ease-in-out infinite;
        }
        @keyframes liveBlink {
          0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--accent-emerald); }
          50% { opacity: 0.3; box-shadow: none; }
        }

        /* ─── Stat Cards ─── */
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
          margin-bottom: 24px;
        }
        @media (max-width: 960px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 540px) { .stats-grid { grid-template-columns: 1fr; } }

        .stat-card {
          background: var(--bg-card);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border: 1px solid var(--border-glass);
          border-radius: var(--radius);
          padding: 22px 24px;
          box-shadow: var(--shadow-card);
          transition: all var(--transition);
          position: relative;
          overflow: hidden;
        }
        .stat-card:hover {
          background: var(--bg-card-hover);
          border-color: var(--border-glow);
          transform: translateY(-3px);
          box-shadow: var(--shadow-card), var(--shadow-glow);
        }
        .stat-card::before {
          content: '';
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 3px;
          border-radius: var(--radius) var(--radius) 0 0;
        }
        .stat-card:nth-child(1)::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
        .stat-card:nth-child(2)::before { background: linear-gradient(90deg, var(--accent-violet), var(--accent-blue)); }
        .stat-card:nth-child(3)::before { background: linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan)); }
        .stat-card:nth-child(4)::before { background: linear-gradient(90deg, var(--accent-amber), var(--accent-rose)); }

        .stat-header {
          display: flex; align-items: center; justify-content: space-between;
          margin-bottom: 12px;
        }
        .stat-label {
          font-size: 12px; font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 1px;
          color: var(--text-secondary);
        }
        .stat-icon {
          width: 36px; height: 36px;
          border-radius: var(--radius-sm);
          display: flex; align-items: center; justify-content: center;
          font-size: 18px;
        }
        .stat-card:nth-child(1) .stat-icon { background: rgba(96, 165, 250, 0.12); }
        .stat-card:nth-child(2) .stat-icon { background: rgba(167, 139, 250, 0.12); }
        .stat-card:nth-child(3) .stat-icon { background: rgba(52, 211, 153, 0.12); }
        .stat-card:nth-child(4) .stat-icon { background: rgba(251, 191, 36, 0.12); }

        .stat-value {
          font-size: 32px; font-weight: 800;
          letter-spacing: -0.5px;
          line-height: 1.1;
          margin-bottom: 4px;
          transition: all 0.4s ease;
        }
        .stat-card:nth-child(1) .stat-value { color: var(--accent-cyan); }
        .stat-card:nth-child(2) .stat-value { color: var(--accent-violet); }
        .stat-card:nth-child(3) .stat-value { color: var(--accent-emerald); }
        .stat-card:nth-child(4) .stat-value { color: var(--accent-amber); }

        .stat-value.flash { animation: valueFlash 0.5s ease; }
        @keyframes valueFlash {
          0% { transform: scale(1); }
          40% { transform: scale(1.08); }
          100% { transform: scale(1); }
        }

        .stat-sub {
          font-size: 12px; color: var(--text-muted);
          display: flex; align-items: center; gap: 4px;
        }
        .stat-trend { font-weight: 600; }
        .stat-trend.up { color: var(--accent-emerald); }
        .stat-trend.down { color: var(--accent-rose); }

        .sparkline-container {
          margin-top: 14px;
          height: 36px;
        }
        .sparkline-container canvas {
          width: 100%; height: 36px;
        }

        /* ─── Panel Cards ─── */
        .panels-row {
          display: grid;
          grid-template-columns: 2fr 1fr;
          gap: 16px;
          margin-bottom: 16px;
        }
        @media (max-width: 860px) { .panels-row { grid-template-columns: 1fr; } }

        .panel {
          background: var(--bg-card);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border: 1px solid var(--border-glass);
          border-radius: var(--radius);
          box-shadow: var(--shadow-card);
          overflow: hidden;
        }
        .panel-header {
          display: flex; align-items: center; justify-content: space-between;
          padding: 20px 24px;
          border-bottom: 1px solid var(--border-glass);
        }
        .panel-title {
          font-size: 16px; font-weight: 700;
          display: flex; align-items: center; gap: 10px;
        }
        .chart-toolbar { display: flex; align-items: center; gap: 6px; }
        .range-btn {
          border: 1px solid var(--border-glass); background: transparent;
          color: var(--text-muted); border-radius: 7px; padding: 6px 9px;
          font: inherit; font-size: 11px; cursor: pointer; transition: all .2s;
        }
        .range-btn:hover { color: var(--text-primary); border-color: var(--accent-cyan); }
        .range-btn.active {
          color: var(--accent-cyan); background: rgba(34, 211, 238, .1);
          border-color: rgba(34, 211, 238, .35);
        }
        .panel-body { padding: 20px 24px; }
        .panel-body-flush { padding: 0; }

        /* ─── Chart ─── */
        #chartContainer {
          position: relative;
          width: 100%;
          padding: 20px 24px 16px;
        }
        #chart {
          width: 100%;
          height: 280px;
          display: block;
        }
        .chart-legend {
          display: flex; gap: 20px;
          padding: 0 24px 16px;
          font-size: 12px; color: var(--text-secondary);
        }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-dot {
          width: 10px; height: 10px;
          border-radius: 50%;
        }

        /* ─── Anomaly section ─── */
        .anomaly-list { display: flex; flex-direction: column; gap: 8px; }
        .anomaly-item {
          display: flex; align-items: center; gap: 12px;
          padding: 12px 14px;
          border-radius: var(--radius-sm);
          background: rgba(251, 113, 133, 0.06);
          border: 1px solid rgba(251, 113, 133, 0.1);
          transition: all var(--transition);
        }
        .anomaly-item:hover {
          background: rgba(251, 113, 133, 0.1);
          border-color: rgba(251, 113, 133, 0.2);
        }
        .anomaly-dot {
          width: 8px; height: 8px;
          background: var(--accent-rose);
          border-radius: 50%;
          flex-shrink: 0;
          box-shadow: 0 0 8px rgba(251, 113, 133, 0.4);
        }
        .anomaly-text { font-size: 13px; color: var(--text-secondary); }
        .anomaly-val { font-weight: 700; color: var(--accent-rose); }
        .anomaly-empty {
          text-align: center;
          padding: 30px;
          color: var(--text-muted);
          font-size: 13px;
        }

        /* ─── Table ─── */
        .table-panel { margin-bottom: 40px; }
        .table-wrap { overflow-x: auto; }
        table {
          width: 100%; border-collapse: collapse;
          font-size: 13px;
        }
        thead th {
          text-align: left;
          font-size: 11px; font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 1px;
          color: var(--text-muted);
          padding: 14px 20px;
          border-bottom: 1px solid var(--border-glass);
          position: sticky; top: 0;
          background: rgba(17, 24, 42, 0.9);
          backdrop-filter: blur(12px);
        }
        tbody td {
          padding: 13px 20px;
          border-bottom: 1px solid rgba(99, 130, 190, 0.06);
          color: var(--text-secondary);
          transition: all var(--transition);
        }
        tbody tr { transition: background var(--transition); }
        tbody tr:hover { background: rgba(96, 165, 250, 0.04); }
        tbody tr:hover td { color: var(--text-primary); }

        .device-tag {
          display: inline-flex; align-items: center; gap: 6px;
          padding: 4px 10px;
          border-radius: 6px;
          background: rgba(96, 165, 250, 0.08);
          border: 1px solid rgba(96, 165, 250, 0.15);
          font-size: 12px; font-weight: 600;
          color: var(--accent-blue);
        }
        .adc-cell { font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text-primary) !important; }

        /* ─── Buttons ─── */
        .btn {
          display: inline-flex; align-items: center; gap: 8px;
          padding: 9px 18px;
          border: 1px solid var(--border-glass);
          border-radius: var(--radius-sm);
          background: rgba(96, 165, 250, 0.08);
          color: var(--accent-blue);
          font-size: 13px; font-weight: 600;
          cursor: pointer;
          transition: all var(--transition);
          font-family: inherit;
        }
        .btn:hover {
          background: rgba(96, 165, 250, 0.15);
          border-color: var(--border-glow);
          transform: translateY(-1px);
        }
        .btn:active { transform: scale(0.97); }

        /* ─── Device Info ─── */
        .device-info { display: flex; flex-direction: column; gap: 14px; }
        .device-row {
          display: flex; justify-content: space-between;
          padding: 10px 0;
          border-bottom: 1px solid rgba(99, 130, 190, 0.06);
          font-size: 13px;
        }
        .device-row:last-child { border-bottom: none; }
        .device-key { color: var(--text-muted); font-weight: 500; }
        .device-val { color: var(--text-primary); font-weight: 600; }
        .status-online {
          display: inline-flex; align-items: center; gap: 6px;
          color: var(--accent-emerald); font-weight: 600;
        }
        .status-online::before {
          content: '';
          width: 7px; height: 7px;
          background: var(--accent-emerald);
          border-radius: 50%;
          box-shadow: 0 0 8px rgba(52, 211, 153, 0.5);
        }

        /* ─── Scrollbar ─── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(99, 130, 190, 0.2); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(99, 130, 190, 0.35); }

        /* ─── Entrance animations ─── */
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-in {
          animation: fadeUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
          opacity: 0;
        }
        .delay-1 { animation-delay: 0.08s; }
        .delay-2 { animation-delay: 0.16s; }
        .delay-3 { animation-delay: 0.24s; }
        .delay-4 { animation-delay: 0.32s; }
        .delay-5 { animation-delay: 0.40s; }
        .delay-6 { animation-delay: 0.48s; }
      </style>
    </head>
    <body>
      <div class="app-container">

        <!-- Header -->
        <header class="header animate-in">
          <div class="header-left">
            <div class="logo-icon">⚡</div>
            <div>
              <h1>Smart Energy Monitor</h1>
              <div class="header-subtitle">ESP-12E · Current, power and energy · ML anomaly detection</div>
            </div>
          </div>
          <div class="live-badge">
            <span class="live-dot"></span>
            Live — auto-refreshing
          </div>
        </header>

        <!-- Stat Cards -->
        <div class="stats-grid">
          <div class="stat-card animate-in delay-1" id="card-adc">
            <div class="stat-header">
              <span class="stat-label">Latest ADC</span>
              <div class="stat-icon">📊</div>
            </div>
            <div class="stat-value" id="adc">--</div>
            <div class="stat-sub">Raw sensor counts</div>
            <div class="sparkline-container"><canvas id="spark-adc"></canvas></div>
          </div>
          <div class="stat-card animate-in delay-2">
            <div class="stat-header">
              <span class="stat-label">Estimated Current</span>
              <div class="stat-icon">〰️</div>
            </div>
            <div class="stat-value" id="current">--</div>
            <div class="stat-sub">Amperes (A)</div>
          </div>
          <div class="stat-card animate-in delay-2">
            <div class="stat-header">
              <span class="stat-label">Estimated Power</span>
              <div class="stat-icon">⚡</div>
            </div>
            <div class="stat-value" id="power">--</div>
            <div class="stat-sub" id="voltage-info">Watts (W)</div>
          </div>
          <div class="stat-card animate-in delay-2">
            <div class="stat-header">
              <span class="stat-label">Accumulated Energy</span>
              <div class="stat-icon">🔋</div>
            </div>
            <div class="stat-value" id="energy">--</div>
            <div class="stat-sub">Watt-hours (Wh), since power data began</div>
          </div>
          <div class="stat-card animate-in delay-3">
            <div class="stat-header">
              <span class="stat-label">Total Readings</span>
              <div class="stat-icon">🗂️</div>
            </div>
            <div class="stat-value" id="count">0</div>
            <div class="stat-sub" id="rate-info">Collecting data…</div>
          </div>
          <div class="stat-card animate-in delay-3">
            <div class="stat-header">
              <span class="stat-label">ML Anomalies</span>
              <div class="stat-icon">🧠</div>
            </div>
            <div class="stat-value" id="anomalies">--</div>
            <div class="stat-sub" id="mlstatus">Waiting for data</div>
          </div>
          <div class="stat-card animate-in delay-4">
            <div class="stat-header">
              <span class="stat-label">ADC Range</span>
              <div class="stat-icon">📐</div>
            </div>
            <div class="stat-value" id="range-val">--</div>
            <div class="stat-sub" id="range-info">Min — Max</div>
          </div>
        </div>

        <!-- Chart + Anomalies -->
        <div class="panels-row animate-in delay-5">
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">📈 Sensor Trend</div>
              <div class="chart-toolbar">
                <button class="range-btn" data-minutes="5" onclick="setChartRange(5, this)">5m</button>
                <button class="range-btn" data-minutes="10" onclick="setChartRange(10, this)">10m</button>
                <button class="range-btn" data-minutes="30" onclick="setChartRange(30, this)">30m</button>
                <button class="range-btn active" data-minutes="60" onclick="setChartRange(60, this)">1h</button>
                <button class="btn" onclick="refresh()">↻</button>
              </div>
            </div>
            <div id="chartContainer">
              <canvas id="chart"></canvas>
            </div>
            <div class="chart-legend">
              <div class="legend-item"><div class="legend-dot" style="background:var(--accent-cyan)"></div> ADC Value</div>
              <div class="legend-item"><div class="legend-dot" style="background:var(--accent-rose)"></div> Anomaly</div>
              <div class="legend-item"><div class="legend-dot" style="background:rgba(96,165,250,0.15)"></div> Normal Range</div>
            </div>
          </div>

          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">⚠️ Anomalies</div>
            </div>
            <div class="panel-body" id="anomaly-list-container">
              <div class="anomaly-empty">Run ML detection to see anomalies</div>
            </div>
          </div>
        </div>

        <!-- Table + Device Info -->
        <div class="panels-row animate-in delay-6">
          <div class="panel table-panel">
            <div class="panel-header">
              <div class="panel-title">📋 Recent Readings</div>
              <span style="font-size:12px;color:var(--text-muted)" id="table-info">Last 15 readings</span>
            </div>
            <div class="panel-body-flush table-wrap">
              <table>
                <thead><tr><th>Timestamp</th><th>Device</th><th>ADC</th><th>Current (A)</th><th>Power (W)</th></tr></thead>
                <tbody id="rows"></tbody>
              </table>
            </div>
          </div>

          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">🔌 Device Info</div>
            </div>
            <div class="panel-body device-info">
              <div class="device-row">
                <span class="device-key">Status</span>
                <span class="device-val status-online">Online</span>
              </div>
              <div class="device-row">
                <span class="device-key">Board</span>
                <span class="device-val">ESP-12E</span>
              </div>
              <div class="device-row">
                <span class="device-key">ADC Resolution</span>
                <span class="device-val">10-bit (0-1023)</span>
              </div>
              <div class="device-row">
                <span class="device-key">Power formula</span>
                <span class="device-val">Voltage × Current</span>
              </div>
              <div class="device-row">
                <span class="device-key">Energy method</span>
                <span class="device-val">Power over time</span>
              </div>
              <div class="device-row">
                <span class="device-key">ML Model</span>
                <span class="device-val">Isolation Forest</span>
              </div>
              <div class="device-row">
                <span class="device-key">Refresh Rate</span>
                <span class="device-val">3 s</span>
              </div>
              <div class="device-row">
                <span class="device-key">Last Seen</span>
                <span class="device-val" id="last-seen">--</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <script>
        let prevAdc = null;
        let sparkData = [];
        let anomalyTimestamps = new Set();
        let chartRangeMinutes = 60;

        function setChartRange(minutes, button) {
          chartRangeMinutes = minutes;
          document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
          button.classList.add('active');
          refresh();
        }

        async function refresh() {
          try {
            const response = await fetch('/api/readings');
            const data = await response.json();

            // ─ Count
            document.getElementById('count').textContent = data.length;

            // ─ Latest ADC
            const latestAdc = data.length ? Number(data[data.length - 1].adc) : null;
            const adcEl = document.getElementById('adc');
            if (latestAdc !== null) {
              adcEl.textContent = latestAdc.toFixed(1);
              if (prevAdc !== null && prevAdc !== latestAdc) {
                adcEl.classList.remove('flash');
                void adcEl.offsetWidth;
                adcEl.classList.add('flash');
              }
              prevAdc = latestAdc;
            } else {
              adcEl.textContent = '--';
            }

            // ─ Current, power and accumulated energy
            const latest = data.length ? data[data.length - 1] : null;
            document.getElementById('current').textContent =
              latest && latest.current_a !== null && latest.current_a !== undefined
                ? Number(latest.current_a).toFixed(3) : '--';
            document.getElementById('power').textContent =
              latest && latest.power_w !== null && latest.power_w !== undefined
                ? Number(latest.power_w).toFixed(3) : '--';
            document.getElementById('energy').textContent =
              latest && latest.energy_wh !== null && latest.energy_wh !== undefined && latest.power_w !== null
                ? Number(latest.energy_wh).toFixed(4) : '--';
            document.getElementById('voltage-info').textContent =
              latest && latest.voltage_v !== null && latest.voltage_v !== undefined
                ? Number(latest.voltage_v).toFixed(2) + ' V assumed/measured' : 'Voltage not provided';

            // ─ Range
            if (data.length) {
              const vals = data.map(r => Number(r.adc));
              const mn = Math.min(...vals), mx = Math.max(...vals);
              document.getElementById('range-val').textContent = (mx - mn).toFixed(1);
              document.getElementById('range-info').textContent =
                mn.toFixed(0) + ' — ' + mx.toFixed(0);
            }

            // ─ Rate info
            if (data.length >= 2) {
              const first = new Date(data[0].timestamp);
              const last = new Date(data[data.length - 1].timestamp);
              const spanMin = ((last - first) / 60000).toFixed(1);
              document.getElementById('rate-info').textContent =
                data.length + ' in ' + spanMin + ' min';
            }

            // ─ Last seen
            if (data.length) {
              const lastTs = new Date(data[data.length - 1].timestamp);
              document.getElementById('last-seen').textContent =
                lastTs.toLocaleTimeString();
            }

            // ─ Sparkline
            if (latestAdc !== null) {
              sparkData.push(latestAdc);
              if (sparkData.length > 30) sparkData.shift();
              drawSparkline('spark-adc', sparkData, '#22d3ee');
            }

            // ─ Table
            document.getElementById('rows').innerHTML =
              data.slice(-15).reverse().map(r => {
                const isAnomaly = anomalyTimestamps.has(r.timestamp);
                return '<tr>' +
                  '<td>' + new Date(r.timestamp).toLocaleTimeString() + '</td>' +
                  '<td><span class="device-tag">⬡ ' + r.device + '</span></td>' +
                  '<td class="adc-cell"' + (isAnomaly ? ' style="color:var(--accent-rose)"' : '') + '>' +
                  Number(r.adc).toFixed(2) +
                  (isAnomaly ? ' ⚠' : '') +
                  '</td>' +
                  '<td>' + (r.current_a !== null && r.current_a !== undefined ? Number(r.current_a).toFixed(3) : '--') + '</td>' +
                  '<td>' + (r.power_w !== null && r.power_w !== undefined ? Number(r.power_w).toFixed(3) : '--') + '</td>' +
                  '</tr>';
              }).join('');

            // ─ Chart
            const rangeStart = Date.now() - chartRangeMinutes * 60 * 1000;
            drawChart(data.filter(r => new Date(r.timestamp).getTime() >= rangeStart));

            // ─ ML
            const ml = await fetch('/api/ml').then(r => r.json());
            document.getElementById('anomalies').textContent =
              ml.status === 'complete' ? ml.anomaly_count : '--';
            document.getElementById('mlstatus').textContent =
              ml.message || (ml.algorithm + ': ' + ml.samples + ' samples');

            // Anomaly list & timestamps set
            anomalyTimestamps.clear();
            if (ml.status === 'complete' && ml.anomalies.length) {
              ml.anomalies.forEach(a => anomalyTimestamps.add(a.timestamp));
              document.getElementById('anomaly-list-container').innerHTML =
                '<div class="anomaly-list">' +
                ml.anomalies.slice(-8).reverse().map(a =>
                  '<div class="anomaly-item">' +
                  '<span class="anomaly-dot"></span>' +
                  '<div class="anomaly-text">' +
                  '<span class="anomaly-val">' + Number(a.adc).toFixed(2) + '</span>' +
                  ' · ' + new Date(a.timestamp).toLocaleTimeString() +
                  '</div></div>'
                ).join('') +
                '</div>';
            } else if (ml.status === 'complete') {
              document.getElementById('anomaly-list-container').innerHTML =
                '<div class="anomaly-empty">✅ No anomalies detected</div>';
            }
          } catch (e) {
            document.getElementById('mlstatus').textContent = 'Connection error';
          }
        }

        function drawSparkline(id, data, color) {
          const c = document.getElementById(id);
          if (!c) return;
          const dpr = window.devicePixelRatio || 1;
          const rect = c.parentElement.getBoundingClientRect();
          c.width = rect.width * dpr;
          c.height = 36 * dpr;
          c.style.width = rect.width + 'px';
          c.style.height = '36px';
          const ctx = c.getContext('2d');
          ctx.scale(dpr, dpr);
          const w = rect.width, h = 36;
          ctx.clearRect(0, 0, w, h);
          if (data.length < 2) return;
          const mn = Math.min(...data), mx = Math.max(...data);
          const rng = Math.max(mx - mn, 0.1);
          const pad = 3;
          ctx.beginPath();
          data.forEach((v, i) => {
            const x = i * (w / (data.length - 1));
            const y = pad + (1 - (v - mn) / rng) * (h - pad * 2);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.8;
          ctx.lineJoin = 'round';
          ctx.stroke();
          // fill
          const lastX = (data.length - 1) * (w / (data.length - 1));
          ctx.lineTo(lastX, h);
          ctx.lineTo(0, h);
          ctx.closePath();
          const grd = ctx.createLinearGradient(0, 0, 0, h);
          grd.addColorStop(0, color.replace(')', ',0.2)').replace('rgb', 'rgba'));
          grd.addColorStop(1, 'transparent');
          ctx.fillStyle = grd;
          ctx.fill();
        }

        function drawChart(data) {
          const c = document.getElementById('chart');
          const container = document.getElementById('chartContainer');
          const dpr = window.devicePixelRatio || 1;
          const cw = container.clientWidth - 48;
          c.width = cw * dpr;
          c.height = 280 * dpr;
          c.style.width = cw + 'px';
          c.style.height = '280px';
          const ctx = c.getContext('2d');
          ctx.scale(dpr, dpr);
          const w = cw, h = 280;
          ctx.clearRect(0, 0, w, h);

          if (!data.length) {
            ctx.fillStyle = '#5e6e88';
            ctx.font = '14px Inter, sans-serif';
            ctx.fillText('No readings in this time range yet', 20, 30);
            return;
          }

          const vals = data.map(r => Number(r.adc));
          const mn = Math.min(...vals), mx = Math.max(...vals);
          const range = Math.max(mx - mn, 1);
          const padLeft = 52, padRight = 16, padTop = 20, padBottom = 30;
          const plotW = w - padLeft - padRight;
          const plotH = h - padTop - padBottom;

          // Grid lines
          ctx.font = '11px Inter, sans-serif';
          for (let i = 0; i <= 5; i++) {
            const y = padTop + i * plotH / 5;
            ctx.strokeStyle = 'rgba(99, 130, 190, 0.08)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(w - padRight, y);
            ctx.stroke();
            ctx.fillStyle = '#5e6e88';
            ctx.textAlign = 'right';
            ctx.fillText((mx - i * range / 5).toFixed(0), padLeft - 8, y + 4);
          }

          // Mean line
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          const meanY = padTop + (mx - mean) / range * plotH;
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = 'rgba(167, 139, 250, 0.3)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(padLeft, meanY);
          ctx.lineTo(w - padRight, meanY);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(167, 139, 250, 0.5)';
          ctx.textAlign = 'left';
          ctx.fillText('avg ' + mean.toFixed(0), w - padRight + 4, meanY + 4);

          // Points array
          const points = vals.map((v, i) => ({
            x: padLeft + i * plotW / Math.max(vals.length - 1, 1),
            y: padTop + (mx - v) / range * plotH,
            ts: data[i].timestamp,
            val: v
          }));

          // Area fill
          ctx.beginPath();
          points.forEach((p, i) => {
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
          });
          ctx.lineTo(points[points.length - 1].x, padTop + plotH);
          ctx.lineTo(points[0].x, padTop + plotH);
          ctx.closePath();
          const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
          grad.addColorStop(0, 'rgba(34, 211, 238, 0.18)');
          grad.addColorStop(0.6, 'rgba(34, 211, 238, 0.04)');
          grad.addColorStop(1, 'transparent');
          ctx.fillStyle = grad;
          ctx.fill();

          // Line
          ctx.beginPath();
          points.forEach((p, i) => {
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
          });
          ctx.strokeStyle = '#22d3ee';
          ctx.lineWidth = 2.2;
          ctx.lineJoin = 'round';
          ctx.lineCap = 'round';
          ctx.stroke();

          // Anomaly markers
          points.forEach(p => {
            if (anomalyTimestamps.has(p.ts)) {
              ctx.beginPath();
              ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(251, 113, 133, 0.9)';
              ctx.fill();
              ctx.beginPath();
              ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
              ctx.strokeStyle = 'rgba(251, 113, 133, 0.3)';
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          });

          // Latest point glow
          if (points.length) {
            const last = points[points.length - 1];
            ctx.beginPath();
            ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#22d3ee';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(last.x, last.y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(34, 211, 238, 0.35)';
            ctx.lineWidth = 2;
            ctx.stroke();
          }

          // X-axis time labels
          const labelCount = Math.min(6, data.length);
          ctx.fillStyle = '#5e6e88';
          ctx.textAlign = 'center';
          ctx.font = '10px Inter, sans-serif';
          for (let i = 0; i < labelCount; i++) {
            const idx = labelCount === 1 ? 0 : Math.round(i * (data.length - 1) / (labelCount - 1));
            const x = padLeft + idx * plotW / Math.max(vals.length - 1, 1);
            ctx.fillText(
              new Date(data[idx].timestamp).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}),
              x, h - 6
            );
          }
        }

        refresh();
        setInterval(refresh, 3000);
      </script>
    </body>
    </html
    """
