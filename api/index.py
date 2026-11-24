"""
Vercel Serverless Function Entrypoint for Korba PM2.5 & PM10 Forecasting System.
Provides a modern web landing portal, system health check, and REST API endpoints.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from urllib.parse import urlparse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path.rstrip("/")

        if path in ("", "/", "/index", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            
            html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Korba PM2.5 & PM10 Air Quality Forecasting System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2980B9;
            --primary-dark: #1F618D;
            --accent: #27AE60;
            --dark: #2C3E50;
            --light: #F8F9FA;
            --card-bg: #FFFFFF;
            --border: #E5E8E8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #F4F6F7 0%, #EAEDED 100%);
            color: var(--dark);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background: linear-gradient(135deg, #1A5276 0%, #2980B9 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        header h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: 10px; }
        header p { font-size: 1.1rem; opacity: 0.9; max-width: 800px; margin: 0 auto; }
        .badge-bar { margin-top: 15px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.3);
        }
        main {
            flex: 1;
            max-width: 1000px;
            width: 100%;
            margin: 30px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 25px;
        }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 28px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
            border: 1px solid var(--border);
        }
        .card h2 {
            font-size: 1.4rem;
            color: #1A5276;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .cta-box {
            background: linear-gradient(135deg, #EBF5FB 0%, #D4E6F1 100%);
            border: 2px solid #A9CCE3;
            text-align: center;
            padding: 30px;
        }
        .cta-btn {
            display: inline-block;
            background: #27AE60;
            color: white;
            text-decoration: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 15px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        }
        .cta-btn:hover {
            background: #219653;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(39, 174, 96, 0.4);
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid-2 { grid-template-columns: 1fr; }
            header h1 { font-size: 1.6rem; }
        }
        ul { padding-left: 20px; line-height: 1.8; }
        code {
            background: #F2F4F4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: #C0392B;
        }
        .api-link {
            display: inline-block;
            background: #2980B9;
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 8px;
        }
        footer {
            text-align: center;
            padding: 20px;
            font-size: 0.85rem;
            color: #7F8C8D;
            border-top: 1px solid var(--border);
            background: white;
        }
    </style>
</head>
<body>
    <header>
        <h1>🌫️ Korba Air Quality Forecasting System</h1>
        <p>Operational 72-Hour PM2.5 & PM10 Deep Learning Forecaster for Industrial Mining & Ambient Baselines</p>
        <div class="badge-bar">
            <span class="badge">✅ Production Ready</span>
            <span class="badge">🧠 Multi-Station LSTM v1.0</span>
            <span class="badge">🎯 CPCB / CECB Aligned</span>
            <span class="badge">⚡ Vercel Serverless Ready</span>
        </div>
    </header>

    <main>
        <div class="card cta-box">
            <h2>🚀 Full Interactive Dashboard Deployment</h2>
            <p>The stateful interactive dashboard features 4-stage guided forecasting, interactive Plotly charts, 10th–90th percentile conformal intervals, and automated PDF health bulletin downloads.</p>
            <p style="margin-top: 10px; font-size: 0.95rem; color: #566573;">For complete WebSocket & reactive session support, access the live dashboard or launch with Streamlit Cloud / Docker:</p>
            <a href="http://localhost:8501" class="cta-btn" target="_blank">🌐 Open Live Dashboard (Port 8501)</a>
        </div>

        <div class="grid-2">
            <div class="card">
                <h2>📡 REST API Endpoints</h2>
                <p>This Vercel serverless deployment provides live JSON REST endpoints:</p>
                <div style="margin-top: 15px;">
                    <strong>1. Health Check:</strong><br>
                    <code>GET /api/health</code><br>
                    <a href="/api/health" class="api-link" target="_blank">Test /api/health →</a>
                </div>
                <div style="margin-top: 15px;">
                    <strong>2. Monitoring Stations:</strong><br>
                    <code>GET /api/stations</code><br>
                    <a href="/api/stations" class="api-link" target="_blank">Test /api/stations →</a>
                </div>
                <div style="margin-top: 15px;">
                    <strong>3. Model Specs & Schema:</strong><br>
                    <code>GET /api/info</code><br>
                    <a href="/api/info" class="api-link" target="_blank">Test /api/info →</a>
                </div>
            </div>

            <div class="card">
                <h2>⚙️ Architecture Highlights</h2>
                <ul>
                    <li><strong>4 Korba Stations:</strong> Banki Colliery, SECL Gevra, Rampur, Urja Nagar.</li>
                    <li><strong>Adaptive Scaling:</strong> Dynamically preserves regional concentration regimes (Korba ~30 µg/m³ vs Delhi ~150 µg/m³).</li>
                    <li><strong>Strict Monotonicity:</strong> Non-crossing quantiles q<sub>10</sub> &le; q<sub>50</sub> &le; q<sub>90</sub> with physical non-negativity floor.</li>
                    <li><strong>Diurnal Modulation:</strong> 24h harmonic profile with smooth exponential boundary continuity.</li>
                </ul>
            </div>
        </div>

        <div class="card">
            <h2>🚀 Production Deployment Recommendations</h2>
            <p style="margin-bottom: 12px;">Depending on your hosting preference:</p>
            <ul>
                <li><strong>Streamlit Community Cloud (Free & Recommended):</strong> Push repository to GitHub, go to <a href="https://share.streamlit.io" target="_blank">share.streamlit.io</a>, select <code>app/dashboard.py</code> and deploy in 1 click with persistent WebSocket connections!</li>
                <li><strong>Docker / Container (Render, Railway, Fly.io, Cloud Run):</strong> Use the included multi-stage <code>Dockerfile</code> and <code>Procfile</code>.</li>
                <li><strong>Vercel:</strong> This serverless entrypoint runs instantly via <code>vercel.json</code>.</li>
            </ul>
        </div>
    </main>

    <footer>
        Korba PM2.5 & PM10 Air Quality Forecasting System • Production Build v1.0 • Built with PyTorch, Streamlit & ReportLab
    </footer>
</body>
</html>"""
            self.wfile.write(html_content.encode("utf-8"))

        elif path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "Korba AQI Forecasting Engine",
                "version": "1.0.0",
                "model": "MultiStationLSTM",
                "quantiles": ["q10", "q50", "q90"],
                "horizon_hours": 72,
                "environment": os.environ.get("VERCEL_ENV", "production")
            }
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

        elif path == "/api/stations":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "district": "Korba, Chhattisgarh, India",
                "regulatory_body": "CECB / CPCB",
                "stations": [
                    {
                        "name": "Banki Colliery, Korba - CECB",
                        "type": "Underground Coal Mine",
                        "coordinates": {"lat": 22.3833, "lon": 82.7167}
                    },
                    {
                        "name": "SECL Gevra, Korba - CECB",
                        "type": "Open-Cast Mega Coal Mine",
                        "coordinates": {"lat": 22.3333, "lon": 82.5833}
                    },
                    {
                        "name": "Rampur, Korba - CECB",
                        "type": "Industrial & Urban Core",
                        "coordinates": {"lat": 22.3500, "lon": 82.6833}
                    },
                    {
                        "name": "Urja Nagar, Korba - CECB",
                        "type": "Power Plant Residential & Commercial",
                        "coordinates": {"lat": 22.3667, "lon": 82.7333}
                    }
                ]
            }
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

        elif path == "/api/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "project": "Korba PM2.5 & PM10 Air Quality Forecasting System",
                "architecture": "Multi-Station Dilated Conv1D + Bidirectional LSTM + Quantile Output Heads",
                "parameters": {
                    "lookback_hours": 360,
                    "forecast_hours": 72,
                    "target_pollutants": ["PM2.5", "PM10"],
                    "frequency": "Hourly continuous"
                },
                "regulatory_standards": {
                    "PM2.5_24h_standard_ug_m3": 60.0,
                    "PM10_24h_standard_ug_m3": 100.0,
                    "source": "CPCB National Ambient Air Quality Standards (NAAQS)"
                }
            }
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "error": "Not Found",
                "message": f"Endpoint '{self.path}' does not exist.",
                "available_endpoints": ["/", "/api/health", "/api/stations", "/api/info"]
            }
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))
