## Live Demo

🚀 Try the application here: https://ev-station-placement-emegxm295cskbs4lejeksp.streamlit.app/

# ⚡ Hyperlocal EV Charging Station Placement Optimizer

> **DS Layer**: Commuter GPS clustering (DBSCAN) + demand heatmap &nbsp;|&nbsp; **OR Layer**: Set Cover ILP — minimum cost stations to cover all demand zones

## 🎯 Problem Statement
Where should EV charging stations be placed in a city to ensure every commuter demand hotspot has a station within 1.5 km — while minimising total installation cost under a budget cap?

## 🏗️ Architecture
```
Commuter GPS / Traffic Data
        │
        ▼
  DBSCAN Demand Zone Clustering     ← DS Layer
  (identify EV demand hotspots)
        │
        ▼
  Coverage Matrix
  (which candidate site covers which zone within radius?)
        │
        ▼
  Set Cover ILP                      ← OR Layer
  Minimise: Σ install_cost · y_s
  Subject to: every demand zone covered by ≥1 station
        │
        ▼
  Folium heatmap + coverage circles + sensitivity analysis
```

## 📦 Tech Stack
| Layer | Tool |
|-------|------|
| DS | `scikit-learn` DBSCAN, `folium` HeatMap |
| OR | `PuLP` CBC — Binary Set Cover ILP |
| Geo | Haversine distance, Folium coverage circles |
| Dashboard | `streamlit`, `plotly` |
| Data Source | Uber H3 / HERE Traffic API (synthetic here) |

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python ev_charging_optimizer.py
streamlit run app.py
```

## 📊 Sensitivity Analysis
The model also runs sensitivity analysis on coverage radius (0.5km → 3.0km) showing the trade-off between fewer stations (higher radius) vs. better accessibility:

| Radius (km) | Stations Opened | Total Cost (₹ lakh) |
|-------------|-----------------|----------------------|
| 0.5 | 18 | 72.4 |
| 1.0 | 10 | 41.2 |
| 1.5 | 6  | 26.8 |
| 2.0 | 4  | 18.3 |

## 📁 Files
```
├── ev_charging_optimizer.py   # Full pipeline + sensitivity analysis
├── app.py                     # Streamlit interactive map dashboard
├── requirements.txt
└── README.md
```

## 🌍 Real Data Sources
- [Uber Movement](https://movement.uber.com/) — city-level speed & OD data
- [HERE Maps Traffic API](https://developer.here.com/) — live traffic density
- [OpenStreetMap](https://www.openstreetmap.org/) — parking lots, road network
- [Kaggle EV datasets](https://www.kaggle.com/search?q=electric+vehicle+charging) — historical charging session data
