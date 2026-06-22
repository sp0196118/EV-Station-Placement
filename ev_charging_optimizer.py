"""
Hyperlocal EV Charging Station Placement Optimizer
====================================================
DS Layer  : GPS/commuter clustering (HDBSCAN / DBSCAN) + demand heatmap
OR Layer  : Set Cover ILP — place minimum stations to cover all demand zones
"""

import numpy as np
import pandas as pd
import pulp
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC COMMUTER GPS DATA
#    (replace with Uber H3 movement data / HERE traffic API)
# ─────────────────────────────────────────────
CITY = "Gurugram, Haryana"
CITY_CENTER = (28.4595, 77.0266)   # Gurugram

COMMUTER_HOTSPOTS = [
    (28.4949, 77.0878, "Cyber City DLF"),
    (28.4226, 77.0478, "Golf Course Road"),
    (28.4673, 77.0285, "Sohna Road"),
    (28.5079, 77.0956, "Sector 29"),
    (28.4380, 77.1025, "Sector 57"),
]

def generate_commuter_gps(n=1200):
    lats, lons, weights = [], [], []
    for _ in range(n):
        cx, cy, _ = random.choice(COMMUTER_HOTSPOTS)
        # Gaussian spread around each hotspot
        lats.append(cx + np.random.normal(0, 0.015))
        lons.append(cy + np.random.normal(0, 0.015))
        weights.append(random.randint(1, 5))   # trip frequency weight
    return pd.DataFrame({"lat": lats, "lon": lons, "weight": weights})

gps = generate_commuter_gps(1200)
print(f"[DATA] {len(gps)} commuter GPS points generated around {CITY}")

# ─────────────────────────────────────────────
# 2. DS LAYER — DEMAND ZONE CLUSTERING
# ─────────────────────────────────────────────
coords_scaled = StandardScaler().fit_transform(gps[["lat","lon"]].values)
db = DBSCAN(eps=0.12, min_samples=10).fit(coords_scaled)
gps["cluster"] = db.labels_

n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"[DBSCAN] {n_clusters} EV demand zones | noise: {(db.labels_==-1).sum()} points")

demand_zones = (
    gps[gps.cluster >= 0]
    .groupby("cluster")
    .agg(lat=("lat","mean"), lon=("lon","mean"), demand=("weight","sum"))
    .reset_index()
    .rename(columns={"cluster": "zone_id"})
)
print(f"\n[ZONES] Top 5 demand zones by weight:")
print(demand_zones.nlargest(5,"demand")[["zone_id","lat","lon","demand"]].to_string(index=False))

# ─────────────────────────────────────────────
# 3. CANDIDATE CHARGING STATION LOCATIONS
#    (e.g. parking lots, malls, petrol station sites)
# ─────────────────────────────────────────────
def candidate_stations(n=25):
    stations = []
    for i in range(n):
        lat = CITY_CENTER[0] + np.random.uniform(-0.07, 0.07)
        lon = CITY_CENTER[1] + np.random.uniform(-0.07, 0.07)
        install_cost = random.randint(200000, 600000)   # INR
        capacity     = random.choice([4, 8, 12])        # charging ports
        stations.append({
            "station_id": i,
            "lat": lat,
            "lon": lon,
            "install_cost": install_cost,
            "capacity_ports": capacity,
        })
    return pd.DataFrame(stations)

stations = candidate_stations(25)

# ─────────────────────────────────────────────
# 4. COVERAGE MATRIX
#    Zone z is covered by station s if distance(z,s) ≤ MAX_RANGE_KM
# ─────────────────────────────────────────────
MAX_RANGE_KM = 1.5    # a driver won't detour more than 1.5km to charge

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

Z = list(demand_zones.zone_id)
S = list(stations.station_id)

# coverage[z][s] = 1 if station s covers zone z
coverage = {
    z: {
        s: 1 if haversine(
            demand_zones.loc[demand_zones.zone_id==z,"lat"].values[0],
            demand_zones.loc[demand_zones.zone_id==z,"lon"].values[0],
            stations.loc[stations.station_id==s,"lat"].values[0],
            stations.loc[stations.station_id==s,"lon"].values[0]
        ) <= MAX_RANGE_KM else 0
        for s in S
    }
    for z in Z
}

covered_zones = [z for z in Z if any(coverage[z][s] for s in S)]
print(f"\n[COVERAGE] {len(covered_zones)} / {len(Z)} zones reachable within {MAX_RANGE_KM}km")

# ─────────────────────────────────────────────
# 5. OR LAYER — SET COVER ILP
#    Minimise: Σ install_cost_s * y_s
#    Subject to:
#      Σ_s coverage[z][s] * y_s ≥ 1   ∀ reachable zone z
#      y_s ∈ {0,1}
# ─────────────────────────────────────────────
install_cost = stations.set_index("station_id")["install_cost"].to_dict()

prob = pulp.LpProblem("EVSetCover", pulp.LpMinimize)
y = pulp.LpVariable.dicts("open", S, cat="Binary")

# Objective: minimise total installation cost
prob += pulp.lpSum(install_cost[s] * y[s] for s in S)

# Coverage constraint: every reachable demand zone must be covered
for z in covered_zones:
    prob += pulp.lpSum(coverage[z][s] * y[s] for s in S) >= 1, f"cover_zone_{z}"

# Budget constraint (optional — comment out to ignore)
BUDGET = 5_000_000   # ₹50 lakh
prob += pulp.lpSum(install_cost[s] * y[s] for s in S) <= BUDGET

prob.solve(pulp.PULP_CBC_CMD(msg=0))

status = pulp.LpStatus[prob.status]
open_stations = [s for s in S if pulp.value(y[s]) == 1]
total_cost    = pulp.value(prob.objective)

print(f"\n[OR] Status     : {status}")
print(f"[OR] Stations   : {len(open_stations)} / {len(S)} opened")
print(f"[OR] Total cost : ₹{total_cost:,.0f}")
print(f"[OR] Zones covered: {len(covered_zones)} / {len(Z)}")
print(f"[OR] Station IDs: {open_stations}")

# ─────────────────────────────────────────────
# 6. FOLIUM MAP VISUALISATION
# ─────────────────────────────────────────────
m = folium.Map(location=CITY_CENTER, zoom_start=13, tiles="CartoDB positron")

# Heatmap of commuter demand
heat_data = [[row.lat, row.lon, row.weight] for _, row in gps.iterrows()]
HeatMap(heat_data, radius=15, blur=10, max_zoom=13, min_opacity=0.3).add_to(m)

# All candidate stations (grey)
for _, row in stations.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=6, color="#aaa", fill=True, fill_opacity=0.4,
        tooltip=f"Candidate {int(row.station_id)} | ₹{row.install_cost:,}"
    ).add_to(m)

# Open stations (green EV icon)
for s in open_stations:
    row = stations.loc[stations.station_id == s].iloc[0]
    folium.Marker(
        location=[row.lat, row.lon],
        icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
        tooltip=f"⚡ Station {s} | {int(row.capacity_ports)} ports | ₹{int(row.install_cost):,}"
    ).add_to(m)

# Demand zone centroids (orange)
for _, row in demand_zones.iterrows():
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=8, color="orange", fill=True, fill_opacity=0.7,
        tooltip=f"Zone {int(row.zone_id)} | Demand {int(row.demand)}"
    ).add_to(m)

# Coverage circles around open stations
for s in open_stations:
    row = stations.loc[stations.station_id == s].iloc[0]
    folium.Circle(
        location=[row.lat, row.lon],
        radius=MAX_RANGE_KM * 1000,
        color="green", fill=True, fill_opacity=0.05, weight=1
    ).add_to(m)

m.save("ev_charging_map.html")
print("[MAP] Saved → ev_charging_map.html")

# ─────────────────────────────────────────────
# 7. SENSITIVITY ANALYSIS — How cost changes with coverage radius
# ─────────────────────────────────────────────
sensitivity = []
for radius in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    cov = {
        z: {s: 1 if haversine(
                demand_zones.loc[demand_zones.zone_id==z,"lat"].values[0],
                demand_zones.loc[demand_zones.zone_id==z,"lon"].values[0],
                stations.loc[stations.station_id==s,"lat"].values[0],
                stations.loc[stations.station_id==s,"lon"].values[0]) <= radius else 0
            for s in S} for z in Z
    }
    cz = [z for z in Z if any(cov[z][s] for s in S)]
    if not cz:
        continue
    p = pulp.LpProblem(f"EV_{radius}", pulp.LpMinimize)
    y2 = pulp.LpVariable.dicts("y", S, cat="Binary")
    p += pulp.lpSum(install_cost[s]*y2[s] for s in S)
    for z in cz:
        p += pulp.lpSum(cov[z][s]*y2[s] for s in S) >= 1
    p.solve(pulp.PULP_CBC_CMD(msg=0))
    sensitivity.append({
        "Coverage Radius (km)": radius,
        "Stations Opened": int(sum(pulp.value(y2[s]) for s in S)),
        "Zones Covered": len(cz),
        "Total Cost (₹ lakh)": round(pulp.value(p.objective)/100000, 2)
    })

df_sens = pd.DataFrame(sensitivity)
print("\n[SENSITIVITY] Cost vs Coverage Radius:")
print(df_sens.to_string(index=False))
df_sens.to_csv("sensitivity_analysis.csv", index=False)
print("[DONE]")
