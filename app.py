"""
Streamlit Dashboard — EV Charging Station Optimizer
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import random

st.set_page_config(page_title="EV Charging Optimizer", layout="wide")
st.title("⚡ Hyperlocal EV Charging Station Optimizer")
st.markdown("**DS Layer**: DBSCAN demand clustering &nbsp;|&nbsp; **OR Layer**: Set Cover ILP")

CITY_CENTER = (28.4595, 77.0266)

with st.sidebar:
    st.header("⚙️ Parameters")
    n_gps      = st.slider("Commuter GPS points", 300, 2000, 1000, 100)
    n_stations = st.slider("Candidate station sites", 10, 40, 25)
    max_range  = st.slider("Coverage radius (km)", 0.5, 3.0, 1.5, 0.25)
    budget_lakh = st.slider("Budget (₹ lakh)", 20, 100, 50)
    run = st.button("🚀 Run Optimizer", type="primary")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

if "optimizer_ran" not in st.session_state:
    st.session_state.optimizer_ran = False
    
if run:
    st.session_state.optimizer_ran = True

if st.session_state.optimizer_ran:
    random.seed(42); np.random.seed(42)
    HOTSPOTS = [(28.4949,77.0878),(28.4226,77.0478),(28.4673,77.0285),(28.5079,77.0956),(28.4380,77.1025)]
    lats, lons, weights = [], [], []
    for _ in range(n_gps):
        cx, cy = random.choice(HOTSPOTS)
        lats.append(cx + np.random.normal(0, 0.015))
        lons.append(cy + np.random.normal(0, 0.015))
        weights.append(random.randint(1, 5))
    gps = pd.DataFrame({"lat": lats, "lon": lons, "weight": weights})

    coords_s = StandardScaler().fit_transform(gps[["lat","lon"]].values)
    db = DBSCAN(eps=0.12, min_samples=10).fit(coords_s)
    gps["cluster"] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    zones = (gps[gps.cluster >= 0].groupby("cluster")
             .agg(lat=("lat","mean"), lon=("lon","mean"), demand=("weight","sum"))
             .reset_index().rename(columns={"cluster":"zone_id"}))

    np.random.seed(99)
    stations = pd.DataFrame([{
        "station_id": i,
        "lat": CITY_CENTER[0] + np.random.uniform(-0.07, 0.07),
        "lon": CITY_CENTER[1] + np.random.uniform(-0.07, 0.07),
        "install_cost": random.randint(200000, 600000),
        "capacity_ports": random.choice([4, 8, 12])
    } for i in range(n_stations)])

    Z = list(zones.zone_id); S = list(stations.station_id)
    coverage = {z: {s: 1 if haversine(zones.loc[zones.zone_id==z,"lat"].values[0],
                                       zones.loc[zones.zone_id==z,"lon"].values[0],
                                       stations.loc[stations.station_id==s,"lat"].values[0],
                                       stations.loc[stations.station_id==s,"lon"].values[0]) <= max_range else 0
                    for s in S} for z in Z}
    covered_zones = [z for z in Z if any(coverage[z][s] for s in S)]
    install_cost = stations.set_index("station_id")["install_cost"].to_dict()

    prob = pulp.LpProblem("EV", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", S, cat="Binary")
    prob += pulp.lpSum(install_cost[s]*y[s] for s in S)
    for z in covered_zones:
        prob += pulp.lpSum(coverage[z][s]*y[s] for s in S) >= 1
    prob += pulp.lpSum(install_cost[s]*y[s] for s in S) <= budget_lakh * 100000
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    open_stations = [s for s in S if pulp.value(y[s]) == 1]
    total_cost = pulp.value(prob.objective) or 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("GPS Points", n_gps)
    c2.metric("Demand Zones", n_clusters)
    c3.metric("Stations Opened", len(open_stations))
    c4.metric("Total Cost", f"₹{total_cost/100000:.1f}L")

    m = folium.Map(location=CITY_CENTER, zoom_start=13, tiles="CartoDB positron")
    HeatMap([[r.lat, r.lon, r.weight] for _, r in gps.iterrows()], radius=15, blur=10).add_to(m)
    for _, r in stations.iterrows():
        folium.CircleMarker([r.lat,r.lon], radius=5, color="#aaa", fill=True, fill_opacity=0.4,
                            tooltip=f"Candidate {int(r.station_id)}").add_to(m)
    for s in open_stations:
        r = stations.loc[stations.station_id==s].iloc[0]
        folium.Marker([r.lat,r.lon], icon=folium.Icon(color="green",icon="bolt",prefix="fa"),
                      tooltip=f"⚡ Station {s} | {int(r.capacity_ports)} ports").add_to(m)
        folium.Circle([r.lat,r.lon], radius=max_range*1000, color="green",
                      fill=True, fill_opacity=0.05, weight=1).add_to(m)
    for _, r in zones.iterrows():
        folium.CircleMarker([r.lat,r.lon], radius=8, color="orange", fill=True, fill_opacity=0.7,
                            tooltip=f"Zone {int(r.zone_id)} | Demand {int(r.demand)}").add_to(m)
    st_folium(m, width=900, height=500)

    st.subheader("📋 Opened Stations")
    opened_df = stations[stations.station_id.isin(open_stations)][["station_id","lat","lon","install_cost","capacity_ports"]]
    st.dataframe(opened_df, use_container_width=True)

    st.subheader("📊 Cost by Zone Demand")
    fig = px.bar(zones.sort_values("demand",ascending=False), x="zone_id", y="demand",
                 title="Demand per Zone", labels={"zone_id":"Zone","demand":"Demand Weight"},
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.session_state.optimizer_ran = False
    ...
else:
    st.info("👈 Set parameters and click **Run Optimizer**")
