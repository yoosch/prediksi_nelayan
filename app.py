import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import folium
from streamlit_folium import st_folium
import math

SEQ_LEN = 12
HISTORY = SEQ_LEN - 1
st.set_page_config(
    page_title="Prediksi Hasil Tangkapan Ikan", 
    page_icon="🐟", 
)

# --------------------------
# 1. Definisi model
# --------------------------
class STNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --------------------------
# 2. Load Model & Scaler
# --------------------------
@st.cache_resource
def load_models_and_scalers():
    rf_chl = joblib.load("rf_chl.joblib")
    rf_sst = joblib.load("rf_sst.joblib")
    scaler_env = joblib.load("scaler_env.joblib")
    scaler_X_stnn = joblib.load("scaler_X_stnn.joblib")
    scaler_y = joblib.load("scaler_y.joblib")

    if hasattr(scaler_X_stnn, "n_features_in_"):
        n_feats = int(scaler_X_stnn.n_features_in_)
    elif hasattr(scaler_X_stnn, "scale_"):
        n_feats = int(scaler_X_stnn.scale_.shape[0])
    else:
        n_feats = 6

    model = STNN_LSTM(input_size=n_feats, hidden_size=64, num_layers=2)
    state_dict = torch.load("stnn.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return rf_chl, rf_sst, scaler_env, scaler_X_stnn, scaler_y, model, n_feats

try:
    rf_chl, rf_sst, scaler_env, scaler_X_stnn, scaler_y, model, N_FEATS = load_models_and_scalers()
except Exception as e:
    st.error(f"⚠️ Gagal load model / scaler / STNN: {e}")
    st.stop()

# --------------------------
# 3. Fungsi prediksi
# --------------------------
def predict_location(year, month, lat, lon):
    init_row = np.zeros(N_FEATS, dtype=float)
    last_seq = [init_row.copy() for _ in range(HISTORY)]

    sin_m = float(np.sin(2 * np.pi * month / 12))
    cos_m = float(np.cos(2 * np.pi * month / 12))

    Xenv = np.array([[lat, lon, sin_m, cos_m]], dtype=float)
    try:
        Xenv_scaled = scaler_env.transform(Xenv)
    except Exception:
        Xenv_scaled = Xenv

    chl_pred = float(rf_chl.predict(Xenv_scaled)[0])
    sst_pred = float(rf_sst.predict(Xenv_scaled)[0])

    new_row = np.array([chl_pred, sst_pred, sin_m, cos_m, lat, lon], dtype=float)
    if new_row.shape[0] != N_FEATS:
        if new_row.shape[0] > N_FEATS:
            new_row = new_row[:N_FEATS]
        else:
            new_row = np.concatenate([new_row, np.zeros(N_FEATS - new_row.shape[0])])

    seq_full = np.vstack(last_seq + [new_row])
    if seq_full.shape[0] > SEQ_LEN:
        seq_full = seq_full[-SEQ_LEN:, :]

    try:
        seq_full_scaled = scaler_X_stnn.transform(seq_full)
    except Exception:
        seq_full_scaled = seq_full

    X_in = torch.tensor(seq_full_scaled.reshape(1, SEQ_LEN, N_FEATS), dtype=torch.float32)
    with torch.no_grad():
        y_scaled_pred = model(X_in).cpu().numpy()

    try:
        catch_pred = float(scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1))[0, 0])
    except Exception:
        catch_pred = float(y_scaled_pred.reshape(-1)[0])

    return {"Chlorophyll": chl_pred, "SST": sst_pred, "Catch_Pred": catch_pred}

# --------------------------
# 4. Fungsi jarak
# --------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# --------------------------
# 5. Streamlit UI
# --------------------------
st.title("🎣 Prediksi Hasil Tangkapan Ikan")
with st.sidebar:
    st.header("Input Waktu")
    year = st.number_input("Tahun Prediksi", value=2025, step=1)
    month_names = [
        "Januari","Februari","Maret","April","Mei","Juni",
        "Juli","Agustus","September","Oktober","November","Desember"
    ]
    start_month_name = st.selectbox("Bulan Prediksi", month_names, index=0)
    start_month = month_names.index(start_month_name) + 1

# --------------------------
# 6. Grid A1–C4
# --------------------------
grids = [
    {"id":"A1","lon":109.93749,"lat":-6.81249},
    {"id":"A2","lon":109.93749,"lat":-6.77083},
    {"id":"A3","lon":109.93749,"lat":-6.72916},
    {"id":"A4","lon":109.93749,"lat":-6.68749},
    {"id":"B1","lon":109.97916,"lat":-6.81249},
    {"id":"B2","lon":109.97916,"lat":-6.77083},
    {"id":"B3","lon":109.97916,"lat":-6.72916},
    {"id":"B4","lon":109.97916,"lat":-6.68749},
    {"id":"C1","lon":110.02083,"lat":-6.81249},
    {"id":"C2","lon":110.02083,"lat":-6.77083},
    {"id":"C3","lon":110.02083,"lat":-6.72916},
    {"id":"C4","lon":110.02083,"lat":-6.68749},
]
GRID_SIZE_LAT = 0.0417
GRID_SIZE_LON = 0.0417

# --------------------------
# 7. Hitung prediksi semua grid
# --------------------------
predictions = {}
for g in grids:
    pred = predict_location(year, start_month, g["lat"], g["lon"])
    predictions[g["id"]] = pred

values = [p["Catch_Pred"] for p in predictions.values()]
low_th, high_th = np.percentile(values, [33,66])
def get_color(val):
    if val <= low_th:
        return "blue"
    elif val <= high_th:
        return "orange"
    else:
        return "red"

# --------------------------
# 8. Titik awal Rowosari
# --------------------------
rowosari_lat, rowosari_lon = -6.90527, 110.0409

# --------------------------
# 9. Cek koneksi internet (online mode)
# --------------------------
import requests
def internet_on():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

if internet_on():
    st.subheader("📍 Peta Prediksi")
    # Map online
    m = folium.Map(location=[-6.75, 109.95], zoom_start=12)
    folium.Marker(
        location=[rowosari_lat, rowosari_lon],
        popup="📍 Rowosari (Titik Awal)",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(m)

    # Tambah grid
    for g in grids:
        lat0, lon0 = g["lat"], g["lon"]
        polygon_coords = [
            [lat0, lon0],
            [lat0, lon0 + GRID_SIZE_LON],
            [lat0 + GRID_SIZE_LAT, lon0 + GRID_SIZE_LON],
            [lat0 + GRID_SIZE_LAT, lon0],
        ]
        val = predictions[g["id"]]["Catch_Pred"]
        folium.Polygon(
            locations=polygon_coords,
            color="black",
            weight=1,
            fill=True,
            fill_color=get_color(val),
            fill_opacity=0.5,
            tooltip=f"Grid {g['id']} → {val:.2f} kg"
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 160px; height: 110px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white;
        opacity: 0.8;
        padding: 10px;
    ">
    <b><span style="color:black">Keterangan Warna Grid</span></b><br>
    <i style="background: red; width: 12px; height: 12px; display: inline-block;"></i> <span style="color:black">Tinggi</span><br>
    <i style="background: orange; width: 12px; height: 12px; display: inline-block;"></i> <span style="color:black">Sedang</span><br>
    <i style="background: blue; width: 12px; height: 12px; display: inline-block;"></i> <span style="color:black">Rendah</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Render map
    map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"], key="main_map")

    # Handle klik + rute
    if map_data and map_data.get("last_clicked"):
        lat_clicked = map_data["last_clicked"]["lat"]
        lon_clicked = map_data["last_clicked"]["lng"]
        inside_grid = None
        for g in grids:
            lat0, lon0 = g["lat"], g["lon"]
            if lat0 <= lat_clicked <= lat0 + GRID_SIZE_LAT and lon0 <= lon_clicked <= lon0 + GRID_SIZE_LON:
                inside_grid = g
                break
        if inside_grid:
            pred = predictions[inside_grid["id"]]
            center_lat = inside_grid["lat"] + GRID_SIZE_LAT / 2
            center_lon = inside_grid["lon"] + GRID_SIZE_LON / 2
            distance_km = haversine(rowosari_lat, rowosari_lon, center_lat, center_lon)
            # Tambah marker & garis
            folium.Marker(
                location=[center_lat, center_lon],
                popup=f"Grid {inside_grid['id']}",
                icon=folium.Icon(color="blue", icon="fish")
            ).add_to(m)
            folium.PolyLine(
                locations=[[rowosari_lat, rowosari_lon],[center_lat,center_lon]],
                color="green", weight=3, dash_array="5,5",
                tooltip=f"Rute ke Grid {inside_grid['id']}"
            ).add_to(m)
            st_folium(m, width=700, height=500, returned_objects=["last_clicked"], key="main_map_rute")
            # Info box
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #4CAF50;
                    border-radius: 5px;
                    padding: 15px;
                    background-color: #d4edda;
                    color: #155724;
                    box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
                    max-width: 400px;
                ">
                    📍 Grid {inside_grid['id']} ({start_month_name} {year})<br>
                    🌿 Chlorophyll: <b>{pred['Chlorophyll']:.3f}</b><br>
                    🌡 SST: <b>{pred['SST']:.2f} °C</b><br>
                    🐟 Catch Pred: <b>{pred['Catch_Pred']:.2f} kg</b><br>
                    📏 Jarak dari Rowosari: <b>{distance_km:.2f} km</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Klik hanya di dalam grid!")
else:
    # Offline mode
    st.subheader("⚠️ Offline Mode - Peta Tidak Tersedia")
    st.info("Anda dapat melihat prediksi dalam bentuk tabel berikut:")

    # Tampilkan tabel semua grid + jarak
    grid_data = []
    for g in grids:
        pred = predictions[g["id"]]
        center_lat = g["lat"] + GRID_SIZE_LAT/2
        center_lon = g["lon"] + GRID_SIZE_LON/2
        distance = haversine(rowosari_lat, rowosari_lon, center_lat, center_lon)
        grid_data.append({
            "Grid": g["id"],
            "Chlorophyll": round(pred["Chlorophyll"],3),
            "SST (°C)": round(pred["SST"],2),
            "Catch Pred (kg)": round(pred["Catch_Pred"],2),
            "Jarak dari Rowosari (km)": round(distance,2)
        })
    st.table(grid_data)
