import os
import requests
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Folder & path model
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_ENV_PATH = os.path.join(CACHE_DIR, "model_env.pkl")
MODEL_CATCH_PATH = os.path.join(CACHE_DIR, "model_catch.pkl")

# ======================================
# Update model dari Google Drive
# ======================================
def update_from_server():
    try:
        files = {
            "env": "1_aK_dh7xbU0krFTFWnWcZZXW7CsFG6ph",   # model_env.pkl
            "catch": "13TT2PCTcuVQv8FvgtNIA6K453Z8LK-sV" # model_catch.pkl
        }

        for name, file_id in files.items():
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                st.error(f"Gagal download {name}: status {r.status_code}")
                continue

            save_path = MODEL_ENV_PATH if name == "env" else MODEL_CATCH_PATH
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)

            st.success(f"✅ {os.path.basename(save_path)} berhasil diperbarui")
    except Exception as e:
        st.error(f"Gagal update: {e}")

# ======================================
# Training ulang model dari dataset
# ======================================
def train_models():
    dataset_path = os.path.join(CACHE_DIR, "data_ikan.xlsx")
    if not os.path.exists(dataset_path):
        st.error("⚠️ Data tidak ditemukan. Pastikan 'data_ikan.xlsx' ada di folder cache/")
        return None, None

    df = pd.read_excel(dataset_path)

    # Mapping kolom dataset baru
    # Model_env: Longitude, Latitude -> Klorofil & SST  
    X_env = df[["Longitude", "Latitude"]]
    y_env = df[["Klorofil", "SST"]]
    model_env = RandomForestRegressor(n_estimators=100, random_state=42)
    model_env.fit(X_env, y_env)

    # Model_catch: Klorofil & SST    -> HasilTangkapan
    X_catch = df[["Klorofil", "SST"]]
    y_catch = df["HasilTangkapan"]
    model_catch = RandomForestRegressor(n_estimators=100, random_state=42)
    model_catch.fit(X_catch, y_catch)

    # Simpan model
    joblib.dump(model_env, MODEL_ENV_PATH)
    joblib.dump(model_catch, MODEL_CATCH_PATH)

    return model_env, model_catch

# ======================================
# Load model (download -> fallback training jika gagal)
# ======================================
def load_models():
    try:
        model_env = joblib.load(MODEL_ENV_PATH)
        model_catch = joblib.load(MODEL_CATCH_PATH)
        return model_env, model_catch
    except Exception as e:
        st.warning(f"Gagal load model lama ({e}), training ulang...")
        return train_models()

# ======================================
# Aplikasi Streamlit
# ======================================
st.title("Prediksi Hasil Tangkapan Ikan 🎣")

# Tombol update model dari server
if st.button("🔄 Update Model dari Server"):
    update_from_server()

# Load atau training model
model_env, model_catch = load_models()

if model_env is not None and model_catch is not None:
    st.success("✅ Model siap digunakan")

    # Input prediksi
    lon = st.number_input("Longitude", value=120.0)
    lat = st.number_input("Latitude", value=-5.0)

    if st.button("Prediksi"):
        # Prediksi Chlorophyll & SST
        chl, sst = model_env.predict([[lon, lat]])[0]

        # Prediksi hasil tangkapan
        catch = model_catch.predict([[chl, sst]])[0]

        st.write(f"🌱 Perkiraan Klorofil: **{chl:.3f}**")
        st.write(f"🌡️ Perkiraan (SST): **{sst:.3f}**")
        st.write(f"🎣 Perkiraan Hasil Tangkapan: **{catch:.2f} kg**")
