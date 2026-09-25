import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="NanoGNN-X Dashboard", page_icon="⚛️", layout="wide")

st.title("⚛️ NanoGNN-X: Crystal Property Predictor")
st.markdown("""
Welcome to the NanoGNN-X Hackathon Dashboard! 
This interface connects to our high-performance FastAPI backend, running a physics-grounded Graph Neural Network (GNN).
The model utilizes a Three-Body Message Passing architecture with Bessel Radial Basis and Chebyshev Angle Basis functions to predict Bandgap and Formation Energy directly from raw `.cif` files.
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🧪 Live Inference")
    uploaded_file = st.file_uploader("Upload a Crystal Information File (.cif)", type="cif")
    
    if uploaded_file is not None:
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Processing through ONNX Runtime..."):
                try:
                    # Send to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "chemical/x-cif")}
                    response = requests.post("http://localhost:8000/predict", files=files)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"Success! Inference took {res['inference_time_ms']} ms using {res['backend']}.")
                        
                        mcol1, mcol2 = st.columns(2)
                        mcol1.metric("Bandgap (eV)", f"{res['bandgap_eV']:.4f}")
                        mcol2.metric("Formation Energy (eV/atom)", f"{res['formation_energy_eV_per_atom']:.4f}")
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Connection failed. Is the FastAPI server running? {e}")

with col2:
    st.subheader("📊 System Health & Metrics")
    try:
        health_res = requests.get("http://localhost:8000/health").json()
        st.json(health_res)
    except:
        st.warning("Could not reach FastAPI health endpoint.")
        
    st.markdown("### Architecture Highlights")
    st.markdown("""
    * **Graph Construction**: Dynamic periodic boundary conditions via `pymatgen`
    * **Features**: Edge lengths + triplet angles
    * **Scaling**: O(N) linear scaling with neighbor truncation
    * **Deployment**: ONNX export with dynamic shape carrier tensors
    """)
