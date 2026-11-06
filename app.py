import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="💧 Calidad del Agua — IRCA",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    with open("model_rf.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return modelo, scaler

FEATURES = ['Color Aparente', 'Turbiedad', 'pH', 'Cloro Residual Libre', 'Coliformes Totales']

try:
    modelo, scaler = cargar_modelo()
    modelo_cargado = True
except FileNotFoundError:
    modelo_cargado = False

st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #1a3a5c, #2196F3);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-apta {
        background: #1b5e20;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-apta h2, .result-apta p, .result-apta strong {
        color: white !important;
    }
    .result-no-apta {
        background: #b71c1c;
        border: 2px solid #F44336;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-no-apta h2, .result-no-apta p, .result-no-apta strong {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>💧 Sistema de Detección de Calidad del Agua</h1>
    <p>Clasificación automática de muestras mediante Machine Learning (CRISP-ML(Q))</p>
    <small>Random Forest</small>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Predicción Individual", "📊 Predicción por Lote (CSV)", "ℹ️ Acerca del Modelo"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ingrese los parámetros de la muestra de agua")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🧪 Parámetros Fisicoquímicos")
        color   = st.slider("Color Aparente (UPC)", 0.0, 50.0, 3.5, 0.1, help="Límite aceptable: < 15 UPC")
        turbied = st.slider("Turbiedad (NTU)", 0.0, 10.0, 0.4, 0.05, help="Límite: < 2 NTU")
        ph      = st.slider("pH", 6.0, 9.0, 7.0, 0.01, help="Rango aceptable: 6.5 – 9.0")
        cloro   = st.slider("Cloro Residual Libre (mg/L)", 0.0, 2.0, 1.5, 0.01, help="Rango: 0.3 – 2.0 mg/L")
        st.markdown("#### 🦠 Parámetros Microbiológicos")
        colifor = st.selectbox("Coliformes Totales (UFC/100 mL)", [0, 1, 2], help="0 = Ausencia (aceptable)")

    with col2:
        st.markdown("#### 📋 Resumen de la muestra")
        resumen = pd.DataFrame({
            'Parámetro': FEATURES,
            'Valor ingresado': [color, turbied, ph, cloro, colifor],
            'Límite referencia': ['< 15 UPC', '< 2 NTU', '6.5–9.0', '0.3–2.0 mg/L', '0 (ausencia)']
        })
        st.dataframe(resumen, use_container_width=True, hide_index=True)

        if st.button("🔎 Clasificar Muestra", type="primary", use_container_width=True):
            if not modelo_cargado:
                st.error("⚠️ Ejecuta primero el notebook para generar model_rf.pkl y scaler.pkl")
            else:
                entrada = np.array([[color, turbied, ph, cloro, colifor]])
                entrada_s = scaler.transform(entrada)
                pred  = modelo.predict(entrada_s)[0]
                proba = modelo.predict_proba(entrada_s)[0]

                st.markdown("---")
                if pred == 0:
                    st.markdown(f"""
                    <div class="result-apta">
                        <h2 style="color:white;">✅ AGUA APTA</h2>
                        <p style="color:white;">La muestra <strong>cumple</strong> con los parámetros de calidad.</p>
                        <p style="color:white;">Confianza: <strong>{proba[0]*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-no-apta">
                        <h2 style="color:white;">🚨 AGUA NO APTA</h2>
                        <p style="color:white;">La muestra <strong>NO cumple</strong> con los parámetros. IRCA &gt; 0.</p>
                        <p style="color:white;">Probabilidad de riesgo: <strong>{proba[1]*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### Probabilidad por clase")
                col_a, col_b = st.columns(2)
                col_a.metric("P(Apta)", f"{proba[0]*100:.1f}%")
                col_b.metric("P(No Apta)", f"{proba[1]*100:.1f}%")

                fig, ax = plt.subplots(figsize=(6, 1.5))
                ax.barh([''], [proba[0]], color='#4CAF50', label='Apta')
                ax.barh([''], [proba[1]], left=[proba[0]], color='#F44336', label='No Apta')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilidad')
                ax.legend(loc='upper right', fontsize=9)
                ax.set_title('Distribución de probabilidad', fontsize=10)
                st.pyplot(fig)
                plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Carga un archivo CSV con múltiples muestras")
    st.markdown(f"El archivo debe contener las columnas: **{', '.join(FEATURES)}**")

    archivo = st.file_uploader("Subir CSV", type=["csv"])
    if archivo is not None:
        try:
            df_upload = pd.read_csv(archivo)
            for col in FEATURES:
                if col in df_upload.columns:
                    df_upload[col] = pd.to_numeric(
                        df_upload[col].astype(str).str.replace(',', '.'), errors='coerce'
                    )
            faltantes = [c for c in FEATURES if c not in df_upload.columns]
            if faltantes:
                st.error(f"Columnas faltantes: {faltantes}")
            else:
                df_pred = df_upload[FEATURES].dropna()
                st.write(f"Muestras cargadas: **{len(df_pred)}**")
                st.dataframe(df_pred.head(10), use_container_width=True)

                if modelo_cargado and st.button("🔎 Clasificar todas las muestras", type="primary"):
                    X_s    = scaler.transform(df_pred)
                    preds  = modelo.predict(X_s)
                    probas = modelo.predict_proba(X_s)[:, 1]

                    df_pred = df_pred.copy()
                    df_pred['Predicción'] = np.where(preds == 0, '✅ Apta', '🚨 No Apta')
                    df_pred['P(No Apta)'] = (probas * 100).round(1).astype(str) + '%'
                    st.dataframe(df_pred, use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total muestras", len(preds))
                    c2.metric("✅ Aptas", (preds == 0).sum())
                    c3.metric("🚨 No Aptas", (preds == 1).sum())

                    csv_out = df_pred.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Descargar resultados CSV", csv_out,
                                       "resultados_clasificacion.csv", "text/csv")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Información del Modelo y Metodología")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🤖 Modelo: Random Forest
        - **Árboles:** 100 estimadores
        - **Balanceo:** SMOTE (k_neighbors=3)
        - **Escalado:** RobustScaler
        - **División:** 60% Train / 20% Val / 20% Test

        ### 📊 Métricas en Test
        | Métrica | Score |
        |---|---|
        | Accuracy | ~0.99 |
        | F1 Score (weighted) | ~0.99 |
        | Precision (clase No Apta) | ~0.89 |
        | Recall (clase No Apta) | ~0.89 |
        """)

    with col2:
        st.markdown("""
        ### 📋 Metodología CRISP-ML(Q)
        1. **Comprensión del negocio:** Detección de IRCA > 0
        2. **EDA:** Análisis uni/bivariable, correlación
        3. **Preprocesamiento:** Limpieza, RobustScaler, SMOTE
        4. **Modelamiento:** Regresión Logística, RF, XGBoost
        5. **Evaluación:** Accuracy, F1, Precision, Recall, ROC
        6. **Despliegue:** Esta app Streamlit

        ### 🔬 Variables del modelo
        | Variable | Importancia |
        |---|---|
        | Color Aparente | ⭐⭐⭐ Alta |
        | Turbiedad | ⭐⭐⭐ Alta |
        | Coliformes Totales | ⭐⭐ Media |
        | Cloro Residual Libre | ⭐ Baja |
        | pH | ⭐ Baja |
        """)

    st.markdown("---")
    st.markdown("**Semana de la Ingeniería 2026** · Dataset: Calidad del Agua IRCA Colombia")
