# 💧 Detección de Calidad del Agua — CRISP-ML(Q)
**Semana de la Ingeniería 2026**

## Estructura del proyecto
```
├── CRISP_MLQ_Calidad_Agua.ipynb   ← Notebook principal (EDA + 3 modelos)
├── app.py                          ← Aplicativo Streamlit
├── CALIDAD_DEL_AGUA_20260516.csv  ← Dataset
├── requirements.txt                ← Dependencias
├── model_rf.pkl                    ← Generado al ejecutar el notebook
└── scaler.pkl                      ← Generado al ejecutar el notebook
```

## Instrucciones

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar el notebook
Abre `CRISP_MLQ_Calidad_Agua.ipynb` y ejecuta **todas las celdas**.
Esto genera `model_rf.pkl` y `scaler.pkl` en la misma carpeta.

### 3. Lanzar la app Streamlit
```bash
streamlit run app.py
```

La app abre automáticamente en http://localhost:8501

## Contenido del notebook (CRISP-ML(Q))
- **Fase 2 — EDA:** Análisis univariable, bivariable y matriz de correlación
- **Fase 3 — Preprocesamiento:** Limpieza, RobustScaler, SMOTE
- **Fase 4 — Modelamiento:** Regresión Logística, Random Forest, XGBoost
- **Fase 5 — Evaluación:** Accuracy, F1, Precision, Recall, ROC, Matrices de confusión

## Dataset
- **Registros:** 1.718 muestras de agua (2012–2026)
- **Variable objetivo:** IRCA_label (0=Apta, 1=No Apta)
- **Desbalance:** ~2.5% de muestras No Aptas → se usa SMOTE
