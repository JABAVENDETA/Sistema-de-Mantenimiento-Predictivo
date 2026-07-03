# Sistema-de-Mantenimiento-Predictivo
Aplicacion de un sistema predictivo de mantenimiento

# 🛡️ Mantenimiento Predictivo para la Disponibilidad Operacional
 
> Sistema de Machine Learning para anticipar fallas en componentes críticos navales, priorizar mantenimiento y apoyar decisiones operacionales antes del despliegue de una unidad.
 
[English version below](#-predictive-maintenance-for-operational-availability)
 
Link: [link](https://sistema-de-mantenimiento-predictivo.streamlit.app/)
 
## 📌 Contexto de misión
 
Las fallas inesperadas en componentes críticos reducen la disponibilidad de la flota, generan sobrecostos y pueden comprometer el cumplimiento de misiones. Este proyecto adapta un dataset de sensores industriales para simular un **sistema de monitoreo de componentes críticos de unidades navales**, respondiendo a la pregunta:
 
> *"Antes de autorizar una misión, ¿la unidad presenta riesgo de falla?"*
 
El modelo **no reemplaza** el criterio del comandante ni del ingeniero responsable: entrega evidencia basada en datos para priorizar recursos y reducir riesgo operacional.
 
---
 
## 🎯 Objetivo
 
Predecir la probabilidad de fallo de un componente crítico (`failure`) a partir de lecturas de sensores, y traducir esa probabilidad en un nivel de decisión accionable: **Bajo / Medio / Alto**.
 
---
 
## 📊 Dataset
 
- **Fuente:** [Predictive Maintenance Dataset (Kaggle)](https://www.kaggle.com/), reencuadrado como monitoreo de unidades navales.
- **124,494 registros** · **1,169 dispositivos** · lecturas diarias del **1/1/2015 al 2/11/2015**.
- **9 métricas de sensores** (`metric1`...`metric9`).
- Variable objetivo `failure`: **106 fallas de 124,494 filas (~0.085%)** — desbalance extremo, característica inherente al mantenimiento predictivo (las fallas son eventos raros por diseño).
---
 
## 🔬 Metodología
 
### 1. Análisis Exploratorio (EDA)
- Confirmación del desbalance extremo de clases.
- Identificación de sensores sin señal (`metric3`, plano en cero) y variables duplicadas (`metric7` = `metric8`, correlación perfecta).
- Detección de un patrón clave: varios sensores permanecen en cero durante operación normal y se "activan" ante una falla — los outliers **son la señal**, no ruido a eliminar.
- Validación de que no hay un evento sistémico único explicando los picos de fallas por día (dispositivos de familias distintas, fallas independientes).
### 2. Preprocesamiento
- Eliminación de `metric8` (redundante) y `metric3` (sin señal) → 7 variables predictoras finales.
- Split **estratificado** 80/20 (obligatorio dado que solo hay 106 casos positivos en todo el dataset).
- Escalado con `RobustScaler` (resistente a outliers, preservando la señal real de falla).
- Tres estrategias de balanceo comparadas: `class_weight='balanced'`, **Random Undersampling** y **SMOTE**.
### 3. Modelado
Comparación de **Logistic Regression, Random Forest y XGBoost** contra las 3 estrategias de balanceo (9 combinaciones). Métrica principal: **Recall** y **ROC-AUC**, no accuracy ni F1 puro — con este nivel de desbalance, priorizar precision/F1 penaliza injustamente la detección real de fallas.
 
### 4. Evaluación de riesgo
**Modelo seleccionado: Random Forest + Undersampling**
- Recall: **76.2%** (detecta 16 de 21 fallas reales en test)
- ROC-AUC: **0.865** (el más alto entre las 9 combinaciones)
- Precision baja (0.5%) es esperable e inherente al desbalance extremo del problema, no un defecto del modelo.
### 5. Traducción a niveles de decisión
Umbrales de probabilidad calibrados con la curva Precision-Recall:
 
| Nivel | Probabilidad | Recomendación |
|---|---|---|
| 🟢 Bajo | < 0.15 | Operación autorizada. Continuar monitoreo normal. |
| 🟡 Medio | 0.15 – 0.55 | Inspección programada. Revisar en la siguiente ventana técnica. |
| 🔴 Alto | ≥ 0.55 | No desplegar. Requiere inspección especializada antes de misión. |
 
El nivel **Alto** concentra el **71% de las fallas reales en solo el 11% de las unidades**, permitiendo enfocar recursos de inspección de forma eficiente.
 
### 6. Despliegue
Aplicación web construida en **Google Antigravity** (Streamlit) con formulario de consulta: se ingresan las 7 lecturas de sensores y la app devuelve la probabilidad de falla, el nivel de riesgo y la recomendación operacional en tiempo real, cargando el modelo y el scaler ya entrenados (`.pkl`).
 
---
 
## ⚠️ Limitaciones
 
- El modelo es conservador por diseño: prioriza no perder fallas reales (recall alto) a costa de generar alertas de más (falsos positivos). Esto es intencional dado el contexto de decisión — el costo de una falla no detectada supera el costo de una inspección innecesaria.
- Con solo 106 casos de falla en el histórico completo, ninguna estrategia de balanceo elimina por completo la incertidumbre estadística inherente a una clase tan escasa.
- El enfoque trata cada lectura como una observación independiente (no se modelaron tendencias temporales por dispositivo); una extensión futura podría incorporar features de series temporales (medias móviles, lags previos a la falla).
- Como toda herramienta de apoyo a la decisión, **no reemplaza el criterio del comandante ni del ingeniero responsable**.
---
 
## 🛠️ Stack técnico
 
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` (SMOTE / RandomUnderSampler) · `Matplotlib` · `Seaborn` · `Google Colab` · `Streamlit` (vía Google Antigravity)
 
---
 
## 📁 Estructura del repositorio
 
```
├── notebooks/
│   └── Mantenimiento_Predictivo.ipynb   # Fases 0-5: EDA, preprocesamiento, modelado, evaluación
├── modelo/
│   ├── modelo_mantenimiento_predictivo.pkl
│   ├── scaler_mantenimiento_predictivo.pkl
│   └── feature_cols.pkl
├── app/
│   └── (aplicación Streamlit desplegada en Antigravity)
└── README.md
```
 
---
 
## 👤 Autor
 
**Jesús** — [@JABAVENDETA](https://github.com/JABAVENDETA)
Proyecto personal de portafolio en ciencia de datos aplicada a contextos operacionales/tácticos.
 
---
---
 
# 🛡️ Predictive Maintenance for Operational Availability
 
> A Machine Learning system to anticipate failures in critical naval components, prioritize maintenance, and support operational decisions before deploying a unit.
 
## Context
 
Unexpected failures in critical components reduce fleet availability, generate cost overruns, and can compromise mission readiness. This project reframes an industrial sensor dataset as a **naval critical-component monitoring system**, answering: *"Before authorizing a mission, does the unit present a risk of failure?"* The model does not replace the commander's or engineer's judgment — it provides data-driven evidence to prioritize resources and reduce operational risk.
 
## Key results
 
- **Dataset:** 124,494 readings, 1,169 devices, extreme class imbalance (106 failures, ~0.085%).
- **Selected model:** Random Forest + Random Undersampling — Recall 76.2%, ROC-AUC 0.865.
- **3-tier risk system** (Low/Medium/High) calibrated via Precision-Recall curve: the High tier concentrates 71% of real failures in only 11% of units.
- **Deployed** as an interactive Streamlit form in Google Antigravity, validated against real sensor readings from the training set.
## Tech stack
 
Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn · Matplotlib · Seaborn · Google Colab · Streamlit
 
## Limitations
 
The model is intentionally conservative (optimized for recall over precision) given that missing a real failure is costlier than an unnecessary inspection. It treats each sensor reading as an independent observation rather than modeling per-device time series — a natural extension for future work. As with any decision-support tool, it does not replace human judgment.
 
---
 
**Author:** Jesús — [@JABAVENDETA](https://github.com/JABAVENDETA)
