# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
import os

# Configuración de página
st.set_page_config(
    page_title="Mantenimiento Predictivo — Armada",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS personalizados para lograr una estética naval premium con vidrio y gradientes
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Gradiente de fondo del sistema */
.stApp {
    background: radial-gradient(circle at top left, #0A192F 0%, #112240 60%, #020C1B 100%);
    color: #E6F1FF;
}

/* Contenedor principal de la app */
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Tarjetas con efecto glassmorphism */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    margin-bottom: 25px;
}

/* Título y subtítulos */
.app-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #64FFDA 0%, #00B4DB 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
    text-shadow: 0px 2px 10px rgba(100, 255, 218, 0.15);
}

.app-subtitle {
    font-size: 1.1rem;
    color: #8892B0;
    margin-bottom: 30px;
}

/* Estilización de tarjetas de resultados */
.result-card-container {
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    margin-top: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-card-container:hover {
    transform: translateY(-3px);
}

.badge {
    color: #FFFFFF !important;
    padding: 8px 16px;
    border-radius: 50px;
    font-weight: 800;
    text-transform: uppercase;
    font-size: 0.9rem;
    letter-spacing: 1.5px;
    display: inline-block;
    margin-bottom: 15px;
}

.badge-bajo {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
}

.badge-medio {
    background: linear-gradient(135deg, #f0932b 0%, #f1c40f 100%);
    box-shadow: 0 4px 15px rgba(240, 147, 43, 0.3);
}

.badge-alto {
    background: linear-gradient(135deg, #eb4d4b 0%, #ff7979 100%);
    box-shadow: 0 4px 15px rgba(235, 77, 75, 0.3);
}

.rec-title {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.3;
    margin-top: 15px;
    color: #FFFFFF;
}

.prob-val {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 10px 0;
    letter-spacing: -1px;
}

.disclaimer-text {
    font-size: 0.8rem;
    color: #8892B0;
    text-align: center;
    margin-top: 40px;
    padding: 15px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    font-style: italic;
}

/* Modificaciones de inputs y botones de Streamlit */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 25px !important;
}

.stButton>button {
    background: linear-gradient(135deg, #64FFDA 0%, #00B4DB 100%) !important;
    color: #0A192F !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(100, 255, 218, 0.25) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-top: 10px;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(100, 255, 218, 0.45) !important;
}

.stButton>button:active {
    transform: translateY(1px) !important;
}

/* Inputs numéricos */
.stNumberInput input {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: #E6F1FF !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

.stNumberInput label {
    color: #CCD6F6 !important;
    font-weight: 500 !important;
}

</style>
""", unsafe_allow_html=True)

# Carga de modelos al iniciar la aplicación (se cachean los recursos)
@st.cache_resource
def load_ml_components():
    feature_cols = joblib.load('feature_cols.pkl')
    scaler = joblib.load('scaler_mantenimiento_predictivo.pkl')
    model = joblib.load('modelo_mantenimiento_predictivo.pkl')
    return feature_cols, scaler, model

try:
    feature_cols, scaler, model = load_ml_components()
except Exception as e:
    st.error(f"Error al cargar los modelos o archivos pickle: {e}")
    st.stop()

# Estructura de la aplicación
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Encabezado principal
st.markdown('<h1 class="app-title">⚓ Sistema de Mantenimiento Predictivo</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="app-subtitle">Disponibilidad Operacional — Evaluación de Riesgo para Misiones Navales</h3>', unsafe_allow_html=True)

# División en dos columnas: Izquierda para el formulario, Derecha para los resultados de la predicción
col1, col2 = st.columns([3, 2], gap="large")

# Inicializamos el estado para almacenar las predicciones si existen
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Función callback para leer los valores actuales en el momento del clic y evaluar
def evaluar_callback():
    # 1. Leer los valores ACTUALES de los 7 inputs en el momento del clic (desde session_state)
    input_values = [st.session_state[f"input_{col}"] for col in feature_cols]
    
    # 2. Construir el DataFrame en el orden exacto de feature_cols
    input_df = pd.DataFrame([input_values], columns=feature_cols)
    
    # 3. Ejecutar scaler.transform() y modelo.predict_proba() de nuevo en cada clic
    scaled_data = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)
    prob = model.predict_proba(scaled_df)[0, 1]
    
    # Clasificación de riesgo según la lógica exacta provista
    if prob < 0.15:
        riesgo = "Bajo"
        recomendacion = "Operación autorizada. Continuar monitoreo normal."
        color_class = "badge-bajo"
        bg_color = "rgba(56, 239, 125, 0.08)"
        border_color = "rgba(56, 239, 125, 0.4)"
    elif prob < 0.55:
        riesgo = "Medio"
        recomendacion = "Inspección programada. Revisar en la siguiente ventana técnica."
        color_class = "badge-medio"
        bg_color = "rgba(240, 147, 43, 0.08)"
        border_color = "rgba(240, 147, 43, 0.4)"
    else:
        riesgo = "Alto"
        recomendacion = "No desplegar. Requiere inspección especializada antes de misión."
        color_class = "badge-alto"
        bg_color = "rgba(235, 77, 75, 0.08)"
        border_color = "rgba(235, 77, 75, 0.4)"
        
    st.session_state.prediction = {
        "prob": prob,
        "riesgo": riesgo,
        "recomendacion": recomendacion,
        "color_class": color_class,
        "bg_color": bg_color,
        "border_color": border_color
    }

with col1:
    st.markdown('<h4 style="color: #64FFDA; margin-bottom: 15px;">Parámetros del Sensor</h4>', unsafe_allow_html=True)
    
    # Formulario para el ingreso de variables
    with st.form("mantenimiento_form"):
        st.write("Ingrese las lecturas de los sensores del componente naval a continuación:")
        
        # Grid para ordenar los campos
        grid_col1, grid_col2 = st.columns(2)
        
        # Asignamos campos según el orden de feature_cols
        for i, col_name in enumerate(feature_cols):
            # Determinamos en qué columna colocar el input para equilibrar visualmente
            target_col = grid_col1 if i % 2 == 0 else grid_col2
            
            with target_col:
                st.number_input(
                    label=f"Lectura de {col_name}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{col_name}"
                )
        
        # Botón de envío
        submitted = st.form_submit_button("Evaluar Riesgo Operacional", on_click=evaluar_callback)
        


with col2:
    st.markdown('<h4 style="color: #64FFDA; margin-bottom: 15px;">Evaluación de Riesgo</h4>', unsafe_allow_html=True)
    
    if st.session_state.prediction is not None:
        p = st.session_state.prediction
        
        # Tarjeta dinámica según el nivel de riesgo
        st.markdown(f"""
        <div class="result-card-container" style="background-color: {p['bg_color']}; border-color: {p['border_color']};">
            <span class="badge {p['color_class']}">{p['riesgo']}</span>
            <div style="font-size: 0.9rem; color: #8892B0; text-transform: uppercase; letter-spacing: 1px;">Riesgo de Falla</div>
            <div class="prob-val">{p['prob'] * 100:.2f}%</div>
            <hr style="border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 20px 0;">
            <div style="font-size: 0.9rem; color: #8892B0; text-transform: uppercase; letter-spacing: 1px;">Recomendación Operacional</div>
            <div class="rec-title">{p['recomendacion']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Estado inicial / sin evaluar
        st.markdown("""
        <div class="glass-card" style="text-align: center; color: #8892B0; padding: 50px 30px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">📋</div>
            <h4>Esperando Lecturas de Sensores</h4>
            <p style="font-size: 0.9rem;">Complete el formulario técnico de la izquierda y haga clic en "Evaluar Riesgo Operacional" para obtener el diagnóstico en tiempo real.</p>
        </div>
        """, unsafe_allow_html=True)

# 5. Texto de descargo de responsabilidad (disclaimer)
st.markdown('<div class="disclaimer-text">Este sistema es una herramienta de apoyo a la decisión. No reemplaza el criterio del comandante ni del ingeniero responsable.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
