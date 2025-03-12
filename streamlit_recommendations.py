import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar los datos y embeddings
data = np.load("data.npy", allow_pickle=True)
embeddings = np.load("embeddings.npy")

# Función para filtrar recomendaciones según la selección del usuario
def filtrar_recomendaciones(data, objetivo, nivel, dieta):
    recomendaciones = []
    
    for item in data:
        if objetivo == "Ganar músculo" and ("proteína" in item.lower() or "entrenamiento" in item.lower()):
            recomendaciones.append(item)
        elif objetivo == "Perder grasa" and ("cardio" in item.lower() or "déficit calórico" in item.lower()):
            recomendaciones.append(item)
        elif objetivo == "Mantenerse":
            recomendaciones.append(item)  # Se muestran todas las opciones
        
        if dieta == "Vegetariana" and ("carne" in item.lower() or "pollo" in item.lower() or "pescado" in item.lower()):
            continue  # Excluye carnes para dieta vegetariana
        
    return recomendaciones[:5]  # Limitamos a 5 recomendaciones

# Configuración de la app Streamlit
st.title("Recomendador de Ejercicios y Alimentación con IA")

# Filtros de usuario
objetivo = st.selectbox("Selecciona tu objetivo", ["Ganar músculo", "Perder grasa", "Mantenerse"])
nivel = st.selectbox("Selecciona tu nivel de experiencia", ["Principiante", "Intermedio", "Avanzado"])
dieta = st.selectbox("Selecciona tu tipo de dieta", ["Equilibrada", "Vegetariana", "Cetogénica"])

# Mostrar recomendaciones filtradas
st.subheader("Recomendaciones Personalizadas")
recomendaciones = filtrar_recomendaciones(data, objetivo, nivel, dieta)

for rec in recomendaciones:
    st.write(f"✅ {rec}")
