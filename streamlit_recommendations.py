import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar los datos y embeddings
data = np.load("data.npy", allow_pickle=True)
embeddings = np.load("embeddings.npy")

# Función para filtrar recomendaciones según los filtros seleccionados
def filtrar_recomendaciones(data, objetivo, nivel, dieta):
    recomendaciones = []
    
    for item in data:
        if objetivo == "Ganar músculo" and ("proteína" in item.lower() or "fuerza" in item.lower()):
            recomendaciones.append(item)
        elif objetivo == "Perder grasa" and ("cardio" in item.lower() or "quemar grasa" in item.lower()):
            recomendaciones.append(item)
        elif objetivo == "Mantenerse":
            recomendaciones.append(item)  # Se muestran todas las opciones
        
        # Si el usuario elige dieta vegetariana, filtramos alimentos vegetarianos
    if dieta == "Vegetariana":
        recomendaciones = [item for item in recomendaciones if not any(x in item.lower() for x in ["carne", "pollo", "atún", "pescado"])]
        # Añadimos alimentos vegetarianos manualmente si no hay suficientes
        vegetarianos = [x for x in data if any(y in x.lower() for y in ["quinoa", "avena", "almendras", "brócoli", "espinaca"])]
        recomendaciones.extend(vegetarianos[:3])  # Añadimos hasta 3 alimentos extra

    return recomendaciones[:5]  # Limitamos a 5 recomendaciones

# Interfaz con Streamlit
st.title("Recomendador de Ejercicios y Alimentación")

# Filtros de usuario
objetivo = st.selectbox("Selecciona tu objetivo", ["Ganar músculo", "Perder grasa", "Mantenerse"])
nivel = st.selectbox("Selecciona tu nivel de experiencia", ["Principiante", "Intermedio", "Avanzado"])
dieta = st.selectbox("Selecciona tu tipo de dieta", ["Equilibrada", "Vegetariana", "Cetogénica"])

# Mostrar recomendaciones
st.subheader("Recomendacioness Personalizadas")
recomendaciones = filtrar_recomendaciones(data, objetivo, nivel, dieta)

for rec in recomendaciones:
    st.write(f"✅ {rec}")