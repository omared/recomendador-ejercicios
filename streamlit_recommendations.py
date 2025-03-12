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

     # Listas de alimentos por tipo de dieta
    alimentos_vegetarianos = ["quinoa", "avena", "almendras", "brócoli", "espinaca", "batata", "frutos rojos", "yogur griego"]
    alimentos_no_vegetarianos = ["carne", "pollo", "atún", "pescado", "huevo"]
    
    for item in data:
        # Filtro por objetivo
        if objetivo == "Ganar músculo" and ("proteína" in item.lower() or "fuerza" in item.lower()):
            recomendaciones.append(item)
        elif objetivo == "Perder grasa" and ("cardio" in item.lower() or "quemar grasa" in item.lower()):
            recomendaciones.append(item)
        elif objetivo == "Mantenerse":
            recomendaciones.append(item)

    # Aplicar el filtro de dieta
    if dieta == "Vegetariana":
        recomendaciones = [item for item in recomendaciones if not any(x in item.lower() for x in alimentos_no_vegetarianos)]
        
        # Si hay menos de 5 recomendaciones, añadimos alimentos vegetarianos extra
        if len(recomendaciones) < 5:
            extra_vegetarianos = [x for x in data if any(y in x.lower() for y in alimentos_vegetarianos)]
            recomendaciones.extend(extra_vegetarianos[:5 - len(recomendaciones)])

    return recomendaciones[:5]  # Limitamos a 5 recomendaciones

# Interfaz con Streamlit
st.title("Recomendador de Ejercicios y Alimentaciónn")

# Filtros de usuario
objetivo = st.selectbox("Selecciona tu objetivo", ["Ganar músculo", "Perder grasa", "Mantenerse"])
nivel = st.selectbox("Selecciona tu nivel de experiencia", ["Principiante", "Intermedio", "Avanzado"])
dieta = st.selectbox("Selecciona tu tipo de dieta", ["Equilibrada", "Vegetariana", "Cetogénica"])

# Mostrar recomendaciones
st.subheader("Recomendacioness Personalizadas")
recomendaciones = filtrar_recomendaciones(data, objetivo, nivel, dieta)

for rec in recomendaciones:
    st.write(f"✅ {rec}")