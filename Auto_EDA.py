import streamlit as st
import pandas as pd
import numpy as np
import sqlalchemy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import io
import base64
import plotly.express as px

# Establece la configuración de la página antes de cualquier otro comando de Streamlit
st.set_page_config(page_title="AutoEDA", layout="wide")
st.title("🔍 Auto Procesador de Datos")

# Función para obtener el string base64 de un archivo binario
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Función para inyectar el fondo en la aplicación
def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
      background-image: url("data:image/jpg;base64,{bin_str}");
      background-size: cover;
      background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Llama a la función para establecer el fondo.
# Asegúrate de que la imagen "tengen-uzui-losing-his-hand-6ynpundzh79dh642.jpg" esté en el mismo directorio que este script.
set_background('tengen-uzui-losing-his-hand-6ynpundzh79dh642.jpg')

# Inicializamos la variable df para evitar advertencias
df = None

# Selección de la fuente de datos en la barra lateral
data_source = st.sidebar.radio("Selecciona la fuente de datos", ["Subir CSV", "Conectar a SQL"])

# Carga de datos desde CSV
if data_source == "Subir CSV":
    file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

# Conexión y consulta a base de datos SQL
elif data_source == "Conectar a SQL":
    user = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Contraseña", type="password")
    host = st.sidebar.text_input("Host")
    database = st.sidebar.text_input("Base de datos")
    query = st.sidebar.text_area("Consulta SQL")
    if st.sidebar.button("Conectar y cargar"):
        try:
            engine = sqlalchemy.create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
            df = pd.read_sql(query, engine)
        except Exception as e:
            st.error(f"Error en la conexión: {e}")
            df = None

if df is not None:
    # Sección de Exploración de Datos
    st.subheader("📊 Exploración de Datos")
    st.dataframe(df.head())

    st.subheader("📋 Información General")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write(df.describe(include='all'))

    # Detección de variables numéricas y categóricas
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    st.write(f"🔢 Variables numéricas ({len(num_cols)}): {num_cols}")
    st.write(f"🔤 Variables categóricas ({len(cat_cols)}): {cat_cols}")

    st.subheader("🧼 Valores Nulos")
    st.write(df.isnull().sum())
    if st.button("Eliminar filas con valores nulos"):
        df.dropna(inplace=True)
        st.success("Filas con valores nulos eliminadas.")

    

    # Visualización interactivas para variables numéricas utilizando Plotly Express
    st.subheader("📈 Distribuciones de Variables Numéricas")
    for col in num_cols:
        st.write(f"**{col}**")
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribución de {col}", marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    # Visualización interactivas para variables categóricas
    st.subheader("📊 Distribuciones de Variables Categóricas")
    for col in cat_cols:
        st.write(f"**{col}**")
        value_df = df[col].value_counts().reset_index()
        value_df.columns = [col, 'count']
        fig = px.bar(value_df, x=col, y='count',
                     title=f"Conteo de {col}",
                     labels={col: col, 'count': 'Conteo'})
        st.plotly_chart(fig, use_container_width=True)

    # Matriz de correlación interactiva
    if len(num_cols) >= 2:
        st.subheader("🔗 Matriz de Correlación")
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Matriz de Correlación", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    # Sección de Reducción de Dimensionalidad en un expander
    with st.expander("Reducción de Dimensionalidad", expanded=False):
        st.markdown(
            """
            **Nota:** Selecciona un método y define el número de componentes para transformar las variables numéricas.  
            Puedes probar con un mayor número de componentes (hasta 20 o el máximo disponible de variables numéricas).
            """
        )
        if num_cols:
            reduction_method = st.selectbox("Método de reducción", ["PCA", "t-SNE"])
            max_pc = min(len(num_cols), 20)
            num_components = st.slider("Número de componentes", min_value=2, max_value=max_pc, value=2)
            
            if st.button("Aplicar Reducción", key="reduction"):
                X = df[num_cols].dropna()
                if reduction_method == "PCA":
                    pca = PCA(n_components=num_components)
                    X_reduced = pca.fit_transform(X)
                    st.write("Varianza explicada de cada componente:")
                    st.write(pca.explained_variance_ratio_)
                elif reduction_method == "t-SNE":
                    tsne = TSNE(n_components=num_components, random_state=42)
                    X_reduced = tsne.fit_transform(X)
                
                if num_components >= 2:
                    df_reduced = pd.DataFrame(X_reduced, columns=[f"Componente {i+1}" for i in range(num_components)])
                    fig = px.scatter(df_reduced, x="Componente 1", y="Componente 2",
                                     title=f"Reducción con {reduction_method}",
                                     hover_data=df_reduced.columns)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Para visualizar la reducción, selecciona al menos 2 componentes.")
        else:
            st.error("No se encontraron variables numéricas para la reducción.")

    # Sección de Segmentación (Clustering) en otro expander
    with st.expander("Segmentación (Clustering)", expanded=False):
        st.markdown(
            """
            **Nota:** Se recomienda aplicar previamente una reducción dimensional (por ejemplo, PCA a 2 o 3 componentes)  
            para facilitar la visualización óptima de los clusters.  
            Si tus datos son de alta dimensión, considera reducir la cantidad de dimensiones para una mejor interpretación.
            """
        )
        if num_cols:
            clustering_method = st.selectbox("Método de Clustering", ["KMeans", "DBSCAN", "Agglomerative Clustering"])
            X = df[num_cols].dropna()  # Eliminamos filas con nulos
            
            if clustering_method == "KMeans":
                k = st.slider("Número de clusters (K)", min_value=2, max_value=10, value=3)
                model = KMeans(n_clusters=k, random_state=42)
                clusters = model.fit_predict(X)
            elif clustering_method == "DBSCAN":
                eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=10.0, step=0.1, value=0.5)
                min_samples = st.slider("Mínimo número de muestras", min_value=2, max_value=20, value=5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(X)
            elif clustering_method == "Agglomerative Clustering":
                k = st.slider("Número de clusters (K)", min_value=2, max_value=10, value=3)
                model = AgglomerativeClustering(n_clusters=k)
                clusters = model.fit_predict(X)

            df_clusters = X.copy()
            df_clusters["Cluster"] = clusters
            st.write("Distribución de clusters:")
            st.write(df_clusters["Cluster"].value_counts().sort_index())

            # Visualización interactiva: Proyección a 2D con PCA si hay más de dos variables
            if len(num_cols) > 2:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
                df_2d = pd.DataFrame(X_2d, columns=["PCA 1", "PCA 2"])
                df_2d["Cluster"] = clusters.astype(str)
                fig = px.scatter(df_2d, x="PCA 1", y="PCA 2", color="Cluster",
                                 title=f"Segmentación con {clustering_method}",
                                 hover_data=df_2d.columns)
                st.plotly_chart(fig, use_container_width=True)
            else:
                df_2d = X.copy()
                df_2d["Cluster"] = clusters.astype(str)
                fig = px.scatter(df_2d, x=df_2d.columns[0], y=df_2d.columns[1], color="Cluster",
                                 title=f"Segmentación con {clustering_method}",
                                 hover_data=df_2d.columns)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No se encontraron variables numéricas para la segmentación.")
else:
    st.info("Por favor, carga un dataset para continuar.")


##RUNEAR LA APP.PY CON LA SIGUENTE LINEA DE COMANDO EN SU TERMINAL##
###streamlit run Auto_EDA.py####