import streamlit as st
import gpxpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic  # Pour le calcul de la distance géodésique
from sklearn.ensemble import IsolationForest  # Pour la détection des outliers

# Fonction pour charger et traiter le fichier GPX
def load_gpx(uploaded_file):
    try:
        gpx = gpxpy.parse(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GPX : {e}")
        return None

    # Extraction des données depuis le fichier GPX
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'altitude': point.elevation,
                    'time': point.time
                })
    return pd.DataFrame(data)

# Fonction pour calculer les métriques principales
def compute_metrics(df):
    alt_dif, dist_geo_no_alt, dist_dif_geo_2d = [0], [0], [0]
    
    for index in range(1, len(df)):
        start, stop = df.iloc[index - 1], df.iloc[index]
        # Distance géodésique 2D (sans altitude)
        distance_geo_2d = geodesic((start.latitude, start.longitude), (stop.latitude, stop.longitude)).m
        dist_dif_geo_2d.append(distance_geo_2d)
        dist_geo_no_alt.append(dist_geo_no_alt[-1] + distance_geo_2d)
        # Différence d'altitude
        alt_dif.append(stop.altitude - start.altitude)

    df['distance'] = dist_geo_no_alt
    df['dist_point'] = df['distance'].diff().fillna(0)
    df['alti_point'] = alt_dif
    return df

# Fonction pour segmenter les données par tranche de distance
def segment_data(df, segment_length=50):
    cumulative_distance = 0
    segment = 1
    segments = []
    
    for dist in df['dist_point']:
        cumulative_distance += dist
        if cumulative_distance >= segment_length:
            segment += 1
            cumulative_distance = 0
        segments.append(segment)
    
    df['segment'] = segments
    return df

# Fonction pour calculer les gradients, allure et autres dérivées par segment
def compute_segment_metrics(df):
    df['time'] = pd.to_datetime(df['time'])
    df['seconds'] = df['time'].dt.hour * 3600 + df['time'].dt.minute * 60 + df['time'].dt.second

    # Regrouper par segment
    df_segment = df.groupby('segment').last().reset_index()
    df_segment['distance_diff'] = df_segment['distance'].diff()
    df_segment['altitude_diff'] = df_segment['altitude'].diff()
    df_segment['sec_diff'] = df_segment['seconds'].diff()

    # Calcul des gradients et allures
    df_segment['grad'] = (df_segment['altitude_diff'] / df_segment['distance_diff']) * 100
    df_segment['pace'] = df_segment['sec_diff'] / df_segment['distance_diff']

    return df_segment

# Fonction pour détecter et marquer les outliers avec Isolation Forest
def detect_outliers(df_segment):
    features = df_segment[['grad', 'pace']]
    model = IsolationForest(contamination=0.05, random_state=42)
    df_segment['is_outlier'] = model.fit_predict(features)
    return df_segment

# Fonction pour couper en n parties
def assign_parts(df_segment, num_parts):
    max_seconds = df_segment['seconds'].max()
    part_duration = max_seconds / num_parts

    def assign_part(seconds):
        for part in range(num_parts):
            if seconds <= (part + 1) * part_duration:
                return part
        return num_parts - 1

    df_segment['part'] = df_segment['seconds'].apply(assign_part)
    return df_segment

# Fonction pour tracer le graphique avec ou sans outliers
def plot_scatter(df_segment, show_outliers, num_parts):

    # Supprimer données manquantes pour grade et pace
    df_segment.dropna(subset=["grad", "pace"], inplace=True)
    
    # Filtrer les outliers si nécessaire
    if not show_outliers:
        df_segment = df_segment[df_segment['is_outlier'] == 1]

    # Couleurs dynamiques pour chaque partie
    colors = plt.cm.viridis(np.linspace(0, 1, num_parts))

    fig, ax = plt.subplots()

    for part in range(num_parts):
        df_part = df_segment[df_segment['part'] == part]
        if len(df_part) > 0:
            # Régression polynomiale
            model = np.poly1d(np.polyfit(df_part['grad'], df_part['pace'], deg=3))
            polyline = np.linspace(df_segment['grad'].min(), df_segment['grad'].max(), 50)

            # Tracer les points et la courbe
            ax.scatter(df_part['grad'], df_part['pace'], label=f'Partie {part + 1}', color=colors[part])
            ax.plot(polyline, model(polyline), color=colors[part])

    ax.set_title('Allure vs Gradient')
    ax.set_xlabel('Gradient (%)')
    ax.set_ylabel('Allure (sec/m)')
    ax.legend()
    ax.grid(True)

    return fig

# Application Streamlit principale
st.title("Analyse de fichier GPX")

uploaded_file = st.file_uploader("Téléchargez un fichier GPX", type=["gpx"])

if uploaded_file:
    # Chargement et traitement du fichier GPX
    df = load_gpx(uploaded_file)
    if df is not None:
        st.write("Aperçu des données :")
        st.dataframe(df.head())

        # Calcul des métriques et segmentation
        df = compute_metrics(df)
        df = segment_data(df)
        df_segment = compute_segment_metrics(df)

        # Détection des outliers avec Isolation Forest
        df_segment = detect_outliers(df_segment)

        # Sélection du nombre de parties
        num_parts = st.slider("Nombre de parties", min_value=2, max_value=4, value=2)
        df_segment = assign_parts(df_segment, num_parts)

        # Ajouter un bouton ON/OFF pour afficher ou masquer les outliers
        show_outliers = st.checkbox("Afficher les outliers")

        # Tracer le graphique
        fig = plot_scatter(df_segment, show_outliers, num_parts)
        st.pyplot(fig)
