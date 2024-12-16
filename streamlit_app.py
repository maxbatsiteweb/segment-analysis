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

def clean_data(df):
    return df.dropna()

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

    # Ajouter une colonne indiquant la partie (première ou deuxième moitié du parcours)
    df_segment["seconds_start"] = df_segment['seconds'] - df_segment['seconds'].min()
    halfway = df_segment["seconds_start"].max() / 2
    df_segment["part_2"] = df_segment['seconds_start'].apply(lambda x: 0 if x < halfway else 1)
    return df_segment

# Fonction pour détecter et marquer les outliers avec Isolation Forest
def detect_outliers(df_segment):
    features = df_segment[['grad', 'pace']]
    model = IsolationForest(contamination=0.05, random_state=42)
    df_segment['is_outlier'] = model.fit_predict(features)
    return df_segment

# Fonction pour tracer le graphique avec ou sans outliers
def plot_scatter(df_segment, show_outliers):
    # Filtrer les données qui ont des Nan pace / gradient
    df_segment.dropna(subset=['grad', 'pace'], inplace=True)
    
    # Filtrer les outliers si nécessaire
    if not show_outliers:
        df_segment = df_segment[df_segment['is_outlier'] == 1]

    # Régression polynomiale pour chaque partie
    df_part_1 = df_segment[df_segment.part_2 == 0]
    df_part_2 = df_segment[df_segment.part_2 == 1]

    model1 = np.poly1d(np.polyfit(df_part_1['grad'], df_part_1['pace'], deg=2))
    model2 = np.poly1d(np.polyfit(df_part_2['grad'], df_part_2['pace'], deg=2))

    polyline = np.linspace(df_segment['grad'].min(), df_segment['grad'].max(), 50)

    # Tracer le nuage de points
    fig, ax = plt.subplots()
    color_map = {0: 'lime', 1: 'lightcoral'}
    colors = df_segment['part_2'].map(color_map)

    ax.scatter(df_segment['grad'], df_segment['pace'], c=colors)
    ax.plot(polyline, model1(polyline), color='green', label="Partie 1")
    ax.plot(polyline, model2(polyline), color='red', label="Partie 2")

    ax.set_title('Allure vs Gradient')
    ax.set_xlabel('Gradient (%)')
    ax.set_ylabel('Allure (sec/m)')
    ax.legend()
    ax.grid(True)

    return fig, model1, model2, df_segment

# Fonction pour calculer le pourcentage de variation d'allure basé sur la régression
def compute_pace_variation_from_regression(model1, model2, gradient_range):
    part_1_pace = model1(gradient_range)
    part_2_pace = model2(gradient_range)
    pace_variation = ((part_2_pace - part_1_pace) / part_1_pace) * 100
    return gradient_range, pace_variation

# Fonction pour tracer le graphique de variation d'allure
def plot_variation_curve(gradient_range, pace_variation):
    fig, ax = plt.subplots()
    ax.plot(gradient_range, pace_variation, color='gold', linestyle='-')
    ax.set_title('Variation d\'Allure entre Partie 1 et Partie 2 par Gradient')
    ax.set_xlabel('Gradient (%)')
    ax.set_ylabel('Variation d\'Allure (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
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

        # Clean data
        df = clean_data(df)

        # Calcul des métriques et segmentation
        df = compute_metrics(df)
        df = segment_data(df)
        df_segment = compute_segment_metrics(df)

        # Détection des outliers avec Isolation Forest
        df_segment = detect_outliers(df_segment)

        # Ajouter un bouton ON/OFF pour afficher ou masquer les outliers
        show_outliers = st.checkbox("Afficher les outliers")

        # Tracer le nuage de points
        scatter_fig, model1, model2, filtered_df_segment = plot_scatter(df_segment, show_outliers)
        st.pyplot(scatter_fig)

        # Calculer la variation d'allure par gradient basé sur les modèles de régression
        gradient_range = np.linspace(filtered_df_segment['grad'].min(), filtered_df_segment['grad'].max(), 50)
        gradient_range, pace_variation = compute_pace_variation_from_regression(model1, model2, gradient_range)

        # Tracer le graphique de variation d'allure
        variation_fig = plot_variation_curve(gradient_range, pace_variation)
        st.pyplot(variation_fig)
