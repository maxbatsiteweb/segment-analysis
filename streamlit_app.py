import streamlit as st
import os
import gpxpy
import pandas as pd
import matplotlib.pyplot as plt
from geopy import distance #for the vincenty distance calculation
import datetime
import numpy as np
import gpxo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image




import gpxpy
import pandas as pd

# Charger le fichier GPX
gpx_file = 'alsace.gpx'  # Remplacez par le chemin de votre fichier GPX
with open(gpx_file, 'r') as f:
    gpx = gpxpy.parse(f)

# Extraire les points de trace
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

# Convertir en DataFrame
df = pd.DataFrame(data)

# Afficher les premières lignes



#initialize empty lists
alt_dif = [0]
time_dif = [0]
dist_geo_no_alt = [0] #cum dist
dist_dif_geo_2d = [0] #dist between points

#not sure why it is on the data and not the df
for index in range(len(df)):
    if index == 0:
        pass
    else:
        start = df.iloc[index-1]
        stop = df.iloc[index]
        #calculates distance between index-1 point and index point
        distance_geo_2d = distance.geodesic((start.latitude, start.longitude), (stop.latitude, stop.longitude)).m
        #appends the value to the list during each iteration
        dist_dif_geo_2d.append(distance_geo_2d) 
        #cumulative distance
        dist_geo_no_alt.append(dist_geo_no_alt[-1] + distance_geo_2d)
        
        #calculate difference in elevation for each between points
        alt_d = stop.altitude - start.altitude #should be stop - start
        alt_dif.append(alt_d)

#add the lists to df
df['distance'] = dist_geo_no_alt 

# normalisaiton de la distance

df['dist_point'] = df.distance.diff().fillna(0) 
df['alti_point'] = alt_dif

# split le df par segment de 50 mètres

# Ajouter une colonne segment
segment = 1
cumulative_distance = 0
segments = []  # Pour stocker les numéros de segment

for dist in df['dist_point']:
    cumulative_distance += dist
    if cumulative_distance >= 50:
        segment += 1
        cumulative_distance = 0  # Réinitialiser la distance cumulée
    segments.append(segment)

df['segment'] = segments

# Convertir la colonne 'time' en datetime
df['time'] = pd.to_datetime(df['time'])

# Calculer les secondes totales depuis le début de la journée
df['seconds'] = (
    df['time'].dt.hour * 3600 +
    df['time'].dt.minute * 60 +
    df['time'].dt.second
)

df_segment = df.groupby('segment').last().reset_index()
df_segment['distance_diff'] = df_segment['distance'].diff()
df_segment['altitude_diff'] = df_segment['altitude'].diff()
df_segment['sec_diff'] = df_segment['seconds'].diff()

grad_dif = []
for index in range(len(df_segment)):    
    try:    
        grad_d = df_segment.iloc[index].alti_point / df_segment.iloc[index].dist_point * 100
        grad_dif.append(grad_d)
        #or no need to make empty list (?)
        #df['grad_point'][index] = df['alti_point'][index] / df['dist_point'][index]
    except: #dividing by zero
        grad_dif.append(0)   

df_segment['grad'] = grad_dif
df_segment['segment_distance'] = df_segment['distance'].diff()
df_segment['pace'] = df_segment.sec_diff / df_segment.segment_distance

df_segment['gap'] =  df_segment.apply(lambda row: row.pace *
                                   
                                   (0.98462 +
                                   (0.030266 * row.grad) + 
                                   (0.0018814 * row.grad ** 2) + 
                                   (-3.3882e-06 * row.grad ** 3) + 
                                   (-4.5704e-07 * row.grad ** 4)
                                   ),
                                    axis=1)

df_segment["seconds_start"] = df_segment.apply(lambda x: x.seconds - df_segment.seconds.min(), axis=1)

# filtres

# drop les pentes au dessus et en dessous de 20%
df_segment = df_segment[(df_segment.grad <= 20) & (df_segment.grad >= -20)]
df_segment.dropna(inplace=True)

df_segment["part_2"] = df_segment.apply(lambda x: 0 if x.seconds_start < df_segment.seconds_start.max()/2 else 1, axis=1)

df_part_2_one = df_segment[df_segment.part_2 == 0]
df_part_2_two = df_segment[df_segment.part_2 == 1]

# Régression polynomiale (degré 3)
model1 = np.poly1d(np.polyfit(df_part_2_one['grad'], df_part_2_one['pace'], deg=4))
model2 = np.poly1d(np.polyfit(df_part_2_two['grad'], df_part_2_two['pace'], deg=4))

polyline = np.linspace(df_segment['grad'].min(), df_segment['grad'].max(), 50)




# Tracer le nuage de points
# Graphique : Élévation en fonction de la distance
fig, ax = plt.subplots()

# Définir les couleurs pour chaque catégorie
color_map = {0: 'lime', 1: 'lightcoral'}

# Appliquer les couleurs en fonction de la catégorie
colors = df_segment['part_2'].map(color_map)

ax.scatter(df_segment['grad'], df_segment['pace'], c=colors, label='Première partie')
#add fitted polynomial lines to scatterplot 
ax.plot(polyline, model1(polyline), color='green')
ax.plot(polyline, model2(polyline), color='red')

ax.set_ylim(0.1, 1)  # Limites de l'axe Y entre 0 et 70


# Ajouter des titres et des labels
ax.set_title('Allure vs Pente', fontsize=16)
ax.set_xlabel('Gradient %', fontsize=14)
ax.set_ylabel('Pace sec/mètres', fontsize=14)
ax.grid(True)

st.pyplot(fig)


