#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilitaires pour le traitement des MNT et la visualisation 3D de ravins.

Ce module fournit des fonctions d'aide couramment utilisées dans le workflow:
- Calcul de métriques de terrain (pente, orientation, etc.)
- Conversion de formats
- Manipulation des données géospatiales
- Fonctions d'aide pour les visualisations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage


def calculate_slope(dem_data, cell_size=1.0):
    """
    Calcule la pente du terrain en degrés.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        cell_size (float): Taille des cellules du MNT en mètres.
        
    Returns:
        numpy.ndarray: Pente en degrés.
    """
    # Calculer les gradients dans les directions x et y
    dy, dx = np.gradient(dem_data, cell_size)
    
    # Calculer la pente en radians puis convertir en degrés
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg


def calculate_aspect(dem_data, cell_size=1.0):
    """
    Calcule l'orientation (aspect) du terrain en degrés (0-360).
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        cell_size (float): Taille des cellules du MNT en mètres.
        
    Returns:
        numpy.ndarray: Orientation en degrés (0=nord, 90=est, etc.).
    """
    # Calculer les gradients dans les directions x et y
    dy, dx = np.gradient(dem_data, cell_size)
    
    # Calculer l'orientation en radians puis convertir en degrés
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad)
    
    # Ajuster pour obtenir l'intervalle 0-360 degrés
    aspect_deg = 90.0 - aspect_deg
    aspect_deg = np.mod(aspect_deg, 360.0)
    
    return aspect_deg


def calculate_hillshade(dem_data, azimuth=315, altitude=45, cell_size=1.0):
    """
    Calcule un relief ombré (hillshade) pour la visualisation du terrain.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        azimuth (float): Azimut de la source de lumière en degrés.
        altitude (float): Élévation de la source de lumière en degrés.
        cell_size (float): Taille des cellules du MNT en mètres.
        
    Returns:
        numpy.ndarray: Relief ombré (valeurs entre 0 et 1).
    """
    # Convertir azimut et altitude en radians
    azimuth_rad = np.radians(360.0 - azimuth)
    altitude_rad = np.radians(altitude)
    
    # Calculer les gradients dans les directions x et y
    dy, dx = np.gradient(dem_data, cell_size)
    
    # Calculer la pente et l'orientation en radians
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dx, dy)
    
    # Calculer le relief ombré
    hillshade = np.sin(altitude_rad) * np.cos(slope_rad) + np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    
    # Normaliser entre 0 et 1
    hillshade = np.clip(hillshade, 0, 1)
    
    return hillshade


def calculate_curvature(dem_data, cell_size=1.0):
    """
    Calcule la courbure du terrain.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        cell_size (float): Taille des cellules du MNT en mètres.
        
    Returns:
        numpy.ndarray: Courbure (valeurs positives = convexe, négatives = concave).
    """
    # Utiliser un noyau Laplacien pour calculer la courbure
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / (cell_size**2)
    curvature = ndimage.convolve(dem_data, kernel)
    
    return curvature


def calculate_roughness(dem_data, window_size=3):
    """
    Calcule la rugosité du terrain (écart-type local des élévations).
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        window_size (int): Taille de la fenêtre pour le calcul (en pixels).
        
    Returns:
        numpy.ndarray: Rugosité (écart-type local).
    """
    # Utiliser le filtre d'écart-type local
    roughness = ndimage.generic_filter(dem_data, np.std, size=window_size)
    
    return roughness


def get_dem_info(dem_path):
    """
    Récupère les informations d'un fichier MNT.
    
    Args:
        dem_path (str): Chemin vers le fichier MNT.
        
    Returns:
        dict: Dictionnaire contenant les informations du MNT.
    """
    with rasterio.open(dem_path) as src:
        info = {
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'crs': src.crs.to_string() if src.crs else None,
            'transform': src.transform,
            'bounds': src.bounds,
            'resolution_x': src.transform[0],
            'resolution_y': abs(src.transform[4]),
            'nodata': src.nodata
        }
        
        # Lire les statistiques si le fichier n'est pas trop grand
        if src.width * src.height < 10000000:  # Limite à environ 10 millions de pixels
            data = src.read(1)
            valid_data = data[data != src.nodata] if src.nodata is not None else data
            
            info.update({
                'min': float(np.min(valid_data)) if len(valid_data) > 0 else None,
                'max': float(np.max(valid_data)) if len(valid_data) > 0 else None,
                'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else None,
                'std': float(np.std(valid_data)) if len(valid_data) > 0 else None
            })
    
    return info


def create_colored_hillshade(dem_data, cmap='terrain', azimuth=315, altitude=45, blend_mode='soft', vertical_exaggeration=1.0):
    """
    Crée un relief ombré coloré pour la visualisation.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        cmap (str): Nom de la colormap matplotlib.
        azimuth (float): Azimut de la source de lumière en degrés.
        altitude (float): Élévation de la source de lumière en degrés.
        blend_mode (str): Mode de mélange ('soft', 'hsv', 'overlay').
        vertical_exaggeration (float): Facteur d'exagération pour l'ombrage.
        
    Returns:
        numpy.ndarray: Image RGBA du relief ombré coloré.
    """
    from matplotlib.colors import LightSource
    
    # Appliquer l'exagération verticale pour l'ombrage
    dem_exag = dem_data * vertical_exaggeration
    
    # Créer une source de lumière
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    
    # Normaliser le MNT pour la colormap
    vmin, vmax = np.min(dem_data), np.max(dem_data)
    norm_dem = (dem_data - vmin) / (vmax - vmin)
    
    # Créer le relief ombré coloré
    colored_hillshade = ls.shade(dem_exag, cmap=plt.cm.get_cmap(cmap), blend_mode=blend_mode, 
                                vert_exag=vertical_exaggeration, vmin=vmin, vmax=vmax)
    
    return colored_hillshade


def save_geotiff(data, output_path, transform, crs=None, nodata=None):
    """
    Sauvegarde un array numpy en tant que GeoTIFF.
    
    Args:
        data (numpy.ndarray): Données à sauvegarder (2D ou 3D).
        output_path (str): Chemin de sortie pour le fichier GeoTIFF.
        transform (Affine): Transformation affine.
        crs (str, optional): Système de coordonnées.
        nodata (float, optional): Valeur NoData.
        
    Returns:
        str: Chemin vers le fichier sauvegardé.
    """
    # Déterminer le nombre de bandes
    if data.ndim == 2:
        # Une seule bande
        count = 1
        height, width = data.shape
        data = data.reshape(1, height, width)
    elif data.ndim == 3:
        # Plusieurs bandes
        count, height, width = data.shape
    else:
        raise ValueError("Les données doivent être 2D ou 3D")
    
    # Déterminer le type de données
    dtype = data.dtype
    
    # Créer le fichier GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data)
    
    return output_path


def create_empty_dem(width, height, xmin, ymax, resolution, fill_value=0, output_path=None):
    """
    Crée un MNT vide avec les dimensions et l'emprise spécifiées.
    
    Args:
        width (int): Largeur en pixels.
        height (int): Hauteur en pixels.
        xmin (float): Coordonnée X minimale.
        ymax (float): Coordonnée Y maximale.
        resolution (float): Résolution en unités/pixel.
        fill_value (float): Valeur de remplissage pour le MNT.
        output_path (str, optional): Chemin de sortie pour le fichier GeoTIFF.
        
    Returns:
        tuple: (numpy.ndarray, Affine) ou (str, Affine) si output_path est spécifié.
    """
    # Créer un array rempli avec la valeur spécifiée
    dem_data = np.full((height, width), fill_value, dtype=np.float32)
    
    # Créer la transformation affine
    transform = from_origin(xmin, ymax, resolution, resolution)
    
    # Sauvegarder en tant que GeoTIFF si un chemin est spécifié
    if output_path:
        save_geotiff(dem_data, output_path, transform)
        return output_path, transform
    
    return dem_data, transform


def calculate_volume_below_surface(dem_data, reference_elevation=None, cell_size=1.0):
    """
    Calcule le volume sous une surface de référence (ex: volume d'un ravin).
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        reference_elevation (float, optional): Élévation de référence.
                                               Si None, utilise le maximum.
        cell_size (float): Taille des cellules en mètres.
        
    Returns:
        float: Volume en mètres cubes.
    """
    # Déterminer l'élévation de référence
    if reference_elevation is None:
        reference_elevation = np.max(dem_data)
    
    # Calculer la différence de hauteur
    height_diff = reference_elevation - dem_data
    
    # Ne considérer que les valeurs positives (sous la référence)
    height_diff = np.maximum(height_diff, 0)
    
    # Calculer le volume (hauteur * aire)
    volume = np.sum(height_diff) * (cell_size**2)
    
    return volume


def dem_to_stl(dem_data, output_path, scale_z=1.0, scale_xy=1.0):
    """
    Convertit un MNT en fichier STL pour l'impression 3D.
    Nécessite numpy-stl.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        output_path (str): Chemin de sortie pour le fichier STL.
        scale_z (float): Facteur d'échelle verticale.
        scale_xy (float): Facteur d'échelle horizontale.
        
    Returns:
        str: Chemin vers le fichier STL.
    """
    try:
        from stl import mesh
    except ImportError:
        raise ImportError("Le module 'numpy-stl' est requis. Installez-le avec 'pip install numpy-stl'.")
    
    # Dimensions
    rows, cols = dem_data.shape
    
    # Créer les faces (2 triangles par cellule)
    faces = []
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Coordonnées des 4 sommets de la cellule
            v1 = (j * scale_xy, i * scale_xy, dem_data[i, j] * scale_z)
            v2 = ((j + 1) * scale_xy, i * scale_xy, dem_data[i, j + 1] * scale_z)
            v3 = (j * scale_xy, (i + 1) * scale_xy, dem_data[i + 1, j] * scale_z)
            v4 = ((j + 1) * scale_xy, (i + 1) * scale_xy, dem_data[i + 1, j + 1] * scale_z)
            
            # Premier triangle (v1, v2, v3)
            faces.append([v1, v2, v3])
            
            # Deuxième triangle (v2, v4, v3)
            faces.append([v2, v4, v3])
    
    # Créer le maillage
    faces = np.array(faces)
    terrain_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    # Remplir les données du maillage
    for i, face in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = face[j]
    
    # Sauvegarder le fichier STL
    terrain_mesh.save(output_path)
    
    return output_path


def plot_dem_profile(dem_data, start_point, end_point, num_points=100, title="Profil d'élévation", 
                   exaggeration=1.0, ax=None, show=True, output_path=None):
    """
    Trace un profil d'élévation le long d'une ligne dans un MNT.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        start_point (tuple): Point de départ (col, row).
        end_point (tuple): Point d'arrivée (col, row).
        num_points (int): Nombre de points dans le profil.
        title (str): Titre du graphique.
        exaggeration (float): Facteur d'exagération verticale.
        ax (matplotlib.Axes, optional): Axes matplotlib existants.
        show (bool): Afficher le graphique.
        output_path (str, optional): Chemin pour sauvegarder l'image.
        
    Returns:
        tuple: (matplotlib.Figure, numpy.ndarray, numpy.ndarray) Figure, distances, élévations.
    """
    # Interpoler les points le long de la ligne
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    
    # Extraire les élévations
    # Utiliser map_coordinates pour une interpolation bilinéaire
    elevations = ndimage.map_coordinates(dem_data, [y, x], order=1)
    
    # Calculer les distances
    distances = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
    
    # Créer le graphique
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    
    # Tracer le profil
    ax.plot(distances, elevations * exaggeration)
    
    # Ajouter des étiquettes et un titre
    ax.set_xlabel('Distance (pixels)')
    ax.set_ylabel('Élévation')
    ax.set_title(title)
    
    # Ajouter une indication sur l'exagération verticale si nécessaire
    if exaggeration != 1.0:
        ax.text(0.95, 0.05, f"Exagération verticale: x{exaggeration}", 
               transform=ax.transAxes, ha='right', fontsize=9, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder si un chemin est fourni
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Afficher si demandé
    if show:
        plt.show()
    elif output_path:
        plt.close(fig)
    
    return fig, distances, elevations


def convert_dem_units(dem_data, input_unit='meters', output_unit='meters', z_factor=None):
    """
    Convertit les unités d'élévation d'un MNT.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        input_unit (str): Unité d'entrée ('meters', 'feet', 'cm', etc.).
        output_unit (str): Unité de sortie ('meters', 'feet', 'cm', etc.).
        z_factor (float, optional): Facteur de conversion personnalisé.
                                    Si spécifié, remplace les unités.
        
    Returns:
        numpy.ndarray: Données d'élévation converties.
    """
    # Facteurs de conversion vers les mètres (unité intermédiaire)
    to_meters = {
        'meters': 1.0,
        'm': 1.0,
        'centimeters': 0.01,
        'cm': 0.01,
        'millimeters': 0.001,
        'mm': 0.001,
        'kilometers': 1000.0,
        'km': 1000.0,
        'feet': 0.3048,
        'ft': 0.3048,
        'inches': 0.0254,
        'in': 0.0254
    }
    
    # Facteurs de conversion depuis les mètres
    from_meters = {unit: 1.0 / factor for unit, factor in to_meters.items()}
    
    # Si un facteur personnalisé est fourni, l'utiliser directement
    if z_factor is not None:
        return dem_data * z_factor
    
    # Vérifier que les unités sont supportées
    if input_unit.lower() not in to_meters:
        raise ValueError(f"Unité d'entrée non reconnue: {input_unit}")
    if output_unit.lower() not in from_meters:
        raise ValueError(f"Unité de sortie non reconnue: {output_unit}")
    
    # Convertir en mètres puis dans l'unité de sortie
    factor = to_meters[input_unit.lower()] * from_meters[output_unit.lower()]
    return dem_data * factor
