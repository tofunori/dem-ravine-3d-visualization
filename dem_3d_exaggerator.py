#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module principal pour la visualisation 3D de ravins de thermo-érosion
avec exagération verticale.

Ce script permet de charger un MNT (Modèle Numérique de Terrain) et une
texture (orthophoto), d'appliquer une exagération verticale pour accentuer
les reliefs, puis de générer un modèle 3D texturé.
"""

import os
import argparse
import yaml
import rasterio
from rasterio.windows import Window
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from datetime import datetime

# Constantes et configuration par défaut
DEFAULT_CONFIG = {
    'exaggeration_factor': 5.0,
    'chunk_size': 1000,
    'output_format': 'obj',
    'optimization_factor': 0.5,
    'output_directory': 'output',
}


def load_config(config_path=None):
    """
    Charge la configuration depuis un fichier YAML ou utilise les valeurs par défaut.
    
    Args:
        config_path (str, optional): Chemin vers le fichier de configuration.
        
    Returns:
        dict: Configuration chargée ou valeurs par défaut.
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as file:
            loaded_config = yaml.safe_load(file)
            if loaded_config:
                config.update(loaded_config)
    
    return config


def load_chunk(dem_path, texture_path=None, window=None, chunk_size=1000):
    """
    Charge une portion du MNT et de la texture si disponible.
    
    Args:
        dem_path (str): Chemin vers le fichier MNT (GeoTIFF).
        texture_path (str, optional): Chemin vers le fichier de texture (GeoTIFF).
        window (rasterio.windows.Window, optional): Fenêtre de lecture spécifique.
        chunk_size (int, optional): Taille du morceau à charger si window n'est pas spécifié.
        
    Returns:
        tuple: (dem_data, texture_data, transform)
    """
    with rasterio.open(dem_path) as dem_src:
        # Utiliser la fenêtre spécifiée ou créer une fenêtre de taille chunk_size
        if window is None:
            window = Window(0, 0, min(chunk_size, dem_src.width), min(chunk_size, dem_src.height))
        
        # Lire les données du MNT
        dem_data = dem_src.read(1, window=window)
        transform = dem_src.transform
        
        # Lire la texture si un chemin est fourni
        texture_data = None
        if texture_path and os.path.exists(texture_path):
            with rasterio.open(texture_path) as tex_src:
                # Ajuster la fenêtre si nécessaire pour correspondre à la résolution de la texture
                scale_x = tex_src.width / dem_src.width
                scale_y = tex_src.height / dem_src.height
                
                tex_window = Window(
                    int(window.col_off * scale_x),
                    int(window.row_off * scale_y),
                    int(window.width * scale_x),
                    int(window.height * scale_y)
                )
                
                texture_data = tex_src.read(window=tex_window)
                
                # Normaliser si nécessaire (8-bit à flottant 0-1)
                if texture_data.max() > 1:
                    texture_data = texture_data / 255.0
                
                # Réorganiser pour PyVista: (bands, height, width) -> (height, width, bands)
                texture_data = texture_data.transpose(1, 2, 0)
                
                # Si une seule bande, répéter pour créer un RGB
                if texture_data.shape[2] == 1:
                    texture_data = np.repeat(texture_data, 3, axis=2)
    
    return dem_data, texture_data, transform


def exaggerate_depth(dem_data, factor=5.0, base_level=None):
    """
    Exagère la profondeur du MNT par un facteur multiplicatif.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation.
        factor (float): Facteur d'exagération (>1 pour amplifier).
        base_level (float, optional): Niveau de base pour l'exagération.
            Si non spécifié, utilise le point le plus bas.
    
    Returns:
        numpy.ndarray: MNT avec profondeur exagérée.
    """
    # Ignorer les valeurs NoData si présentes (souvent représentées par des valeurs très négatives)
    valid_mask = dem_data > -9999
    
    if base_level is None:
        # Utiliser le point le plus bas comme niveau de référence
        base_level = np.min(dem_data[valid_mask])
    
    # Appliquer l'exagération aux valeurs valides uniquement
    exaggerated_dem = dem_data.copy()
    exaggerated_dem[valid_mask] = base_level + (dem_data[valid_mask] - base_level) * factor
    
    return exaggerated_dem


def create_3d_mesh(dem_data, texture_data=None, dem_spacing=(1.0, 1.0)):
    """
    Crée un maillage 3D à partir des données du MNT et applique une texture si disponible.
    
    Args:
        dem_data (numpy.ndarray): Données d'élévation exagérées.
        texture_data (numpy.ndarray, optional): Données de texture (RGB).
        dem_spacing (tuple, optional): Espacement des cellules (x, y) en unités du MNT.
        
    Returns:
        pyvista.StructuredGrid: Maillage 3D avec texture.
    """
    # Créer une grille 2D avec l'espacement approprié
    nx, ny = dem_data.shape[1], dem_data.shape[0]
    x = np.arange(0, nx * dem_spacing[0], dem_spacing[0])
    y = np.arange(0, ny * dem_spacing[1], dem_spacing[1])
    x, y = np.meshgrid(x, y)
    
    # Créer le maillage 3D
    grid = pv.StructuredGrid(x, y, dem_data)
    
    # Appliquer la texture si disponible
    if texture_data is not None:
        grid.texture_map_to_plane(inplace=True)
        texture = pv.Texture(texture_data)
        grid.textures["texture"] = texture
    
    return grid


def optimize_mesh(grid, reduction_factor=0.5):
    """
    Optimise le maillage en réduisant le nombre de faces.
    
    Args:
        grid (pyvista.StructuredGrid): Maillage 3D à optimiser.
        reduction_factor (float): Facteur de réduction (0-1).
        
    Returns:
        pyvista.PolyData: Maillage optimisé.
    """
    # Convertir en PolyData pour décimation
    surface = grid.extract_surface()
    decimated = surface.decimate(reduction_factor)
    return decimated


def export_model(grid, output_path, format='obj'):
    """
    Exporte le modèle 3D dans le format spécifié.
    
    Args:
        grid (pyvista.StructuredGrid or pyvista.PolyData): Maillage 3D à exporter.
        output_path (str): Chemin de base pour l'exportation.
        format (str): Format d'exportation ('obj', 'ply', 'stl', etc.).
    
    Returns:
        str: Chemin complet du fichier exporté.
    """
    # Assurer que le répertoire existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Construire le chemin complet
    full_path = f"{output_path}.{format}"
    
    # Exporter selon le format
    grid.save(full_path)
    print(f"✓ Modèle exporté vers {full_path}")
    
    return full_path


def visualize_3d_mesh(grid, show=True, screenshot_path=None):
    """
    Visualise le maillage 3D interactivement et/ou sauvegarde une capture d'écran.
    
    Args:
        grid (pyvista.StructuredGrid or pyvista.PolyData): Maillage 3D à visualiser.
        show (bool): Si True, affiche une fenêtre interactive.
        screenshot_path (str, optional): Chemin pour sauvegarder une capture d'écran.
        
    Returns:
        pyvista.Plotter: L'objet plotter utilisé.
    """
    # Créer un plotter
    plotter = pv.Plotter(off_screen=not show)
    
    # Ajouter le maillage avec sa texture s'il en a une
    if "texture" in grid.textures:
        plotter.add_mesh(grid, texture=grid.textures["texture"])
    else:
        plotter.add_mesh(grid, cmap="terrain", scalars="z")
    
    # Ajouter une échelle de couleur
    if "texture" not in grid.textures:
        plotter.add_scalar_bar(title="Élévation (m)")
    
    # Configurer l'éclairage
    plotter.add_light(pv.Light(position=(0, 0, 10), focal_point=(0, 0, 0), color='white'))
    plotter.add_light(pv.Light(position=(10, 10, 10), focal_point=(0, 0, 0), color='white', intensity=0.6))
    
    # Sauvegarder une capture si demandé
    if screenshot_path:
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        plotter.screenshot(screenshot_path)
        print(f"✓ Capture d'écran sauvegardée: {screenshot_path}")
    
    # Afficher si demandé
    if show:
        plotter.show()
    
    return plotter


def process_dem_to_3d(dem_file, texture_file=None, config=None):
    """
    Fonction principale pour traiter un MNT en modèle 3D avec exagération verticale.
    
    Args:
        dem_file (str): Chemin vers le fichier MNT.
        texture_file (str, optional): Chemin vers le fichier de texture.
        config (dict, optional): Configuration personnalisée.
        
    Returns:
        str: Chemin vers le fichier de modèle 3D généré.
    """
    # Charger la configuration
    if config is None:
        config = load_config()
    
    print(f"🌋 Traitement du MNT avec exagération verticale x{config['exaggeration_factor']}...")
    
    # Charger les données
    dem_data, texture_data, transform = load_chunk(
        dem_file, 
        texture_file, 
        chunk_size=config['chunk_size']
    )
    
    # Calculer la résolution spatiale
    dem_spacing = (transform[0], abs(transform[4]))
    print(f"📏 Résolution spatiale: {dem_spacing[0]:.2f}m x {dem_spacing[1]:.2f}m")
    
    # Afficher quelques statistiques sur le MNT
    print(f"📊 Statistiques du MNT original:")
    print(f"   - Min: {np.min(dem_data):.2f}m")
    print(f"   - Max: {np.max(dem_data):.2f}m")
    print(f"   - Plage: {np.max(dem_data) - np.min(dem_data):.2f}m")
    
    # Exagérer la profondeur
    exaggerated_dem = exaggerate_depth(
        dem_data, 
        factor=config['exaggeration_factor']
    )
    
    print(f"📊 Statistiques du MNT exagéré:")
    print(f"   - Min: {np.min(exaggerated_dem):.2f}m")
    print(f"   - Max: {np.max(exaggerated_dem):.2f}m")
    print(f"   - Plage: {np.max(exaggerated_dem) - np.min(exaggerated_dem):.2f}m")
    
    # Créer le maillage 3D
    grid = create_3d_mesh(exaggerated_dem, texture_data, dem_spacing)
    
    # Optimiser si demandé
    if config.get('optimization_factor', 0) > 0:
        print(f"🔧 Optimisation du maillage (facteur: {config['optimization_factor']})...")
        grid = optimize_mesh(grid, reduction_factor=config['optimization_factor'])
    
    # Créer le répertoire de sortie si nécessaire
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output_directory'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Nom de base pour les fichiers de sortie
    base_filename = os.path.splitext(os.path.basename(dem_file))[0]
    output_base = os.path.join(output_dir, f"{base_filename}_exag{config['exaggeration_factor']}")
    
    # Exporter le modèle
    output_path = export_model(
        grid, 
        output_base, 
        format=config['output_format']
    )
    
    # Générer une visualisation si demandé
    if config.get('create_screenshot', True):
        screenshot_path = f"{output_base}_preview.png"
        visualize_3d_mesh(grid, show=False, screenshot_path=screenshot_path)
    
    # Afficher interactivement si demandé
    if config.get('show_interactive', False):
        print("🖥️ Affichage de la visualisation 3D (fermer la fenêtre pour continuer)...")
        visualize_3d_mesh(grid, show=True)
    
    return output_path


def main():
    """Point d'entrée principal pour l'exécution en ligne de commande."""
    # Définir les arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description='Génère un modèle 3D à partir d\'un MNT avec exagération verticale.'
    )
    parser.add_argument('--dem_file', required=True, help='Chemin vers le fichier MNT (GeoTIFF)')
    parser.add_argument('--texture_file', help='Chemin vers le fichier de texture (GeoTIFF)')
    parser.add_argument('--config', help='Chemin vers le fichier de configuration YAML')
    parser.add_argument('--exaggeration', type=float, help='Facteur d\'exagération verticale')
    parser.add_argument('--output_dir', help='Répertoire de sortie pour les fichiers générés')
    parser.add_argument('--format', choices=['obj', 'ply', 'stl', 'vtk'], help='Format d\'exportation')
    parser.add_argument('--show', action='store_true', help='Afficher la visualisation interactive')
    
    args = parser.parse_args()
    
    # Charger la configuration de base
    config = load_config(args.config)
    
    # Remplacer les valeurs par les arguments en ligne de commande si spécifiés
    if args.exaggeration:
        config['exaggeration_factor'] = args.exaggeration
    if args.output_dir:
        config['output_directory'] = args.output_dir
    if args.format:
        config['output_format'] = args.format
    if args.show:
        config['show_interactive'] = True
    
    # Traiter et générer le modèle 3D
    output_path = process_dem_to_3d(args.dem_file, args.texture_file, config)
    
    print(f"✅ Traitement terminé! Modèle généré: {output_path}")


if __name__ == "__main__":
    main()
