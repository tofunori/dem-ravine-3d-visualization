#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de visualisation interactive pour les modèles 3D de ravins
avec exagération verticale.

Ce module fournit des fonctions pour:
- Visualiser interactivement des modèles 3D
- Comparer des modèles avec différents niveaux d'exagération
- Générer des rendus avec éclairages et effets
- Créer des animations de survol
"""

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import rasterio
from rasterio.plot import show
import imageio
import tempfile
from datetime import datetime


def visualize_dem_2d(dem_path, output_path=None, hillshade=True, cmap='terrain',
                     title="Modèle Numérique de Terrain", show_plot=True):
    """
    Visualise un MNT en 2D avec option d'ombrage.
    
    Args:
        dem_path (str): Chemin vers le fichier MNT.
        output_path (str, optional): Chemin pour sauvegarder l'image.
        hillshade (bool): Appliquer un ombrage (relief ombré).
        cmap (str): Palette de couleurs pour la visualisation.
        title (str): Titre du graphique.
        show_plot (bool): Afficher le graphique.
        
    Returns:
        matplotlib.figure.Figure: L'objet figure créé.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hillshade:
            # Calculer l'ombrage
            ls = LightSource(azdeg=315, altdeg=45)
            
            # Normaliser les données pour l'ombrage
            norm_dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
            
            # Appliquer l'ombrage
            rgb = ls.shade(norm_dem, cmap=plt.cm.get_cmap(cmap), blend_mode='soft')
            im = ax.imshow(rgb)
        else:
            # Afficher simplement les valeurs d'élévation
            im = show(dem, ax=ax, cmap=cmap, title=title)
        
        # Ajouter une barre de couleur
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Élévation (m)')
        
        # Ajouter un titre et des étiquettes
        ax.set_title(title)
        ax.set_xlabel('Colonne (pixel)')
        ax.set_ylabel('Ligne (pixel)')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualisation 2D sauvegardée: {output_path}")
        
        # Afficher si demandé
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig


def compare_exaggeration_2d(original_dem, exaggerated_dem, output_path=None, 
                           titles=None, cmap='terrain', show_plot=True):
    """
    Compare visuellement un MNT original et sa version exagérée en 2D.
    
    Args:
        original_dem (str): Chemin vers le MNT original.
        exaggerated_dem (str): Chemin vers le MNT exagéré.
        output_path (str, optional): Chemin pour sauvegarder l'image.
        titles (tuple): Titres pour les deux graphiques (original, exagéré).
        cmap (str): Palette de couleurs.
        show_plot (bool): Afficher le graphique.
        
    Returns:
        matplotlib.figure.Figure: L'objet figure créé.
    """
    # Titres par défaut
    if titles is None:
        titles = ("MNT Original", "MNT avec Exagération Verticale")
    
    # Ouvrir les MNT
    with rasterio.open(original_dem) as src1, rasterio.open(exaggerated_dem) as src2:
        dem1 = src1.read(1)
        dem2 = src2.read(1)
        
        # Créer la figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculer l'ombrage pour les deux MNT
        ls = LightSource(azdeg=315, altdeg=45)
        
        # Normaliser les données pour l'ombrage
        norm_dem1 = (dem1 - np.min(dem1)) / (np.max(dem1) - np.min(dem1))
        norm_dem2 = (dem2 - np.min(dem2)) / (np.max(dem2) - np.min(dem2))
        
        # Appliquer l'ombrage et afficher
        rgb1 = ls.shade(norm_dem1, cmap=plt.cm.get_cmap(cmap), blend_mode='soft')
        rgb2 = ls.shade(norm_dem2, cmap=plt.cm.get_cmap(cmap), blend_mode='soft')
        
        im1 = ax1.imshow(rgb1)
        im2 = ax2.imshow(rgb2)
        
        # Ajouter des barres de couleur
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
        cbar1.set_label('Élévation (m)')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cbar2.set_label('Élévation (m)')
        
        # Ajouter des titres
        ax1.set_title(titles[0])
        ax2.set_title(titles[1])
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparaison 2D sauvegardée: {output_path}")
        
        # Afficher si demandé
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig


def visualize_3d_mesh(grid, output_path=None, show=True, 
                     window_size=(1024, 768), screenshot=True,
                     background_color='white', lighting=True):
    """
    Visualise interactivement un maillage 3D.
    
    Args:
        grid (pyvista.StructuredGrid ou pyvista.PolyData): Maillage 3D à visualiser.
        output_path (str, optional): Chemin pour sauvegarder la capture d'écran.
        show (bool): Afficher une fenêtre interactive.
        window_size (tuple): Taille de la fenêtre (largeur, hauteur).
        screenshot (bool): Prendre une capture d'écran.
        background_color (str): Couleur d'arrière-plan.
        lighting (bool): Activer l'éclairage avancé.
        
    Returns:
        pyvista.Plotter: L'objet plotter utilisé.
    """
    # Créer un plotter
    plotter = pv.Plotter(off_screen=not show, window_size=window_size)
    
    # Configurer l'arrière-plan
    plotter.set_background(background_color)
    
    # Ajouter le maillage avec texture si disponible
    if "texture" in grid.textures:
        plotter.add_mesh(grid, texture=grid.textures["texture"])
    else:
        plotter.add_mesh(grid, cmap="terrain", scalars="z", show_edges=False)
        plotter.add_scalar_bar(title="Élévation (m)")
    
    # Configurer l'éclairage
    if lighting:
        # Lumière principale (soleil)
        main_light = pv.Light(
            position=(10, 10, 10), 
            focal_point=(0, 0, 0), 
            color='white', 
            intensity=0.8
        )
        # Lumière d'appoint (ambiance)
        fill_light = pv.Light(
            position=(-10, -10, 10), 
            focal_point=(0, 0, 0), 
            color='blue', 
            intensity=0.3
        )
        # Lumière d'ambiance
        ambient_light = pv.Light(
            position=(0, 0, 5), 
            focal_point=(0, 0, 0), 
            color='white', 
            intensity=0.2
        )
        
        plotter.add_light(main_light)
        plotter.add_light(fill_light)
        plotter.add_light(ambient_light)
    
    # Ajuster la caméra
    plotter.view_isometric()
    plotter.camera.elevation = 30
    plotter.camera.azimuth = 225
    
    # Prendre une capture d'écran si demandé
    if output_path and screenshot:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plotter.screenshot(output_path)
        print(f"✓ Capture d'écran 3D sauvegardée: {output_path}")
    
    # Afficher si demandé
    if show:
        plotter.show()
    
    return plotter


def create_animation(grid, output_path, duration=10, fps=30, orbit=True, zoom=False):
    """
    Crée une animation du modèle 3D (orbite ou survol).
    
    Args:
        grid (pyvista.StructuredGrid or pyvista.PolyData): Maillage 3D à animer.
        output_path (str): Chemin pour sauvegarder l'animation (GIF ou MP4).
        duration (float): Durée de l'animation en secondes.
        fps (int): Images par seconde.
        orbit (bool): Si True, orbite autour du modèle; sinon, survol.
        zoom (bool): Ajouter un effet de zoom.
        
    Returns:
        str: Chemin vers l'animation générée.
    """
    # Créer un répertoire temporaire pour les images
    temp_dir = tempfile.mkdtemp()
    
    # Initialiser le plotter
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
    
    # Ajouter le maillage
    if "texture" in grid.textures:
        plotter.add_mesh(grid, texture=grid.textures["texture"])
    else:
        plotter.add_mesh(grid, cmap="terrain", scalars="z")
    
    # Configurer l'éclairage
    main_light = pv.Light(position=(10, 10, 10), focal_point=(0, 0, 0), color='white')
    plotter.add_light(main_light)
    
    # Nombre total d'images
    n_frames = int(fps * duration)
    
    # Animation
    if orbit:
        # Orbite autour du modèle
        angle_step = 360.0 / n_frames
        for i in range(n_frames):
            # Calculer l'angle actuel
            angle = i * angle_step
            
            # Positionner la caméra
            plotter.camera.azimuth = angle
            plotter.camera.elevation = 30
            
            # Ajouter un zoom si demandé
            if zoom:
                zoom_factor = 1.0 - 0.3 * np.sin(np.pi * i / n_frames)
                plotter.camera.zoom(zoom_factor)
            
            # Sauvegarder l'image
            img_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            plotter.screenshot(img_path)
    else:
        # Animation de survol
        # TODO: Implémenter une trajectoire de survol
        pass
    
    # Assembler les images en animation
    images = []
    for i in range(n_frames):
        img_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        images.append(imageio.imread(img_path))
    
    # Sauvegarder l'animation
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Déterminer le format en fonction de l'extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.gif':
        imageio.mimsave(output_path, images, fps=fps)
    elif ext in ['.mp4', '.avi', '.mov']:
        # Pour les formats vidéo, utiliser un encodeur approprié
        imageio.mimsave(output_path, images, fps=fps, codec='h264')
    else:
        # Format par défaut
        output_path = output_path + '.gif'
        imageio.mimsave(output_path, images, fps=fps)
    
    print(f"✓ Animation 3D sauvegardée: {output_path}")
    return output_path


def compare_exaggeration_factors(dem_path, texture_path=None, factors=(1.0, 3.0, 5.0, 10.0),
                                output_dir='comparison', screenshot=True, show_interactive=False):
    """
    Compare visuellement différents facteurs d'exagération verticale.
    
    Args:
        dem_path (str): Chemin vers le fichier MNT.
        texture_path (str, optional): Chemin vers le fichier de texture.
        factors (tuple): Facteurs d'exagération à comparer.
        output_dir (str): Répertoire pour les captures d'écran.
        screenshot (bool): Sauvegarder des captures d'écran.
        show_interactive (bool): Afficher une visualisation interactive.
        
    Returns:
        list: Chemins vers les captures d'écran générées.
    """
    import rasterio
    from rasterio.windows import Window
    import numpy as np
    
    # Fonction de chargement
    def load_data(dem_path, texture_path=None):
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            transform = src.transform
            
            texture_data = None
            if texture_path:
                with rasterio.open(texture_path) as tex_src:
                    texture_data = tex_src.read()
                    if texture_data.max() > 1:
                        texture_data = texture_data / 255.0
                    texture_data = texture_data.transpose(1, 2, 0)
            
            return dem_data, texture_data, transform
    
    # Fonction d'exagération
    def exaggerate(dem_data, factor):
        min_height = np.min(dem_data)
        return min_height + (dem_data - min_height) * factor
    
    # Fonction de création de maillage
    def create_mesh(dem_data, texture_data=None):
        x, y = np.meshgrid(np.arange(dem_data.shape[1]), np.arange(dem_data.shape[0]))
        grid = pv.StructuredGrid(x, y, dem_data)
        
        if texture_data is not None:
            grid.texture_map_to_plane(inplace=True)
            texture = pv.Texture(texture_data)
            grid.textures["texture"] = texture
        
        return grid
    
    # Charger les données
    dem_data, texture_data, transform = load_data(dem_path, texture_path)
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer les visualisations pour chaque facteur
    screenshot_paths = []
    for factor in factors:
        print(f"Traitement du facteur d'exagération: {factor}")
        
        # Exagérer le MNT
        exaggerated_dem = exaggerate(dem_data, factor)
        
        # Créer le maillage
        grid = create_mesh(exaggerated_dem, texture_data)
        
        # Nom du fichier de sortie
        output_path = os.path.join(output_dir, f"exaggeration_factor_{factor:.1f}.png")
        
        # Visualiser
        visualize_3d_mesh(
            grid, 
            output_path=output_path if screenshot else None,
            show=show_interactive,
            window_size=(1024, 768),
            background_color='white'
        )
        
        if screenshot:
            screenshot_paths.append(output_path)
    
    # Créer une image comparative combinant toutes les vues
    if screenshot and len(screenshot_paths) > 1:
        from PIL import Image, ImageDraw, ImageFont
        
        # Disposer les images en grille
        n_images = len(screenshot_paths)
        cols = min(2, n_images)
        rows = (n_images + cols - 1) // cols
        
        # Charger les images
        images = [Image.open(path) for path in screenshot_paths]
        img_width, img_height = images[0].size
        
        # Créer une grande image pour la compilation
        combined = Image.new('RGB', (cols * img_width, rows * img_height + 50), 'white')
        draw = ImageDraw.Draw(combined)
        
        # Ajouter un titre
        # font = ImageFont.truetype("arial.ttf", 36)  # Ajustez selon la police disponible
        # draw.text((10, 10), "Comparaison des Facteurs d'Exagération Verticale", fill="black", font=font)
        
        # Placer les images
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            combined.paste(img, (col * img_width, row * img_height + 50))
            
            # Ajouter une étiquette pour chaque image
            label = f"Facteur: {factors[i]:.1f}"
            # draw.text((col * img_width + 10, row * img_height + 60), label, fill="black", font=font)
        
        # Sauvegarder l'image combinée
        combined_path = os.path.join(output_dir, "exaggeration_comparison.png")
        combined.save(combined_path)
        screenshot_paths.append(combined_path)
        print(f"✓ Comparaison des facteurs d'exagération sauvegardée: {combined_path}")
    
    return screenshot_paths


def main():
    """Point d'entrée principal pour les tests de visualisation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualisation de MNT et modèles 3D de ravins.'
    )
    parser.add_argument('--dem', required=True, help='Chemin vers le fichier MNT')
    parser.add_argument('--texture', help='Chemin vers le fichier de texture')
    parser.add_argument('--output_dir', default='visualizations', help='Répertoire de sortie')
    parser.add_argument('--mode', choices=['2d', '3d', 'compare', 'animation'], 
                       default='3d', help='Mode de visualisation')
    parser.add_argument('--exaggeration', type=float, default=5.0, 
                       help='Facteur d\'exagération verticale')
    parser.add_argument('--show', action='store_true', help='Afficher la visualisation interactive')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Timestamp pour les noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.mode == '2d':
        # Visualisation 2D
        output_path = os.path.join(args.output_dir, f"{timestamp}_dem_2d.png")
        visualize_dem_2d(args.dem, output_path=output_path, show_plot=args.show)
    
    elif args.mode == '3d':
        # Visualisation 3D
        # Importer les fonctions nécessaires
        from dem_3d_exaggerator import exaggerate_depth, create_3d_mesh, load_chunk
        
        # Charger les données
        dem_data, texture_data, transform = load_chunk(args.dem, args.texture)
        
        # Exagérer le MNT
        exaggerated_dem = exaggerate_depth(dem_data, factor=args.exaggeration)
        
        # Créer le maillage
        grid = create_3d_mesh(exaggerated_dem, texture_data)
        
        # Visualiser
        output_path = os.path.join(args.output_dir, f"{timestamp}_dem_3d.png")
        visualize_3d_mesh(grid, output_path=output_path, show=args.show)
    
    elif args.mode == 'compare':
        # Comparaison de facteurs d'exagération
        compare_dir = os.path.join(args.output_dir, f"{timestamp}_comparison")
        compare_exaggeration_factors(
            args.dem, 
            args.texture, 
            factors=(1.0, 3.0, 5.0, 10.0),
            output_dir=compare_dir,
            show_interactive=args.show
        )
    
    elif args.mode == 'animation':
        # Animation
        # Importer les fonctions nécessaires
        from dem_3d_exaggerator import exaggerate_depth, create_3d_mesh, load_chunk
        
        # Charger les données
        dem_data, texture_data, transform = load_chunk(args.dem, args.texture)
        
        # Exagérer le MNT
        exaggerated_dem = exaggerate_depth(dem_data, factor=args.exaggeration)
        
        # Créer le maillage
        grid = create_3d_mesh(exaggerated_dem, texture_data)
        
        # Créer l'animation
        output_path = os.path.join(args.output_dir, f"{timestamp}_animation.gif")
        create_animation(grid, output_path, duration=5, fps=20)
    
    print(f"✅ Visualisation terminée!")


if __name__ == "__main__":
    main()
