#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemple de flux de travail complet pour la visualisation 3D de ravins.

Ce script montre un exemple d'utilisation des différentes étapes:
1. Prétraitement
2. Exagération verticale
3. Visualisation
4. Comparaison de différents facteurs d'exagération

Utilisation:
    python example_workflow.py --dem path/to/dem.tif --texture path/to/texture.tif

Note: Ce script suppose que les MNT et textures sont placés dans le dossier data/raw/
"""

import os
import argparse
import sys
import yaml
import time
from datetime import datetime

# Ajouter le dossier parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les fonctions personnalisées
from preprocessing import crop_raster, smooth_dem, align_rasters
from dem_3d_exaggerator import process_dem_to_3d, load_config
from visualization import compare_exaggeration_factors, visualize_dem_2d


def create_directory_structure():
    """Crée la structure de dossiers nécessaire pour l'exemple."""
    directories = [
        '../data/raw',
        '../data/preprocessed',
        '../data/output',
        '../visualizations'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(os.path.dirname(__file__), directory), exist_ok=True)


def preprocess_data(dem_path, texture_path, output_dir):
    """
    Prétraite les données pour l'exemple.
    
    Args:
        dem_path: Chemin vers le MNT
        texture_path: Chemin vers la texture
        output_dir: Répertoire de sortie
        
    Returns:
        tuple: (processed_dem_path, processed_texture_path)
    """
    print("\n" + "="*60)
    print("ÉTAPE 1: PRÉTRAITEMENT DES DONNÉES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Noms des fichiers de sortie
    smoothed_dem = os.path.join(output_dir, 'dem_smoothed.tif')
    aligned_dem = os.path.join(output_dir, 'dem_aligned.tif')
    aligned_texture = os.path.join(output_dir, 'texture_aligned.tif')
    
    # 1. Lisser le MNT
    print("\n🔄 Application d'un léger lissage au MNT pour réduire le bruit...")
    smooth_dem(dem_path, smoothed_dem, sigma=1.0, method='gaussian')
    
    # 2. Aligner le MNT et la texture
    if texture_path:
        print("\n🔄 Alignement du MNT et de la texture...")
        aligned_dem, aligned_texture = align_rasters(
            smoothed_dem,
            texture_path,
            aligned_dem,
            aligned_texture
        )
        return aligned_dem, aligned_texture
    else:
        return smoothed_dem, None


def generate_models(dem_path, texture_path):
    """
    Génère des modèles 3D avec différents facteurs d'exagération.
    
    Args:
        dem_path: Chemin vers le MNT prétraité
        texture_path: Chemin vers la texture prétraitée
        
    Returns:
        list: Chemins vers les modèles générés
    """
    print("\n" + "="*60)
    print("ÉTAPE 2: GÉNÉRATION DES MODÈLES 3D AVEC EXAGÉRATION VERTICALE")
    print("="*60)
    
    # Charger la configuration par défaut
    config = load_config()
    
    # Dossier de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), f'../data/output/{timestamp}')
    
    # Liste pour stocker les chemins des modèles générés
    model_paths = []
    
    # Générer des modèles avec différents facteurs d'exagération
    exaggeration_factors = [1.0, 3.0, 5.0, 10.0]
    
    for factor in exaggeration_factors:
        print(f"\n🔄 Génération du modèle avec facteur d'exagération x{factor}...")
        
        # Mettre à jour la configuration
        config['exaggeration_factor'] = factor
        config['output_directory'] = output_dir
        
        # Traiter le MNT
        output_path = process_dem_to_3d(dem_path, texture_path, config)
        model_paths.append(output_path)
        
        # Courte pause pour la lisibilité
        time.sleep(0.5)
    
    print(f"\n✅ {len(model_paths)} modèles générés dans {output_dir}")
    return model_paths


def visualize_results(dem_path, texture_path, model_paths):
    """
    Visualise les résultats de différentes manières.
    
    Args:
        dem_path: Chemin vers le MNT prétraité
        texture_path: Chemin vers la texture prétraitée
        model_paths: Liste des chemins vers les modèles générés
    """
    print("\n" + "="*60)
    print("ÉTAPE 3: VISUALISATION ET COMPARAISON")
    print("="*60)
    
    # Dossier pour les visualisations
    vis_dir = os.path.join(os.path.dirname(__file__), '../visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Visualisation 2D du MNT
    print("\n🔄 Création d'une visualisation 2D du MNT...")
    vis_2d_path = os.path.join(vis_dir, 'dem_2d_visualization.png')
    visualize_dem_2d(
        dem_path, 
        output_path=vis_2d_path,
        hillshade=True,
        show_plot=False
    )
    
    # 2. Comparaison des facteurs d'exagération
    print("\n🔄 Génération d'une comparaison des différents facteurs d'exagération...")
    comparison_dir = os.path.join(vis_dir, 'exaggeration_comparison')
    
    screenshot_paths = compare_exaggeration_factors(
        dem_path,
        texture_path,
        factors=(1.0, 3.0, 5.0, 10.0),
        output_dir=comparison_dir,
        screenshot=True,
        show_interactive=False
    )
    
    print("\n✅ Visualisations terminées!")
    print(f"   - Visualisation 2D: {vis_2d_path}")
    print(f"   - Comparaison des facteurs: {comparison_dir}")


def main():
    """Fonction principale pour l'exemple de flux de travail."""
    parser = argparse.ArgumentParser(
        description='Exemple de flux de travail pour la visualisation 3D de ravins'
    )
    parser.add_argument('--dem', help='Chemin vers le fichier MNT (si non spécifié, cherche dans data/raw)')
    parser.add_argument('--texture', help='Chemin vers le fichier de texture')
    
    args = parser.parse_args()
    
    # Créer la structure de répertoires
    create_directory_structure()
    
    # Chercher un MNT par défaut si non spécifié
    if not args.dem:
        raw_data_dir = os.path.join(os.path.dirname(__file__), '../data/raw')
        tif_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.tif')]
        
        if tif_files:
            # Essayer de trouver un fichier qui pourrait être un MNT
            dem_candidates = [f for f in tif_files if 'dem' in f.lower() or 'mnt' in f.lower() or 'elev' in f.lower()]
            
            if dem_candidates:
                args.dem = os.path.join(raw_data_dir, dem_candidates[0])
                print(f"MNT automatiquement détecté: {args.dem}")
            else:
                # Prendre le premier fichier TIF
                args.dem = os.path.join(raw_data_dir, tif_files[0])
                print(f"Utilisation du premier fichier TIF trouvé comme MNT: {args.dem}")
        else:
            print("❌ Aucun fichier TIF trouvé dans data/raw/")
            print("Veuillez spécifier un chemin vers un MNT avec --dem")
            sys.exit(1)
    
    # Chercher une texture par défaut si non spécifiée
    if not args.texture:
        raw_data_dir = os.path.join(os.path.dirname(__file__), '../data/raw')
        tif_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.tif') and not os.path.samefile(os.path.join(raw_data_dir, f), args.dem)]
        
        if tif_files:
            # Essayer de trouver un fichier qui pourrait être une texture
            texture_candidates = [f for f in tif_files if 'tex' in f.lower() or 'ortho' in f.lower() or 'rgb' in f.lower()]
            
            if texture_candidates:
                args.texture = os.path.join(raw_data_dir, texture_candidates[0])
                print(f"Texture automatiquement détectée: {args.texture}")
            else:
                # Prendre le premier fichier TIF qui n'est pas le MNT
                args.texture = os.path.join(raw_data_dir, tif_files[0])
                print(f"Utilisation du premier fichier TIF disponible comme texture: {args.texture}")
        else:
            print("⚠️ Aucune texture trouvée dans data/raw/")
            print("Le modèle sera généré sans texture.")
    
    print("\n===== DÉMARRAGE DU FLUX DE TRAVAIL EXEMPLE =====\n")
    print(f"MNT: {args.dem}")
    print(f"Texture: {args.texture if args.texture else 'Non spécifiée'}")
    
    # 1. Prétraitement
    preprocessed_dir = os.path.join(os.path.dirname(__file__), '../data/preprocessed')
    dem_preprocessed, texture_preprocessed = preprocess_data(args.dem, args.texture, preprocessed_dir)
    
    # 2. Génération des modèles
    model_paths = generate_models(dem_preprocessed, texture_preprocessed)
    
    # 3. Visualisation
    visualize_results(dem_preprocessed, texture_preprocessed, model_paths)
    
    print("\n" + "="*60)
    print("FLUX DE TRAVAIL TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print("\nConsultez les dossiers suivants pour les résultats:")
    print(f" - Données prétraitées: {preprocessed_dir}")
    print(f" - Modèles 3D générés: {os.path.dirname(model_paths[0])}")
    print(f" - Visualisations: {os.path.join(os.path.dirname(__file__), '../visualizations')}")


if __name__ == "__main__":
    main()
