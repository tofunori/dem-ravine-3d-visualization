#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilitaires pour le prétraitement des MNT et des orthophotos avant 
la visualisation 3D et l'exagération verticale.

Ce module fournit des fonctions pour:
- Découper les MNT en zones d'intérêt
- Réduire la résolution pour les fichiers trop volumineux
- Aligner les MNT et les orthophotos
- Filtrer/lisser les MNT pour réduire le bruit
"""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import subprocess
from scipy import ndimage


def crop_raster(input_path, output_path, bounds):
    """
    Découpe un raster selon les coordonnées spécifiées.
    
    Args:
        input_path (str): Chemin vers le fichier raster d'entrée.
        output_path (str): Chemin où sauvegarder le raster découpé.
        bounds (tuple): Coordonnées de découpage (minx, miny, maxx, maxy).
        
    Returns:
        str: Chemin vers le fichier de sortie si succès, None sinon.
    """
    try:
        # Utiliser GDAL via subprocess pour le découpage
        cmd = [
            'gdal_translate', '-projwin',
            str(bounds[0]), str(bounds[3]), str(bounds[2]), str(bounds[1]),
            input_path, output_path
        ]
        
        # Exécuter la commande
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Raster découpé sauvegardé à {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Erreur lors du découpage du raster: {e}")
        print(f"Détails: {e.stderr}")
        return None


def resample_raster(input_path, output_path, target_resolution, method='bilinear'):
    """
    Rééchantillonne un raster à une résolution cible.
    
    Args:
        input_path (str): Chemin vers le fichier raster d'entrée.
        output_path (str): Chemin où sauvegarder le raster rééchantillonné.
        target_resolution (tuple): Résolution cible (x_res, y_res) en unités du raster.
        method (str): Méthode de rééchantillonnage ('nearest', 'bilinear', 'cubic', 'lanczos').
        
    Returns:
        str: Chemin vers le fichier de sortie si succès, None sinon.
    """
    # Mapping des méthodes de rééchantillonnage
    resampling_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'lanczos': Resampling.lanczos
    }
    
    resampling = resampling_methods.get(method.lower(), Resampling.bilinear)
    
    try:
        with rasterio.open(input_path) as src:
            # Calculer les nouvelles dimensions
            x_factor = src.transform[0] / target_resolution[0]
            y_factor = abs(src.transform[4]) / target_resolution[1]
            
            width = int(src.width / x_factor)
            height = int(src.height / y_factor)
            
            # Calculer la nouvelle transformation
            transform = rasterio.transform.from_origin(
                src.bounds.left, src.bounds.top,
                target_resolution[0], target_resolution[1]
            )
            
            # Définir les métadonnées de sortie
            out_kwargs = src.meta.copy()
            out_kwargs.update({
                'height': height,
                'width': width,
                'transform': transform
            })
            
            # Rééchantillonner et sauvegarder
            with rasterio.open(output_path, 'w', **out_kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=resampling
                    )
            
            print(f"✓ Raster rééchantillonné sauvegardé à {output_path}")
            print(f"  Nouvelle résolution: {target_resolution[0]}m x {target_resolution[1]}m")
            print(f"  Nouvelles dimensions: {width}x{height} pixels")
            return output_path
    
    except Exception as e:
        print(f"⚠️ Erreur lors du rééchantillonnage du raster: {e}")
        return None


def align_rasters(dem_path, texture_path, output_dem=None, output_texture=None):
    """
    Aligne un MNT et une texture (orthophoto) pour qu'ils aient la même emprise et résolution.
    
    Args:
        dem_path (str): Chemin vers le fichier MNT.
        texture_path (str): Chemin vers le fichier de texture.
        output_dem (str, optional): Chemin pour sauvegarder le MNT aligné.
        output_texture (str, optional): Chemin pour sauvegarder la texture alignée.
        
    Returns:
        tuple: (dem_path, texture_path) chemins vers les fichiers alignés.
    """
    # Générer des noms de fichiers par défaut si non spécifiés
    if output_dem is None:
        base, ext = os.path.splitext(dem_path)
        output_dem = f"{base}_aligned{ext}"
    
    if output_texture is None:
        base, ext = os.path.splitext(texture_path)
        output_texture = f"{base}_aligned{ext}"
    
    try:
        # Ouvrir les fichiers source
        with rasterio.open(dem_path) as dem_src, rasterio.open(texture_path) as tex_src:
            # Déterminer l'emprise commune (intersection)
            intersection = rasterio.coords.BoundingBox(
                max(dem_src.bounds.left, tex_src.bounds.left),
                max(dem_src.bounds.bottom, tex_src.bounds.bottom),
                min(dem_src.bounds.right, tex_src.bounds.right),
                min(dem_src.bounds.top, tex_src.bounds.top)
            )
            
            # Vérifier si les rasters se chevauchent
            if (intersection.left >= intersection.right or 
                intersection.bottom >= intersection.top):
                print("⚠️ Les rasters ne se chevauchent pas!")
                return dem_path, texture_path
            
            # Déterminer la résolution cible (la plus grossière des deux)
            dem_res = (dem_src.transform[0], abs(dem_src.transform[4]))
            tex_res = (tex_src.transform[0], abs(tex_src.transform[4]))
            
            target_res = (
                max(dem_res[0], tex_res[0]),
                max(dem_res[1], tex_res[1])
            )
            
            # Découper et rééchantillonner le MNT
            temp_dem = crop_raster(
                dem_path,
                f"{os.path.splitext(output_dem)[0]}_cropped.tif",
                (intersection.left, intersection.bottom, intersection.right, intersection.top)
            )
            
            aligned_dem = resample_raster(
                temp_dem,
                output_dem,
                target_res
            )
            
            # Découper et rééchantillonner la texture
            temp_texture = crop_raster(
                texture_path,
                f"{os.path.splitext(output_texture)[0]}_cropped.tif",
                (intersection.left, intersection.bottom, intersection.right, intersection.top)
            )
            
            aligned_texture = resample_raster(
                temp_texture,
                output_texture,
                target_res,
                method='lanczos'  # Meilleure qualité pour les textures
            )
            
            # Nettoyer les fichiers temporaires
            if os.path.exists(temp_dem) and temp_dem != aligned_dem:
                os.remove(temp_dem)
            
            if os.path.exists(temp_texture) and temp_texture != aligned_texture:
                os.remove(temp_texture)
            
            print(f"✓ Rasters alignés avec succès")
            print(f"  Emprise commune: {intersection}")
            print(f"  Résolution commune: {target_res[0]}m x {target_res[1]}m")
            
            return aligned_dem, aligned_texture
    
    except Exception as e:
        print(f"⚠️ Erreur lors de l'alignement des rasters: {e}")
        return dem_path, texture_path


def smooth_dem(input_path, output_path, sigma=1.0, method='gaussian'):
    """
    Lisse un MNT pour réduire le bruit.
    
    Args:
        input_path (str): Chemin vers le fichier MNT.
        output_path (str): Chemin où sauvegarder le MNT lissé.
        sigma (float): Paramètre de lissage (rayon ou écart-type).
        method (str): Méthode de lissage ('gaussian', 'median', 'uniform').
        
    Returns:
        str: Chemin vers le fichier de sortie si succès, None sinon.
    """
    try:
        with rasterio.open(input_path) as src:
            dem_data = src.read(1)
            
            # Appliquer le filtre selon la méthode choisie
            if method.lower() == 'gaussian':
                smoothed = ndimage.gaussian_filter(dem_data, sigma=sigma)
            elif method.lower() == 'median':
                smoothed = ndimage.median_filter(dem_data, size=int(sigma*2)+1)
            elif method.lower() == 'uniform':
                smoothed = ndimage.uniform_filter(dem_data, size=int(sigma*2)+1)
            else:
                print(f"⚠️ Méthode de lissage inconnue: {method}. Utilisation de gaussian.")
                smoothed = ndimage.gaussian_filter(dem_data, sigma=sigma)
            
            # Sauvegarder le résultat
            out_kwargs = src.meta.copy()
            with rasterio.open(output_path, 'w', **out_kwargs) as dst:
                dst.write(smoothed, 1)
            
            print(f"✓ MNT lissé sauvegardé à {output_path}")
            print(f"  Méthode: {method}, Paramètre: {sigma}")
            return output_path
    
    except Exception as e:
        print(f"⚠️ Erreur lors du lissage du MNT: {e}")
        return None


def fill_nodata(input_path, output_path, max_distance=10):
    """
    Remplit les valeurs NoData dans un MNT en interpolant à partir des valeurs voisines.
    
    Args:
        input_path (str): Chemin vers le fichier MNT.
        output_path (str): Chemin où sauvegarder le MNT sans NoData.
        max_distance (int): Distance maximale (en pixels) pour l'interpolation.
        
    Returns:
        str: Chemin vers le fichier de sortie si succès, None sinon.
    """
    try:
        # Utiliser GDAL via subprocess pour combler les NoData
        cmd = [
            'gdal_fillnodata.py',
            '-md', str(max_distance),
            input_path,
            output_path
        ]
        
        # Exécuter la commande
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ NoData comblés dans le MNT: {output_path}")
        return output_path
    
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Erreur lors du remplissage des NoData: {e}")
        print(f"Détails: {e.stderr}")
        return None


def extract_ravine_area(dem_path, output_path, ravine_threshold=-0.5, min_size=100):
    """
    Extrait automatiquement la zone d'un ravin basée sur la courbure du terrain.
    
    Args:
        dem_path (str): Chemin vers le fichier MNT.
        output_path (str): Chemin où sauvegarder le masque du ravin.
        ravine_threshold (float): Seuil de courbure pour détecter les ravins.
        min_size (int): Taille minimale (en pixels) pour un ravin.
        
    Returns:
        tuple: (output_path, bounds) où bounds est l'emprise du ravin.
    """
    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            
            # Calculer la courbure (approximation par Laplacien)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            curvature = ndimage.convolve(dem, kernel)
            
            # Créer un masque binaire pour les zones de ravin
            ravine_mask = curvature < ravine_threshold
            
            # Supprimer les petits objets
            labeled_mask, num_features = ndimage.label(ravine_mask)
            sizes = ndimage.sum(ravine_mask, labeled_mask, range(1, num_features+1))
            mask_sizes = sizes >= min_size
            cleaned_mask = mask_sizes[labeled_mask-1]
            
            # Trouver les limites du ravin
            rows, cols = np.where(cleaned_mask)
            if len(rows) > 0 and len(cols) > 0:
                row_min, row_max = rows.min(), rows.max()
                col_min, col_max = cols.min(), cols.max()
                
                # Ajouter une marge
                margin = 20  # pixels
                row_min = max(0, row_min - margin)
                row_max = min(dem.shape[0]-1, row_max + margin)
                col_min = max(0, col_min - margin)
                col_max = min(dem.shape[1]-1, col_max + margin)
                
                # Convertir en coordonnées géographiques
                transform = src.transform
                minx = transform[0] + col_min * transform[1]
                maxx = transform[0] + col_max * transform[1]
                miny = transform[3] + row_max * transform[5]
                maxy = transform[3] + row_min * transform[5]
                
                bounds = (minx, miny, maxx, maxy)
                
                # Sauvegarder le masque
                out_meta = src.meta.copy()
                out_meta.update({
                    'dtype': 'uint8',
                    'count': 1
                })
                
                with rasterio.open(output_path, 'w', **out_meta) as dst:
                    dst.write(cleaned_mask.astype('uint8'), 1)
                
                print(f"✓ Zone de ravin extraite: {output_path}")
                print(f"  Emprise du ravin: {bounds}")
                return output_path, bounds
            else:
                print("⚠️ Aucun ravin détecté avec les paramètres actuels.")
                return None, None
    
    except Exception as e:
        print(f"⚠️ Erreur lors de l'extraction de la zone de ravin: {e}")
        return None, None


def main():
    """Point d'entrée pour tester les fonctions de prétraitement."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prétraitement des MNT et orthophotos pour visualisation 3D.'
    )
    parser.add_argument('--dem', required=True, help='Chemin vers le fichier MNT')
    parser.add_argument('--texture', help='Chemin vers le fichier de texture')
    parser.add_argument('--output_dir', default='preprocessed', help='Répertoire de sortie')
    parser.add_argument('--align', action='store_true', help='Aligner les rasters')
    parser.add_argument('--smooth', action='store_true', help='Lisser le MNT')
    parser.add_argument('--extract_ravine', action='store_true', help='Extraire la zone du ravin')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    dem_path = args.dem
    texture_path = args.texture
    
    # Extraire la zone du ravin si demandé
    if args.extract_ravine:
        mask_path = os.path.join(args.output_dir, 'ravine_mask.tif')
        _, bounds = extract_ravine_area(dem_path, mask_path)
        
        if bounds:
            # Découper le MNT et la texture à la zone du ravin
            dem_path = crop_raster(
                dem_path,
                os.path.join(args.output_dir, 'dem_ravine.tif'),
                bounds
            )
            
            if texture_path:
                texture_path = crop_raster(
                    texture_path,
                    os.path.join(args.output_dir, 'texture_ravine.tif'),
                    bounds
                )
    
    # Aligner les rasters si demandé
    if args.align and texture_path:
        dem_path, texture_path = align_rasters(
            dem_path,
            texture_path,
            os.path.join(args.output_dir, 'dem_aligned.tif'),
            os.path.join(args.output_dir, 'texture_aligned.tif')
        )
    
    # Lisser le MNT si demandé
    if args.smooth:
        dem_path = smooth_dem(
            dem_path,
            os.path.join(args.output_dir, 'dem_smoothed.tif'),
            sigma=1.0
        )
    
    print(f"✅ Prétraitement terminé!")
    print(f"  MNT final: {dem_path}")
    if texture_path:
        print(f"  Texture finale: {texture_path}")


if __name__ == "__main__":
    main()
