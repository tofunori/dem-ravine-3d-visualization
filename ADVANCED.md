# Guide Avancé pour la Visualisation 3D de Ravins

Ce document fournit des informations détaillées sur les fonctionnalités avancées et les techniques d'optimisation pour la visualisation 3D de ravins de thermo-érosion.

## Table des matières

1. [Optimisation pour très grands jeux de données](#1-optimisation-pour-très-grands-jeux-de-données)
2. [Prétraitement avancé avec GDAL](#2-prétraitement-avancé-avec-gdal)
3. [Techniques de lissage et filtrage](#3-techniques-de-lissage-et-filtrage)
4. [Détection automatique des ravins](#4-détection-automatique-des-ravins)
5. [Rendu avancé et éclairage](#5-rendu-avancé-et-éclairage)
6. [Intégration avec d'autres logiciels](#6-intégration-avec-dautres-logiciels)
7. [Analyses quantitatives](#7-analyses-quantitatives)
8. [Traitement par lots](#8-traitement-par-lots)

## 1. Optimisation pour très grands jeux de données

### 1.1 Morcellement (Chunking) adaptatif

Pour les MNT extrêmement volumineux (>10 000 x 10 000 pixels), vous pouvez utiliser un système de chunking adaptatif qui ajuste automatiquement la taille des morceaux en fonction de la mémoire disponible:

```python
# Exemple de code pour chunking adaptatif
import os
import psutil
import rasterio
from rasterio.windows import Window

def adaptive_chunk_size(dem_path, target_memory_usage=0.5):
    """
    Détermine une taille de chunk optimale basée sur la mémoire disponible.
    
    Args:
        dem_path: Chemin vers le fichier MNT
        target_memory_usage: Fraction de la mémoire disponible à utiliser (0-1)
        
    Returns:
        tuple: (chunk_width, chunk_height)
    """
    # Obtenir la mémoire disponible
    available_memory = psutil.virtual_memory().available
    target_memory = available_memory * target_memory_usage
    
    # Lire les métadonnées du MNT
    with rasterio.open(dem_path) as src:
        dtype_size = src.dtypes[0].itemsize
        width, height = src.width, src.height
        
    # Calculer la taille de chunk maximale
    pixel_memory = dtype_size * 2  # Pour le MNT et les calculs intermédiaires
    max_pixels = target_memory / pixel_memory
    
    # Maintenir le ratio d'aspect
    aspect_ratio = width / height
    chunk_height = int((max_pixels / aspect_ratio) ** 0.5)
    chunk_width = int(chunk_height * aspect_ratio)
    
    # Limiter à la taille du fichier
    chunk_width = min(chunk_width, width)
    chunk_height = min(chunk_height, height)
    
    return chunk_width, chunk_height
```

### 1.2 Partitionnement et traitement parallèle

Pour les systèmes avec plusieurs cœurs, vous pouvez implémenter un traitement parallèle:

```python
import multiprocessing
from functools import partial
import numpy as np

def process_chunk(args):
    """Traite un chunk individuel avec exagération verticale."""
    window, dem_path, texture_path, exag_factor, output_dir = args
    # ... traitement du chunk ...
    return result_path

def parallel_process_dem(dem_path, texture_path, exag_factor, chunk_size, output_dir, n_processes=None):
    """Traite un grand MNT en parallèle par morceaux."""
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)
    
    with rasterio.open(dem_path) as src:
        width, height = src.width, src.height
        
    # Générer les fenêtres de chunking
    windows = []
    for j in range(0, height, chunk_size[1]):
        for i in range(0, width, chunk_size[0]):
            w = min(chunk_size[0], width - i)
            h = min(chunk_size[1], height - j)
            windows.append(Window(i, j, w, h))
    
    # Préparer les arguments pour chaque processus
    process_args = [(window, dem_path, texture_path, exag_factor, output_dir) for window in windows]
    
    # Traitement parallèle
    with multiprocessing.Pool(n_processes) as pool:
        results = pool.map(process_chunk, process_args)
    
    # Fusionner les résultats si nécessaire
    # ...
    
    return results
```

### 1.3 Décimation progressive

Pour optimiser davantage les maillages très denses, utilisez une décimation progressive qui préserve les caractéristiques importantes:

```python
def progressive_decimate(mesh, target_reduction=0.5, quality_threshold=0.8):
    """
    Réduit progressivement le maillage tout en préservant les caractéristiques importantes.
    
    Args:
        mesh: Maillage PyVista
        target_reduction: Réduction cible (0-1)
        quality_threshold: Seuil de qualité pour arrêter la décimation
        
    Returns:
        pyvista.PolyData: Maillage décimé
    """
    # Extraire la surface
    surface = mesh.extract_surface()
    
    # Calculer la courbure pour identifier les zones importantes
    surface.compute_normals(inplace=True)
    curv = surface.curvature(curv_type='mean')
    
    # Normaliser la courbure
    curv_range = curv.max() - curv.min()
    if curv_range > 0:
        importance = (curv - curv.min()) / curv_range
    else:
        importance = np.ones_like(curv)
    
    # Ajouter l'importance comme scalaire
    surface.point_data['importance'] = importance
    
    # Décimation progressive
    reduction_step = 0.1
    current_reduction = 0
    
    while current_reduction < target_reduction:
        # Calculer le prochain niveau de réduction
        next_reduction = min(current_reduction + reduction_step, target_reduction)
        
        # Décimer
        decimated = surface.decimate_pro(
            target_reduction=next_reduction,
            feature_angle=60,
            preserve_topology=True,
            boundary_vertex_deletion=False
        )
        
        # Vérifier la qualité
        quality = evaluate_mesh_quality(decimated, surface)
        
        if quality < quality_threshold:
            # Revenir au niveau précédent si la qualité est insuffisante
            break
        
        surface = decimated
        current_reduction = next_reduction
    
    return surface
```

## 2. Prétraitement avancé avec GDAL

### 2.1 Alignement précis des orthophotos

Pour un alignement parfait entre MNT et orthophoto avec des résolutions différentes:

```bash
# Rééchantillonnage d'une orthophoto pour correspondre exactement à un MNT
gdalwarp -r lanczos -tr 0.02 0.02 -te $XMIN $YMIN $XMAX $YMAX -t_srs EPSG:32632 input_ortho.tif aligned_ortho.tif
```

### 2.2 Fusion de MNT

Pour combiner plusieurs MNT adjacents:

```bash
# Fusionner plusieurs fichiers en un seul
gdal_merge.py -o merged_dem.tif -n -9999 input_dem1.tif input_dem2.tif input_dem3.tif

# Harmoniser les valeurs aux bordures
gdal_fillnodata.py -md 10 merged_dem.tif merged_dem_filled.tif
```

### 2.3 Traitement des artefacts et valeurs aberrantes

Pour éliminer les valeurs aberrantes qui peuvent affecter la visualisation:

```bash
# Identifier et filtrer les valeurs aberrantes
gdal_calc.py --calc="numpy.where((A<$MIN_ELEV) | (A>$MAX_ELEV), -9999, A)" --outfile=filtered_dem.tif -A input_dem.tif --NoDataValue=-9999

# Combler les zones NoData résultantes
gdal_fillnodata.py -md 20 filtered_dem.tif cleaned_dem.tif
```

## 3. Techniques de lissage et filtrage

### 3.1 Lissage adaptatif

Le lissage adaptatif préserve les caractéristiques importantes (comme les bords du ravin) tout en lissant les zones de bruit:

```python
def adaptive_smoothing(dem_data, sigma_min=0.5, sigma_max=2.0, curvature_threshold=0.1):
    """
    Lissage adaptatif qui préserve les bords importants.
    
    Args:
        dem_data: Données d'élévation
        sigma_min: Sigma minimum pour les zones à forte courbure
        sigma_max: Sigma maximum pour les zones plates
        curvature_threshold: Seuil pour identifier les caractéristiques importantes
        
    Returns:
        numpy.ndarray: MNT lissé
    """
    from scipy import ndimage
    
    # Calculer la courbure (approximation par Laplacien)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    curvature = np.abs(ndimage.convolve(dem_data, kernel))
    
    # Normaliser la courbure
    curvature_norm = curvature / curvature.max()
    
    # Calculer le sigma adaptatif pour chaque pixel
    # Forte courbure -> faible sigma, faible courbure -> fort sigma
    sigma_map = sigma_max - (sigma_max - sigma_min) * np.minimum(curvature_norm / curvature_threshold, 1.0)
    
    # Appliquer le lissage adaptatif
    smoothed = np.copy(dem_data)
    
    # Pour chaque valeur de sigma utilisée
    sigma_values = np.linspace(sigma_min, sigma_max, 10)
    
    for sigma in sigma_values:
        # Créer un masque pour les pixels à traiter avec ce sigma
        mask = (sigma_map >= sigma - 0.1) & (sigma_map <= sigma + 0.1)
        
        if np.any(mask):
            # Lisser seulement les zones avec ce sigma
            smoothed_part = ndimage.gaussian_filter(dem_data, sigma=sigma)
            smoothed[mask] = smoothed_part[mask]
    
    return smoothed
```

### 3.2 Filtrage morphologique pour la préservation des ravins

```python
def ravine_preserving_filter(dem_data, filter_size=3):
    """
    Filtre qui préserve les structures linéaires comme les ravins.
    
    Args:
        dem_data: Données d'élévation
        filter_size: Taille du filtre
        
    Returns:
        numpy.ndarray: MNT filtré
    """
    from skimage import morphology
    
    # Détection des structures linéaires (ravins)
    # Créer un élément structurant linéaire dans différentes directions
    angles = range(0, 180, 15)
    ravine_mask = np.zeros_like(dem_data, dtype=bool)
    
    for angle in angles:
        # Créer un élément structurant linéaire
        selem = morphology.disk(filter_size)
        
        # Appliquer une ouverture morphologique
        opened = morphology.opening(dem_data, selem)
        
        # La différence identifie les structures plus petites que filter_size
        diff = dem_data - opened
        
        # Identifier les zones de ravins potentiels
        ravine_mask |= (diff < -0.2)  # Ajuster le seuil selon vos données
    
    # Appliquer un filtrage médian partout sauf dans les ravins
    filtered = ndimage.median_filter(dem_data, size=filter_size)
    
    # Préserver les valeurs originales dans les zones de ravins
    filtered[ravine_mask] = dem_data[ravine_mask]
    
    return filtered
```

## 4. Détection automatique des ravins

### 4.1 Détection par analyse de courbure

```python
def detect_ravines(dem_data, curvature_threshold=-0.2, min_size=100):
    """
    Détecte automatiquement les ravins dans un MNT en utilisant l'analyse de courbure.
    
    Args:
        dem_data: Données d'élévation
        curvature_threshold: Seuil de courbure négative pour identifier les ravins
        min_size: Taille minimale (en pixels) pour un ravin
        
    Returns:
        tuple: (mask, boundaries) 
    """
    from scipy import ndimage
    from skimage import measure
    
    # Calculer la courbure
    # Une courbure négative indique un "creux" comme un ravin
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    curvature = ndimage.convolve(dem_data, kernel)
    
    # Créer un masque binaire
    ravine_mask = curvature < curvature_threshold
    
    # Supprimer les petits objets
    labeled_mask, num_features = ndimage.label(ravine_mask)
    sizes = ndimage.sum(ravine_mask, labeled_mask, range(1, num_features+1))
    mask_sizes = sizes >= min_size
    cleaned_mask = mask_sizes[labeled_mask-1]
    
    # Trouver les contours des ravins
    contours = measure.find_contours(cleaned_mask.astype(float), 0.5)
    
    return cleaned_mask, contours
```

### 4.2 Analyse morphologique des écoulements

```python
def flow_accumulation_analysis(dem_data, flow_threshold=100):
    """
    Utilise l'analyse d'accumulation d'écoulement pour détecter les ravins.
    Nécessite PySheds ou RichDEM.
    """
    import richdem as rd
    
    # Convertir en format RichDEM
    rddem = rd.rdarray(dem_data, no_data=-9999)
    
    # Combler les dépressions
    filled = rd.FillDepressions(rddem, epsilon=True)
    
    # Calculer la direction d'écoulement
    flow_dir = rd.FlowDirectionD8(filled)
    
    # Calculer l'accumulation d'écoulement
    flow_acc = rd.FlowAccumulation(flow_dir)
    
    # Créer un masque pour les zones d'écoulement significatif
    ravine_mask = flow_acc > flow_threshold
    
    return ravine_mask
```

## 5. Rendu avancé et éclairage

### 5.1 Configuration d'éclairage multi-source

Pour un rendu plus réaliste des détails du ravin:

```python
def setup_advanced_lighting(plotter):
    """
    Configure un système d'éclairage avancé pour mettre en valeur les ravins.
    
    Args:
        plotter: Objet PyVista Plotter
    """
    # Lumière principale (soleil) - direction 315° azimut, 45° élévation
    main_light = pv.Light(
        position=(10, 10, 10),
        focal_point=(0, 0, 0),
        color='white',
        intensity=0.7
    )
    
    # Lumière secondaire côté opposé (remplissage) - bleutée
    fill_light = pv.Light(
        position=(-8, -8, 12),
        focal_point=(0, 0, 0),
        color=[0.8, 0.85, 1.0],  # Bleu très léger
        intensity=0.3
    )
    
    # Lumière d'accentuation pour les ravins (rasante) - ambrée
    accent_light = pv.Light(
        position=(0, -15, 3),
        focal_point=(0, 0, 0),
        color=[1.0, 0.9, 0.7],  # Ambrée
        intensity=0.4,
        cone_angle=30
    )
    
    # Lumière ambiante faible
    ambient_light = pv.Light(
        position=(0, 0, 15),
        focal_point=(0, 0, 0),
        color='white',
        intensity=0.15
    )
    
    # Ajouter toutes les lumières
    plotter.add_light(main_light)
    plotter.add_light(fill_light)
    plotter.add_light(accent_light)
    plotter.add_light(ambient_light)
    
    # Configuration générale de l'éclairage
    plotter.enable_shadows()
    plotter.set_environment_texture('skybox')  # Utiliser une texture d'environnement
```

### 5.2 Effets d'ombrage et de texture avancés

```python
def enhance_terrain_texture(dem_data, texture_data=None):
    """
    Améliore les textures du terrain en ajoutant des effets d'ombrage détaillés.
    """
    from matplotlib.colors import LightSource
    
    # Si pas de texture, créer une texture basée sur l'altitude
    if texture_data is None:
        # Normaliser les altitudes pour la texture
        norm_dem = (dem_data - np.min(dem_data)) / (np.max(dem_data) - np.min(dem_data))
        
        # Créer une colormap de terrain
        cmap = plt.cm.terrain
        rgb_texture = cmap(norm_dem)[:, :, :3]
    else:
        rgb_texture = texture_data
    
    # Calculer l'ombrage avec différents angles pour plus de détails
    ls1 = LightSource(azdeg=315, altdeg=45)
    ls2 = LightSource(azdeg=135, altdeg=30)
    
    # Combiner les ombrages
    rgb_shaded1 = ls1.shade_rgb(rgb_texture, dem_data, blend_mode='soft')
    rgb_shaded = ls2.shade_rgb(rgb_shaded1, dem_data, blend_mode='overlay', fraction=0.3)
    
    # Ajouter des détails de pente
    slope = np.gradient(dem_data)
    slope_intensity = np.sqrt(slope[0]**2 + slope[1]**2)
    norm_slope = slope_intensity / slope_intensity.max()
    
    # Assombrir légèrement les pentes fortes
    rgb_shaded = rgb_shaded * (1.0 - 0.2 * norm_slope[:, :, np.newaxis])
    
    return rgb_shaded
```

## 6. Intégration avec d'autres logiciels

### 6.1 Export pour Blender

Script Python pour automatiser l'importation dans Blender:

```python
def export_for_blender(obj_path, output_blend, exaggeration_factor):
    """
    Crée un fichier .blend avec le modèle correctement configuré.
    Nécessite bpy (Blender Python API).
    """
    import bpy
    import os
    
    # Réinitialiser Blender
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Supprimer tous les objets existants
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Importer l'OBJ
    bpy.ops.import_scene.obj(filepath=obj_path)
    
    # Configurer le matériau
    for obj in bpy.context.selected_objects:
        # Activer les matériaux de texture
        for slot in obj.material_slots:
            if slot.material:
                slot.material.use_nodes = True
                nodes = slot.material.node_tree.nodes
                
                # Configurer le shader de terrain
                shader = nodes.get('Principled BSDF')
                if shader:
                    shader.inputs['Roughness'].default_value = 0.7
                    shader.inputs['Specular'].default_value = 0.1
    
    # Configurer l'éclairage
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.object
    sun.location = (10, 10, 10)
    sun.data.energy = 2.0
    sun.rotation_euler = (0.6, 0.8, 1.2)
    
    # Ajouter une lumière d'appoint
    bpy.ops.object.light_add(type='AREA')
    fill = bpy.context.object
    fill.location = (-5, -5, 8)
    fill.data.energy = 0.8
    
    # Configurer la caméra
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.location = (15, -15, 10)
    camera.rotation_euler = (0.8, 0, 0.6)
    
    # Rendre la caméra active
    bpy.context.scene.camera = camera
    
    # Configurer le rendu
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    
    # Ajouter une note sur l'exagération verticale
    text_info = f"Exagération verticale: {exaggeration_factor}x"
    bpy.ops.object.text_add()
    text_obj = bpy.context.object
    text_obj.data.body = text_info
    text_obj.location = (-5, 5, 0)
    
    # Sauvegarder le fichier blend
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    
    return output_blend
```

### 6.2 Export pour Unity

```python
def prepare_for_unity(obj_path, output_dir):
    """
    Prépare les fichiers pour l'importation dans Unity.
    """
    import shutil
    import os
    from PIL import Image
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Copier le fichier OBJ
    shutil.copy(obj_path, os.path.join(output_dir, os.path.basename(obj_path)))
    
    # Convertir les textures au format PNG
    texture_dir = os.path.dirname(obj_path)
    for file in os.listdir(texture_dir):
        if file.endswith('.jpg') or file.endswith('.tif'):
            tex_path = os.path.join(texture_dir, file)
            img = Image.open(tex_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.png")
            img.save(output_path, 'PNG')
    
    # Créer un fichier README pour Unity
    with open(os.path.join(output_dir, 'README_UNITY.txt'), 'w') as f:
        f.write("Instructions pour Unity:\n\n")
        f.write("1. Importer l'ensemble des fichiers dans votre projet Unity\n")
        f.write("2. Faire glisser le fichier OBJ dans votre scène\n")
        f.write("3. Les matériaux devraient être automatiquement configurés\n")
        f.write("4. Pour un meilleur rendu, activer l'URP ou le HDRP\n")
        f.write("5. Ajouter des effets de post-processing pour améliorer l'apparence\n")
    
    return output_dir
```

## 7. Analyses quantitatives

### 7.1 Calcul de métriques de ravins

```python
def calculate_ravine_metrics(dem_data, ravine_mask, cell_size):
    """
    Calcule des métriques quantitatives pour les ravins.
    
    Args:
        dem_data: Données d'élévation
        ravine_mask: Masque binaire des zones de ravins
        cell_size: Taille des cellules en mètres
        
    Returns:
        dict: Métriques des ravins
    """
    from scipy import ndimage
    from skimage import measure
    
    # Mesurer les propriétés de chaque ravin
    labeled_ravines, num_ravines = ndimage.label(ravine_mask)
    props = measure.regionprops(labeled_ravines, dem_data)
    
    metrics = {
        'total_count': num_ravines,
        'total_area': np.sum(ravine_mask) * cell_size**2,  # m²
        'ravines': []
    }
    
    for i, prop in enumerate(props):
        # Créer un masque pour ce ravin
        ravine_i_mask = labeled_ravines == prop.label
        
        # Calculer les élévations dans ce ravin
        ravine_elevations = dem_data[ravine_i_mask]
        
        # Calculer les métriques
        ravine_metrics = {
            'id': i + 1,
            'area': prop.area * cell_size**2,  # m²
            'perimeter': prop.perimeter * cell_size,  # m
            'length': prop.major_axis_length * cell_size,  # m
            'width': prop.minor_axis_length * cell_size,  # m
            'orientation': prop.orientation,  # radians
            'min_elevation': np.min(ravine_elevations),
            'max_elevation': np.max(ravine_elevations),
            'depth': np.max(ravine_elevations) - np.min(ravine_elevations),
            'mean_depth': np.mean(ravine_elevations),
            'volume': np.sum(np.max(ravine_elevations) - ravine_elevations) * cell_size**2  # m³
        }
        
        metrics['ravines'].append(ravine_metrics)
    
    return metrics
```

### 7.2 Analyse de l'effet de l'exagération

```python
def analyze_exaggeration_effect(original_dem, exaggerated_dem, exag_factor):
    """
    Analyse l'effet de l'exagération sur différentes propriétés du terrain.
    """
    # Calcul des pentes
    original_slope = calculate_slope(original_dem)
    exag_slope = calculate_slope(exaggerated_dem)
    
    # Calcul des courbures
    original_curvature = calculate_curvature(original_dem)
    exag_curvature = calculate_curvature(exaggerated_dem)
    
    # Statistiques
    metrics = {
        'exaggeration_factor': exag_factor,
        'elevation_range_original': np.max(original_dem) - np.min(original_dem),
        'elevation_range_exaggerated': np.max(exaggerated_dem) - np.min(exaggerated_dem),
        'mean_slope_original': np.mean(original_slope),
        'mean_slope_exaggerated': np.mean(exag_slope),
        'max_slope_original': np.max(original_slope),
        'max_slope_exaggerated': np.max(exag_slope),
        'mean_curvature_original': np.mean(np.abs(original_curvature)),
        'mean_curvature_exaggerated': np.mean(np.abs(exag_curvature)),
        'slope_amplification': np.mean(exag_slope) / np.mean(original_slope),
        'curvature_amplification': np.mean(np.abs(exag_curvature)) / np.mean(np.abs(original_curvature))
    }
    
    return metrics
```

## 8. Traitement par lots

### 8.1 Script pour traitement de plusieurs MNT

```python
def batch_process_dems(input_dir, output_dir, config=None):
    """
    Traite tous les MNT dans un répertoire avec les mêmes paramètres.
    
    Args:
        input_dir: Répertoire contenant les MNT
        output_dir: Répertoire de sortie
        config: Configuration (dict)
    """
    import os
    import glob
    from dem_3d_exaggerator import process_dem_to_3d, load_config
    
    # Charger la configuration
    if config is None:
        config = load_config()
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Trouver tous les fichiers MNT
    dem_files = glob.glob(os.path.join(input_dir, "*.tif"))
    
    # Trouver les textures correspondantes
    for dem_file in dem_files:
        base_name = os.path.splitext(os.path.basename(dem_file))[0]
        
        # Chercher une texture avec un nom similaire
        texture_file = None
        possible_texture_names = [
            f"{base_name}_texture.tif",
            f"{base_name}_ortho.tif",
            f"{base_name.replace('dem', 'texture')}.tif",
            f"{base_name.replace('dem', 'ortho')}.tif"
        ]
        
        for tex_name in possible_texture_names:
            tex_path = os.path.join(input_dir, tex_name)
            if os.path.exists(tex_path):
                texture_file = tex_path
                break
        
        # Traiter le MNT
        print(f"Traitement de {dem_file}...")
        process_dem_to_3d(
            dem_file,
            texture_file,
            config
        )
    
    print(f"✅ Traitement par lots terminé!")
```

### 8.2 Automatisation des comparaisons

```python
def batch_compare_exaggeration_factors(dem_file, texture_file, factors=(1.0, 2.0, 5.0, 10.0), output_dir=None):
    """
    Génère des modèles avec différents facteurs d'exagération et les compare.
    """
    from visualization import compare_exaggeration_factors
    
    if output_dir is None:
        output_dir = f"comparison_{os.path.splitext(os.path.basename(dem_file))[0]}"
    
    # Générer les comparaisons
    compare_exaggeration_factors(
        dem_file,
        texture_file,
        factors=factors,
        output_dir=output_dir
    )
    
    # Créer un rapport HTML
    create_comparison_report(output_dir, dem_file, factors)
    
    return output_dir
```

Ces fonctionnalités avancées vous permettent d'optimiser le traitement des MNT de ravins de thermo-érosion, d'améliorer la qualité visuelle des rendus et d'extraire des métriques quantitatives pour vos analyses scientifiques.
