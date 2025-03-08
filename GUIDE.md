# Guide d'utilisation détaillé pour la visualisation 3D de ravins de thermo-érosion

Ce guide vous explique en détail comment utiliser ce dépôt pour visualiser des ravins de thermo-érosion en 3D avec exagération verticale. Il inclut des explications pas à pas ainsi que des recommandations pour optimiser vos résultats.

## Table des matières

1. [Installation](#1-installation)
2. [Préparation des données](#2-préparation-des-données)
3. [Flux de travail de base](#3-flux-de-travail-de-base)
4. [Configuration avancée](#4-configuration-avancée)
5. [Visualisation des résultats](#5-visualisation-des-résultats)
6. [Analyse comparative](#6-analyse-comparative)
7. [Exportation pour d'autres logiciels](#7-exportation-pour-dautres-logiciels)
8. [Résolution de problèmes](#8-résolution-de-problèmes)

## 1. Installation

### Prérequis

Avant de commencer, assurez-vous d'avoir installé:
- Python 3.8 ou supérieur
- Git

### Étapes d'installation

1. **Cloner le dépôt**:
   ```bash
   git clone https://github.com/tofunori/dem-ravine-3d-visualization.git
   cd dem-ravine-3d-visualization
   ```

2. **Créer un environnement virtuel**:
   
   Sur Linux/macOS:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   
   Sur Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Installer les dépendances**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Installation de GDAL**:
   
   GDAL peut être difficile à installer via pip. Si vous rencontrez des problèmes:
   
   Sur Linux:
   ```bash
   sudo apt-get install libgdal-dev
   pip install gdal==$(gdal-config --version)
   ```
   
   Sur macOS (avec Homebrew):
   ```bash
   brew install gdal
   pip install gdal
   ```
   
   Sur Windows, utilisez les wheels précompilées:
   ```bash
   pip install GDAL‑<version>‑cp<python_version>‑cp<python_version>‑win_amd64.whl
   ```
   
   Les wheels sont disponibles sur: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal

## 2. Préparation des données

### Formats de données supportés

- **MNT (Modèle Numérique de Terrain)**: Fichiers GeoTIFF (.tif) avec une seule bande représentant les élévations
- **Texture/Orthophoto**: Fichiers GeoTIFF (.tif) RVB ou RVBA

### Organisation recommandée des données

```
dem-ravine-3d-visualization/
├── data/
│   ├── raw/             # Données brutes
│   ├── preprocessed/    # Données prétraitées
│   └── output/          # Résultats générés
```

### Prétraitement des données

Si vos données sont très volumineuses ou nécessitent un nettoyage, utilisez le script de prétraitement:

```bash
python preprocessing.py --dem data/raw/mon_mnt.tif --texture data/raw/ma_texture.tif --output_dir data/preprocessed --extract_ravine --align --smooth
```

Options principales:
- `--extract_ravine`: Détecte et extrait automatiquement la zone du ravin
- `--align`: Aligne le MNT et la texture pour qu'ils aient la même emprise et résolution
- `--smooth`: Applique un lissage gaussien pour réduire le bruit

## 3. Flux de travail de base

Voici la séquence typique d'utilisation:

1. **Prétraiter les données** (si nécessaire)
2. **Configurer les paramètres** (éditer `config.yml` ou utiliser des arguments en ligne de commande)
3. **Exécuter le script principal**
4. **Visualiser les résultats**

### Exemple d'utilisation simple

```bash
python dem_3d_exaggerator.py --dem data/preprocessed/dem_ravine.tif --texture data/preprocessed/texture_ravine.tif --exaggeration 5.0
```

Cela générera un modèle 3D avec un facteur d'exagération verticale de 5.0, ce qui rendra un ravin de 2m comme s'il avait 10m de profondeur.

### Vérification des résultats

Les résultats seront sauvegardés dans le dossier `output/` (ou celui spécifié avec `--output_dir`), avec un sous-dossier horodaté pour chaque exécution.

## 4. Configuration avancée

### Fichier de configuration YAML

Pour une configuration plus détaillée, modifiez le fichier `config.yml`. Les paramètres les plus importants sont:

- `exaggeration_factor`: Facteur d'exagération verticale
- `chunk_size`: Taille des morceaux pour le traitement (réduire si problèmes de mémoire)
- `optimization_factor`: Facteur de décimation du maillage (0-1)

### Ajustement du facteur d'exagération

Le facteur optimal dépend de:
- La profondeur du ravin (plus il est peu profond, plus l'exagération peut être forte)
- L'étendue horizontale de la zone
- L'objectif de visualisation

Recommandations:
- Ravins peu profonds (1-2m): facteur 5-10
- Ravins moyens (2-5m): facteur 3-5
- Ravins profonds (>5m): facteur 1-3

### Optimisation pour les grands jeux de données

Si vous travaillez avec des fichiers très volumineux:

1. **Réduire la résolution en prétraitement**:
   ```bash
   python preprocessing.py --dem input.tif --output_dir preprocessed --resample 0.5
   ```

2. **Ajuster les paramètres de chunking**:
   Modifiez `chunk_size` dans config.yml à une valeur plus petite (ex: 500)

3. **Augmenter l'optimisation du maillage**:
   Modifiez `optimization_factor` à une valeur plus élevée (ex: 0.7)

## 5. Visualisation des résultats

### Visualisation interactive

Pour visualiser interactivement le modèle 3D:

```bash
python visualization.py --dem data/preprocessed/dem_ravine.tif --texture data/preprocessed/texture_ravine.tif --mode 3d --exaggeration 5.0 --show
```

Cela ouvrira une fenêtre PyVista où vous pourrez:
- Faire pivoter le modèle avec la souris
- Zoomer avec la molette
- Ajuster l'éclairage et autres paramètres via l'interface

### Création d'une animation

Pour créer une animation du modèle (orbite autour du modèle):

```bash
python visualization.py --dem data/preprocessed/dem_ravine.tif --texture data/preprocessed/texture_ravine.tif --mode animation --output_dir animations
```

### Visualisation 2D pour comparaison

Pour voir une représentation 2D du MNT avec ombrage:

```bash
python visualization.py --dem data/preprocessed/dem_ravine.tif --mode 2d --output_dir visualizations
```

## 6. Analyse comparative

### Comparaison de différents facteurs d'exagération

Pour comparer visuellement l'effet de différents facteurs d'exagération:

```bash
python visualization.py --dem data/preprocessed/dem_ravine.tif --texture data/preprocessed/texture_ravine.tif --mode compare --output_dir comparisons
```

Cela générera des visualisations avec des facteurs de 1.0, 3.0, 5.0 et 10.0, ainsi qu'une image combinée pour faciliter la comparaison.

### Évaluation quantitative

Pour une analyse quantitative de l'exagération:

1. Générer des modèles avec différents facteurs
2. Comparer les statistiques d'élévation
3. Évaluer l'impact visuel des détails du ravin

## 7. Exportation pour d'autres logiciels

### Formats d'exportation supportés

Le script peut exporter les modèles 3D dans les formats suivants:
- OBJ (.obj) - Format le plus compatible, avec texture
- PLY (.ply) - Format polyvalent pour l'analyse
- STL (.stl) - Pour l'impression 3D
- VTK (.vtk) - Pour l'analyse scientifique

Exemple:
```bash
python dem_3d_exaggerator.py --dem data/dem.tif --texture data/texture.tif --format obj
```

### Utilisation dans d'autres logiciels

#### Blender
1. Importez le fichier OBJ généré (File > Import > Wavefront OBJ)
2. Les textures devraient être automatiquement liées
3. Ajustez l'éclairage pour mettre en valeur les reliefs

#### QGIS
1. Utilisez le plugin "Qgis2threejs" pour visualiser le modèle
2. Importez le MNT exagéré généré

#### Unity/Unreal Engine
1. Importez le fichier OBJ
2. Configurez les matériaux avec la texture
3. Ajoutez des effets d'éclairage pour améliorer le rendu

## 8. Résolution de problèmes

### Problèmes de mémoire

**Symptôme**: Erreur "MemoryError" ou "Out of memory"

**Solutions**:
1. Réduire `chunk_size` dans config.yml
2. Prétraiter les données pour réduire leur résolution
3. Augmenter le facteur d'optimisation du maillage

### Problèmes d'alignement texture/MNT

**Symptôme**: La texture ne s'aligne pas correctement avec le relief

**Solutions**:
1. Utilisez l'option `--align` du script de prétraitement
2. Vérifiez que les systèmes de coordonnées sont identiques
3. Assurez-vous que les emprise spatiales se chevauchent

### Problèmes visuels

**Symptôme**: Artefacts visuels ou effet d'escalier dans le rendu

**Solutions**:
1. Appliquez un lissage au MNT avec `--smooth`
2. Réduisez le facteur d'exagération
3. Ajustez l'éclairage dans les paramètres de visualisation

### Erreurs GDAL

**Symptôme**: Erreurs relatives à GDAL ou à la lecture des fichiers GeoTIFF

**Solutions**:
1. Vérifiez l'installation de GDAL
2. Assurez-vous que les versions de GDAL et rasterio sont compatibles
3. Validez que vos fichiers GeoTIFF ne sont pas corrompus

## Ressources additionnelles

- [Documentation PyVista](https://docs.pyvista.org/)
- [Documentation Rasterio](https://rasterio.readthedocs.io/)
- [Tutoriels GDAL](https://gdal.org/tutorials/index.html)
- [Guide des modèles numériques de terrain](https://opentopography.org/learn)
