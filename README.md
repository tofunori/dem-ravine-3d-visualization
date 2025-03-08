# Visualisation 3D de Ravins de Thermo-érosion

Ce dépôt contient des outils et scripts pour créer des visualisations 3D de ravins de thermo-érosion avec **exagération verticale**, permettant de mettre en valeur des caractéristiques topographiques subtiles (comme des ravins de 2m de profondeur) dans les présentations et analyses.

![Exemple de visualisation d'un ravin exagéré verticalement](images/demo_visualization.jpg)

## 🌟 Fonctionnalités

- Charge des MNT (Modèles Numériques de Terrain) et des orthophotos pour la texture
- Applique une exagération verticale configurable pour accentuer les reliefs
- Génère des maillages 3D texturés exportables vers divers formats (OBJ, PLY, etc.)
- Optimise les données lourdes par un traitement par morceaux (chunking)
- Compatible avec les données à haute résolution (p. ex. 2cm/pixel)

## 📋 Prérequis

- Python 3.8+
- Les bibliothèques suivantes:
  - rasterio
  - numpy
  - pyvista
  - gdal (optionnel, pour le prétraitement)
  - matplotlib (pour la visualisation)

## 🚀 Installation

1. Clonez ce dépôt:
```bash
git clone https://github.com/tofunori/dem-ravine-3d-visualization.git
cd dem-ravine-3d-visualization
```

2. Créez un environnement virtuel et installez les dépendances:
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Guide d'utilisation

### Utilisation basique

1. Placez vos fichiers MNT et de texture dans le dossier `data/`
2. Ajustez les paramètres dans le fichier de configuration ou directement dans le script
3. Exécutez le script principal:
```bash
python dem_3d_exaggerator.py
```

### Configuration des paramètres d'exagération

Modifiez `config.yml` ou passez des arguments en ligne de commande:

```bash
python dem_3d_exaggerator.py --dem_file data/mon_mnt.tif --texture_file data/ma_texture.tif --exaggeration 5.0
```

- `exaggeration_factor`: Multiplicateur pour les hauteurs (ex: 5.0 rend un ravin de 2m comme ayant 10m de profondeur)
- `chunk_size`: Taille des blocs pour le traitement (réduire en cas de mémoire limitée)
- `output_format`: Format d'exportation du modèle 3D ("obj", "ply", etc.)

### Paramètres avancés

Consultez le fichier [ADVANCED.md](ADVANCED.md) pour:
- Optimisation pour les très grands jeux de données
- Prétraitement avec GDAL
- Techniques de lissage et de filtrage
- Intégration avec d'autres logiciels (Blender, QGIS, etc.)

## 🔧 Structure du code

- `dem_3d_exaggerator.py`: Script principal pour l'exagération et la génération de modèles 3D
- `preprocessing.py`: Utilitaires pour préparer et optimiser les données en amont
- `visualization.py`: Module pour la visualisation interactive des résultats
- `utils/`: Fonctions utilitaires diverses
- `examples/`: Exemples d'applications et cas d'utilisation

## 📝 Notes sur l'exagération verticale

L'exagération verticale est une technique courante en cartographie et visualisation 3D qui amplifie artificiellement les variations d'altitude. Elle est particulièrement utile pour:

- Rendre visibles des caractéristiques topographiques subtiles
- Mettre en évidence des phénomènes d'érosion ou de dégradation
- Améliorer l'impact visuel dans les présentations scientifiques

**Important**: Toujours mentionner dans vos présentations que l'échelle verticale a été exagérée, et préciser le facteur d'exagération utilisé pour éviter toute interprétation erronée.

## 🔍 Exemples de résultats

Voir le dossier [examples/results](examples/results) pour des visualisations comparatives:
- Avant/après exagération
- Différents facteurs d'exagération (2x, 5x, 10x)
- Rendus avec différents éclairages

## 👥 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
- Signaler des bugs
- Proposer des améliorations
- Soumettre des pull requests

## 📜 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.
