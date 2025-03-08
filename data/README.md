# Dossier de données

Ce dossier contient toutes les données nécessaires pour la visualisation 3D de ravins.

## Structure des sous-dossiers

- **raw/** : Placez ici vos fichiers MNT et textures d'origine
- **preprocessed/** : Contient les données après prétraitement (alignement, lissage, etc.)
- **output/** : Contient les modèles 3D générés et autres résultats

## Formats de données supportés

### MNT (Modèle Numérique de Terrain)
- Format : GeoTIFF (.tif)
- Type : Fichier raster à une seule bande représentant les élévations
- Résolution typique : 1-10 cm/pixel pour les ravins de petite taille

### Textures (Orthophotos)
- Format : GeoTIFF (.tif)
- Type : Images RVB ou RVBA géoréférencées
- Résolution : Idéalement similaire ou légèrement supérieure à celle du MNT

## Notes importantes

1. Assurez-vous que vos MNT et textures partagent le même système de coordonnées et la même emprise spatiale, ou utilisez la fonction d'alignement du script de prétraitement.

2. Si vous travaillez avec des fichiers très volumineux (>1 GB), considérez un prétraitement pour réduire leur taille avant de les utiliser avec ce système.

3. Les fichiers MNT doivent contenir des valeurs d'élévation réelles (et non des indices ou des codes).
