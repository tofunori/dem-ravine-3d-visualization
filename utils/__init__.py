"""
Module d'utilitaires pour la visualisation 3D de ravins de thermo-érosion.
"""

from .dem_utils import (
    calculate_slope,
    calculate_aspect,
    calculate_hillshade,
    calculate_curvature,
    calculate_roughness,
    get_dem_info,
    create_colored_hillshade,
    save_geotiff,
    create_empty_dem,
    calculate_volume_below_surface,
    dem_to_stl,
    plot_dem_profile,
    convert_dem_units
)

__all__ = [
    'calculate_slope',
    'calculate_aspect',
    'calculate_hillshade',
    'calculate_curvature',
    'calculate_roughness',
    'get_dem_info',
    'create_colored_hillshade',
    'save_geotiff',
    'create_empty_dem',
    'calculate_volume_below_surface',
    'dem_to_stl',
    'plot_dem_profile',
    'convert_dem_units'
]
