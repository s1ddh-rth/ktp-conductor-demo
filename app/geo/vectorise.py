"""Vector output: linestrings → GeoJSON FeatureCollection.

The TTPLA imagery is not georeferenced, so for the demo we map pixel
coordinates onto a synthetic latitude/longitude box centred on a sample
location. This is purely for visualisation on a Leaflet map; for the real
KTP project, segmenter outputs would be georeferenced via the camera-pose
metadata of the aerial flight.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, mapping


def linestrings_to_geojson(
    lines: list[LineString],
    image_size: tuple[int, int],
    bbox_latlon: tuple[float, float, float, float] = (
        51.4030,  # south
        -1.2700,  # west
        51.4090,  # north
        -1.2580,  # east  (≈ small box near Thatcham)
    ),
    properties_fn: Any = None,
) -> dict:
    """Convert pixel-space linestrings to GeoJSON in the given lat/lon box.

    The bbox defaults to a small box near Thatcham (the SSEN base location)
    purely for visual context. This is *not* claimed as a real geo-reference.
    """
    h, w = image_size
    south, west, north, east = bbox_latlon

    def to_lonlat(x: float, y: float) -> tuple[float, float]:
        lon = west + (x / w) * (east - west)
        lat = north - (y / h) * (north - south)
        return lon, lat

    features = []
    for i, ls in enumerate(lines):
        coords = [to_lonlat(x, y) for x, y in ls.coords]
        props: dict = {"id": i, "length_px": float(ls.length)}
        if properties_fn:
            props.update(properties_fn(ls))
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(LineString(coords)),
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}
