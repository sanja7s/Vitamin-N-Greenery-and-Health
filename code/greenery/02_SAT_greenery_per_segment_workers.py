"""Satellite (NDVI) green inside a buffer around each OSM road, per osmid (Methods, eq. 6-8).

Also defines the road-centrality weight `choice_log` used by 03 (aerial, weighted) and 04 (street view, weighted).
Inputs : road edges (data/raw_data/road_network/London-roads/edges.shp), NDVI tiles, M25 choice shapefile
Outputs: data/greenery_metrics/SAT_greenery_per_segment.csv (+ per-tile cache)
Run    : from 02_SAT_greenery_per_segment.ipynb (joblib workers import this module)
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

REPO_ROOT = Path(__file__).resolve().parents[2]

BUFFER_M = 10   # road strip: 10 m on each side of the road centreline

ROADS_PATH = REPO_ROOT / "data" / "raw_data" / "road_network" / "London-roads" / "edges.shp"
NDVI_TILES_DIR = REPO_ROOT / "data" / "raw_data" / "green_cover_tiles"
M25_PATH = REPO_ROOT / "data" / "raw_data" / "M25" / "M25_seg_p.shp"
TILES_OUT_DIR = REPO_ROOT / "data" / "greenery_metrics" / "SAT_greenery_per_segment_tiles"
BUFFERS_PATH = TILES_OUT_DIR / "road_buffers.gpkg"
OUT_CSV = REPO_ROOT / "data" / "greenery_metrics" / "SAT_greenery_per_segment.csv"

# NDVI tiles are natively in British National Grid (metres).
CRS = 27700
OUT_COLUMNS = ["osmid", "sat_green_area", "sat_green_pct", "segment_tot_area"]


def build_road_buffers() -> None:
    """One dissolved buffer polygon per osmid, so overlapping edge buffers are counted once."""
    if BUFFERS_PATH.exists():
        return
    roads = gpd.read_file(ROADS_PATH, columns=["osmid"]).to_crs(CRS)
    roads["geometry"] = roads.buffer(BUFFER_M)
    buffers = roads.dissolve("osmid", as_index=False)
    buffers["segment_tot_area"] = buffers.area
    TILES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BUFFERS_PATH.with_name("tmp_" + BUFFERS_PATH.name)
    buffers.to_file(tmp, driver="GPKG")
    os.replace(tmp, BUFFERS_PATH)


# M25 and OSM centrelines never coincide (they only cross), so an edge is matched to the
# M25 segment running longest within M25_MATCH_M of it, if that covers >= M25_MIN_COVER of its length.
M25_MATCH_M = 10.0
M25_MIN_COVER = 0.5


# Road-centrality weight (paper eq. 8). M25 stores LogChR800 = log10(choice_800m + 1). As described in
# Methods, the choice value is log-transformed, scaled to 0-100 and log-transformed again for the weight;
# roads with a scaled value <= 1 (dead ends, cul-de-sacs) get weight 0.
def _choice_weight(log_choice: pd.Series) -> pd.Series:
    choice = np.exp(log_choice)
    scaled = (choice - choice.min()) / (choice.max() - choice.min()) * 100.0
    return np.log(scaled.where(scaled > 0)).fillna(0.0).clip(lower=0.0)


def edge_choice() -> gpd.GeoDataFrame:
    """Road edges with centrality weight `choice_log` (shared by the weighted measures in 03 and 04).

    Each edge takes LogChR800 of its best-matching M25 segment; unmatched edges are dropped.
    """
    roads = gpd.read_file(ROADS_PATH, columns=["osmid"]).to_crs(CRS).reset_index(drop=True)
    m25 = gpd.read_file(M25_PATH, columns=["LogChR800"]).to_crs(CRS).reset_index(drop=True)
    corridor = gpd.GeoDataFrame(geometry=roads.buffer(M25_MATCH_M), crs=CRS)
    pairs = gpd.sjoin(corridor, m25, predicate="intersects")
    pairs["overlap"] = shapely.length(shapely.intersection(
        m25.geometry.values[pairs["index_right"]], corridor.geometry.values[pairs.index]
    ))
    best = pairs.sort_values("overlap").groupby(level=0).tail(1)
    best = best[best["overlap"] >= M25_MIN_COVER * roads.length.loc[best.index]]
    return roads.loc[best.index].assign(choice_log=_choice_weight(best["LogChR800"]))


def osmid_choice() -> pd.Series:
    """Length-weighted mean edge `choice_log` per osmid."""
    edges = edge_choice()
    length = edges.length
    return (edges["choice_log"] * length).groupby(edges["osmid"]).sum() / length.groupby(edges["osmid"]).sum()


def tile_paths() -> list[Path]:
    return sorted(NDVI_TILES_DIR.glob("ndvi_landcover_merge_*.shp"))


def tile_out(tile: Path) -> Path:
    return TILES_OUT_DIR / f"{tile.stem}.csv"


def process_tile(tile: Path) -> Path:
    """Green m² per osmid inside this tile; the output CSV doubles as the checkpoint."""
    out = tile_out(tile)
    if out.exists():
        return out
    green = gpd.read_file(tile)
    green = green.loc[green["cover_type"] == "green", ["geometry"]].to_crs(CRS)
    res = pd.DataFrame(columns=["osmid", "sat_green_area"])
    if not green.empty:
        green["geometry"] = green.make_valid(method="structure", keep_collapsed=False)
        buffers = gpd.read_file(BUFFERS_PATH, bbox=tuple(green.total_bounds), columns=["osmid"])
        inter = gpd.overlay(buffers, green, how="intersection", keep_geom_type=True)
        res = inter.assign(sat_green_area=inter.area).groupby("osmid", as_index=False)["sat_green_area"].sum()
    tmp = out.with_suffix(".tmp")
    res.to_csv(tmp, index=False)
    os.replace(tmp, out)
    return out


def merge_tiles() -> pd.DataFrame:
    """One row per road osmid; segments without any green get sat_green_area = 0."""
    missing = [t.name for t in tile_paths() if not tile_out(t).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} tile(s) not processed, e.g. {missing[:5]}")
    frames = [pd.read_csv(tile_out(t), dtype={"osmid": str}) for t in tile_paths()]
    green = pd.concat([f for f in frames if not f.empty]).groupby("osmid")["sat_green_area"].sum()
    out = gpd.read_file(BUFFERS_PATH, columns=["osmid", "segment_tot_area"], ignore_geometry=True)
    out["sat_green_area"] = out["osmid"].map(green).fillna(0.0)
    out["sat_green_pct"] = out["sat_green_area"] / out["segment_tot_area"]
    # Tiles share sub-m² slivers at their seams; anything beyond that is a real error.
    assert out["sat_green_pct"].max() <= 1 + 1e-3, out["sat_green_pct"].max()
    out["sat_green_pct"] = out["sat_green_pct"].clip(0.0, 1.0)
    return out[OUT_COLUMNS]


def write_csv(df: pd.DataFrame, out_path: Path | str = OUT_CSV) -> Path:
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    return out_path
