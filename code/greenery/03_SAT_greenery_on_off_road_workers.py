"""Satellite on-road / off-road green per LSOA and ward, per NDVI tile.

on-road = green ∩ union of road buffers (04); accessible off-road = (green − buffers) ∩ public
parks/gardens; inaccessible = total (03) − on-road − accessible.  Shares divide by road-strip
area; the weighted versions weight each road piece by `choice_log` (02 workers).
Inputs : 02 buffers + per-segment CSV, 01 totals, NDVI tiles, data/raw_data/ward_public_green/
Outputs: data/greenery_metrics/SAT_greenery_{on_off_road,off_road_accessibility}_{lsoa,ward}.csv
Run    : from 03_SAT_greenery_on_off_road.ipynb (joblib workers import this module)
"""
from __future__ import annotations

import importlib
import os
from functools import cache
from pathlib import Path

import geopandas as gpd
import pandas as pd

_seg = importlib.import_module("02_SAT_greenery_per_segment_workers")
_tot = importlib.import_module("01_SAT_greenery_total_workers")

CRS = _seg.CRS
CONFIGS = {"lsoa": _tot.LSOA_CONFIG, "ward": _tot.WARD_CONFIG}
ACCESSIBLE_MASK_PATH = _seg.REPO_ROOT / "data/raw_data/ward_public_green/ward_public_green.shp"
_GM = _seg.REPO_ROOT / "data" / "greenery_metrics"
TILES_OUT_DIR = _GM / "SAT_greenery_on_off_road_tiles"
ON_OFF_CSV = {label: _GM / f"SAT_greenery_on_off_road_{label}.csv" for label in CONFIGS}
ACCESS_CSV = {label: _GM / f"SAT_greenery_off_road_accessibility_{label}.csv" for label in CONFIGS}
KINDS = ["on_road_m2", "accessible_m2"]


@cache
def _areas(label: str) -> gpd.GeoDataFrame:
    cfg = CONFIGS[label]
    return _tot.load_areas(cfg)[[cfg.area_id_col, "geometry"]].to_crs(CRS)


@cache
def _accessible_mask() -> gpd.GeoDataFrame:
    """Park/garden polygons dissolved into non-overlapping parts, so overlaps count once."""
    mask = gpd.read_file(ACCESSIBLE_MASK_PATH, columns=[]).to_crs(CRS)
    return gpd.GeoDataFrame(geometry=[mask.make_valid().union_all()], crs=CRS).explode(ignore_index=True)


def tile_out(tile: Path, label: str) -> Path:
    return TILES_OUT_DIR / f"{tile.stem}_{label}.csv"


def tile_done(tile: Path) -> bool:
    return all(tile_out(tile, label).exists() for label in CONFIGS)


def _green_parts(tile: Path) -> gpd.GeoDataFrame:
    """On-road and accessible off-road green polygons in this tile, tagged by `kind`."""
    green = gpd.read_file(tile)
    green = green.loc[green["cover_type"] == "green", ["geometry"]].to_crs(CRS)
    if green.empty:
        return gpd.GeoDataFrame({"kind": []}, geometry=[], crs=CRS)
    green["geometry"] = green.make_valid(method="structure", keep_collapsed=False)
    bbox = tuple(green.total_bounds)
    roads = gpd.read_file(_seg.BUFFERS_PATH, bbox=bbox, columns=[])
    if roads.empty:
        on_road, off_road = green.iloc[:0], green
    else:
        roads = gpd.GeoDataFrame(geometry=[roads.union_all()], crs=CRS)
        on_road = gpd.overlay(green, roads, how="intersection", keep_geom_type=True)
        off_road = gpd.overlay(green, roads, how="difference", keep_geom_type=True)
    mask = _accessible_mask().cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    accessible = off_road.iloc[:0]
    if not mask.empty and not off_road.empty:
        accessible = gpd.overlay(off_road, mask, how="intersection", keep_geom_type=True)
    return pd.concat([on_road.assign(kind="on_road_m2"), accessible.assign(kind="accessible_m2")], ignore_index=True)


def process_tile(tile: Path) -> None:
    """Green m² per kind per LSOA and per ward inside this tile; the CSVs double as checkpoints."""
    if tile_done(tile):
        return
    parts = _green_parts(tile)
    TILES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, cfg in CONFIGS.items():
        res = pd.DataFrame(columns=[cfg.area_id_col, *KINDS])
        inter = gpd.overlay(parts, _areas(label), how="intersection", keep_geom_type=True) if not parts.empty else parts
        if not inter.empty:
            res = (
                inter.assign(m2=inter.area)
                .pivot_table(index=cfg.area_id_col, columns="kind", values="m2", aggfunc="sum")
                .reindex(columns=KINDS, fill_value=0.0)
                .fillna(0.0)
                .reset_index()
            )
        out = tile_out(tile, label)
        tmp = out.with_suffix(".tmp")
        res.to_csv(tmp, index=False)
        os.replace(tmp, out)


@cache
def _road_union() -> gpd.GeoDataFrame:
    roads = gpd.read_file(_seg.BUFFERS_PATH, columns=[])
    return gpd.GeoDataFrame(geometry=[roads.union_all()], crs=CRS).explode(ignore_index=True)


def road_strip_m2(label: str) -> pd.Series:
    """Area of the union of all road buffers inside each LSOA / ward (cached in TILES_OUT_DIR)."""
    cfg = CONFIGS[label]
    path = TILES_OUT_DIR / f"road_strip_{label}.csv"
    if not path.exists():
        inter = gpd.overlay(_road_union(), _areas(label), how="intersection", keep_geom_type=True)
        TILES_OUT_DIR.mkdir(parents=True, exist_ok=True)
        inter.assign(road_strip_m2=inter.area).groupby(cfg.area_id_col)["road_strip_m2"].sum().to_csv(path)
    return pd.read_csv(path, index_col=cfg.area_id_col)["road_strip_m2"]


def osmid_strip_m2(label: str) -> pd.DataFrame:
    """Area of each road's (osmid's) buffer inside each LSOA / ward (cached in TILES_OUT_DIR)."""
    area_id = CONFIGS[label].area_id_col
    path = TILES_OUT_DIR / f"osmid_strip_{label}.csv"
    if not path.exists():
        buffers = gpd.read_file(_seg.BUFFERS_PATH, columns=["osmid"])
        inter = gpd.overlay(buffers, _areas(label), how="intersection", keep_geom_type=True)
        TILES_OUT_DIR.mkdir(parents=True, exist_ok=True)
        inter.assign(strip_m2=inter.area).groupby(["osmid", area_id], as_index=False)["strip_m2"].sum().to_csv(path, index=False)
    return pd.read_csv(path, dtype={"osmid": str})


@cache
def _osmid_choice() -> pd.Series:
    return _seg.osmid_choice()


def on_road_weighted(label: str) -> pd.Series:
    """
    Pooled choice-weighted share of the road strip that is green, per LSOA / ward.
    A road crossing a boundary is split by its strip area on each side, with its green
    share (sat_green_pct from 04) assumed even along it.
    """
    pieces = osmid_strip_m2(label)
    green_pct = pd.read_csv(_seg.OUT_CSV, dtype={"osmid": str}).set_index("osmid")["sat_green_pct"]
    strip_w = pieces["strip_m2"] * pieces["osmid"].map(_osmid_choice()).fillna(0.0)
    green_w = strip_w * pieces["osmid"].map(green_pct).fillna(0.0)
    g = pd.DataFrame({"green_w": green_w, "strip_w": strip_w}).groupby(pieces[CONFIGS[label].area_id_col]).sum()
    return g["green_w"] / g["strip_w"].where(g["strip_w"] > 0)


def merge_tiles(label: str) -> pd.DataFrame:
    """Area fractions of on-road, accessible / inaccessible off-road and total green for every area in 03."""
    cfg = CONFIGS[label]
    area_id, area = cfg.area_id_col, cfg.area_area_col
    tiles = _seg.tile_paths()
    missing = [t.name for t in tiles if not tile_out(t, label).exists()]
    if missing:
        raise FileNotFoundError(f"[{label}] {len(missing)} tile(s) not processed, e.g. {missing[:5]}")
    frames = [pd.read_csv(tile_out(t, label)) for t in tiles]
    m2 = pd.concat([f for f in frames if not f.empty]).groupby(area_id)[KINDS].sum()

    out = pd.read_csv(cfg.out_csv)
    out["total_green_area"] = out["total_green_area"] / out[area]
    on_road_m2 = out[area_id].map(m2["on_road_m2"]).fillna(0.0)
    strip_m2 = out[area_id].map(road_strip_m2(label))
    out["on_road_green_area"] = on_road_m2 / out[area]
    out["on_road_green_share"] = on_road_m2 / strip_m2
    out["on_road_green_share_weighted"] = out[area_id].map(on_road_weighted(label))
    # area = share × road-strip density, so weighting the share alone gives the area counterpart.
    out["on_road_green_area_weighted"] = out["on_road_green_share_weighted"] * strip_m2 / out[area]
    out["accessible_off_road_green"] = out[area_id].map(m2["accessible_m2"]).fillna(0.0) / out[area]
    out["off_road_green_area"] = out["total_green_area"] - out["on_road_green_area"]
    out["inaccessible_off_road_green"] = out["off_road_green_area"] - out["accessible_off_road_green"]
    # 03 measures in EPSG:32630 and 04/05 in EPSG:27700; areas differ by ~0.06%.
    for col in ("off_road_green_area", "inaccessible_off_road_green"):
        assert out[col].min() > -1e-3, (col, out[col].min())
        out[col] = out[col].clip(lower=0.0)
    return out


def write_csvs(df: pd.DataFrame, label: str) -> list[Path]:
    area_id, area = CONFIGS[label].area_id_col, CONFIGS[label].area_area_col
    tables = {
        ON_OFF_CSV[label]: [
            area_id, area, "on_road_green_area", "on_road_green_area_weighted",
            "on_road_green_share", "on_road_green_share_weighted", "off_road_green_area", "total_green_area",
        ],
        ACCESS_CSV[label]: [area_id, "accessible_off_road_green", "inaccessible_off_road_green"],
    }
    for path, cols in tables.items():
        tmp = path.with_suffix(".tmp")
        df[cols].to_csv(tmp, index=False)
        os.replace(tmp, path)
    return list(tables)
