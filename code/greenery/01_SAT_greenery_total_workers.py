"""Total satellite (NDVI) green area per LSOA or ward (Methods, eq. 3), per NDVI tile.

Inputs : NDVI tiles (data/raw_data/green_cover_tiles/), LSOA / ward polygons
Outputs: data/greenery_metrics/SAT_greenery_total_{lsoa,ward}.csv (+ one cached CSV per tile,
         written atomically: a tile is done iff its CSV exists; delete the folder to recompute)
Run    : from the 01_SAT_greenery_total_{lsoa,ward} notebooks (joblib workers, 03 and 04 import this module)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
NDVI_TILES_DIR = REPO_ROOT / "data" / "raw_data" / "green_cover_tiles"
LONDON_WARDS_PATH = (
    REPO_ROOT / "data/raw_data/geo_wards/London_Ward.shp"
)
# UTM 30N, unlike 27700 in steps 02-04: areas differ by about 0.06 %, below the NDVI map's own accuracy.
WORK_CRS = 32630
N_BUCKETS = 30


@dataclass(frozen=True)
class TotalAreaConfig:
    label: str
    area_id_col: str
    area_area_col: str
    polygons_path: Path
    polygon_id_field: str
    chunks_dir: Path
    out_csv: Path
    # NDVI tiles cover Greater London only; areas outside it would get a spurious 0.
    london_only: bool = False
    # Hectares of water (tidal Thames) inside each polygon, excluded from the area denominator.
    # LSOA BGC boundaries are already clipped to the riverbank, so only wards need this.
    non_land_ha_field: str | None = None


LSOA_CONFIG = TotalAreaConfig(
    label="lsoa",
    area_id_col="lsoa_code",
    area_area_col="lsoa_area",
    polygons_path=REPO_ROOT / "data/raw_data/geo_LSOAs/LSOA_(Dec_2011)_Boundaries_Generalised_Clipped_(BGC)_EW_V3.shp",
    polygon_id_field="LSOA11CD",
    chunks_dir=REPO_ROOT / "data/greenery_metrics/SAT_greenery_total_lsoa_chunks",
    out_csv=REPO_ROOT / "data/greenery_metrics/SAT_greenery_total_lsoa.csv",
    london_only=True,
)

WARD_CONFIG = TotalAreaConfig(
    label="ward",
    area_id_col="ward_code",
    area_area_col="ward_area",
    polygons_path=LONDON_WARDS_PATH,
    polygon_id_field="GSS_CODE",
    chunks_dir=REPO_ROOT / "data/greenery_metrics/SAT_greenery_total_ward_chunks",
    out_csv=REPO_ROOT / "data/greenery_metrics/SAT_greenery_total_ward.csv",
    non_land_ha_field="NONLD_AREA",
)


def load_areas(config: TotalAreaConfig) -> gpd.GeoDataFrame:
    """Load polygon units in metric CRS with (land) area column."""
    areas = gpd.read_file(config.polygons_path)
    if config.london_only:
        london = gpd.read_file(LONDON_WARDS_PATH).to_crs(areas.crs).union_all()
        areas = areas[areas.representative_point().within(london)]
    land_frac = 1.0
    if config.non_land_ha_field:
        if not areas.crs.is_projected:
            raise ValueError(f"{config.polygons_path} must be in a metric CRS to use {config.non_land_ha_field}")
        land_frac = (1 - areas[config.non_land_ha_field] * 1e4 / areas.geometry.area).to_numpy()
    areas = areas.to_crs(epsg=WORK_CRS)
    areas = areas[[config.polygon_id_field, "geometry"]].rename(
        columns={config.polygon_id_field: config.area_id_col}
    )
    areas[config.area_area_col] = areas.geometry.area * land_frac
    return areas


_TILE_PREFIX = "ndvi_landcover_merge_"


def tile_id_from_path(tile_path: Path) -> str | None:
    """Numeric id for ndvi_landcover_merge_<digits>.shp, or None if the name does not match."""
    if tile_path.suffix != ".shp" or not tile_path.stem.startswith(_TILE_PREFIX):
        return None
    tile_id = tile_path.stem[len(_TILE_PREFIX):]
    return tile_id if tile_id.isdigit() else None


def tile_csv_path(chunks_dir: Path, tile_id: str) -> Path:
    return chunks_dir / f"{tile_id}.csv"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def list_tiles_for_bucket(
    thread_id: int,
    input_folder: Path | str = NDVI_TILES_DIR,
) -> list[tuple[str, str, Path]]:
    """Return (basename, tile_id, path) for tiles in this bucket."""
    input_folder = Path(input_folder)
    tiles: list[tuple[str, str, Path]] = []
    for file_name in sorted(os.listdir(input_folder)):
        tile_path = input_folder / file_name
        tile_id = tile_id_from_path(tile_path)
        if tile_id is None or (int(tile_id) % N_BUCKETS) != thread_id:
            continue
        tiles.append((file_name, tile_id, tile_path))
    return tiles


def list_all_tile_ids(input_folder: Path | str = NDVI_TILES_DIR) -> set[str]:
    input_folder = Path(input_folder)
    ids: set[str] = set()
    for file_name in os.listdir(input_folder):
        tile_id = tile_id_from_path(input_folder / file_name)
        if tile_id is not None:
            ids.add(tile_id)
    return ids


def overlay_one_tile(
    tile_path: Path,
    areas: gpd.GeoDataFrame,
    area_id_col: str,
) -> pd.DataFrame:
    """Intersect green NDVI with area polygons; sum green m² per area unit."""
    tile = gpd.read_file(tile_path)
    tile = tile.loc[tile["cover_type"] == "green", ["geometry"]]
    if tile.empty:
        return pd.DataFrame(columns=[area_id_col, "green_area"])

    tile = tile.to_crs(epsg=WORK_CRS)
    minx, miny, maxx, maxy = tile.total_bounds
    areas_sub = areas.cx[minx:maxx, miny:maxy]
    if areas_sub.empty:
        return pd.DataFrame(columns=[area_id_col, "green_area"])

    inter = gpd.overlay(
        tile,
        areas_sub[[area_id_col, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    inter["green_area"] = inter.geometry.area
    return inter.groupby(area_id_col, as_index=False)["green_area"].sum()


def pending_tiles(config: TotalAreaConfig, thread_id: int) -> list[tuple[str, str, Path]]:
    return [
        tile for tile in list_tiles_for_bucket(thread_id)
        if not tile_csv_path(config.chunks_dir, tile[1]).exists()
    ]


def buckets_to_run(config: TotalAreaConfig, force: bool = False) -> list[int]:
    return [i for i in range(N_BUCKETS) if force or pending_tiles(config, i)]


def print_status(config: TotalAreaConfig) -> None:
    tile_ids = list_all_tile_ids()
    n_done = sum(tile_csv_path(config.chunks_dir, t).exists() for t in tile_ids)
    print(f"[{config.label}] {n_done}/{len(tile_ids)} tiles done under {config.chunks_dir}")


def run_bucket(thread_id: int, config: TotalAreaConfig, force: bool = False) -> None:
    """Joblib entry point: one bucket of tiles for LSOA or ward."""
    tiles = list_tiles_for_bucket(thread_id) if force else pending_tiles(config, thread_id)
    if not tiles:
        return
    areas = load_areas(config)
    for i, (basename, tile_id, tile_path) in enumerate(tiles, 1):
        print(f"[{config.label} bucket {thread_id}] {basename} ({i}/{len(tiles)})", flush=True)
        df = overlay_one_tile(tile_path, areas, config.area_id_col)
        _atomic_write_csv(df, tile_csv_path(config.chunks_dir, tile_id))


def finalize_total(config: TotalAreaConfig) -> pd.DataFrame:
    """Concatenate tile chunks and build final area-level CSV."""
    paths = [tile_csv_path(config.chunks_dir, t) for t in sorted(list_all_tile_ids(), key=int)]
    missing = [p.stem for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[{config.label}] {len(missing)}/{len(paths)} tile chunk(s) missing, "
            f"e.g. {missing[:5]}. Run the parallel step first."
        )

    parts = [df for df in map(pd.read_csv, paths) if not df.empty]
    totals = (
        pd.concat(parts, ignore_index=True)
        .groupby(config.area_id_col, as_index=False)["green_area"]
        .sum()
        .rename(columns={"green_area": "total_green_area"})
    )

    areas = load_areas(config)
    out = areas[[config.area_id_col, config.area_area_col]].merge(
        totals, on=config.area_id_col, how="left"
    )
    out["total_green_area"] = out["total_green_area"].fillna(0.0)
    out["percent_green"] = out["total_green_area"] / out[config.area_area_col]
    return out


def write_total_csv(df: pd.DataFrame, config: TotalAreaConfig) -> Path:
    _atomic_write_csv(df, config.out_csv)
    return config.out_csv
