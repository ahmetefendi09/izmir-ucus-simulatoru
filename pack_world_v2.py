#!/usr/bin/env python3
"""
pack_world_v2.py — Complete Overture Maps + OSM + 30m Terrarium DEM Binary World Packer (v2).
Produces izmir_osm_world.bin ("IZMIROSM" v2) for izmir_flight_sim.cpp containing:
  - 1024x1024 bilinearly sampled 30m Terrarium DEM
  - 2048x2048 exact Overture Land / Landcover / Landuse / Water / Port / Salt-Pan GIS Grid
  - Mapped real trees (4,341 Overture trees)
  - Active operating rail tracks (İZBAN/TCDD, Konak/Karşıyaka/Çiğli Tram, surface Metro; bridges flagged, tunnels omitted)
  - Operating road network (Motorways, Trunks, Primary, Secondary, Tertiary, Residential + TomTom missing streets, bridges flagged, tunnels omitted)
  - ~421,000 real Overture building footprints (exact oriented OBBs, heights, functional categories & Aegean palette)
"""

import glob
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import orjson
import pyarrow.parquet as pq
import shapely
import shapely.wkb
from PIL import Image, ImageDraw
from shapely.geometry import Point
from shapely.ops import substring
from shapely.strtree import STRtree

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "osm_cache"
OVERTURE_DIR = CACHE_DIR / "overture"
DEM_DIR = CACHE_DIR / "dem_tiles"
BIN_PATH = BASE_DIR / "izmir_osm_world.bin"

# Exact coordinate constants matching izmir_flight_sim.cpp
REF_LAT = 38.418850
REF_LON = 27.128680
LON_SCALE = 87215.0
LAT_SCALE = 111132.0
WORLD_X0, WORLD_X1 = -17500.0, 17500.0
WORLD_Z0, WORLD_Z1 = -26000.0, 19000.0
DEM_N = 1024
MASK_N = 2048

LON_MIN = REF_LON + WORLD_X0 / LON_SCALE
LON_MAX = REF_LON + WORLD_X1 / LON_SCALE
LAT_MIN = REF_LAT - WORLD_Z1 / LAT_SCALE
LAT_MAX = REF_LAT - WORLD_Z0 / LAT_SCALE

# GIS Surface Land Class IDs (shared with izmir_flight_sim.cpp)
LC_SEA = 0
LC_SALT_WETLAND = 1
LC_WATER = 2
LC_URBAN = 3
LC_INDUSTRIAL = 4
LC_PARK = 5
LC_FOREST = 6
LC_SHRUB = 7
LC_CROP = 8
LC_BARREN = 9

# Building categories (b_kind)
BK_APT = 0
BK_TOWER = 1
BK_INDUSTRIAL = 2
BK_EDUCATION = 3
BK_MOSQUE = 4
BK_MEDICAL = 5
BK_MALL = 6
BK_HILLSIDE = 7
BK_GREENHOUSE = 8


def lon_to_x(lon: float) -> float:
    return (lon - REF_LON) * LON_SCALE


def lat_to_z(lat: float) -> float:
    return -(lat - REF_LAT) * LAT_SCALE


def split_segment_by_flags(geom, flags_list, level_rules):
    if not flags_list and not level_rules:
        return [(geom, set(), 0)]

    pts_set = {0.0, 1.0}
    if flags_list:
        for item in flags_list:
            btw = item.get("between")
            if btw and len(btw) == 2:
                pts_set.add(max(0.0, min(1.0, float(btw[0]))))
                pts_set.add(max(0.0, min(1.0, float(btw[1]))))
    if level_rules:
        for item in level_rules:
            btw = item.get("between")
            if btw and len(btw) == 2:
                pts_set.add(max(0.0, min(1.0, float(btw[0]))))
                pts_set.add(max(0.0, min(1.0, float(btw[1]))))

    cuts = sorted(pts_set)
    if len(cuts) == 2:
        spans = [(0.0, 1.0, geom)]
    else:
        spans = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 1e-6:
                continue
            sub = substring(geom, a, b, normalized=True)
            if not sub.is_empty and sub.geom_type == "LineString" and len(sub.coords) >= 2:
                spans.append((a, b, sub))

    out = []
    for a, b, sub in spans:
        mid = 0.5 * (a + b)
        fset = set()
        if flags_list:
            for item in flags_list:
                btw = item.get("between")
                if btw is None or (float(btw[0]) - 1e-7 <= mid <= float(btw[1]) + 1e-7):
                    fset.update(item.get("values") or [])
        lvl = 0
        if level_rules:
            for item in level_rules:
                btw = item.get("between")
                if btw is None or (float(btw[0]) - 1e-7 <= mid <= float(btw[1]) + 1e-7):
                    v = item.get("value")
                    if v is not None:
                        lvl = int(v)
        if "is_bridge" in fset and lvl <= 0:
            lvl = 1
        if "is_tunnel" in fset and lvl >= 0:
            lvl = -1
        out.append((sub, fset, lvl))
    return out


def build_dem_grid() -> np.ndarray:
    t0 = time.time()
    print("[1/6] Vectorized 1024x1024 30m Terrarium DEM sampling...", flush=True)
    xs = np.linspace(WORLD_X0, WORLD_X1, DEM_N, dtype=np.float64)
    zs = np.linspace(WORLD_Z0, WORLD_Z1, DEM_N, dtype=np.float64)
    lons = REF_LON + xs / LON_SCALE
    lats = REF_LAT - zs / LAT_SCALE

    n_zoom = 1 << 12
    x_tile_f = (lons + 180.0) / 360.0 * n_zoom
    lat_rads = np.radians(lats)
    y_tile_f = (1.0 - np.arcsinh(np.tan(lat_rads)) / math.pi) / 2.0 * n_zoom

    txs = np.floor(x_tile_f).astype(np.int32)
    tys = np.floor(y_tile_f).astype(np.int32)

    tile_elevs = {}
    for ty in np.unique(tys):
        for tx in np.unique(txs):
            p = DEM_DIR / f"z12_{tx}_{ty}.png"
            if p.exists():
                arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                tile_elevs[(int(tx), int(ty))] = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0

    # Assemble mosaic covering all needed tiles for clean cross-tile bilinear interpolation
    tx_min, tx_max = int(txs.min()), int(txs.max())
    ty_min, ty_max = int(tys.min()), int(tys.max())
    mw = (tx_max - tx_min + 1) * 256
    mh = (ty_max - ty_min + 1) * 256
    mosaic = np.full((mh, mw), 2.0, dtype=np.float32)
    for (tx, ty), elev in tile_elevs.items():
        ox = (tx - tx_min) * 256
        oy = (ty - ty_min) * 256
        mosaic[oy:oy + 256, ox:ox + 256] = elev

    gx = np.clip((x_tile_f - tx_min) * 256.0 - 0.5, 0.0, mw - 1.001)
    gy = np.clip((y_tile_f - ty_min) * 256.0 - 0.5, 0.0, mh - 1.001)
    gx_grid, gy_grid = np.meshgrid(gx, gy)

    x0 = np.floor(gx_grid).astype(np.int32)
    y0 = np.floor(gy_grid).astype(np.int32)
    fx = (gx_grid - x0).astype(np.float32)
    fy = (gy_grid - y0).astype(np.float32)

    e00 = mosaic[y0, x0]
    e10 = mosaic[y0, x0 + 1]
    e01 = mosaic[y0 + 1, x0]
    e11 = mosaic[y0 + 1, x0 + 1]
    dem = (e00 * (1.0 - fx) + e10 * fx) * (1.0 - fy) + (e01 * (1.0 - fx) + e11 * fx) * fy
    print(f"  DEM ready in {time.time() - t0:.2f}s (min={dem.min():.1f}m, max={dem.max():.1f}m)", flush=True)
    return dem.astype(np.float32)


def coords_to_mask_px(coords_seq):
    sx = (MASK_N - 1) / (WORLD_X1 - WORLD_X0)
    sz = (MASK_N - 1) / (WORLD_Z1 - WORLD_Z0)
    return [
        ((lon - REF_LON) * LON_SCALE - WORLD_X0) * sx
        for lon, lat in coords_seq
    ], [
        (-(lat - REF_LAT) * LAT_SCALE - WORLD_Z0) * sz
        for lon, lat in coords_seq
    ]


def draw_poly_on_arr(arr: np.ndarray, geom, fill_val: int):
    sx = (MASK_N - 1) / (WORLD_X1 - WORLD_X0)
    sz = (MASK_N - 1) / (WORLD_Z1 - WORLD_Z0)
    ox_off = -WORLD_X0 * sx
    oz_off = -WORLD_Z0 * sz
    xs_scale = LON_SCALE * sx
    zs_scale = -LAT_SCALE * sz

    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        if poly.is_empty:
            continue
        ext = list(poly.exterior.coords)
        if len(ext) < 3:
            continue
        pts = [((lon - REF_LON) * xs_scale + ox_off, (lat - REF_LAT) * zs_scale + oz_off) for lon, lat in ext]
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        bx0 = max(0, int(math.floor(min(xs))) - 1)
        bx1 = min(MASK_N, int(math.ceil(max(xs))) + 2)
        bz0 = max(0, int(math.floor(min(zs))) - 1)
        bz1 = min(MASK_N, int(math.ceil(max(zs))) + 2)
        if bx1 <= bx0 or bz1 <= bz0:
            continue

        if not poly.interiors and (bx1 - bx0) * (bz1 - bz0) > 65536:
            # Fast global PIL draw for simple hole-free polygons? Still crisp via local crop!
            pass
        bw, bh = bx1 - bx0, bz1 - bz0
        loc = Image.new("L", (bw, bh), 0)
        ld = ImageDraw.Draw(loc)
        ld.polygon([(x - bx0, z - bz0) for x, z in pts], fill=255)
        for interior in poly.interiors:
            ipts = [((lon - REF_LON) * xs_scale + ox_off - bx0, (lat - REF_LAT) * zs_scale + oz_off - bz0) for lon, lat in interior.coords]
            if len(ipts) >= 3:
                ld.polygon(ipts, fill=0)
        m = np.asarray(loc, dtype=np.uint8) == 255
        arr[bz0:bz1, bx0:bx1][m] = fill_val


def build_gis_grid_and_trees():
    t0 = time.time()
    print("[2/6] Rasterizing 2048x2048 Overture GIS Surface Class Grid & real trees...", flush=True)
    arr = np.full((MASK_N, MASK_N), LC_SEA, dtype=np.uint8)
    land_mask = np.zeros((MASK_N, MASK_N), dtype=np.uint8)
    ocean_mask = np.zeros((MASK_N, MASK_N), dtype=np.uint8)

    # 2A. Land polygons & mapped trees
    land_tbl = pq.read_table(OVERTURE_DIR / "land.parquet", columns=["subtype", "geometry"])
    trees = []
    for st, gb in zip(land_tbl["subtype"].to_pylist(), land_tbl["geometry"].to_pylist()):
        g = shapely.wkb.loads(gb)
        if st == "land" and g.geom_type in ("Polygon", "MultiPolygon"):
            draw_poly_on_arr(land_mask, g, 1)
        elif st == "tree" and g.geom_type == "Point":
            x, z = lon_to_x(g.x), lat_to_z(g.y)
            if WORLD_X0 + 50 <= x <= WORLD_X1 - 50 and WORLD_Z0 + 50 <= z <= WORLD_Z1 - 50:
                trees.append((x, z))
    arr[land_mask == 1] = LC_SHRUB

    # 2B. Land cover (filter min_zoom >= 8 to eliminate coarse low-zoom tile seams, sort by sort_key)
    lc_tbl = pq.read_table(OVERTURE_DIR / "land_cover.parquet", columns=["subtype", "cartography", "geometry"])
    lc_rows = []
    for st, cart, gb in zip(lc_tbl["subtype"].to_pylist(), lc_tbl["cartography"].to_pylist(), lc_tbl["geometry"].to_pylist()):
        mz = int((cart or {}).get("min_zoom") or 0)
        if mz < 8:
            continue
        sk = int((cart or {}).get("sort_key") or 0)
        lc_rows.append((sk, st, shapely.wkb.loads(gb)))
    lc_rows.sort(key=lambda r: r[0])
    lc_map = {
        "urban": LC_URBAN,
        "forest": LC_FOREST,
        "shrub": LC_SHRUB,
        "grass": LC_PARK,
        "crop": LC_CROP,
        "barren": LC_BARREN,
        "wetland": LC_SALT_WETLAND,
    }
    for _, st, g in lc_rows:
        if g.geom_type in ("Polygon", "MultiPolygon"):
            draw_poly_on_arr(arr, g, lc_map.get(st, LC_SHRUB))

    # 2C. Land use (skip broad 'protected'/'nature_reserve' bounding polys that cover sea/wetlands)
    lu_tbl = pq.read_table(OVERTURE_DIR / "land_use.parquet", columns=["subtype", "class", "geometry"])
    for st, cl, gb in zip(lu_tbl["subtype"].to_pylist(), lu_tbl["class"].to_pylist(), lu_tbl["geometry"].to_pylist()):
        g = shapely.wkb.loads(gb)
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        s = (st or "").lower()
        c = (cl or "").lower()
        if s == "protected" or c == "nature_reserve":
            continue
        if s in ("agriculture", "farmland") or c in ("farmland", "orchard", "meadow", "horticulture", "vineyard", "olive_grove"):
            val = LC_CROP
        elif s in ("forest",) or c in ("forest", "wood"):
            val = LC_FOREST
        elif s in ("park", "recreation", "cemetery", "horticulture") or c in ("park", "recreation_ground", "stadium", "golf_course", "pitch", "cemetery", "garden", "village_green"):
            val = LC_PARK
        elif s in ("industrial", "resource_extraction", "transportation", "military", "landfill") or c in ("industrial", "quarry", "port", "logistics", "aerodrome", "airport", "military", "railway", "marina"):
            val = LC_INDUSTRIAL
        elif s in ("residential", "commercial", "civic", "institutional", "education", "medical", "religious", "developed"):
            val = LC_URBAN
        else:
            continue
        draw_poly_on_arr(arr, g, val)

    # 2D. Water: Exact Gulf Ocean cut + Salt ponds + inland reservoirs/lakes/rivers
    sx = (MASK_N - 1) / (WORLD_X1 - WORLD_X0)
    sz = (MASK_N - 1) / (WORLD_Z1 - WORLD_Z0)
    w_tbl = pq.read_table(OVERTURE_DIR / "water.parquet", columns=["subtype", "class", "geometry"])
    inland_polys = []
    water_lines = []
    for st, cl, gb in zip(w_tbl["subtype"].to_pylist(), w_tbl["class"].to_pylist(), w_tbl["geometry"].to_pylist()):
        g = shapely.wkb.loads(gb)
        s = (st or "").lower()
        c = (cl or "").lower()
        if s == "ocean":
            draw_poly_on_arr(ocean_mask, g, 1)
            continue
        if s == "physical" or c == "sea" or c == "bay":
            continue
        if g.geom_type in ("Polygon", "MultiPolygon"):
            if g.area > 0.005:
                continue
            inland_polys.append((s, c, g))
        elif g.geom_type in ("LineString", "MultiLineString"):
            water_lines.append((c, g))

    # Enforce strict Overture sea/ocean cutout before inland water & piers
    arr[(land_mask == 0) | (ocean_mask == 1)] = LC_SEA

    for s, c, g in inland_polys:
        if c in ("salt_pond",) or s in ("wetland", "marsh"):
            draw_poly_on_arr(arr, g, LC_SALT_WETLAND)
        elif s in ("lake", "pond", "reservoir", "river", "stream", "canal", "water", "spring") or c in ("reservoir", "lake", "pond", "river", "stream", "canal", "basin", "wastewater"):
            draw_poly_on_arr(arr, g, LC_WATER)

    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)
    for c, g in water_lines:
        lines = [g] if g.geom_type == "LineString" else list(g.geoms)
        wpx = 2 if c in ("river", "canal") else 1
        for ln in lines:
            pts = [((lon_to_x(lon) - WORLD_X0) * sx, (lat_to_z(lat) - WORLD_Z0) * sz) for lon, lat in ln.coords]
            if len(pts) >= 2:
                draw.line(pts, fill=LC_WATER, width=wpx)

    # 2E. Infrastructure piers & airport surfaces
    inf_tbl = pq.read_table(OVERTURE_DIR / "infrastructure.parquet", columns=["subtype", "class", "geometry"])
    for st, cl, gb in zip(inf_tbl["subtype"].to_pylist(), inf_tbl["class"].to_pylist(), inf_tbl["geometry"].to_pylist()):
        if st not in ("pier", "airport") and cl not in ("pier", "runway", "taxiway", "apron", "terminal"):
            continue
        g = shapely.wkb.loads(gb)
        if g.geom_type in ("Polygon", "MultiPolygon"):
            arr = np.asarray(img, dtype=np.uint8).copy()
            draw_poly_on_arr(arr, g, LC_INDUSTRIAL)
            img = Image.fromarray(arr, mode="L")
            draw = ImageDraw.Draw(img)
        elif g.geom_type == "LineString" and st == "pier":
            pts = [((lon_to_x(lon) - WORLD_X0) * sx, (lat_to_z(lat) - WORLD_Z0) * sz) for lon, lat in g.coords]
            if len(pts) >= 2:
                draw.line(pts, fill=LC_INDUSTRIAL, width=2)

    grid = np.asarray(img, dtype=np.uint8).copy()
    print(f"  GIS Grid 2048x2048 ready in {time.time() - t0:.2f}s (Trees={len(trees)}, sea%={100.0*(grid==0).mean():.1f}%)", flush=True)
    return grid, trees


def load_roads_and_rails():
    t0 = time.time()
    print("[3/6] Processing Overture operating rails & road network (bridges/TomTom dedup)...", flush=True)
    seg_tbl = pq.read_table(
        OVERTURE_DIR / "segment.parquet",
        columns=["subtype", "class", "road_flags", "rail_flags", "level_rules", "sources", "geometry"],
    )
    d = seg_tbl.to_pydict()
    n = len(d["subtype"])

    rails = []  # (rtype: int, is_bridge: int, pts: list[(x,z)])
    osm_road_geoms = []
    osm_spans_raw = []
    tt_roads_raw = []

    road_spec = {
        "motorway": (0, 10.5),
        "trunk": (0, 9.0),
        "primary": (1, 7.0),
        "secondary": (2, 5.8),
        "tertiary": (3, 4.5),
        "unclassified": (4, 3.6),
        "residential": (4, 3.3),
        "living_street": (4, 2.9),
        "pedestrian": (4, 3.2),
        "service": (4, 2.4),
        "track": (4, 2.6),
        "unknown": (4, 3.0),
    }

    world_clip_box = shapely.geometry.box(
        WORLD_X0 + 12.0, WORLD_Z0 + 12.0, WORLD_X1 - 12.0, WORLD_Z1 - 12.0
    )

    def clip_metric_lines(m_pts, tol):
        if len(m_pts) < 2:
            return []
        ln = shapely.LineString(m_pts)
        b0, b1, b2, b3 = ln.bounds
        if b2 < WORLD_X0 + 12.0 or b0 > WORLD_X1 - 12.0 or b3 < WORLD_Z0 + 12.0 or b1 > WORLD_Z1 - 12.0:
            return []
        if b0 < WORLD_X0 + 12.0 or b2 > WORLD_X1 - 12.0 or b1 < WORLD_Z0 + 12.0 or b3 > WORLD_Z1 - 12.0:
            geom = ln.intersection(world_clip_box)
        else:
            geom = ln
        if geom.is_empty:
            return []
        parts = [geom] if geom.geom_type == "LineString" else (
            [p for p in geom.geoms if p.geom_type == "LineString"]
            if hasattr(geom, "geoms") else []
        )
        out = []
        for p in parts:
            if p.length < 1.0:
                continue
            sp = p.simplify(tol, preserve_topology=False)
            coords = list(sp.coords)
            if len(coords) >= 2:
                out.append(coords)
        return out

    for i in range(n):
        st = d["subtype"][i]
        cl = d["class"][i] or "unknown"
        g = shapely.wkb.loads(d["geometry"][i])
        if g.is_empty or g.geom_type != "LineString":
            continue
        bx0, by0, bx1, by1 = g.bounds
        if bx1 < LON_MIN - 0.01 or bx0 > LON_MAX + 0.01 or by1 < LAT_MIN - 0.01 or by0 > LAT_MAX + 0.01:
            continue

        if st == "rail":
            if cl not in ("standard_gauge", "tram", "subway"):
                continue
            rtype = 0 if cl == "standard_gauge" else (1 if cl == "tram" else 2)
            for sub, fset, lvl in split_segment_by_flags(g, d["rail_flags"][i], d["level_rules"][i]):
                if fset & {"is_abandoned", "is_disused", "is_under_construction", "is_tunnel"}:
                    continue
                if lvl < 0:
                    continue
                m_pts = [(lon_to_x(x), lat_to_z(y)) for x, y in sub.coords]
                is_br = 1 if ("is_bridge" in fset or lvl > 0) else 0
                for coords in clip_metric_lines(m_pts, 0.5):
                    rails.append((rtype, is_br, coords))

        elif st == "road":
            src = d["sources"][i]
            ds = src[0].get("dataset") if src else "OpenStreetMap"
            spans = split_segment_by_flags(g, d["road_flags"][i], d["level_rules"][i])
            if ds != "OpenStreetMap" and cl == "unknown":
                tt_roads_raw.append((g, spans))
            else:
                if cl not in road_spec:
                    continue
                if ds == "OpenStreetMap":
                    osm_road_geoms.append(g)
                tier, hw = road_spec[cl]
                for sub, fset, lvl in spans:
                    osm_spans_raw.append((tier, hw, fset, lvl, sub))

    # Deduplicate TomTom unknown missing streets against OSM roads via buffer overlap ratio
    osm_tree = STRtree(osm_road_geoms) if osm_road_geoms else None
    osm_road_bufs = [og.buffer(0.000085) for og in osm_road_geoms] if osm_road_geoms else []
    tt_kept, tt_dropped = 0, 0
    roads = []  # (tier: int, is_bridge: int, level: int, hw_dm: int, pts: list[(x,z)])

    def emit_road(tier, hw, fset, lvl, sub_geom):
        if fset & {"is_abandoned", "is_disused", "is_under_construction"}:
            return
        m_pts = [(lon_to_x(x), lat_to_z(y)) for x, y in sub_geom.coords]
        if len(m_pts) < 2:
            return
        if "is_tunnel" in fset:
            seg_len = sum(
                math.hypot(m_pts[k + 1][0] - m_pts[k][0], m_pts[k + 1][1] - m_pts[k][1])
                for k in range(len(m_pts) - 1)
            )
            # Keep short (<85m) surface/overpass underpasses continuous; skip deep tunnels
            if tier == 0 or seg_len >= 85.0:
                return
            is_br = 0
            lvl_clamped = 0
        elif lvl < 0:
            # Open depressed surface underpass (not flagged as tunnel)
            is_br = 0
            lvl_clamped = 0
        else:
            is_br = 1 if ("is_bridge" in fset or lvl > 0) else 0
            lvl_clamped = max(-5, min(5, int(lvl)))
        hw_dm = max(15, min(255, int(round(hw * 10.0))))
        for coords in clip_metric_lines(m_pts, 0.6):
            roads.append((tier, is_br, lvl_clamped, hw_dm, coords))

    for tier, hw, fset, lvl, sub in osm_spans_raw:
        emit_road(tier, hw, fset, lvl, sub)

    for g, spans in tt_roads_raw:
        g_len = g.length
        if g_len <= 1e-8:
            tt_dropped += 1
            continue
        dup = False
        if osm_tree is not None:
            near_idx = osm_tree.query(g.buffer(0.00010))
            if len(near_idx) > 0:
                near_buf = shapely.unary_union([osm_road_bufs[ni] for ni in near_idx])
                if (g.intersection(near_buf).length / g_len) > 0.50:
                    dup = True
        if dup:
            tt_dropped += 1
            continue
        tt_kept += 1
        for sub, fset, lvl in spans:
            emit_road(4, 3.2, fset, lvl, sub)

    print(f"  Rails={len(rails)}, Roads={len(roads)} (TomTom kept={tt_kept}, dropped={tt_dropped}) in {time.time() - t0:.2f}s", flush=True)
    return rails, roads


def load_buildings():
    t0 = time.time()
    print("[4/6] Processing 421K+ Overture building footprints via vectorized OBBs...", flush=True)
    bt = pq.read_table(
        OVERTURE_DIR / "building.parquet",
        columns=["subtype", "class", "names", "height", "num_floors", "is_underground", "geometry"],
    )
    geoms_raw = bt["geometry"].to_pylist()
    subtypes = bt["subtype"].to_pylist()
    classes = bt["class"].to_pylist()
    names = bt["names"].to_pylist()
    heights = bt["height"].to_pylist()
    floors = bt["num_floors"].to_pylist()
    undergrounds = bt["is_underground"].to_pylist()

    geoms = shapely.from_wkb(geoms_raw)
    # Transform all polygons cleanly to metric (x, z)
    metric_geoms = shapely.transform(
        geoms,
        lambda pts: np.column_stack([(pts[:, 0] - REF_LON) * LON_SCALE, -(pts[:, 1] - REF_LAT) * LAT_SCALE]),
    )
    cents = shapely.centroid(metric_geoms)
    cxs = shapely.get_x(cents)
    czs = shapely.get_y(cents)

    valid_mask = (
        (cxs >= WORLD_X0 + 50.0) & (cxs <= WORLD_X1 - 50.0) &
        (czs >= WORLD_Z0 + 50.0) & (czs <= WORLD_Z1 - 50.0)
    )
    # Exclude underground structures
    for i, ug in enumerate(undergrounds):
        if ug:
            valid_mask[i] = False

    idxs = np.nonzero(valid_mask)[0]
    m_sub = metric_geoms[idxs]
    areas = shapely.area(m_sub)
    mrrs = shapely.minimum_rotated_rectangle(m_sub)

    # Enrich generic Overture footprints with OSM special POI buildings
    b_tree = STRtree(geoms[idxs])
    enriched_cat = {}  # pos in idxs -> (target_kind, osm_levels)
    osm_pts = []
    for p in sorted(glob.glob(str(CACHE_DIR / "izmir_bldgs_*.json"))):
        with open(p, "rb") as f:
            elems = orjson.loads(f.read()).get("elements", [])
        for e in elems:
            tags = e.get("tags", {})
            am = tags.get("amenity", "")
            sh = tags.get("shop", "")
            btag = tags.get("building", "")
            t_kind = None
            if am == "place_of_worship" or btag == "mosque":
                t_kind = BK_MOSQUE
            elif am in ("hospital", "clinic") or btag == "hospital":
                t_kind = BK_MEDICAL
            elif am in ("school", "university", "college", "kindergarten") or btag in ("school", "university"):
                t_kind = BK_EDUCATION
            elif sh in ("mall", "department_store") or btag == "mall":
                t_kind = BK_MALL
            lv = 0
            if "building:levels" in tags:
                try:
                    lv = int(float(tags["building:levels"]))
                except Exception:
                    lv = 0
            if t_kind is None and lv <= 0:
                continue
            gpts = e.get("geometry")
            if not gpts:
                continue
            lon_c = sum(pt["lon"] for pt in gpts) / len(gpts)
            lat_c = sum(pt["lat"] for pt in gpts) / len(gpts)
            osm_pts.append((Point(lon_c, lat_c), t_kind, lv))

    if osm_pts:
        sub_geoms = geoms[idxs]
        q_res = b_tree.query([op[0] for op in osm_pts], predicate="within")
        for q_i, g_i in zip(q_res[0], q_res[1]):
            _, t_kind, lv = osm_pts[q_i]
            prev_k, prev_lv = enriched_cat.get(int(g_i), (None, 0))
            enriched_cat[int(g_i)] = (t_kind if t_kind is not None else prev_k, max(prev_lv, lv))

    # Bayrakli / Konak centers for urban height distribution
    bay_x, bay_z = lon_to_x(27.176), lat_to_z(38.451)
    mav_x, mav_z = lon_to_x(27.074), lat_to_z(38.474)

    packed_records = bytearray()
    pack_b = struct.Struct("<ffffffBBBB").pack_into
    n_out = len(idxs)
    packed_records = bytearray(n_out * 28)
    write_pos = 0

    for k, src_i in enumerate(idxs):
        mrr = mrrs[k]
        if mrr.geom_type == "Polygon":
            coords = list(mrr.exterior.coords)
            if len(coords) >= 4:
                x0, z0 = coords[0][0], coords[0][1]
                x1, z1 = coords[1][0], coords[1][1]
                x2, z2 = coords[2][0], coords[2][1]
                e0x, e0z = x1 - x0, z1 - z0
                e1x, e1z = x2 - x1, z2 - z1
                l0 = math.hypot(e0x, e0z)
                l1 = math.hypot(e1x, e1z)
                if l0 >= l1:
                    w, d = l0, l1
                    vx, vz = e0x, e0z
                else:
                    w, d = l1, l0
                    vx, vz = e1x, e1z
                cx = 0.5 * (x0 + x2)
                cz = 0.5 * (z0 + z2)
                yaw = math.atan2(vz, vx) if (w > 1e-4) else 0.0
            else:
                cx, cz, w, d, yaw = float(cxs[src_i]), float(czs[src_i]), 8.0, 8.0, 0.0
        else:
            cx, cz = float(cxs[src_i]), float(czs[src_i])
            w, d, yaw = 8.0, 8.0, 0.0

        w = max(4.0, min(240.0, float(w)))
        d = max(4.0, min(200.0, float(d)))
        area = float(areas[k])

        st = subtypes[src_i] or ""
        cl = classes[src_i] or ""
        nm_d = names[src_i]
        nm = (nm_d.get("primary") or "") if nm_d else ""
        nm_low = nm.lower()
        ht = heights[src_i]
        nf = floors[src_i]

        enr_k, enr_lv = enriched_cat.get(k, (None, 0))

        # Determine building functional kind
        b_kind = BK_APT
        if enr_k is not None:
            b_kind = enr_k
        elif cl == "mosque" or st == "religious" or "cami" in nm_low or "mescit" in nm_low:
            b_kind = BK_MOSQUE
        elif cl in ("hospital", "clinic") or st == "medical" or "hastane" in nm_low:
            b_kind = BK_MEDICAL
        elif cl in ("school", "university", "college", "kindergarten") or st == "education" or "okul" in nm_low or "lise" in nm_low or "üniversite" in nm_low:
            b_kind = BK_EDUCATION
        elif cl == "greenhouse" or st == "agricultural":
            b_kind = BK_GREENHOUSE
        elif st == "industrial" or cl in ("industrial", "warehouse", "manufacture"):
            b_kind = BK_INDUSTRIAL
        elif cl in ("mall", "retail") or any(kw in nm_low for kw in ("avm", "optimum", "mavibahçe", "agora", "forum", "hilltown", "istinye", "westpark")):
            b_kind = BK_MALL
        elif cl in ("house", "detached", "semidetached_house"):
            b_kind = BK_HILLSIDE
        elif st == "commercial" or cl == "commercial":
            b_kind = BK_MALL if area > 900.0 else BK_APT
        elif area > 1600.0 and st == "":
            # Heuristic for untagged huge industrial/warehouse slabs vs apartment blocks
            if cx < -4000.0 or cx > 7000.0 or cz > 6000.0:
                b_kind = BK_INDUSTRIAL

        # Height assignment (real Overture height/floors -> OSM levels -> smart architectural estimate)
        h_val = 0.0
        if ht is not None and float(ht) > 2.5:
            h_val = max(3.5, min(220.0, float(ht)))
        elif nf is not None and int(nf) > 0:
            h_val = max(3.8, min(220.0, float(nf) * 3.15))
        elif enr_lv > 0:
            h_val = max(3.8, min(220.0, float(enr_lv) * 3.15))

        if h_val <= 0.0:
            hv = (abs(int(cx * 13.7)) ^ abs(int(cz * 29.3)) ^ k) & 0xFFFF
            if b_kind == BK_GREENHOUSE:
                h_val = 4.2
            elif b_kind == BK_MOSQUE:
                h_val = 11.5 if area > 250.0 else 8.5
            elif b_kind == BK_INDUSTRIAL:
                h_val = 9.0 + (hv % 6)
            elif b_kind == BK_MEDICAL:
                h_val = 22.0 if area > 1200.0 else 13.0
            elif b_kind == BK_EDUCATION:
                h_val = 12.5 + (hv % 5)
            elif b_kind == BK_MALL:
                h_val = 16.0 if area > 1500.0 else 11.0
            elif b_kind == BK_HILLSIDE:
                h_val = 6.5 + (hv % 4)
            else:
                # Realistic İzmir density heights
                dist_bay = math.hypot(cx - bay_x, cz - bay_z)
                dist_mav = math.hypot(cx - mav_x, cz - mav_z)
                if dist_bay < 1500.0 and area > 450.0:
                    h_val = 28.0 + (hv % 32)
                elif dist_mav < 1400.0 and area > 350.0:
                    h_val = 22.0 + (hv % 22)
                elif area < 105.0:
                    h_val = 7.0 + (hv % 5)
                elif area < 240.0:
                    h_val = 12.0 + (hv % 8)
                else:
                    h_val = 16.5 + (hv % 12)

        if h_val >= 36.0 and b_kind in (BK_APT, BK_MALL):
            b_kind = BK_TOWER

        # Warm Aegean + modern facade tint
        hv_c = (abs(int(cx * 7.1)) + abs(int(cz * 11.3))) & 0xFF
        if b_kind == BK_TOWER:
            r_col, g_col, b_col = 185 + (hv_c % 25), 215 + (hv_c % 25), 240 + (hv_c % 15)
        elif b_kind == BK_INDUSTRIAL:
            r_col, g_col, b_col = 168 + (hv_c % 25), 182 + (hv_c % 25), 205 + (hv_c % 25)
        elif b_kind == BK_MOSQUE:
            r_col, g_col, b_col = 242, 236, 224
        elif b_kind == BK_GREENHOUSE:
            r_col, g_col, b_col = 235, 245, 248
        else:
            r_col = 225 + (hv_c % 28)
            g_col = 215 + ((hv_c >> 2) % 30)
            b_col = 200 + ((hv_c >> 4) % 35)

        pack_b(packed_records, write_pos, cx, cz, w, d, yaw, h_val, b_kind, r_col, g_col, b_col)
        write_pos += 28

    print(f"  Buildings ready: {n_out} real OBBs ({len(packed_records)/1e6:.2f} MB) in {time.time() - t0:.2f}s", flush=True)
    return bytes(packed_records), n_out


def main():
    t_start = time.time()
    print("=== Building Complete Overture + OSM + 30m DEM World v2 ===", flush=True)
    dem = build_dem_grid()
    gis_grid, trees = build_gis_grid_and_trees()
    rails, roads = load_roads_and_rails()
    bldg_bytes, n_bldgs = load_buildings()

    print(f"\n[5/6] Writing IZMIROSM v2 binary to {BIN_PATH}...", flush=True)
    with open(BIN_PATH, "wb") as f:
        # 1. Header
        f.write(b"IZMIROSM")
        f.write(struct.pack("<I", 2))  # Version 2
        f.write(struct.pack("<ffffff", REF_LAT, REF_LON, WORLD_X0, WORLD_X1, WORLD_Z0, WORLD_Z1))

        # 2. DEM (1024x1024 float32)
        f.write(struct.pack("<II", DEM_N, DEM_N))
        f.write(dem.tobytes())

        # 3. GIS Surface Class Grid (2048x2048 uint8)
        f.write(struct.pack("<I", MASK_N))
        f.write(gis_grid.tobytes())

        # 4. Real Mapped Trees
        f.write(struct.pack("<I", len(trees)))
        if trees:
            tbuf = bytearray(len(trees) * 8)
            off = 0
            for x, z in trees:
                struct.pack_into("<ff", tbuf, off, x, z)
                off += 8
            f.write(tbuf)

        # 5. Active Rails (standard_gauge / tram / surface subway)
        f.write(struct.pack("<I", len(rails)))
        for rtype, is_br, coords in rails:
            np_pts = len(coords)
            f.write(struct.pack("<BBH", rtype, is_br, np_pts))
            f.write(np.asarray(coords, dtype=np.float32).tobytes())

        # 6. Operating Roads & Bridges (tunnels stripped, TomTom deduped)
        f.write(struct.pack("<I", len(roads)))
        for tier, is_br, lvl, hw_dm, coords in roads:
            np_pts = len(coords)
            f.write(struct.pack("<BBbBH", tier, is_br, lvl, hw_dm, np_pts))
            f.write(np.asarray(coords, dtype=np.float32).tobytes())

        # 7. Real Buildings (28 bytes each)
        f.write(struct.pack("<I", n_bldgs))
        f.write(bldg_bytes)

    size_mb = BIN_PATH.stat().st_size / (1024.0 * 1024.0)
    print(f"[6/6] DONE! {BIN_PATH} written cleanly ({size_mb:.2f} MB) in {time.time() - t_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
