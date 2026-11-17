#!/usr/bin/env python3
"""
Izmir Complete Cartographic Map Builder V2 (100% GIS-backed)
Sources:
  - Overture Maps GeoParquet: building (OSM + Microsoft ML), segment (OSM + TomTom + rail flags + level_rules),
    water, land, land_cover, land_use, infrastructure, place, division, division_area
  - Local Z12 Terrarium DEM tiles for analytical relief hillshading & elevation contours
  - OpenStreetMap core station/amenity enrichment (osm_cache/izmir_core.json + izmir_bldgs_*.json)
Outputs high-res Master 6K Metro Map + 6x 2x-supersampled District Detail Maps.
"""
import os
import sys
import math
import time
import glob
import shutil
from collections import defaultdict

import numpy as np
import orjson
import pyarrow.parquet as pq
import shapely
import shapely.wkb
from shapely.geometry import box, Polygon, MultiPolygon, LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OVERTURE_DIR = os.path.join(BASE_DIR, "osm_cache", "overture")
DEM_DIR = os.path.join(BASE_DIR, "osm_cache", "dem_tiles")
OUT_DIR = BASE_DIR
ARTIFACT_DIR = os.environ.get("IZMIR_OUT_DIR", "")

# Full Metro Izmir Viewport
MASTER_BBOX = (26.860, 38.220, 27.340, 38.650)  # lon0, lat0, lon1, lat1

DETAIL_VIEWS = [
    ("detail_v2_egekent2_ulukent.png", "EGEKENT-2 · ULUKENT · İSTİKLAL · HARMANDALI", (27.022, 38.545, 27.078, 38.582)),
    ("detail_v2_menemen_koyundere.png", "MENEMEN · KOYUNDERE · ASARLIK · GEDİZ", (27.042, 38.578, 27.105, 38.628)),
    ("detail_v2_karsiyaka_cigli.png", "KARŞIYAKA · BOSTANLI · MAVİŞEHİR · ÇİĞLİ", (27.055, 38.445, 27.135, 38.505)),
    ("detail_v2_konak_alsancak_bayrakli.png", "KONAK · ALSANCAK · BAYRAKLI · KADİFEKALE", (27.118, 38.404, 27.198, 38.468)),
    ("detail_v2_bornova_sehirhastanesi.png", "BORNOVA · EGE ÜNİVERSİTESİ · BAYRAKLI · ŞEHİR HASTANESİ", (27.128, 38.438, 27.248, 38.508)),
    ("detail_v2_buca_gaziemir_havalimani.png", "BUCA · ŞİRİNYER · GAZİEMİR · ADNAN MENDERES", (27.115, 38.285, 27.215, 38.398)),
    ("detail_v2_guzelbahce_narlidere_inciralti.png", "GÜZELBAHÇE · NARLIDERE · BALÇOVA · İNCİRALTI", (26.875, 38.352, 27.048, 38.422)),
]

# Metric scaling matching Izmir (~38.42N)
LON_M = 87215.0
LAT_M = 111132.0

FONT_REG = "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/google-carlito-fonts/Carlito-Bold.ttf"
FONT_ITAL = "/usr/share/fonts/google-carlito-fonts/Carlito-Italic.ttf"
if not os.path.exists(FONT_REG):
    FONT_REG = "/usr/share/fonts/gnu-free/FreeSans.ttf"
    FONT_BOLD = "/usr/share/fonts/gnu-free/FreeSansBold.ttf"
    FONT_ITAL = "/usr/share/fonts/gnu-free/FreeSansOblique.ttf"


def hex_rgb(s, alpha=None):
    s = s.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (r, g, b) if alpha is None else (r, g, b, alpha)


# Cartographic Palette
COL_SEA = hex_rgb("#9ecbe6")
COL_SEA_SHALLOW = hex_rgb("#b3daef")
COL_SHORELINE = hex_rgb("#659bbb")
COL_LAND_BASE = hex_rgb("#f3efe6")

LANDCOVER_STYLE = {
    "forest": hex_rgb("#cde0bd"),
    "shrub": hex_rgb("#e0e6cc"),
    "grass": hex_rgb("#e4ebd2"),
    "crop": hex_rgb("#eee9cd"),
    "wetland": hex_rgb("#c7dfd5"),
    "barren": hex_rgb("#e8e0d2"),
    "urban": hex_rgb("#ece8de"),
}

LANDUSE_STYLE = {
    "park": (hex_rgb("#b9e0a6"), hex_rgb("#8ec27b")),
    "garden": (hex_rgb("#b2dd9d"), hex_rgb("#86bd72")),
    "playground": (hex_rgb("#c5e6b1"), hex_rgb("#96c57f")),
    "pitch": (hex_rgb("#a0d998"), hex_rgb("#73b86e")),
    "recreation_ground": (hex_rgb("#bae0a8"), hex_rgb("#8ec27b")),
    "track": (hex_rgb("#d6b99a"), hex_rgb("#b38e6c")),
    "grass": (hex_rgb("#cde6b9"), None),
    "meadow": (hex_rgb("#d7e8c4"), None),
    "forest": (hex_rgb("#b6d6a0"), None),
    "wood": (hex_rgb("#b6d6a0"), None),
    "farmland": (hex_rgb("#e9e6c7"), None),
    "orchard": (hex_rgb("#d8e4b6"), None),
    "vineyard": (hex_rgb("#dde5b4"), None),
    "allotments": (hex_rgb("#d8e2b8"), hex_rgb("#b5c58e")),
    "farmyard": (hex_rgb("#e4dcc6"), None),
    "residential": (hex_rgb("#e7e1d6"), hex_rgb("#d2cabc")),
    "school": (hex_rgb("#f4e5cd"), hex_rgb("#d8be98")),
    "university": (hex_rgb("#f3e0c2"), hex_rgb("#d4b487")),
    "education": (hex_rgb("#f4e5cd"), hex_rgb("#d8be98")),
    "hospital": (hex_rgb("#f7dde0"), hex_rgb("#dba3a9")),
    "religious": (hex_rgb("#e7e0ec"), hex_rgb("#c8bcd1")),
    "cemetery": (hex_rgb("#d2dfd0"), hex_rgb("#aec2ab")),
    "commercial": (hex_rgb("#f2dedd"), hex_rgb("#d9b4b3")),
    "retail": (hex_rgb("#f4d8d6"), hex_rgb("#d8aeab")),
    "industrial": (hex_rgb("#e1dde6"), hex_rgb("#c5bfd0")),
    "works": (hex_rgb("#e1dde6"), hex_rgb("#c5bfd0")),
    "construction": (hex_rgb("#e5ded0"), hex_rgb("#c9beab")),
    "greenfield": (hex_rgb("#e9e4d7"), None),
    "brownfield": (hex_rgb("#dfd8cc"), None),
    "quarry": (hex_rgb("#dcd8d2"), hex_rgb("#b8b3ab")),
    "landfill": (hex_rgb("#dad6cc"), None),
    "military": (hex_rgb("#e7d9d8"), hex_rgb("#cbb4b2")),
    "plaza": (hex_rgb("#eae6de"), hex_rgb("#cdc7bb")),
    "pedestrian": (hex_rgb("#ebe7df"), hex_rgb("#cfc9bd")),
    "railway": (hex_rgb("#dedad6"), None),
    "salt_pond": (hex_rgb("#9bd2e6"), hex_rgb("#7bb8d0")),
}

BLDG_COLORS = {
    "default": (hex_rgb("#e2d4c2"), hex_rgb("#b8a895")),
    "residential": (hex_rgb("#e3d5c3"), hex_rgb("#b6a591")),
    "apartments": (hex_rgb("#dfcfb9"), hex_rgb("#af9d87")),
    "house": (hex_rgb("#e8dac8"), hex_rgb("#bcac98")),
    "industrial": (hex_rgb("#cfd7e2"), hex_rgb("#9da8b8")),
    "agricultural": (hex_rgb("#c7dde0"), hex_rgb("#94b3b8")),
    "greenhouse": (hex_rgb("#c5dfe4"), hex_rgb("#8eb2ba")),
    "education": (hex_rgb("#eed39d"), hex_rgb("#c29f61")),
    "medical": (hex_rgb("#eeb5bc"), hex_rgb("#c47c86")),
    "religious": (hex_rgb("#c7c9ea"), hex_rgb("#8d91c4")),
    "commercial": (hex_rgb("#ebc2b9"), hex_rgb("#bf8b80")),
    "mall": (hex_rgb("#e9b4b0"), hex_rgb("#ba7974")),
    "civic": (hex_rgb("#d4c9e4"), hex_rgb("#9d8eb8")),
    "transportation": (hex_rgb("#d6d0dc"), hex_rgb("#9f96aa")),
    "entertainment": (hex_rgb("#e8cbe0"), hex_rgb("#b68eaa")),
}


class DemMosaic:
    """Loads cached Z12 Terrarium DEM PNG tiles cleanly into a fast numpy array."""

    def __init__(self, dem_dir):
        with open(os.path.join(dem_dir, "meta.json"), "r") as f:
            meta = orjson.loads(f.read())
        self.z = meta["zoom"]
        self.x0, self.x1 = meta["x0"], meta["x1"]
        self.y0, self.y1 = meta["y0"], meta["y1"]
        nx = self.x1 - self.x0 + 1
        ny = self.y1 - self.y0 + 1
        elev = np.zeros((ny * 256, nx * 256), dtype=np.float32)
        for ty in range(self.y0, self.y1 + 1):
            for tx in range(self.x0, self.x1 + 1):
                p = os.path.join(dem_dir, f"z{self.z}_{tx}_{ty}.png")
                if not os.path.exists(p):
                    continue
                arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                h = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
                py = (ty - self.y0) * 256
                px = (tx - self.x0) * 256
                elev[py:py + 256, px:px + 256] = h
        self.elev = elev
        self.H, self.W = elev.shape

    def sample_grid(self, bbox, w, h):
        lon0, lat0, lon1, lat1 = bbox
        n = 2.0 ** self.z
        lons = np.linspace(lon0, lon1, w, dtype=np.float64)
        lats = np.linspace(lat1, lat0, h, dtype=np.float64)  # top to bottom

        fx = ((lons + 180.0) / 360.0 * n - self.x0) * 256.0
        lat_rad = np.radians(lats)
        ty = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) / 2.0 * n
        fy = (ty - self.y0) * 256.0

        fx = np.clip(fx, 0.0, self.W - 1.001)
        fy = np.clip(fy, 0.0, self.H - 1.001)
        ix0 = fx.astype(np.int32)
        iy0 = fy.astype(np.int32)
        dx = (fx - ix0).astype(np.float32)[None, :]
        dy = (fy - iy0).astype(np.float32)[:, None]

        e = self.elev
        v00 = e[iy0[:, None], ix0[None, :]]
        v10 = e[iy0[:, None], ix0[None, :] + 1]
        v01 = e[iy0[:, None] + 1, ix0[None, :]]
        v11 = e[iy0[:, None] + 1, ix0[None, :] + 1]
        return (v00 * (1 - dx) + v10 * dx) * (1 - dy) + (v01 * (1 - dx) + v11 * dx) * dy


def project_coords(coords, bbox, w, h):
    lon0, lat0, lon1, lat1 = bbox
    sx = w / (lon1 - lon0)
    sy = h / (lat1 - lat0)
    return [((x - lon0) * sx, (lat1 - y) * sy) for x, y in coords]


def project_ring_np(coords_arr, bbox, w, h):
    lon0, lat0, lon1, lat1 = bbox
    sx = w / (lon1 - lon0)
    sy = h / (lat1 - lat0)
    xs = (coords_arr[:, 0] - lon0) * sx
    ys = (lat1 - coords_arr[:, 1]) * sy
    return list(zip(xs.tolist(), ys.tolist()))


def draw_poly_with_holes(img, draw, poly, bbox, w, h, fill=None, outline=None, outline_w=1):
    if poly.is_empty:
        return
    if poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            draw_poly_with_holes(img, draw, p, bbox, w, h, fill, outline, outline_w)
        return
    if poly.geom_type != "Polygon":
        return

    lon0, lat0, lon1, lat1 = bbox
    bx0, by0, bx1, by1 = poly.bounds
    if bx1 < lon0 or bx0 > lon1 or by1 < lat0 or by0 > lat1:
        return

    ext_pts = project_coords(poly.exterior.coords, bbox, w, h)
    if len(ext_pts) < 3:
        return

    if len(poly.interiors) == 0:
        draw.polygon(ext_pts, fill=fill, outline=outline)
        if outline is not None and outline_w > 1:
            draw.line(ext_pts, fill=outline, width=outline_w, joint="curve")
        return

    # Polygon with courtyard/island holes: render into a minimal local alpha mask
    px0 = max(0, int(math.floor((bx0 - lon0) / (lon1 - lon0) * w)) - 2)
    px1 = min(w, int(math.ceil((bx1 - lon0) / (lon1 - lon0) * w)) + 2)
    py0 = max(0, int(math.floor((lat1 - by1) / (lat1 - lat0) * h)) - 2)
    py1 = min(h, int(math.ceil((lat1 - by0) / (lat1 - lat0) * h)) + 2)
    mw, mh = px1 - px0, py1 - py0
    if mw <= 0 or mh <= 0:
        return

    mask = Image.new("L", (mw, mh), 0)
    mdraw = ImageDraw.Draw(mask)
    sh = [(x - px0, y - py0) for x, y in ext_pts]
    mdraw.polygon(sh, fill=255)
    holes_pts = []
    for ring in poly.interiors:
        hp = project_coords(ring.coords, bbox, w, h)
        if len(hp) >= 3:
            holes_pts.append(hp)
            mdraw.polygon([(x - px0, y - py0) for x, y in hp], fill=0)

    if fill is not None:
        img.paste(fill[:3], (px0, py0), mask)
    if outline is not None:
        draw.line(ext_pts, fill=outline, width=outline_w, joint="curve")
        for hp in holes_pts:
            draw.line(hp, fill=outline, width=outline_w, joint="curve")


def split_segment_by_flags(geom, flags_list, level_rules):
    """Split a LineString by Overture linear-referenced between=[t0, t1] ranges."""
    if geom.is_empty or geom.geom_type != "LineString":
        return []
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
    # Fast path if no internal split
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


class GISWorldData:
    def __init__(self):
        t0 = time.time()
        print("=== Loading Complete Overture & OSM GIS World ===", flush=True)
        self.dem = DemMosaic(DEM_DIR)

        # 1. Land & Water
        land_tbl = pq.read_table(os.path.join(OVERTURE_DIR, "land.parquet"), columns=["subtype", "class", "names", "geometry"])
        self.land_polys = []
        self.trees = []
        self.tree_rows = []
        self.peaks = []
        for st, cl, nm, gb in zip(land_tbl["subtype"].to_pylist(), land_tbl["class"].to_pylist(),
                                  land_tbl["names"].to_pylist(), land_tbl["geometry"].to_pylist()):
            g = shapely.wkb.loads(gb)
            if st == "land" and g.geom_type in ("Polygon", "MultiPolygon"):
                self.land_polys.append(g)
            elif st == "tree":
                if g.geom_type == "Point":
                    self.trees.append((g.x, g.y))
                elif g.geom_type == "LineString":
                    self.tree_rows.append(list(g.coords))
            elif cl == "peak" and g.geom_type == "Point":
                name = nm.get("primary") if nm else None
                if name:
                    self.peaks.append((g.x, g.y, name))
            elif st in ("forest", "shrub", "grass", "wetland", "sand", "rock") and g.geom_type in ("Polygon", "MultiPolygon"):
                pass

        water_tbl = pq.read_table(os.path.join(OVERTURE_DIR, "water.parquet"), columns=["subtype", "class", "names", "geometry"])
        self.water_polys = []
        self.water_lines = []
        self.salt_ponds = []
        for st, cl, nm, gb in zip(water_tbl["subtype"].to_pylist(), water_tbl["class"].to_pylist(),
                                  water_tbl["names"].to_pylist(), water_tbl["geometry"].to_pylist()):
            if st in ("ocean", "physical"):
                continue  # Sea/Bay handled directly via exact inverse land_mask so coastline never clips
            g = shapely.wkb.loads(gb)
            if g.geom_type in ("Polygon", "MultiPolygon"):
                if cl == "salt_pond":
                    self.salt_ponds.append(g)
                else:
                    self.water_polys.append((st, cl, g))
            elif g.geom_type == "LineString":
                self.water_lines.append((st, cl, g))
        print(f"  Land polys: {len(self.land_polys)}, Salt ponds: {len(self.salt_ponds)}, Inland water polys: {len(self.water_polys)}, Water lines: {len(self.water_lines)}, Trees: {len(self.trees)}", flush=True)

        # 2. Land cover (STRICTLY keep only high-res min_zoom >= 8 layers sorted by sort_key to eliminate quadkey seams!)
        lc_tbl = pq.read_table(os.path.join(OVERTURE_DIR, "land_cover.parquet"), columns=["subtype", "cartography", "geometry"])
        lc_items = []
        for st, ca, gb in zip(lc_tbl["subtype"].to_pylist(), lc_tbl["cartography"].to_pylist(), lc_tbl["geometry"].to_pylist()):
            mz = (ca or {}).get("min_zoom", 8)
            sk = (ca or {}).get("sort_key", 0)
            if mz >= 8:
                lc_items.append((sk, st, shapely.wkb.loads(gb)))
        lc_items.sort(key=lambda x: x[0])
        self.land_cover = [(st, g) for _sk, st, g in lc_items]
        print(f"  Land cover polys (z>=8 high-res): {len(self.land_cover)}", flush=True)

        # 3. Land use (sorted largest to smallest so parks/pitches sit inside residential estates!)
        lu_tbl = pq.read_table(os.path.join(OVERTURE_DIR, "land_use.parquet"), columns=["subtype", "class", "names", "geometry"])
        lu_raw = []
        for st, cl, nm, gb in zip(lu_tbl["subtype"].to_pylist(), lu_tbl["class"].to_pylist(),
                                  lu_tbl["names"].to_pylist(), lu_tbl["geometry"].to_pylist()):
            g = shapely.wkb.loads(gb)
            if g.geom_type in ("Polygon", "MultiPolygon"):
                key = cl if cl in LANDUSE_STYLE else (st if st in LANDUSE_STYLE else None)
                if key:
                    lu_raw.append((g.area, key, nm.get("primary") if nm else None, g))
        lu_raw.sort(key=lambda x: -x[0])  # Largest bottom-most, smallest (playgrounds, gardens, pitches) top-most!
        self.land_use = lu_raw
        print(f"  Land use polys: {len(self.land_use)}", flush=True)

        # 4. Infrastructure (airports, piers, stations)
        inf_tbl = pq.read_table(os.path.join(OVERTURE_DIR, "infrastructure.parquet"), columns=["subtype", "class", "names", "geometry"])
        self.airport_polys = []
        self.airport_lines = []
        self.piers = []
        self.ferry_terminals = []
        for st, cl, nm, gb in zip(inf_tbl["subtype"].to_pylist(), inf_tbl["class"].to_pylist(),
                                  inf_tbl["names"].to_pylist(), inf_tbl["geometry"].to_pylist()):
            g = shapely.wkb.loads(gb)
            name = nm.get("primary") if nm else None
            if st == "airport":
                if g.geom_type in ("Polygon", "MultiPolygon"):
                    self.airport_polys.append((cl, g))
                elif g.geom_type == "LineString":
                    self.airport_lines.append((cl, g))
            elif cl == "pier":
                self.piers.append(g)
            elif cl == "ferry_terminal" and name:
                pt = g.centroid
                self.ferry_terminals.append((pt.x, pt.y, name))
        print(f"  Airport feats: {len(self.airport_polys) + len(self.airport_lines)}, Piers: {len(self.piers)}, Ferry terminals: {len(self.ferry_terminals)}", flush=True)

        # 5. Roads & Active Rails (with fractional bridge/tunnel/level linear splitting + TomTom dedup!)
        seg_tbl = pq.read_table(
            os.path.join(OVERTURE_DIR, "segment.parquet"),
            columns=["subtype", "class", "subclass", "road_flags", "rail_flags", "level_rules", "names", "routes", "sources", "geometry"]
        )
        osm_roads = []
        tt_roads = []
        self.rails = []  # (cls, flags_set, level, name, LineString)
        self.road_tunnels = []
        self.road_under_construction = []
        self.road_surface = defaultdict(list)  # cls -> list of (LineString, name, is_link, lvl)
        self.road_bridges = []  # (level, cls, LineString, name, is_link)

        for st, cl, subc, rf, rlf, lr, nm, rts, src, gb in zip(
            seg_tbl["subtype"].to_pylist(), seg_tbl["class"].to_pylist(), seg_tbl["subclass"].to_pylist(),
            seg_tbl["road_flags"].to_pylist(), seg_tbl["rail_flags"].to_pylist(), seg_tbl["level_rules"].to_pylist(),
            seg_tbl["names"].to_pylist(), seg_tbl["routes"].to_pylist(), seg_tbl["sources"].to_pylist(),
            seg_tbl["geometry"].to_pylist()
        ):
            g = shapely.wkb.loads(gb)
            name = nm.get("primary") if nm else None
            if not name and rts:
                ref = rts[0].get("ref")
                if ref:
                    name = ref
            ds = src[0].get("dataset") if src else "OpenStreetMap"

            if st == "rail":
                if cl not in ("standard_gauge", "tram", "subway"):
                    continue
                for sub_g, fset, lvl in split_segment_by_flags(g, rlf, lr):
                    if fset & {"is_abandoned", "is_disused", "is_under_construction"}:
                        continue
                    self.rails.append((cl, fset, lvl, name, sub_g))
            elif st == "road":
                spans = split_segment_by_flags(g, rf, lr)
                if ds != "OpenStreetMap" and cl == "unknown":
                    tt_roads.append((g, spans, name))
                else:
                    if ds == "OpenStreetMap":
                        osm_roads.append(g)
                    for sub_g, fset, lvl in spans:
                        if fset & {"is_abandoned", "is_disused"}:
                            continue
                        if "is_under_construction" in fset:
                            self.road_under_construction.append((cl, sub_g))
                            continue
                        eff_cls = cl
                        if subc in ("parking_aisle", "driveway", "alley"):
                            eff_cls = "service"
                        elif subc == "sidewalk":
                            eff_cls = "footway"
                        is_link = ("is_link" in fset) or (subc == "link")
                        if "is_tunnel" in fset or lvl < 0:
                            self.road_tunnels.append((eff_cls, sub_g, lvl))
                        elif "is_bridge" in fset or lvl > 0:
                            self.road_bridges.append((max(1, lvl), eff_cls, sub_g, name, is_link))
                        else:
                            self.road_surface[eff_cls].append((sub_g, name, is_link, 0))

        # Deduplicate TomTom roads against OSM roads (~8m buffer check)
        osm_tree = STRtree(osm_roads)
        tt_kept = 0
        tt_dropped = 0
        for full_g, spans, name in tt_roads:
            buf = full_g.buffer(0.000085)
            idxs = osm_tree.query(buf)
            overlap_len = 0.0
            for idx in idxs:
                og = osm_roads[idx]
                if buf.intersects(og):
                    overlap_len += full_g.intersection(og.buffer(0.000085)).length
            if full_g.length > 0 and (overlap_len / full_g.length) > 0.50:
                tt_dropped += 1
                continue
            tt_kept += 1
            for sub_g, fset, lvl in spans:
                if "is_under_construction" in fset:
                    continue
                if "is_tunnel" in fset or lvl < 0:
                    self.road_tunnels.append(("residential", sub_g, lvl))
                elif "is_bridge" in fset or lvl > 0:
                    self.road_bridges.append((max(1, lvl), "residential", sub_g, name, False))
                else:
                    self.road_surface["residential"].append((sub_g, name, False, 0))

        self.road_bridges.sort(key=lambda x: x[0])
        print(f"  Roads loaded -> OSM surf classes={sum(len(v) for v in self.road_surface.values())}, TomTom kept={tt_kept} (dropped {tt_dropped} dup), Bridges={len(self.road_bridges)}, Tunnels={len(self.road_tunnels)}, Rails={len(self.rails)}", flush=True)

        # 6. Buildings (450,921 from Overture: OSM + Microsoft ML)
        b_tbl = pq.read_table(
            os.path.join(OVERTURE_DIR, "building.parquet"),
            columns=["subtype", "class", "names", "height", "num_floors", "is_underground", "geometry"]
        )
        self.buildings = []  # (cat, height, name, geom, cx, cy)
        b_geoms = []
        for st, cl, nm, ht, nf, ug, gb in zip(
            b_tbl["subtype"].to_pylist(), b_tbl["class"].to_pylist(), b_tbl["names"].to_pylist(),
            b_tbl["height"].to_pylist(), b_tbl["num_floors"].to_pylist(), b_tbl["is_underground"].to_pylist(),
            b_tbl["geometry"].to_pylist()
        ):
            if ug:
                continue
            g = shapely.wkb.loads(gb)
            if g.is_empty:
                continue
            name = nm.get("primary") if nm else None
            cat = "default"
            if cl in ("mosque",) or st == "religious":
                cat = "religious"
            elif cl in ("hospital", "clinic") or st == "medical":
                cat = "medical"
            elif cl in ("school", "university", "college", "kindergarten") or st == "education":
                cat = "education"
            elif cl in ("greenhouse",) or st == "agricultural":
                cat = "greenhouse"
            elif st == "industrial" or cl in ("industrial", "warehouse", "manufacture"):
                cat = "industrial"
            elif cl in ("mall", "retail") or (name and any(k in name.lower() for k in ("avm", "optimum", "mavibahçe", "agora", "forum", "hilltown", "istinye", "kipa", "westpark"))):
                cat = "mall"
            elif st == "commercial" or cl == "commercial":
                cat = "commercial"
            elif st == "civic" or cl in ("public", "civic", "fire_station", "government"):
                cat = "civic"
            elif st == "transportation" or cl in ("train_station", "hangar"):
                cat = "transportation"
            elif cl == "apartments":
                cat = "apartments"
            elif cl in ("house", "detached", "semidetached_house"):
                cat = "house"
            elif st == "residential":
                cat = "residential"

            h_val = float(ht) if ht is not None else (float(nf) * 3.1 if nf is not None else 9.0)
            c = g.centroid
            self.buildings.append([cat, h_val, name, g, c.x, c.y])
            b_geoms.append(g)

        # Enrich generic buildings using OSM special tags (izmir_bldgs_*.json) & Overture malls/hospitals
        b_tree = STRtree(b_geoms)
        enriched = 0
        for p in sorted(glob.glob(os.path.join(BASE_DIR, "osm_cache", "izmir_bldgs_*.json"))):
            with open(p, "rb") as f:
                elems = orjson.loads(f.read()).get("elements", [])
            for e in elems:
                tags = e.get("tags", {})
                am = tags.get("amenity", "")
                sh = tags.get("shop", "")
                nm = tags.get("name")
                target_cat = None
                if am == "place_of_worship":
                    target_cat = "religious"
                elif am in ("hospital", "clinic"):
                    target_cat = "medical"
                elif am in ("school", "university", "college", "kindergarten"):
                    target_cat = "education"
                elif sh in ("mall", "department_store"):
                    target_cat = "mall"
                if not target_cat:
                    continue
                geom_pts = e.get("geometry")
                if not geom_pts:
                    continue
                lon_c = sum(pt["lon"] for pt in geom_pts) / len(geom_pts)
                lat_c = sum(pt["lat"] for pt in geom_pts) / len(geom_pts)
                pt = Point(lon_c, lat_c)
                idxs = b_tree.query(pt)
                for idx in idxs:
                    if self.buildings[idx][3].contains(pt):
                        if self.buildings[idx][0] in ("default", "residential", "commercial"):
                            self.buildings[idx][0] = target_cat
                            enriched += 1
                        if nm and not self.buildings[idx][2]:
                            self.buildings[idx][2] = nm
                        break
        print(f"  Buildings loaded: {len(self.buildings)} (OSM tag enriched: {enriched})", flush=True)

        # 7. Stations (strictly active/operating stations matched by track class + distance!)
        rail_by_kind = {
            "izban": [r[4] for r in self.rails if r[0] == "standard_gauge"],
            "metro": [r[4] for r in self.rails if r[0] == "subway"],
            "tram": [r[4] for r in self.rails if r[0] == "tram"],
        }
        rail_trees = {k: STRtree(v) for k, v in rail_by_kind.items() if v}
        self.stations = []  # (kind, lon, lat, name)  kind: 'izban'|'metro'|'tram'|'ferry'
        seen_st = []
        with open(os.path.join(BASE_DIR, "osm_cache", "izmir_core.json"), "rb") as f:
            core_elems = orjson.loads(f.read()).get("elements", [])
        for e in core_elems:
            if e.get("type") != "node":
                continue
            tags = e.get("tags", {})
            rw = tags.get("railway")
            nm = tags.get("name")
            if not rw or not nm:
                continue
            if any(
                k in ("construction", "proposed", "disused", "abandoned")
                or k.startswith(("proposed:", "construction:", "disused:", "abandoned:"))
                for k in tags
            ):
                continue
            if rw in ("proposed", "construction", "disused", "abandoned"):
                continue
            lat, lon = e["lat"], e["lon"]
            op = (tags.get("operator") or tags.get("network") or "").lower()
            st_tag = (tags.get("station") or "").lower()
            if rw == "tram_stop" or "tram" in op:
                kind = "tram"
            elif st_tag == "subway" or "metro" in op or tags.get("subway") == "yes":
                kind = "metro"
            elif rw in ("station", "halt"):
                kind = "izban"
            else:
                continue
            tr = rail_trees.get(kind)
            if tr is not None:
                pt = Point(lon, lat)
                geoms = rail_by_kind[kind]
                idxs = tr.query(pt.buffer(0.0020))
                if not any(pt.distance(geoms[j]) <= 0.0018 for j in idxs):  # ~180m exact distance to active track of same class
                    continue
            seen_st.append((kind, lon, lat, nm))

        # Deduplicate nearby multi-entrance/platform station nodes of same normalized name
        dedup_st = {}
        for kind, lon, lat, nm in seen_st:
            key = (kind, nm.strip().lower())
            if key not in dedup_st:
                dedup_st[key] = [kind, lon, lat, nm, 1]
            else:
                d2 = (dedup_st[key][1] - lon) ** 2 + (dedup_st[key][2] - lat) ** 2
                if d2 < (0.006 ** 2):
                    dedup_st[key][1] += lon
                    dedup_st[key][2] += lat
                    dedup_st[key][4] += 1
                else:
                    dedup_st[(kind, nm.strip().lower(), round(lon, 2))] = [kind, lon, lat, nm, 1]
        for v in dedup_st.values():
            self.stations.append((v[0], v[1] / v[4], v[2] / v[4], v[3]))
        for lon, lat, nm in self.ferry_terminals:
            self.stations.append(("ferry", lon, lat, nm))

        # 8. Districts, Neighborhoods (Mahalleler) & Key Places
        div_tbl = pq.read_table(os.path.join(OVERTURE_DIR, "division.parquet"), columns=["subtype", "class", "names", "geometry"])
        self.counties = []
        self.localities = []
        self.mahalleler = []
        for st, cl, nm, gb in zip(div_tbl["subtype"].to_pylist(), div_tbl["class"].to_pylist(),
                                  div_tbl["names"].to_pylist(), div_tbl["geometry"].to_pylist()):
            name = nm.get("primary") if nm else None
            if not name:
                continue
            pt = shapely.wkb.loads(gb)
            if st == "county":
                self.counties.append((pt.x, pt.y, name))
            elif st == "locality" and cl in ("city", "town", "village"):
                self.localities.append((cl, pt.x, pt.y, name))
            elif st in ("macrohood", "neighborhood"):
                self.mahalleler.append((st, pt.x, pt.y, name))

        # Add named major POIs from buildings (area converted accurately via LON_M * LAT_M -> m^2!)
        self.landmarks = []
        seen_lm = set()
        DEG2_TO_M2 = LON_M * LAT_M
        for cat, h_val, name, g, cx, cy in self.buildings:
            if not name or name.lower() in ("cami", "camii", "mescit", "konut", "okul", "derslik"):
                continue
            area_m2 = g.area * DEG2_TO_M2
            nl = name.lower()
            is_major = (
                cat in ("mall", "medical", "transportation")
                or (cat == "education" and area_m2 > 2200 and any(k in nl for k in ("üniversite", "fakülte", "lise", "kolej", "kampüs", "rektörlük")))
                or any(k in nl for k in ("havalimanı", "stadyum", "arena", "fuar izmir", "gar", "belediye", "kaymakamlık", "otogar", "şehir hastanesi"))
            )
            if is_major and area_m2 > 500:
                k = nl.strip()
                if k not in seen_lm:
                    seen_lm.add(k)
                    self.landmarks.append((cat, cx, cy, name, area_m2))
        self.landmarks.sort(key=lambda x: -x[4])
        print(f"  Stations: {len(self.stations)}, Counties: {len(self.counties)}, Mahalleler: {len(self.mahalleler)}, Landmarks: {len(self.landmarks)}", flush=True)
        print(f"=== GIS World loaded in {time.time() - t0:.1f}s ===", flush=True)


class LabelPlacer:
    """Grid-accelerated collision checker + haloed typography helper."""

    def __init__(self, w, h, draw, y_min=6, y_max=None):
        self.w = w
        self.h = h
        self.y_min = y_min
        self.y_max = (h - 6) if y_max is None else y_max
        self.draw = draw
        self.boxes = []
        self.same_road_cells = set()

    def try_place_box(self, x0, y0, x1, y1, margin=4):
        if x0 < 6 or y0 < self.y_min or x1 > self.w - 6 or y1 > self.y_max:
            return False
        ax0, ay0, ax1, ay1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
        for bx0, by0, bx1, by1 in self.boxes:
            if not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1):
                return False
        self.boxes.append((ax0, ay0, ax1, ay1))
        return True

    def draw_halo_text(self, x, y, text, font, fill, halo=(255, 255, 255), stroke_w=3, anchor="mm", margin=4):
        tb = self.draw.textbbox((x, y), text, font=font, anchor=anchor, stroke_width=stroke_w)
        if not self.try_place_box(tb[0], tb[1], tb[2], tb[3], margin=margin):
            return False
        self.draw.text((x, y), text, font=font, fill=fill, anchor=anchor, stroke_width=stroke_w, stroke_fill=halo)
        return True

    def draw_road_label(self, img, cx, cy, angle_deg, text, font, fill, halo=(255, 255, 255), cell_px=340):
        grid_key = (text.strip().lower(), int(cx // cell_px), int(cy // cell_px))
        if grid_key in self.same_road_cells:
            return False
        # Neighbor cell check to guarantee generous separation between identical street names
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (text.strip().lower(), int(cx // cell_px) + dx, int(cy // cell_px) + dy) in self.same_road_cells:
                    return False

        # Keep text always left-to-right readable
        if angle_deg > 85.0:
            angle_deg -= 180.0
        elif angle_deg < -85.0:
            angle_deg += 180.0

        tmp = Image.new("RGBA", (10, 10))
        tdraw = ImageDraw.Draw(tmp)
        l, t, r, b = tdraw.textbbox((0, 0), text, font=font, stroke_width=3)
        tw, th = (r - l) + 12, (b - t) + 10
        rad = abs(math.radians(angle_deg))
        rw = int(math.ceil(tw * math.cos(rad) + th * math.sin(rad))) + 4
        rh = int(math.ceil(tw * math.sin(rad) + th * math.cos(rad))) + 4

        x0, y0 = int(cx - rw * 0.5), int(cy - rh * 0.5)
        if not self.try_place_box(x0, y0, x0 + rw, y0 + rh, margin=6):
            return False

        patch = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(patch)
        pdraw.text((tw * 0.5, th * 0.5), text, font=font, fill=fill + (255,), anchor="mm",
                   stroke_width=3, stroke_fill=halo + (245,))
        rot = patch.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
        px = int(cx - rot.width * 0.5)
        py = int(cy - rot.height * 0.5)
        img.paste(rot, (px, py), rot)
        self.same_road_cells.add(grid_key)
        return True


def render_map(data: GISWorldData, bbox, w, h, out_path, title, is_detail=False, supersample=1):
    t0 = time.time()
    rw, rh = w * supersample, h * supersample
    lon0, lat0, lon1, lat1 = bbox
    view_box = box(lon0, lat0, lon1, lat1)
    m_per_px = ((lon1 - lon0) * LON_M) / rw
    px_per_m = 1.0 / m_per_px

    # Scale helper: keeps strokes proportional cleanly across zooms & supersampling
    def sw_road(meters, min_px):
        return max(min_px * supersample, int(round(meters * px_per_m)))

    print(f"\nRendering {os.path.basename(out_path)} ({rw}x{rh} -> {w}x{h}, {m_per_px*supersample:.2f} m/px)...", flush=True)

    # 1. Exact Sea/Land Base + 30m DEM Relief Shading on Land
    land_mask_img = Image.new("L", (rw, rh), 0)
    lm_draw = ImageDraw.Draw(land_mask_img)
    for lp in data.land_polys:
        if lp.intersects(view_box):
            draw_poly_with_holes(land_mask_img, lm_draw, lp, bbox, rw, rh, fill=255, outline=None)

    land_mask = np.asarray(land_mask_img, dtype=np.uint8) > 127
    elev = data.dem.sample_grid(bbox, rw, rh)

    # Analytical NW hillshade
    dy, dx = np.gradient(elev, m_per_px)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dy, dx)
    az = math.radians(315.0)
    alt = math.radians(45.0)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip((hs - 0.707) * 28.0, -18.0, 14.0)  # subtle relief shading

    # Warm hypsometric elevation shift on foothills/mountains
    elev_clip = np.clip(elev, 0.0, 1000.0)
    base_rgb = np.empty((rh, rw, 3), dtype=np.float32)
    base_rgb[:, :, :] = COL_SEA
    # Coastal shallows subtle gradient away from shoreline using Gaussian blur on land_mask
    coastal_dist = np.asarray(land_mask_img.filter(ImageFilter.GaussianBlur(radius=max(3, int(8 * supersample)))), dtype=np.float32) / 255.0
    for c in range(3):
        base_rgb[:, :, c] = COL_SEA[c] * (1.0 - coastal_dist * 0.65) + COL_SEA_SHALLOW[c] * (coastal_dist * 0.65)

    land_r = COL_LAND_BASE[0] + hs - elev_clip * 0.016
    land_g = COL_LAND_BASE[1] + hs - elev_clip * 0.020
    land_b = COL_LAND_BASE[2] + hs - elev_clip * 0.026
    base_rgb[land_mask, 0] = np.clip(land_r[land_mask], 180, 250)
    base_rgb[land_mask, 1] = np.clip(land_g[land_mask], 175, 248)
    base_rgb[land_mask, 2] = np.clip(land_b[land_mask], 165, 240)

    # Subtle contour lines on hills (>= 50m intervals on master, >= 25m on detail)
    c_step = 25.0 if is_detail else 75.0
    q = np.floor(np.maximum(elev, 0.0) / c_step)
    contour_mask = np.zeros((rh, rw), dtype=bool)
    contour_mask[:-1, :] |= (q[:-1, :] != q[1:, :])
    contour_mask[:, :-1] |= (q[:, :-1] != q[:, 1:])
    contour_mask &= land_mask & (elev > 15.0)
    base_rgb[contour_mask, :] = base_rgb[contour_mask, :] * 0.90 + np.array([150, 140, 120]) * 0.10

    img = Image.new("RGB", (rw, rh), COL_LAND_BASE)
    draw = ImageDraw.Draw(img)

    # 2. Natural Land Cover (z>=8 sorted cleanly so zero quadkey tile seams exist!)
    for st, g in data.land_cover:
        if st in LANDCOVER_STYLE and g.intersects(view_box):
            draw_poly_with_holes(img, draw, g, bbox, rw, rh, fill=LANDCOVER_STYLE[st], outline=None)

    # 3. Land Use (parks, gardens, pitches, playgrounds, residential complexes, schools, industrial, farmland)
    for _, key, _, g in data.land_use:
        if g.intersects(view_box):
            fill, out = LANDUSE_STYLE[key]
            draw_poly_with_holes(img, draw, g, bbox, rw, rh, fill=fill, outline=out, outline_w=max(1, supersample))

    # Composite 30m DEM shaded relief + subtle contour lines ON ALL LAND (over landcover/landuse), and crisp sea outside!
    land_np = np.asarray(img, dtype=np.float32)
    land_np[land_mask, 0] = np.clip(land_np[land_mask, 0] + hs[land_mask] - elev_clip[land_mask] * 0.012, 120, 252)
    land_np[land_mask, 1] = np.clip(land_np[land_mask, 1] + hs[land_mask] - elev_clip[land_mask] * 0.015, 120, 250)
    land_np[land_mask, 2] = np.clip(land_np[land_mask, 2] + hs[land_mask] - elev_clip[land_mask] * 0.020, 110, 245)
    land_np[contour_mask, :] = land_np[contour_mask, :] * 0.88 + np.array([138.0, 126.0, 108.0]) * 0.12
    land_np[~land_mask] = base_rgb[~land_mask]
    img = Image.fromarray(land_np.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    # 4. Crisp Shoreline & Piers / Maritime
    for lp in data.land_polys:
        if lp.intersects(view_box):
            draw_poly_with_holes(img, draw, lp, bbox, rw, rh, fill=None, outline=COL_SHORELINE, outline_w=max(1, int(1.5 * supersample)))

    for g in data.piers:
        if g.intersects(view_box):
            if g.geom_type in ("Polygon", "MultiPolygon"):
                draw_poly_with_holes(img, draw, g, bbox, rw, rh, fill=hex_rgb("#dfd9ce"), outline=hex_rgb("#9e978b"))
            elif g.geom_type == "LineString":
                pts = project_coords(g.coords, bbox, rw, rh)
                if len(pts) >= 2:
                    draw.line(pts, fill=hex_rgb("#c8c1b4"), width=max(2, int(2.5 * supersample)))

    # 5. Inland Water, Salt Ponds (Çamaltı Tuzlası), Rivers, Streams, Swimming Pools
    for g in data.salt_ponds:
        if g.intersects(view_box):
            draw_poly_with_holes(img, draw, g, bbox, rw, rh, fill=hex_rgb("#9bd1e6"), outline=hex_rgb("#7bb3ca"), outline_w=max(1, supersample))

    for st, cl, g in data.water_polys:
        if g.intersects(view_box):
            col = hex_rgb("#7ec4e8") if cl == "swimming_pool" else hex_rgb("#8ec8e8")
            out = hex_rgb("#5ca3c9") if cl in ("swimming_pool", "pond", "reservoir", "basin") else None
            draw_poly_with_holes(img, draw, g, bbox, rw, rh, fill=col, outline=out, outline_w=max(1, supersample))

    for st, cl, g in data.water_lines:
        if g.intersects(view_box):
            pts = project_coords(g.coords, bbox, rw, rh)
            if len(pts) >= 2:
                wt = sw_road(10 if st == "river" else 4, 2 if st == "river" else 1)
                draw.line(pts, fill=hex_rgb("#82bfe0"), width=wt, joint="curve")

    # 6. Airports (runways, taxiways, aprons)
    for cl, g in data.airport_polys:
        if g.intersects(view_box):
            col = hex_rgb("#c5c9d1") if cl in ("apron", "runway", "taxiway") else hex_rgb("#dadce0")
            draw_poly_with_holes(img, draw, g, bbox, rw, rh, fill=col, outline=hex_rgb("#9aa0ac"))
    for cl, g in data.airport_lines:
        if g.intersects(view_box):
            pts = project_coords(g.coords, bbox, rw, rh)
            if len(pts) >= 2:
                if cl == "runway":
                    rw_w = sw_road(48, 7)
                    draw.line(pts, fill=hex_rgb("#686e7a"), width=rw_w)
                    draw.line(pts, fill=hex_rgb("#e5e7eb"), width=max(1, rw_w // 10))
                elif cl == "taxiway":
                    draw.line(pts, fill=hex_rgb("#a5abb5"), width=sw_road(18, 3))

    # 7. Mapped Individual Trees & Tree Rows
    if is_detail:
        tr_r = max(2, int(2.2 * supersample))
        for tx, ty in data.trees:
            if lon0 <= tx <= lon1 and lat0 <= ty <= lat1:
                px = (tx - lon0) / (lon1 - lon0) * rw
                py = (lat1 - ty) / (lat1 - lat0) * rh
                draw.ellipse([px - tr_r, py - tr_r, px + tr_r, py + tr_r], fill=hex_rgb("#87be73"), outline=hex_rgb("#6aa158"))
        for coords in data.tree_rows:
            pts = project_coords(coords, bbox, rw, rh)
            if len(pts) >= 2:
                draw.line(pts, fill=hex_rgb("#89bf75"), width=max(3, int(3.5 * supersample)), joint="curve")

    # 8. Underground Subsurface Layer (Tunnels: İzmir Metro underground, Karşıyaka İZBAN Tunnel, Konak/Bayraklı Road Tunnels)
    # Drawn BELOW surface streets/buildings with crisp subtle dashed strokes so they NEVER collide/confuse with surface roads!
    def draw_dashed_line(coords_pts, color, width, dash_px=12, gap_px=10):
        for (x0, y0), (x1, y1) in zip(coords_pts[:-1], coords_pts[1:]):
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 1.0:
                continue
            ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
            d = 0.0
            while d < seg_len:
                d_end = min(seg_len, d + dash_px)
                draw.line([(x0 + ux * d, y0 + uy * d), (x0 + ux * d_end, y0 + uy * d_end)], fill=color, width=width)
                d += dash_px + gap_px

    for cl, g, _lvl in data.road_tunnels:
        if g.intersects(view_box):
            pts = project_coords(g.coords, bbox, rw, rh)
            if len(pts) >= 2:
                wt = sw_road(12 if cl in ("motorway", "trunk", "primary") else 6, 2)
                draw_dashed_line(pts, hex_rgb("#b8ada4"), wt, dash_px=10 * supersample, gap_px=7 * supersample)

    for cl, g in data.road_under_construction:
        if g.intersects(view_box):
            pts = project_coords(g.coords, bbox, rw, rh)
            if len(pts) >= 2:
                draw_dashed_line(pts, hex_rgb("#cbb89d"), sw_road(8, 2), dash_px=8 * supersample, gap_px=8 * supersample)

    # 9. Surface Roads (Ordered cleanly by hierarchy: paths/steps -> service -> residential -> tertiary -> secondary -> primary -> trunk -> motorway)
    ROAD_ORDER = [
        ("path", 1.8, 1, hex_rgb("#cbbba8"), None, "dash"),
        ("track", 2.8, 1, hex_rgb("#c4ae92"), None, "dash"),
        ("footway", 2.2, 1, hex_rgb("#e0cfc0"), hex_rgb("#bca996"), "solid" if is_detail else "skip"),
        ("cycleway", 2.6, 1, hex_rgb("#79b8a8"), None, "solid"),
        ("steps", 3.0, 2, hex_rgb("#c78d6e"), None, "steps"),
        ("service", 4.5, 1, hex_rgb("#f7f5f0"), hex_rgb("#cbc4b8"), "solid"),
        ("pedestrian", 6.5, 2, hex_rgb("#f2ede3"), hex_rgb("#c4bcb0"), "solid"),
        ("living_street", 6.5, 2, hex_rgb("#fcfaf5"), hex_rgb("#c2bbae"), "solid"),
        ("unclassified", 7.2, 2, hex_rgb("#ffffff"), hex_rgb("#beb7aa"), "solid"),
        ("residential", 7.5, 2, hex_rgb("#ffffff"), hex_rgb("#beb7aa"), "solid"),
        ("tertiary", 10.0, 3, hex_rgb("#fffdf2"), hex_rgb("#b0a898"), "solid"),
        ("secondary", 12.5, 4, hex_rgb("#fded9c"), hex_rgb("#c2a64d"), "solid"),
        ("primary", 15.0, 5, hex_rgb("#fdc978"), hex_rgb("#bf8135"), "solid"),
        ("trunk", 18.0, 6, hex_rgb("#f79f70"), hex_rgb("#b35a30"), "solid"),
        ("motorway", 21.0, 7, hex_rgb("#ef7d57"), hex_rgb("#9e3f20"), "solid"),
    ]

    # Pre-clip surface segments to viewport for super fast casing+fill passes
    visible_surf = {}
    for cl, wm, min_px, fill_c, case_c, mode in ROAD_ORDER:
        if mode == "skip":
            continue
        segs = data.road_surface.get(cl, [])
        v_list = []
        for g, nm, is_lnk, _lvl in segs:
            if g.intersects(view_box):
                pts = project_coords(g.coords, bbox, rw, rh)
                if len(pts) >= 2:
                    v_list.append((pts, nm, is_lnk, g))
        visible_surf[cl] = v_list

    # Helper to draw road pass smoothly
    def render_road_list(road_items, wm, min_px, fill_c, case_c, mode):
        if not road_items:
            return
        w_base = sw_road(wm, min_px)
        w_link = max(max(1, w_base - 2 * supersample), int(w_base * 0.72))
        if mode == "dash":
            for pts, _nm, _lnk, _g in road_items:
                draw_dashed_line(pts, fill_c, w_base, dash_px=7 * supersample, gap_px=5 * supersample)
            return
        if mode == "steps":
            for pts, _nm, _lnk, _g in road_items:
                draw.line(pts, fill=hex_rgb("#e5d9cc"), width=w_base + 2 * supersample)
                draw_dashed_line(pts, fill_c, w_base, dash_px=2 * supersample, gap_px=3 * supersample)
            return
        # Casing pass first, then fill pass (merges junctions cleanly without internal stroke seams!)
        if case_c is not None:
            case_pad = max(2, int(1.6 * supersample))
            for pts, _nm, is_lnk, _g in road_items:
                wb = w_link if is_lnk else w_base
                draw.line(pts, fill=case_c, width=wb + case_pad * 2, joint="curve")
        for pts, _nm, is_lnk, _g in road_items:
            wb = w_link if is_lnk else w_base
            draw.line(pts, fill=fill_c, width=wb, joint="curve")

    # Draw minor roads first, then major arterials; on each level casing -> fill cleanly merges intersections
    for cl, wm, min_px, fill_c, case_c, mode in ROAD_ORDER:
        if cl in visible_surf:
            render_road_list(visible_surf[cl], wm, min_px, fill_c, case_c, mode)

    # 10. Buildings (All 450,921 conflated OSM + Microsoft ML footprints!)
    # Placed AFTER surface roads & BEFORE bridges/elevated rails so houses never overlap roads while viaducts fly over!
    sh_dx = int(1.8 * supersample) if is_detail else 0
    vis_buildings = []
    for cat, h_val, name, g, cx, cy in data.buildings:
        bx0, by0, bx1, by1 = g.bounds
        if not (bx1 < lon0 or bx0 > lon1 or by1 < lat0 or by0 > lat1):
            vis_buildings.append((cat, g))

    if is_detail and sh_dx > 0:
        sh_col = hex_rgb("#c2baac")
        for _cat, g in vis_buildings:
            if g.geom_type == "Polygon":
                pts = project_coords(g.exterior.coords, bbox, rw, rh)
                if len(pts) >= 3:
                    draw.polygon([(x + sh_dx, y + sh_dx) for x, y in pts], fill=sh_col)
            elif g.geom_type == "MultiPolygon":
                for p in g.geoms:
                    pts = project_coords(p.exterior.coords, bbox, rw, rh)
                    if len(pts) >= 3:
                        draw.polygon([(x + sh_dx, y + sh_dx) for x, y in pts], fill=sh_col)

    out_w = max(1, supersample) if is_detail else 0
    for cat, g in vis_buildings:
        fill_c, out_c = BLDG_COLORS.get(cat, BLDG_COLORS["default"])
        draw_poly_with_holes(img, draw, g, bbox, rw, rh, fill=fill_c,
                             outline=(out_c if is_detail else None), outline_w=out_w)

    # 11. Active Railways (Tunnels subtle dashed vs Surface active railway ties vs Bridges later!)
    def draw_rail_track(pts, cl, is_tunnel=False, is_bridge=False):
        if len(pts) < 2:
            return
        if is_tunnel:
            if cl == "subway":
                draw_dashed_line(pts, hex_rgb("#2563eb"), max(2, int(2.6 * supersample)),
                                 dash_px=9 * supersample, gap_px=7 * supersample)
            elif cl == "standard_gauge":
                draw_dashed_line(pts, hex_rgb("#475569"), max(2, int(2.8 * supersample)),
                                 dash_px=10 * supersample, gap_px=7 * supersample)
            return
        if cl == "standard_gauge":
            w_out = sw_road(5.5, 3)
            w_in = max(2, w_out - 2 * supersample)
            if is_bridge:
                draw.line(pts, fill=hex_rgb("#334155"), width=w_out + 4 * supersample, joint="curve")
            draw.line(pts, fill=hex_rgb("#3f4959"), width=w_out, joint="curve")
            draw_dashed_line(pts, hex_rgb("#f8fafc"), w_in, dash_px=7 * supersample, gap_px=7 * supersample)
        elif cl == "tram":
            w_tr = sw_road(3.5, 2)
            draw.line(pts, fill=hex_rgb("#047857"), width=w_tr + 2 * supersample, joint="curve")
            draw.line(pts, fill=hex_rgb("#34d399"), width=w_tr, joint="curve")
        elif cl == "subway":
            w_sb = sw_road(4.5, 2)
            draw.line(pts, fill=hex_rgb("#1e3a8a"), width=w_sb + 2 * supersample, joint="curve")
            draw.line(pts, fill=hex_rgb("#3b82f6"), width=w_sb, joint="curve")

    # Subsurface rail tunnels (drawn softly AFTER buildings? Wait: slightly translucent/clean dashed AFTER buildings so users can trace Metro/Karşıyaka Tunnel corridor without looking like surface rails crashing over roads — using crisp thin dashed line!)
    for cl, fset, lvl, nm, g in data.rails:
        if ("is_tunnel" in fset or lvl < 0) and g.intersects(view_box):
            draw_rail_track(project_coords(g.coords, bbox, rw, rh), cl, is_tunnel=True)

    # Surface rails (level == 0)
    for cl, fset, lvl, nm, g in data.rails:
        if "is_tunnel" not in fset and "is_bridge" not in fset and lvl == 0 and g.intersects(view_box):
            draw_rail_track(project_coords(g.coords, bbox, rw, rh), cl, is_tunnel=False, is_bridge=False)

    # 12. Bridges, Viaducts & Overpasses (sorted by elevation level 1, 2, 3 -> cleanly cross over surface roads/rails!)
    ROAD_STYLE_MAP = {row[0]: (row[1], row[2], row[3], row[4]) for row in ROAD_ORDER}
    # Group bridges by level
    for lvl_val in sorted(set(b[0] for b in data.road_bridges)):
        lvl_bridges = [b for b in data.road_bridges if b[0] == lvl_val and b[2].intersects(view_box)]
        # Dark bridge parapet casing first
        for _, cl, g, nm, is_lnk in lvl_bridges:
            pts = project_coords(g.coords, bbox, rw, rh)
            if len(pts) < 2:
                continue
            wm, min_px, fill_c, case_c = ROAD_STYLE_MAP.get(cl, (7.5, 2, hex_rgb("#ffffff"), hex_rgb("#b0a898")))
            wb = sw_road(wm, min_px)
            if is_lnk:
                wb = max(max(1, wb - 2 * supersample), int(wb * 0.72))
            draw.line(pts, fill=hex_rgb("#6c665c"), width=wb + 4 * supersample, joint="curve")
        for _, cl, g, nm, is_lnk in lvl_bridges:
            pts = project_coords(g.coords, bbox, rw, rh)
            if len(pts) < 2:
                continue
            wm, min_px, fill_c, case_c = ROAD_STYLE_MAP.get(cl, (7.5, 2, hex_rgb("#ffffff"), hex_rgb("#b0a898")))
            wb = sw_road(wm, min_px)
            if is_lnk:
                wb = max(max(1, wb - 2 * supersample), int(wb * 0.72))
            draw.line(pts, fill=fill_c, width=wb, joint="curve")

    # Rail viaduct bridges on top
    for cl, fset, lvl, nm, g in data.rails:
        if ("is_bridge" in fset or lvl > 0) and g.intersects(view_box):
            draw_rail_track(project_coords(g.coords, bbox, rw, rh), cl, is_tunnel=False, is_bridge=True)

    # Downsample supersampled base map cleanly via LANCZOS BEFORE drawing razor-sharp vector text & UI!
    if supersample > 1:
        img = img.resize((w, h), resample=Image.LANCZOS)
        draw = ImageDraw.Draw(img)

    # 13. Crisp Stations, Landmarks & Smart Non-Overlapping Labels (at native output resolution!)
    f_county = ImageFont.truetype(FONT_BOLD, 21 if not is_detail else 26)
    f_mahalle = ImageFont.truetype(FONT_BOLD, 12 if not is_detail else 16)
    f_sea = ImageFont.truetype(FONT_ITAL, 28 if not is_detail else 26)
    f_peak = ImageFont.truetype(FONT_ITAL, 12 if not is_detail else 15)
    f_st = ImageFont.truetype(FONT_BOLD, 11 if not is_detail else 13)
    f_lm = ImageFont.truetype(FONT_BOLD, 11 if not is_detail else 13)
    f_rd_maj = ImageFont.truetype(FONT_BOLD, 11 if not is_detail else 13)
    f_rd_min = ImageFont.truetype(FONT_REG, 10 if not is_detail else 11)

    bar_h = 48 if is_detail else 104
    leg_h = 42 if is_detail else 78
    lp = LabelPlacer(w, h, draw, y_min=bar_h + 8, y_max=h - leg_h - 8)

    def to_xy(lon, lat):
        return (lon - lon0) / (lon1 - lon0) * w, (lat1 - lat) / (lat1 - lat0) * h

    # Water Labels
    water_labels = [
        (26.985, 38.445, "İZMİR KÖRFEZİ"),
        (26.905, 38.485, "Gediz Deltası & Çamaltı Tuzlası"),
    ]
    for lx, ly, txt in water_labels:
        if lon0 <= lx <= lon1 and lat0 <= ly <= lat1:
            px, py = to_xy(lx, ly)
            lp.draw_halo_text(px, py, txt, f_sea, hex_rgb("#1c587a"), halo=hex_rgb("#cde7f5"), stroke_w=3, margin=10)

    # Counties (İlçeler)
    for lx, ly, name in data.counties:
        if lon0 + 0.01 <= lx <= lon1 - 0.01 and lat0 + 0.01 <= ly <= lat1 - 0.01:
            px, py = to_xy(lx, ly)
            lp.draw_halo_text(px, py, name.upper(), f_county, hex_rgb("#262f3b"), halo=hex_rgb("#fffcf5"), stroke_w=4, margin=12)

    # Transit Stations (İZBAN, Metro, Ferry, Tram)
    st_prio = {"izban": 0, "metro": 1, "ferry": 2, "tram": 3}
    vis_stations = [s for s in data.stations if lon0 <= s[1] <= lon1 and lat0 <= s[2] <= lat1]
    vis_stations.sort(key=lambda s: st_prio.get(s[0], 9))
    for kind, sx, sy, sname in vis_stations:
        if kind == "tram" and not is_detail:
            # Tiny emerald dot on overview without cluttering labels
            px, py = to_xy(sx, sy)
            draw.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5], fill=hex_rgb("#10b981"), outline=hex_rgb("#064e3b"))
            continue
        px, py = to_xy(sx, sy)
        col_map = {
            "izban": (hex_rgb("#0369a1"), "İZBAN"),
            "metro": (hex_rgb("#1d4ed8"), "M"),
            "ferry": (hex_rgb("#0891b2"), "V"),
            "tram": (hex_rgb("#059669"), "T"),
        }
        bg_c, badge = col_map[kind]
        r = 5 if is_detail else 4
        draw.ellipse([px - r, py - r, px + r, py + r], fill=bg_c, outline=(255, 255, 255), width=2)
        lbl = f"{badge} {sname}" if kind in ("izban", "ferry") or is_detail else sname
        lp.draw_halo_text(px, py - r - 8, lbl, f_st, bg_c, halo=(255, 255, 255), stroke_w=3, margin=4)

    # Major Landmarks (Hospitals, Malls, Universities, Airports, Stadiums)
    for cat, lx, ly, lname, area in data.landmarks:
        if lon0 <= lx <= lon1 and lat0 <= ly <= lat1:
            px, py = to_xy(lx, ly)
            col = hex_rgb("#9f1239") if cat == "medical" else (hex_rgb("#9a3412") if cat in ("education", "mall") else hex_rgb("#374151"))
            lp.draw_halo_text(px, py, lname, f_lm, col, halo=(255, 253, 248), stroke_w=3, margin=5)

    # Mahalleler (Neighborhood names)
    mah_list = [m for m in data.mahalleler if lon0 <= m[1] <= lon1 and lat0 <= m[2] <= lat1]
    mah_list.sort(key=lambda m: 0 if m[0] == "macrohood" else 1)
    for _st, mx, my, mname in mah_list:
        px, py = to_xy(mx, my)
        lp.draw_halo_text(px, py, mname, f_mahalle, hex_rgb("#4b5563"), halo=(255, 255, 255), stroke_w=3, margin=8)

    # Peaks (vector triangle instead of font glyph!)
    for kx, ky, kname in data.peaks:
        if lon0 <= kx <= lon1 and lat0 <= ky <= lat1:
            px, py = to_xy(kx, ky)
            if py < bar_h + 16 or py > h - leg_h - 16:
                continue
            if lp.draw_halo_text(px, py - 9, kname, f_peak, hex_rgb("#5c4d3c"), halo=hex_rgb("#f7f4ec"), stroke_w=2, margin=8):
                draw.polygon([(px, py - 2), (px - 4, py + 4), (px + 4, py + 4)], fill=hex_rgb("#6b533b"), outline=(255, 255, 255))

    # Deduplicated Tangential Road Labels (Major roads first, then local streets on detail maps!)
    label_classes = ["motorway", "trunk", "primary", "secondary", "tertiary"]
    if is_detail:
        label_classes += ["residential", "living_street", "unclassified", "pedestrian"]

    for cl in label_classes:
        cand = []
        for g, nm, is_lnk, _lvl in data.road_surface.get(cl, []):
            if not nm or is_lnk or not g.intersects(view_box):
                continue
            cand.append((g.length, nm, g))
        cand.sort(key=lambda x: -x[0])  # Longest road segments first!
        is_maj = cl in ("motorway", "trunk", "primary", "secondary")
        font = f_rd_maj if is_maj else f_rd_min
        text_col = hex_rgb("#5c2811") if cl in ("motorway", "trunk") else (hex_rgb("#3f3526") if is_maj else hex_rgb("#475569"))
        cell_px = 360 if is_detail else 580

        for _ln, nm, g in cand:
            pts = project_coords(g.coords, bbox, w, h)
            if len(pts) < 2:
                continue
            # Sample midpoint along polyline arc
            mid_pt = g.interpolate(0.5, normalized=True)
            if not (lon0 <= mid_pt.x <= lon1 and lat0 <= mid_pt.y <= lat1):
                continue
            mx, my = to_xy(mid_pt.x, mid_pt.y)
            p_prev = g.interpolate(0.45, normalized=True)
            p_next = g.interpolate(0.55, normalized=True)
            x1, y1 = to_xy(p_prev.x, p_prev.y)
            x2, y2 = to_xy(p_next.x, p_next.y)
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            lp.draw_road_label(img, mx, my, ang, nm, font, text_col, cell_px=cell_px)

    # 14. Crisp SOLID Header Banner, Scale Bar & Cartographic Legend
    f_hdr = ImageFont.truetype(FONT_BOLD, 21 if is_detail else 44)
    f_sub = ImageFont.truetype(FONT_REG, 12 if is_detail else 23)
    f_leg = ImageFont.truetype(FONT_REG, 12 if is_detail else 22)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, 0, w, bar_h], fill=(22, 30, 42, 255))
    od.text((22, bar_h // 2 - (8 if is_detail else 16)), f"İZMİR GERÇEK HARİTASI  |  {title}",
            font=f_hdr, fill=(255, 255, 255, 255), anchor="lm")
    od.text((22, bar_h // 2 + (12 if is_detail else 22)),
            f"Overture Maps (OSM + Microsoft ML + TomTom) · 450,899 Bina · 138K Yol/Ray Segmenti · 30m Terrarium DEM",
            font=f_sub, fill=(196, 214, 232, 255), anchor="lm")

    # Bottom Legend + Dynamic Scale Bar
    od.rectangle([0, h - leg_h, w, h], fill=(248, 246, 240, 255))
    od.line([(0, h - leg_h), (w, h - leg_h)], fill=(170, 165, 155, 255), width=2)

    items = [
        ("#ef7d57", "Otoyol / Çevre Yolu"),
        ("#fdc978", "Ana Cadde"),
        ("#ffffff", "Ara Sokak"),
        ("#3f4959", "İZBAN / TCDD"),
        ("#10b981", "Tramvay"),
        ("#2563eb", "Metro (M1 Hat/Tünel)"),
        ("#b9e0a6", "Park & Bahçe"),
        ("#e2d4c2", "Konut"),
        ("#eed39d", "Okul"),
        ("#eeb5bc", "Hastane"),
        ("#c7c9ea", "Cami"),
        ("#9ecbe6", "Deniz / Akarsu"),
    ]
    # Dynamic Metric Scale Bar at Bottom Right (compute first so legend never collides)
    view_w_m = (lon1 - lon0) * LON_M
    target_m = 5000 if view_w_m > 20000 else (1000 if view_w_m > 4000 else 500)
    bar_px = int(round(target_m / view_w_m * w))
    bx1 = w - (36 if not is_detail else 22)
    bx0 = bx1 - bar_px
    by = h - leg_h // 2 + (10 if not is_detail else 5)
    tick_h = 8 if not is_detail else 5
    od.line([(bx0, by), (bx1, by)], fill=(30, 35, 45, 255), width=4 if not is_detail else 3)
    od.line([(bx0, by - tick_h), (bx0, by + 3)], fill=(30, 35, 45, 255), width=3 if not is_detail else 2)
    od.line([(bx1, by - tick_h), (bx1, by + 3)], fill=(30, 35, 45, 255), width=3 if not is_detail else 2)
    s_lbl = f"{target_m // 1000} km" if target_m >= 1000 else f"{target_m} m"
    od.text(((bx0 + bx1) // 2, by - (18 if not is_detail else 12)), s_lbl, font=f_leg, fill=(30, 35, 45, 255), anchor="mm")

    lx = 24 if not is_detail else 16
    ly = h - leg_h // 2
    sw_box = 15 if is_detail else 26
    for hex_c, label in items:
        c = hex_rgb(hex_c) + (255,)
        tb = od.textbbox((lx + sw_box + 6, ly), label, font=f_leg, anchor="lm")
        if tb[2] + 32 > bx0:
            break
        od.rectangle([lx, ly - sw_box // 2, lx + sw_box, ly + sw_box // 2], fill=c, outline=(90, 90, 90, 255), width=1)
        lx += sw_box + 6
        od.text((lx, ly), label, font=f_leg, fill=(32, 38, 46, 255), anchor="lm")
        lx = tb[2] + (14 if is_detail else 26)

    final_img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    final_img.save(out_path, optimize=True)
    if ARTIFACT_DIR and os.path.isdir(ARTIFACT_DIR):
        shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, os.path.basename(out_path)))
    print(f"  Saved {out_path} in {time.time() - t0:.1f}s", flush=True)


def main():
    data = GISWorldData()

    # Master Full Metro Map (5256 x 6000 -> ~8m/px across entire 41.8km x 47.8km Izmir region!)
    master_out = os.path.join(OUT_DIR, "izmir_master_real_map_v2_6k.png")
    render_map(
        data,
        MASTER_BBOX,
        w=5256,
        h=6000,
        out_path=master_out,
        title="METROPOL İZMİR TAM HARİTASI (Menemen ↔ Güzelbahçe / Menderes ↔ Kemalpaşa)",
        is_detail=False,
        supersample=1,
    )

    # High-res 2x Supersampled District Detail Maps
    for fname, title, bbox in DETAIL_VIEWS:
        lon0, lat0, lon1, lat1 = bbox
        aspect = ((lon1 - lon0) * LON_M) / ((lat1 - lat0) * LAT_M)
        h = 2000
        w = int(round(h * aspect))
        out_p = os.path.join(OUT_DIR, fname)
        render_map(data, bbox, w=w, h=h, out_path=out_p, title=title, is_detail=True, supersample=2)

    print("\n=== ALL V2 COMPLETE CARTOGRAPHIC MAPS RENDERED ===", flush=True)


if __name__ == "__main__":
    main()
