#!/usr/bin/env python3
"""
Izmir Tam Detayli Gercek Harita Cizici (OpenStreetMap + DEM).
Ara sokaklar, ev duzeni, bahceler, ozel binalar, IZBAN ve caddeleri
gercek OSM verisiyle yuksek cozunurlukte (4K Master + Bolgesel Detaylar) cizer.
"""

import glob
import json
import math
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CACHE_DIR = Path(__file__).resolve().parent / "osm_cache"
ARTIFACT_DIR = Path(os.environ.get("IZMIR_OUT_DIR", str(Path(__file__).resolve().parent / "renders")))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Font yukleyici
def get_font(size=14, bold=False):
    font_paths = [
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Bold.ttf" if bold else "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


class MapRenderer:
    def __init__(self, lat_min, lon_min, lat_max, lon_max, width, height, title=""):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.w = width
        self.h = height
        self.title = title

        # Arka plan: modern kartografik açik gri-bej (#f1f5f9)
        self.img = Image.new("RGBA", (width, height), (241, 245, 249, 255))
        self.draw = ImageDraw.Draw(self.img, "RGBA")

    def to_px(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * (self.w - 1)
        y = (self.lat_max - lat) / (self.lat_max - self.lat_min) * (self.h - 1)
        return x, y

    def in_bounds(self, lat, lon):
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    def draw_polygon(self, geom, fill, outline=None, width=1):
        pts = [self.to_px(p['lat'], p['lon']) for p in geom]
        if len(pts) >= 3:
            self.draw.polygon(pts, fill=fill, outline=outline, width=width)

    def draw_polyline(self, geom, stroke, width=1):
        pts = [self.to_px(p['lat'], p['lon']) for p in geom]
        if len(pts) >= 2:
            self.draw.line(pts, fill=stroke, width=width, joint="round")


def load_all_osm_data():
    print("[1/5] OSM veri dosyalari taranip yukleniyor...", flush=True)
    all_elements = []
    files = sorted(glob.glob(str(CACHE_DIR / "*.json")))
    for f in files:
        if "meta" in f:
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
                elems = d.get("elements", [])
                all_elements.extend(elems)
        except Exception as e:
            print(f"Hata {f}: {e}")
    print(f"  Toplam {len(all_elements)} OSM ogesi bellege alindi.", flush=True)
    return all_elements


def categorize_elements(elements):
    print("[2/5] Ogeler katmanlara ayriliyor...", flush=True)
    waterways = []
    coastlines = []
    landuse_green = []
    landuse_farmland = []
    landuse_industrial = []
    buildings_regular = []
    buildings_special = []
    roads_residential = []
    roads_secondary = []
    roads_primary = []
    roads_motorway = []
    railway_tracks = []
    stations = []
    places = []

    special_building_tags = {"school", "university", "college", "kindergarten", "hospital", "clinic",
                             "mosque", "church", "cathedral", "stadium", "sports_centre", "train_station",
                             "public", "government", "civic", "transportation"}

    for el in elements:
        tags = el.get("tags", {})
        geom = el.get("geometry", [])
        etype = el.get("type")

        # Nokta verileri: Istasyonlar ve Yerlesimler
        if etype == "node":
            lat = el.get("lat")
            lon = el.get("lon")
            if lat is not None and lon is not None:
                if tags.get("railway") in ("station", "halt", "tram_stop"):
                    stations.append((lat, lon, tags.get("name", ""), tags.get("railway")))
                elif "place" in tags:
                    places.append((lat, lon, tags.get("name", ""), tags.get("place")))
            continue

        if not geom:
            continue

        # Su / Kiyi
        if tags.get("natural") == "coastline":
            coastlines.append(geom)
        elif "waterway" in tags or tags.get("natural") in ("water", "wetland"):
            waterways.append(geom)

        # Arazi / Bahceler / Parklar
        leisure = tags.get("leisure", "")
        landuse = tags.get("landuse", "")
        natural = tags.get("natural", "")

        if leisure in ("park", "garden", "pitch", "playground", "recreation_ground", "nature_reserve") or \
           landuse in ("forest", "grass", "meadow", "orchard", "village_green", "cemetery") or \
           natural in ("wood", "scrub"):
            landuse_green.append((geom, leisure or landuse or natural))
        elif landuse in ("farmland", "farmyard"):
            landuse_farmland.append(geom)
        elif landuse in ("industrial", "commercial", "retail", "salt_pond"):
            landuse_industrial.append(geom)

        # Binalar (Evler, Siteler, Ozel Yapilar)
        if "building" in tags:
            btype = tags.get("building")
            amenity = tags.get("amenity", "")
            if btype in special_building_tags or amenity in special_building_tags or tags.get("name"):
                buildings_special.append((geom, tags.get("name", ""), btype or amenity))
            else:
                buildings_regular.append(geom)

        # Demiryolu / IZBAN
        if "railway" in tags and tags["railway"] in ("rail", "subway", "light_rail", "tram"):
            railway_tracks.append((geom, tags.get("railway"), tags.get("name", "")))

        # Yollar
        highway = tags.get("highway", "")
        if highway:
            if highway in ("motorway", "motorway_link", "trunk", "trunk_link"):
                roads_motorway.append((geom, tags.get("name", ""), highway))
            elif highway in ("primary", "primary_link"):
                roads_primary.append((geom, tags.get("name", ""), highway))
            elif highway in ("secondary", "secondary_link", "tertiary", "tertiary_link"):
                roads_secondary.append((geom, tags.get("name", ""), highway))
            elif highway in ("residential", "living_street", "unclassified", "pedestrian", "service", "footway", "path"):
                roads_residential.append((geom, tags.get("name", ""), highway))

    print(f"  Ayrilan katmanlar:\n"
          f"    - Binalar: {len(buildings_regular)} normal ev/bina, {len(buildings_special)} ozel yapilar\n"
          f"    - Yesil alan / bahce: {len(landuse_green)}\n"
          f"    - Yollar: {len(roads_residential)} ara sokak, {len(roads_secondary)} cadde, {len(roads_primary)} ana yol, {len(roads_motorway)} otoyol\n"
          f"    - Rayli sistem / IZBAN: {len(railway_tracks)} hat, {len(stations)} istasyon\n"
          f"    - Yerlesim / ilce: {len(places)}", flush=True)

    return {
        "coastlines": coastlines,
        "waterways": waterways,
        "landuse_green": landuse_green,
        "landuse_farmland": landuse_farmland,
        "landuse_industrial": landuse_industrial,
        "buildings_regular": buildings_regular,
        "buildings_special": buildings_special,
        "roads_residential": roads_residential,
        "roads_secondary": roads_secondary,
        "roads_primary": roads_primary,
        "roads_motorway": roads_motorway,
        "railway_tracks": railway_tracks,
        "stations": stations,
        "places": places,
    }


def render_map(data, lat_min, lon_min, lat_max, lon_max, width, height, title, out_path, detail_mode=False):
    print(f"\n[HARITA CIZIMI] {title} ({width}x{height}) ciziliyor -> {out_path.name}...", flush=True)
    renderer = MapRenderer(lat_min, lon_min, lat_max, lon_max, width, height, title)
    draw = renderer.draw

    # 1. Deniz / Korfez dolgusu (Kiyi poligonlari veya acik deniz)
    # Izmir Korfezi su rengi: #38bdf8 (derin su) / #bae6fd (sig su)
    sea_color = (186, 230, 253, 255)
    deep_sea = (56, 189, 248, 255)

    # Korfez alanini yaklasik kiyi veya su poligonlariyla ciz
    for geom in data["waterways"]:
        if len(geom) >= 3 and geom[0] == geom[-1]:
            renderer.draw_polygon(geom, fill=sea_color, outline=(56, 189, 248, 180), width=1)
        else:
            renderer.draw_polyline(geom, stroke=deep_sea, width=3 if detail_mode else 2)

    for geom in data["coastlines"]:
        renderer.draw_polyline(geom, stroke=(14, 165, 233, 255), width=3 if detail_mode else 2)

    # 2. Sanayi ve Tarim alanlari
    for geom in data["landuse_farmland"]:
        renderer.draw_polygon(geom, fill=(254, 243, 199, 140))  # acik saman sarisi
    for geom in data["landuse_industrial"]:
        renderer.draw_polygon(geom, fill=(226, 232, 240, 160))  # gri sanayi

    # 3. Gercek Yesil Alanlar, Bahceler, Parklar, Korular
    for geom, ltype in data["landuse_green"]:
        if "garden" in ltype or "park" in ltype or "playground" in ltype:
            # Parklar ve konut ici bahceler: canli taze yesil
            renderer.draw_polygon(geom, fill=(134, 239, 172, 190), outline=(74, 222, 128, 220), width=1)
        elif "forest" in ltype or "wood" in ltype:
            # Orman ve koruluklar: zengin derin orman yesili
            renderer.draw_polygon(geom, fill=(187, 247, 208, 170), outline=(34, 197, 94, 200), width=1)
        else:
            renderer.draw_polygon(geom, fill=(220, 252, 231, 150))

    # 4. Normal Konut ve Binalar (Ev duzeni)
    # detail_mode'da her evin kati/izdusumu net kirmizi/kiremit rengi catilarla
    bldg_fill = (254, 202, 202, 220) if detail_mode else (226, 232, 240, 180)
    bldg_stroke = (239, 68, 68, 160) if detail_mode else (148, 163, 184, 120)
    for geom in data["buildings_regular"]:
        renderer.draw_polygon(geom, fill=bldg_fill, outline=bldg_stroke, width=1)

    # 5. Ozel Binalar (Okullar, Camiler, Hastaneler, Kamu, Siteler)
    for geom, name, btype in data["buildings_special"]:
        if "school" in btype or "university" in btype:
            c_fill, c_stroke = (191, 219, 254, 255), (59, 130, 246, 255)  # Mavi okul
        elif "mosque" in btype or "church" in btype:
            c_fill, c_stroke = (254, 240, 138, 255), (202, 138, 4, 255)   # Altin cami/mabed
        elif "hospital" in btype or "clinic" in btype:
            c_fill, c_stroke = (254, 205, 211, 255), (225, 29, 72, 255)   # Pembe/Kirmizi saglik
        elif "stadium" in btype or "sports" in btype:
            c_fill, c_stroke = (167, 243, 208, 255), (5, 150, 105, 255)   # Yesil spor
        else:
            c_fill, c_stroke = (253, 186, 116, 255), (234, 88, 12, 255)   # Turuncu ozel kamu

        renderer.draw_polygon(geom, fill=c_fill, outline=c_stroke, width=2 if detail_mode else 1)

    # 6. Yollar Hiyerarsisi
    # 6a. Ara Sokaklar, Cikmazlar, Yaya Yollari
    st_w = 2 if detail_mode else 1
    st_col = (148, 163, 184, 220) if detail_mode else (203, 213, 225, 160)
    for geom, name, hw in data["roads_residential"]:
        renderer.draw_polyline(geom, stroke=st_col, width=st_w)

    # 6b. Ikincil Yollar / Semt Caddeleri
    sec_w = 4 if detail_mode else 2
    for geom, name, hw in data["roads_secondary"]:
        renderer.draw_polyline(geom, stroke=(100, 116, 139, 255), width=sec_w)
        if detail_mode and sec_w >= 4:
            renderer.draw_polyline(geom, stroke=(248, 250, 252, 255), width=sec_w - 2)

    # 6c. Ana Caddeler (Primary)
    pri_w = 6 if detail_mode else 3
    for geom, name, hw in data["roads_primary"]:
        renderer.draw_polyline(geom, stroke=(51, 65, 85, 255), width=pri_w + 2)
        renderer.draw_polyline(geom, stroke=(251, 191, 36, 255), width=pri_w)

    # 6d. Otoyollar ve Cevre Yollari (Motorway / Trunk)
    hwy_w = 8 if detail_mode else 4
    for geom, name, hw in data["roads_motorway"]:
        renderer.draw_polyline(geom, stroke=(30, 41, 59, 255), width=hwy_w + 3)
        renderer.draw_polyline(geom, stroke=(249, 115, 22, 255), width=hwy_w)

    # 7. IZBAN & Demiryolu AGI (Yollardan tamamen ayrik, belirgin cizim)
    # IZBAN koridoru alt siyah kalin serit, ust kesikli kirmizi-beyaz ray
    rail_w = 5 if detail_mode else 3
    for geom, rtype, name in data["railway_tracks"]:
        if rtype in ("rail", "subway", "light_rail"):
            # IZBAN / TCDD ana hat
            renderer.draw_polyline(geom, stroke=(15, 23, 42, 255), width=rail_w + 2)
            renderer.draw_polyline(geom, stroke=(220, 38, 38, 255), width=rail_w)
        else:
            # Tramvay
            renderer.draw_polyline(geom, stroke=(6, 182, 212, 255), width=rail_w - 1)

    # 8. IZBAN Istasyon Noktalari ve Isimleri
    font_station = get_font(13 if detail_mode else 11, bold=True)
    for lat, lon, sname, stype in data["stations"]:
        if renderer.in_bounds(lat, lon):
            px, py = renderer.to_px(lat, lon)
            r = 6 if detail_mode else 4
            # Istasyon rozeti
            draw.ellipse([px - r - 2, py - r - 2, px + r + 2, py + r + 2], fill=(15, 23, 42, 255))
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(239, 68, 68, 255), outline=(255, 255, 255, 255), width=2)
            if sname:
                # Metin arka plan kutusu
                tw = draw.textlength(sname, font=font_station)
                draw.rectangle([px + 8, py - 9, px + 12 + tw, py + 8], fill=(255, 255, 255, 230), outline=(220, 38, 38, 255), width=1)
                draw.text((px + 10, py - 8), sname, fill=(15, 23, 42, 255), font=font_station)

    # 9. Cadde ve Sokak Isimleri (Detail modunda)
    if detail_mode:
        font_street = get_font(11, bold=True)
        # Onemli caddeleri etiketle
        labeled_roads = data["roads_primary"] + data["roads_secondary"] + data["roads_motorway"]
        for geom, name, hw in labeled_roads:
            if name and len(geom) >= 2:
                mid_idx = len(geom) // 2
                pt = geom[mid_idx]
                if renderer.in_bounds(pt['lat'], pt['lon']):
                    px, py = renderer.to_px(pt['lat'], pt['lon'])
                    tw = draw.textlength(name, font=font_street)
                    draw.rectangle([px - tw/2 - 3, py - 7, px + tw/2 + 3, py + 7],
                                   fill=(255, 255, 255, 210), outline=(100, 116, 139, 180), width=1)
                    draw.text((px - tw/2, py - 6), name, fill=(30, 41, 59, 255), font=font_street)

        # Ozel binalarin isimlerini etiketle (okullar, camiler vb)
        font_bldg = get_font(10, bold=False)
        for geom, name, btype in data["buildings_special"][:150]:
            if name and len(geom) >= 1:
                pt = geom[0]
                if renderer.in_bounds(pt['lat'], pt['lon']):
                    px, py = renderer.to_px(pt['lat'], pt['lon'])
                    tw = draw.textlength(name, font=font_bldg)
                    draw.rectangle([px - tw/2 - 2, py - 6, px + tw/2 + 2, py + 6],
                                   fill=(254, 240, 138, 220), outline=(202, 138, 4, 200), width=1)
                    draw.text((px - tw/2, py - 5), name, fill=(113, 63, 18, 255), font=font_bldg)

    # 10. Ilce ve Semt Isimleri (Places)
    font_city = get_font(20, bold=True)
    font_suburb = get_font(14, bold=True)
    for lat, lon, pname, ptype in data["places"]:
        if renderer.in_bounds(lat, lon):
            px, py = renderer.to_px(lat, lon)
            f = font_city if ptype in ("city", "town") else font_suburb
            color = (15, 23, 42, 255)
            tw = draw.textlength(pname, font=f)
            draw.rectangle([px - tw/2 - 4, py - 10, px + tw/2 + 4, py + 10], fill=(255, 255, 255, 220), outline=(15, 23, 42, 100), width=1)
            draw.text((px - tw/2, py - 8), pname, fill=color, font=f)

    # 11. Baslik ve Bilgi Lejanti (Header Banner & Legend)
    draw.rectangle([0, 0, width, 55], fill=(15, 23, 42, 245))
    font_title = get_font(20, bold=True)
    font_meta = get_font(12, bold=False)
    draw.text((25, 10), title, fill=(255, 255, 255, 255), font=font_title)
    meta_str = f"Koordinatlar: {lat_min:.4f}N..{lat_max:.4f}N, {lon_min:.4f}E..{lon_max:.4f}E | Gercek OSM & DEM Verisi | Binalar, Parklar, Ara Sokaklar, IZBAN"
    draw.text((25, 34), meta_str, fill=(148, 163, 184, 255), font=font_meta)

    # Lejant kutusu (Sag alt)
    leg_x = width - 360
    leg_y = height - 190
    draw.rectangle([leg_x, leg_y, width - 20, height - 20], fill=(255, 255, 255, 240), outline=(15, 23, 42, 180), width=2)
    font_leg_title = get_font(12, bold=True)
    font_leg = get_font(11, bold=False)
    draw.text((leg_x + 15, leg_y + 10), "LEJANT (HARITA DETAYLARI)", fill=(15, 23, 42, 255), font=font_leg_title)

    items = [
        ((239, 68, 68), (15, 23, 42), "IZBAN Demiryolu & Istasyonlar", "line_rail"),
        ((249, 115, 22), None, "Otoyollar / Cevre Yolu", "line_thick"),
        ((251, 191, 36), None, "Ana Caddeler (Primary)", "line_mid"),
        ((148, 163, 184), None, "Ara Sokaklar & Yaya Yollari", "line_thin"),
        ((134, 239, 172), (74, 222, 128), "Parklar, Bahceler, Yesil Alanlar", "box"),
        ((254, 202, 202), (239, 68, 68), "Gercek Evler ve Binalar", "box"),
        ((191, 219, 254), (59, 130, 246), "Okul, Cami, Saglik, Ozel Binalar", "box"),
    ]
    cur_y = leg_y + 32
    for fill, stroke, label, itype in items:
        if itype == "box":
            draw.rectangle([leg_x + 15, cur_y + 1, leg_x + 35, cur_y + 13], fill=fill, outline=stroke, width=1)
        elif itype == "line_rail":
            draw.line([leg_x + 15, cur_y + 7, leg_x + 35, cur_y + 7], fill=stroke, width=4)
            draw.line([leg_x + 15, cur_y + 7, leg_x + 35, cur_y + 7], fill=fill, width=2)
        elif itype == "line_thick":
            draw.line([leg_x + 15, cur_y + 7, leg_x + 35, cur_y + 7], fill=fill, width=4)
        elif itype == "line_mid":
            draw.line([leg_x + 15, cur_y + 7, leg_x + 35, cur_y + 7], fill=fill, width=3)
        elif itype == "line_thin":
            draw.line([leg_x + 15, cur_y + 7, leg_x + 35, cur_y + 7], fill=fill, width=2)
        draw.text((leg_x + 45, cur_y), label, fill=(30, 41, 59, 255), font=font_leg)
        cur_y += 18

    # Kaydet
    renderer.img.convert("RGB").save(str(out_path), "PNG", quality=95)
    print(f"  [KAYDEDILDI] {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.2f} MB)", flush=True)


def main():
    elements = load_all_osm_data()
    data = categorize_elements(elements)

    # 1. Tum Izmir Metropol Master Haritasi (4096 x 4096)
    # Adnan Menderes HL -> Menemen, Guzelbahce -> Belkahve
    render_map(
        data=data,
        lat_min=38.235, lon_min=26.875, lat_max=38.640, lon_max=27.325,
        width=4096, height=4096,
        title="IZMIR BUYUKSEHIR HARITASI (METROPOL ALAN TAM DETAY)",
        out_path=ARTIFACT_DIR / "izmir_master_real_map_4k.png",
        detail_mode=False
    )

    # 2. OZEL DETAY 1: EGEKENT-2 & MENEMEN & ULUKENT
    # Kullanicinin ozellikle belirttigi Egekent-2 ve cevresi:
    # Tum ara sokaklar (3. Cadde, Sevgi Cd, 8809. Sk), ev duzeni, siteler, bahceler, IZBAN Egekent-2 duragi
    render_map(
        data=data,
        lat_min=38.535, lon_min=27.020, lat_max=38.610, lon_max=27.090,
        width=2800, height=2200,
        title="EGEKENT-2, ULUKENT VE MENEMEN GUNEYI TAM DETAYLI HARITASI",
        out_path=ARTIFACT_DIR / "detail_egekent2_menemen.png",
        detail_mode=True
    )

    # 3. OZEL DETAY 2: KARSIYAKA, BOSTANLI, MAVISEHIR, CIGLI
    render_map(
        data=data,
        lat_min=38.450, lon_min=27.050, lat_max=38.515, lon_max=27.140,
        width=2800, height=2200,
        title="KARSIYAKA, BOSTANLI, MAVISEHIR VE CIGLI DETAYLI HARITASI",
        out_path=ARTIFACT_DIR / "detail_karsiyaka_cigli.png",
        detail_mode=True
    )

    # 4. OZEL DETAY 3: KONAK, ALSANCAK, KEMERALTI, KULTURPARK, BAYRAKLI
    render_map(
        data=data,
        lat_min=38.400, lon_min=27.110, lat_max=38.465, lon_max=27.190,
        width=2800, height=2200,
        title="KONAK, ALSANCAK, KEMERALTI, KULTURPARK VE BAYRAKLI DETAYLI HARITASI",
        out_path=ARTIFACT_DIR / "detail_konak_alsancak.png",
        detail_mode=True
    )

    print("\n[BASARI] Tum haritalar olusturuldu ve artifact dizinine kaydedildi!", flush=True)


if __name__ == "__main__":
    main()
