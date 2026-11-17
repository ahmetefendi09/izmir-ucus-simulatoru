# İzmir 3D Uçuş Simülatörü ve Gerçek GIS Haritası

İzmir bölgesini (`35 km × 45 km` 3D simülatör alanı, `41.8 km × 47.8 km` 2D harita alanı) **Overture Maps + OpenStreetMap + 30m Terrarium DEM** verileriyle 1:1 ölçekte modelleyen C++/OpenGL 6-DOF uçuş simülatörü ve Python GIS harita üreteci.

## Öne Çıkan Özellikler
- **3 Hava Aracı + Drone Modu (`[V]` / `[F]`):** Türk Yıldızları F-5 Süpersonik Jet, Cessna 172 Skyhawk SP, DG-1000 Akrobasi Planörü ve serbest hızlı `[F]` Drone modu.
- **Gerçekçi 100m Bina Çarpışma Penceresi:** Uzak binalar görsel olarak silinmeden çizilir; uçak çevresinde `<= 100m` içindeki tüm yapılar (Overture/OSM binaları, Folkart, Mistral, Ege Perla, Hilton, Saat Kulesi, Liman vb.) anında katı çarpışmaya girer.
- **Patlama, Yangın ve Kamera Titremesi:** Çarpışma ve silah etkilerinde ateş topu, şok dalgası halkası, yükselen siyah duman, kıvılcım ve sönümlü ekran sarsıntısı.
- **HUD Nişangahı, Mermi & Bina-Ölçekli Bomba Yangını:**
  - **Mermi (Seri Atış):** Klavye `[J]` / `[Enter]` veya Gamepad `RB` (`Btn 5`)
  - **Bomba (CCIP Hedeflemeli):** Klavye `[M]` / `[U]` veya Gamepad `LB` (`Btn 4`) — Bomba binaların döndürülmüş OBB çatısına ve cephelerine fiziksel olarak çarpar, bina büyüklüğüne orantılı **30 saniyelik** alev ve duman yangınını yapının şekline göre dağıtır.

## İçerik
- `izmir_flight_sim.cpp`: C++/OpenGL 3D uçuş simülatörü (`IZMIROSM` v2 ikili dünya formatı desteği)
- `izmir_osm_world.bin`: Paketlenmiş `24 MB` v2 dünya verisi (`1024×1024` DEM, `2048×2048` GIS arazi ızgarası, `420,734` bina, `113,167` yol, `624` ray segmenti, `4,239` haritalı ağaç)
- `pack_world_v2.py`: Ham Overture/OSM/DEM önbelleğini `izmir_osm_world.bin` dosyasına paketleyen betik
- `build_izmir_map_v2.py`: 6K ana harita (`izmir_master_real_map_v2_6k.png`) ve 7 yüksek çözünürlüklü ilçe detay haritası üreten betik

## Derleme ve Çalıştırma
```bash
g++ -O3 -std=c++17 -Wall -Wextra izmir_flight_sim.cpp -o izmir_flight_sim -lEGL -lX11 -lGL -lGLU -lm -lpthread
./izmir_flight_sim
```

### Temel Kontroller
- **Gaz / Fren:** `[W]`/`[S]` veya Gamepad `RT`/`LT`
- **Yönlendirme:** Ok Tuşları / `[A]`/`[D]` / `[Q]`/`[E]` veya Gamepad Sol & Sağ Analog
- **Mermi / Bomba:** `[J]`/`[Enter]` (`RB`) ve `[M]`/`[U]` (`LB`)
- **Işınlanma / Kurtarma:** `[1]`–`[0]` bölgeler, `[R]` (`Start`) havada kurtarma, `[F]` Drone modu, `[C]` kamera, `[T]` gündüz/günbatımı/gece

### Doğrulama Modları
```bash
./izmir_flight_sim --test-gen        # Dünya üretimi, çakışmasızlık, 100m çarpışma ve silah/yangın testleri (8 test)
./izmir_flight_sim --inspect-city    # 24 noktalı otonom şehir turu ekran görüntüleri (./renders/)
./izmir_flight_sim --inspect-night   # Gece/günbatımı/gündüz FPS & sahne denetimi
./izmir_flight_sim --inspect-combat  # Nişangah, bomba patlaması ve 30s bina yangını önizlemesi
```
