// =====================================================================================
//  IZMIR 3D UCUS VE SEHIR SIMULATORU  (izmir_flight_sim.cpp)
//  ---------------------------------------------------------------------------------
//  Overture+OSM+DEM (izmir_osm_world.bin) destekli, yedek sentetik uretimli
//  35 km x 45 km Izmir dunyasi + 6-DOF 240 Hz ucus + silah/patlama/yangin fizigi.
//
//  Derleme:
//    g++ -O3 -std=c++17 -Wall -Wextra izmir_flight_sim.cpp -o izmir_flight_sim -lEGL -lX11 -lGL -lGLU -lm -lpthread
//
//  Projeksiyon (WGS84 yerel metrik, referans: Konak Saat Kulesi):
//    x = (lon - REF_LON) * 87215.0   [+X Dogu, -X Bati]
//    z = -(lat - REF_LAT) * 111132.0 [-Z Kuzey, +Z Guney], +Y irtifa
//
//  Komut satiri:
//    ./izmir_flight_sim                   -> simulator
//    ./izmir_flight_sim --test-gen        -> X11 olmadan dunya uretimi + 8 dogrulama testi
//    ./izmir_flight_sim --inspect-city    -> 24 noktali otonom sehir turu gorselleri
//    ./izmir_flight_sim --inspect-night   -> gece/gunbatimi/gunduz performans cekimleri
//    ./izmir_flight_sim --inspect-combat  -> silah, bomba patlamasi ve 30s bina yangini cekimi
// =====================================================================================

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/XKBlib.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <linux/joystick.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <array>
#include <thread>
#include <functional>

#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif
#ifndef GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif
#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif
#ifndef GL_POINT_DISTANCE_ATTENUATION
#define GL_POINT_DISTANCE_ATTENUATION 0x8129
#endif
#ifndef GL_POINT_SIZE_MIN
#define GL_POINT_SIZE_MIN 0x8126
#endif
#ifndef GL_POINT_SIZE_MAX
#define GL_POINT_SIZE_MAX 0x8127
#endif
#ifndef GLX_SAMPLE_BUFFERS
#define GLX_SAMPLE_BUFFERS 100000
#endif
#ifndef GLX_SAMPLES
#define GLX_SAMPLES 100001
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

// =====================================================================================
//  BOLUM 1 : MATEMATIK, GURULTU, YARDIMCILAR
// =====================================================================================
static const float PI_F = 3.14159265358979f;
static const float DEG2RAD = PI_F / 180.0f;
static const float RAD2DEG = 180.0f / PI_F;

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};
static inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float len(const Vec3& a) { return std::sqrt(dot(a, a)); }
static inline Vec3 norm(const Vec3& a) { float l = len(a); return l > 1e-9f ? a / l : Vec3(0, 1, 0); }
static inline Vec3 lerpV(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }

struct Vec2 {
    float x, z;
    Vec2() : x(0), z(0) {}
    Vec2(float a, float b) : x(a), z(b) {}
    Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, z + o.z); }
    Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, z - o.z); }
    Vec2 operator-() const { return Vec2(-x, -z); }
    Vec2 operator*(float s) const { return Vec2(x * s, z * s); }
};
static inline float dot2(const Vec2& a, const Vec2& b) { return a.x * b.x + a.z * b.z; }
static inline float len2(const Vec2& a) { return std::sqrt(a.x * a.x + a.z * a.z); }
static inline Vec2 norm2(const Vec2& a) { float l = len2(a); return l > 1e-9f ? a * (1.0f / l) : Vec2(1, 0); }
static inline float cross2(const Vec2& a, const Vec2& b) { return a.x * b.z - a.z * b.x; }

static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline int clampi(int v, int a, int b) { return v < a ? a : (v > b ? b : v); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float smooth01(float t) { t = clampf(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
static inline float sgnf(float v) { return v < 0 ? -1.0f : 1.0f; }
static inline float wrapAngle(float a) {
    while (a > PI_F) a -= 2 * PI_F;
    while (a < -PI_F) a += 2 * PI_F;
    return a;
}
static inline float sgnpow(float v, float p) { return sgnf(v) * std::pow(std::fabs(v), p); }

// Nokta - dogru parcasi mesafesi (2D), t parametresini de verir
static inline float segDist2D(float px, float pz, float ax, float az, float bx, float bz, float* tOut) {
    float dx = bx - ax, dz = bz - az;
    float l2 = dx * dx + dz * dz;
    float t = 0.0f;
    if (l2 > 1e-9f) t = clampf(((px - ax) * dx + (pz - az) * dz) / l2, 0.0f, 1.0f);
    float qx = ax + dx * t - px, qz = az + dz * t - pz;
    if (tOut) *tOut = t;
    return std::sqrt(qx * qx + qz * qz);
}

// --- Deterministik rastgele sayi ureteci (xorshift) ---
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed = 1234567u) : s(seed ? seed : 1u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uni() { return (next() >> 8) * (1.0f / 16777216.0f); }
    float range(float a, float b) { return a + (b - a) * uni(); }
    int irange(int a, int b) { return a + (int)(next() % (uint32_t)(b - a + 1)); }
};
static inline uint32_t hash2i(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
static inline float hash01(int x, int y) { return (hash2i(x, y) & 0xFFFFFF) * (1.0f / 16777216.0f); }

// --- Perlin gradyan gurultusu (periyodik destekli) ---
static int g_perm[512];
static void initNoise(uint32_t seed) {
    Rng r(seed);
    int p[256];
    for (int i = 0; i < 256; i++) p[i] = i;
    for (int i = 255; i > 0; i--) { int j = (int)(r.next() % (uint32_t)(i + 1)); std::swap(p[i], p[j]); }
    for (int i = 0; i < 512; i++) g_perm[i] = p[i & 255];
}
static inline float fadeN(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
static inline float gradN(int h, float x, float y) {
    switch (h & 7) {
        case 0: return x + y;
        case 1: return -x + y;
        case 2: return x - y;
        case 3: return -x - y;
        case 4: return x;
        case 5: return -x;
        case 6: return y;
        default: return -y;
    }
}
// period <= 0 ise periyodik degil
static float perlin2(float x, float y, int period = 0) {
    float fx = std::floor(x), fy = std::floor(y);
    int xi = (int)fx, yi = (int)fy;
    float xf = x - fx, yf = y - fy;
    int x0 = xi, x1 = xi + 1, y0 = yi, y1 = yi + 1;
    if (period > 0) {
        x0 = ((x0 % period) + period) % period; x1 = ((x1 % period) + period) % period;
        y0 = ((y0 % period) + period) % period; y1 = ((y1 % period) + period) % period;
    }
    x0 &= 255; x1 &= 255; y0 &= 255; y1 &= 255;
    int aa = g_perm[g_perm[x0] + y0], ab = g_perm[g_perm[x0] + y1];
    int ba = g_perm[g_perm[x1] + y0], bb = g_perm[g_perm[x1] + y1];
    float u = fadeN(xf), v = fadeN(yf);
    float l1 = lerpf(gradN(aa, xf, yf), gradN(ba, xf - 1, yf), u);
    float l2 = lerpf(gradN(ab, xf, yf - 1), gradN(bb, xf - 1, yf - 1), u);
    return lerpf(l1, l2, v) * 0.7071f;
}
static float fbm2(float x, float y, int oct, float lac = 2.0f, float gain = 0.5f, int period = 0) {
    float s = 0, a = 1, n = 0;
    int per = period;
    for (int i = 0; i < oct; i++) {
        s += a * perlin2(x, y, per);
        n += a;
        x *= lac; y *= lac; a *= gain;
        if (per > 0) per = (int)(per * lac);
    }
    return s / n;
}
static float ridged2(float x, float y, int oct) {
    float s = 0, a = 1, n = 0;
    for (int i = 0; i < oct; i++) {
        float r = 1.0f - std::fabs(perlin2(x, y) * 1.6f);
        s += a * r * r;
        n += a;
        x *= 2.03f; y *= 2.03f; a *= 0.5f;
    }
    return s / n;
}

// --- Paralel for (std::thread) ---
static void parallelFor(int n, const std::function<void(int, int)>& fn) {
    int th = (int)std::thread::hardware_concurrency();
    if (th < 1) th = 1;
    if (th > 16) th = 16;
    if (n < th * 4) { fn(0, n); return; }
    std::vector<std::thread> ts;
    int chunk = (n + th - 1) / th;
    for (int t = 0; t < th; t++) {
        int a = t * chunk, b = std::min(n, a + chunk);
        if (a >= b) break;
        ts.emplace_back([&fn, a, b]() { fn(a, b); });
    }
    for (auto& t : ts) t.join();
}

static double nowSec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// =====================================================================================
//  BOLUM 2 : GERCEK WGS84 COGRAFI VERI (harici dosya yok, kaynak kodda gomulu)
// =====================================================================================
static const double REF_LAT = 38.418850;   // Konak Saat Kulesi
static const double REF_LON = 27.128680;
static inline float lonToX(double lon) { return (float)((lon - REF_LON) * 87215.0f); }
static inline float latToZ(double lat) { return (float)(-(lat - REF_LAT) * 111132.0f); }
static inline Vec2 geoPt(double lat, double lon) { return Vec2(lonToX(lon), latToZ(lat)); }
static inline double xToLon(float x) { return REF_LON + x / 87215.0; }
static inline double zToLat(float z) { return REF_LAT - z / 111132.0; }

static const float WORLD_X0 = -17500.0f, WORLD_X1 = 17500.0f;
static const float WORLD_Z0 = -26000.0f, WORLD_Z1 = 19000.0f;

struct GeoNamed { const char* name; double lat, lon; };

// Izmir Korfezi kiyi kontrol noktalari (34 adet, siralidir)
static const GeoNamed COAST_CTRL[34] = {
    {"Güzelbahçe", 38.3780, 26.8900}, {"Sahilevleri", 38.3960, 26.9600},
    {"İnciraltı Batı", 38.4135, 27.0120}, {"Yenikale Burnu", 38.4165, 27.0310},
    {"Çakalburnu Lagünü", 38.4095, 27.0465}, {"Deniz Müzesi", 38.4045, 27.0620},
    {"Üçkuyular İskele", 38.4050, 27.0706}, {"Güzelyalı", 38.3995, 27.0830},
    {"Göztepe İskele", 38.3985, 27.0960}, {"Karantina", 38.4045, 27.1090},
    {"Karataş", 38.4105, 27.1185}, {"Konak İskele", 38.4155, 27.1250},
    {"Konak Sahil", 38.4188, 27.1272}, {"Pasaport İskele", 38.4265, 27.1318},
    {"Cumhuriyet Meydanı", 38.4305, 27.1340}, {"Alsancak İskele", 38.4375, 27.1398},
    {"Alsancak Limanı Batı", 38.4425, 27.1435}, {"Liman Rıhtım", 38.4445, 27.1535},
    {"Meles Deltası", 38.4420, 27.1610},
    {"Bayraklı Güney", 38.4485, 27.1655}, {"Bayraklı Sahil", 38.4565, 27.1665},
    {"Turan", 38.4655, 27.1610}, {"Alaybey Tersanesi", 38.4640, 27.1420},
    {"Alaybey Sahil", 38.4585, 27.1280}, {"Karşıyaka İskele", 38.4547, 27.1194},
    {"Karşıyaka Yelken", 38.4535, 27.1080}, {"Bostanlı İskele", 38.4556, 27.0972},
    {"Bostanlı Günbatımı", 38.4620, 27.0870}, {"Mavişehir Barınak", 38.4695, 27.0755},
    {"Mavişehir Flamingo", 38.4740, 27.0640}, {"Çiğli Kuş Cenneti", 38.4680, 27.0350},
    {"Sasalı Tuzla", 38.4780, 26.9850}, {"Çamaltı Tuzlası", 38.5150, 26.9250},
    {"Gediz Deltası", 38.5650, 26.8900}
};
static const float SEA_CORNER_NW_X = -22000.0f, SEA_CORNER_NW_Z = -16500.0f;
static const float SEA_CORNER_SW_X = -22000.0f, SEA_CORNER_SW_Z = 5000.0f;

struct StationDef { const char* name; double lat, lon; float elev; };
static const int NUM_STATIONS = 28;
static const StationDef STATIONS[NUM_STATIONS] = {
    {"Menemen", 38.60317, 27.07672, 18}, {"Egekent 2", 38.56060, 27.04436, 14},
    {"Ulukent", 38.54785, 27.03547, 12}, {"Egekent", 38.50734, 27.04598, 8},
    {"Ata Sanayi", 38.49880, 27.05850, 5}, {"Çiğli", 38.49166, 27.06343, 4},
    {"Mavişehir", 38.47950, 27.07820, 3}, {"Şemikler", 38.47120, 27.09450, 4},
    {"Demirköprü", 38.46650, 27.10420, 6}, {"Nergiz", 38.46300, 27.11280, 5},
    {"Karşıyaka", 38.45790, 27.11513, 4}, {"Alaybey", 38.46140, 27.12920, 4},
    {"Naldöken", 38.46780, 27.14970, 5}, {"Turan", 38.46750, 27.16310, 3},
    {"Bayraklı", 38.46190, 27.16690, 4}, {"Salhane", 38.45220, 27.17140, 4},
    {"Halkapınar", 38.43522, 27.16885, 4}, {"Alsancak", 38.43863, 27.14812, 3},
    {"Hilal", 38.42605, 27.15436, 8}, {"Kemer", 38.42310, 27.14890, 14},
    {"Şirinyer", 38.39080, 27.15060, 58}, {"Koşu", 38.38140, 27.14810, 68},
    {"İnkılap", 38.37360, 27.14440, 78}, {"Semt Garajı", 38.36390, 27.14190, 92},
    {"Esbaş", 38.34860, 27.13890, 108}, {"Gaziemir", 38.32830, 27.13690, 118},
    {"Sarnıç", 38.31360, 27.14560, 122}, {"Adnan Menderes Havalimanı", 38.29360, 27.15190, 125}
};

struct GeoLL { double lat, lon; };
struct RoadDef {
    const char* name;
    int type;
    float halfW;
    int lanes;
    std::vector<GeoLL> pts;
    bool isCoastal = false;
};
enum { CT_RAIL = 0, CT_HIGHWAY = 1, CT_ROAD = 2, CT_RUNWAY = 3, CT_TAXI = 4 };

static std::vector<RoadDef> makeRoadDefs() {
    std::vector<RoadDef> r;
    // 1. Otoyollar ve Cevre Yolu (dugum noktalari birbirine kesintisiz baglidir)
    r.push_back({"O-30 İzmir Çevreyolu", CT_HIGHWAY, 15.0f, 3,
                 {{38.3880, 27.0350}, {38.3680, 27.1050}, {38.3720, 27.1680}, {38.4150, 27.2150},
                  {38.4580, 27.2220}, {38.4780, 27.1750}, {38.4880, 27.1250}, {38.5020, 27.0750},
                  {38.5450, 27.0480}, {38.6050, 27.0680}}, false});
    r.push_back({"D550 Menemen - Çiğli - Anadolu Caddesi", CT_HIGHWAY, 12.5f, 3,
                 {{38.6050, 27.0680}, {38.5650, 27.0540}, {38.5440, 27.0520}, {38.5230, 27.0580},
                  {38.5050, 27.0650}, {38.4950, 27.0740}, {38.4830, 27.0900}, {38.4750, 27.1150},
                  {38.4720, 27.1450}, {38.4660, 27.1620}}, false});
    r.push_back({"Altınyol Sahil Ekspres Yolu", CT_HIGHWAY, 11.5f, 3,
                 {{38.4430, 27.1660}, {38.4525, 27.1672}, {38.4600, 27.1665}, {38.4660, 27.1620},
                  {38.4655, 27.1450}, {38.4605, 27.1305}, {38.4565, 27.1210}}, true});
    r.push_back({"D550 Yeşildere - Akçay Caddesi", CT_HIGHWAY, 12.5f, 3,
                 {{38.4430, 27.1660}, {38.4300, 27.1550}, {38.4210, 27.1460}, {38.4120, 27.1400},
                  {38.3980, 27.1520}, {38.3810, 27.1500}, {38.3710, 27.1450}, {38.3450, 27.1400},
                  {38.3180, 27.1420}, {38.2930, 27.1490}}, false});
    r.push_back({"Mustafa Kemal Sahil Bulvarı", CT_ROAD, 11.0f, 3,
                 {{38.4188, 27.1278}, {38.4148, 27.1252}, {38.4098, 27.1188}, {38.4038, 27.1095},
                  {38.3980, 27.0965}, {38.3990, 27.0835}, {38.4042, 27.0710}, {38.4038, 27.0622},
                  {38.4090, 27.0470}, {38.4158, 27.0310}, {38.4125, 27.0130}, {38.3950, 26.9605},
                  {38.3790, 26.8980}}, true});

    // 2. Kordon ve Sahil Bulvarlari
    r.push_back({"Atatürk Caddesi (Birinci Kordon) & Liman", CT_ROAD, 9.5f, 2,
                 {{38.4188, 27.1278}, {38.4262, 27.1324}, {38.4302, 27.1346}, {38.4372, 27.1404},
                  {38.4420, 27.1442}, {38.4438, 27.1535}, {38.4425, 27.1635}}, true});
    r.push_back({"Cumhuriyet Bulvarı (İkinci Kordon)", CT_ROAD, 9.5f, 2,
                 {{38.4185, 27.1290}, {38.4250, 27.1340}, {38.4300, 27.1365}, {38.4345, 27.1400},
                  {38.4395, 27.1465}}, false});
    r.push_back({"Karşıyaka Cemal Gürsel Sahil Caddesi", CT_ROAD, 10.0f, 2,
                 {{38.4615, 27.1380}, {38.4582, 27.1282}, {38.4550, 27.1196}, {38.4538, 27.1082}}, true});
    r.push_back({"Bostanlı Hasan Ali Yücel Sahil Bulvarı", CT_ROAD, 10.0f, 2,
                 {{38.4538, 27.1082}, {38.4560, 27.0975}, {38.4622, 27.0874}, {38.4695, 27.0760},
                  {38.4742, 27.0650}}, true});
    r.push_back({"Mavişehir Caher Dudayev Bulvarı", CT_ROAD, 11.0f, 2,
                 {{38.4570, 27.0980}, {38.4650, 27.0910}, {38.4735, 27.0810}, {38.4785, 27.0735},
                  {38.4870, 27.0620}}, false});

    // 3. Konak, Alsancak ve Sehir Ici Ana Bulvarlari
    r.push_back({"Şair Eşref Bulvarı", CT_ROAD, 9.5f, 2,
                 {{38.4215, 27.1365}, {38.4270, 27.1415}, {38.4325, 27.1455}, {38.4385, 27.1485}}, false});
    r.push_back({"Talatpaşa Bulvarı", CT_ROAD, 9.5f, 2,
                 {{38.4380, 27.1420}, {38.4340, 27.1440}, {38.4325, 27.1455}, {38.4265, 27.1460}}, false});
    r.push_back({"Gazi Bulvarı", CT_ROAD, 10.0f, 2,
                 {{38.4190, 27.1295}, {38.4225, 27.1375}, {38.4245, 27.1440}}, false});
    r.push_back({"Fevzipaşa Bulvarı", CT_ROAD, 9.5f, 2,
                 {{38.4175, 27.1285}, {38.4195, 27.1350}, {38.4215, 27.1385}, {38.4235, 27.1450}}, false});
    r.push_back({"Mürselpaşa Bulvarı", CT_ROAD, 10.5f, 2,
                 {{38.4240, 27.1460}, {38.4310, 27.1550}, {38.4420, 27.1680}, {38.4440, 27.1550}}, false});
    r.push_back({"Mithatpaşa Caddesi", CT_ROAD, 9.0f, 2,
                 {{38.4160, 27.1285}, {38.4110, 27.1210}, {38.4055, 27.1110}, {38.4010, 27.1000},
                  {38.3995, 27.0880}, {38.4015, 27.0730}}, false});
    r.push_back({"İnönü Caddesi (Hatay Aksı)", CT_ROAD, 10.0f, 2,
                 {{38.4100, 27.1240}, {38.4050, 27.1180}, {38.3995, 27.1090}, {38.3970, 27.0980},
                  {38.3960, 27.0860}, {38.3980, 27.0725}}, false});
    r.push_back({"Halil Rıfat Paşa - Varyant Yolu", CT_ROAD, 8.0f, 2,
                 {{38.4180, 27.1295}, {38.4150, 27.1325}, {38.4110, 27.1345}, {38.4095, 27.1390},
                  {38.4120, 27.1400}}, false});

    // 4. Karsiyaka, Cigli, Sasali ve Camalti Tuzlasi Arterleri
    r.push_back({"Girne Bulvarı", CT_ROAD, 10.5f, 2,
                 {{38.4550, 27.1196}, {38.4610, 27.1165}, {38.4645, 27.1145}, {38.4690, 27.1120},
                  {38.4750, 27.1150}}, false});
    r.push_back({"Ordu Caddesi / Bostanlı Demirköprü", CT_ROAD, 9.5f, 2,
                 {{38.4560, 27.0975}, {38.4625, 27.1005}, {38.4665, 27.1030}, {38.4750, 27.1150}}, false});
    r.push_back({"Çiğli AOSB 10000. Sokak (Ana Sanayi)", CT_ROAD, 11.0f, 2,
                 {{38.4870, 27.0620}, {38.4940, 27.0450}, {38.5020, 27.0300}, {38.5120, 27.0150}}, false});
    r.push_back({"Çiğli AOSB 10006. Sokak (Tır Arteri)", CT_ROAD, 11.0f, 2,
                 {{38.5050, 27.0650}, {38.5040, 27.0420}, {38.4980, 27.0250}, {38.4920, 27.0120}}, false});
    r.push_back({"Çiğli AOSB Mustafa Kemal Atatürk Bulvarı", CT_ROAD, 10.5f, 2,
                 {{38.4870, 27.0620}, {38.4970, 27.0540}, {38.5090, 27.0440}, {38.5230, 27.0580}}, false});
    r.push_back({"Çiğli AOSB Güzelbahçe - Batı Sanayi Caddesi", CT_ROAD, 10.0f, 2,
                 {{38.4920, 27.0120}, {38.5020, 27.0135}, {38.5120, 27.0150}}, false});
    r.push_back({"Sasalı Doğal Yaşam & Çamaltı Tuzlası Yolu", CT_ROAD, 9.0f, 2,
                 {{38.4785, 27.0735}, {38.4755, 27.0480}, {38.4820, 27.0020}, {38.4895, 26.9760},
                  {38.5010, 26.9440}, {38.5120, 26.9320}}, false});

    // 5. Bayrakli ve Bornova Arterleri
    r.push_back({"Ankara Caddesi (D300 Bornova - Manisa)", CT_HIGHWAY, 12.5f, 3,
                 {{38.4430, 27.1660}, {38.4480, 27.1780}, {38.4550, 27.1950}, {38.4580, 27.2220},
                  {38.4680, 27.2320}}, false});
    r.push_back({"Manas Bulvarı (Bayraklı Gökdelenler & Adliye)", CT_ROAD, 11.0f, 2,
                 {{38.4525, 27.1672}, {38.4556, 27.1742}, {38.4562, 27.1778}, {38.4535, 27.1835},
                  {38.4550, 27.1950}}, false});
    r.push_back({"İslam Kerimov & Haydar Aliyev Caddesi", CT_ROAD, 10.0f, 2,
                 {{38.4600, 27.1665}, {38.4615, 27.1710}, {38.4555, 27.1765}, {38.4480, 27.1780}}, false});
    r.push_back({"Bornova Küçükpark - Ege Üniversitesi Kampüs", CT_ROAD, 9.5f, 2,
                 {{38.4620, 27.2120}, {38.4635, 27.2185}, {38.4600, 27.2250}, {38.4580, 27.2340}}, false});
    r.push_back({"Bornova Sanayi & Yüzbaşı İbrahim Hakkı Bulvarı", CT_ROAD, 9.5f, 2,
                 {{38.4430, 27.1660}, {38.4475, 27.1745}, {38.4535, 27.1820}, {38.4620, 27.2120}}, false});

    // 6. Buca, Balcova, Gaziemir ve Menemen-Seyrek Arterleri
    r.push_back({"Buca Menderes Caddesi & Çarşı", CT_ROAD, 9.5f, 2,
                 {{38.3910, 27.1510}, {38.3930, 27.1580}, {38.3880, 27.1720}, {38.3840, 27.1810},
                  {38.3720, 27.1950}}, false});
    r.push_back({"Buca Hipodrom & Uğur Mumcu Caddesi", CT_ROAD, 9.5f, 2,
                 {{38.3810, 27.1500}, {38.3840, 27.1570}, {38.3750, 27.1820}, {38.3720, 27.1950},
                  {38.3680, 27.2150}}, false});
    r.push_back({"İnciraltı Turizm & Kent Ormanı Bulvarı", CT_ROAD, 9.5f, 2,
                 {{38.4042, 27.0710}, {38.4015, 27.0560}, {38.4005, 27.0400}, {38.3970, 27.0260},
                  {38.3880, 27.0350}}, false});
    r.push_back({"Balçova Sakarya Caddesi & Teleferik", CT_ROAD, 9.0f, 2,
                 {{38.3980, 27.0725}, {38.3920, 27.0580}, {38.3880, 27.0480}, {38.3880, 27.0350}}, false});
    r.push_back({"Gaziemir Önder Caddesi & Fuar İzmir Yolu", CT_ROAD, 10.0f, 2,
                 {{38.3280, 27.1415}, {38.3280, 27.1310}, {38.3320, 27.1150}, {38.3480, 27.1080},
                  {38.3680, 27.1050}}, false});
    r.push_back({"Kemeraltı Tarihi Aksı (Anafartalar Caddesi)", CT_ROAD, 7.0f, 1,
                 {{38.4185, 27.1295}, {38.4172, 27.1330}, {38.4166, 27.1370}, {38.4170, 27.1408},
                  {38.4200, 27.1455}}, false});
    r.push_back({"Alsancak Kıbrıs Şehitleri - Plevne Aksı", CT_ROAD, 8.0f, 1,
                 {{38.4302, 27.1346}, {38.4325, 27.1375}, {38.4350, 27.1410}, {38.4375, 27.1440},
                  {38.4395, 27.1465}}, false});
    r.push_back({"Menemen Çanakkale Asfaltı & Şehir İçi", CT_ROAD, 10.5f, 2,
                 {{38.6050, 27.0680}, {38.6105, 27.0725}, {38.6175, 27.0760}, {38.6280, 27.0790}}, false});
    r.push_back({"Menemen - Seyrek - Sasalı Kavşak Yolu", CT_ROAD, 9.5f, 2,
                 {{38.6050, 27.0680}, {38.5910, 27.0460}, {38.5730, 27.0260}, {38.5450, 27.0180},
                  {38.5120, 27.0150}}, false});
    return r;
}

struct DistrictDef { const char* name; double lat, lon; float radius, density; };
static const int NUM_DISTRICTS = 30;
static const DistrictDef DISTRICTS[NUM_DISTRICTS] = {
    {"Konak", 38.4189, 27.1287, 1700, 1.00}, {"Alsancak", 38.4370, 27.1440, 1200, 1.00},
    {"Kemeraltı", 38.4180, 27.1390, 800, 1.00}, {"Karşıyaka", 38.4590, 27.1150, 2000, 0.95},
    {"Bostanlı", 38.4620, 27.0950, 1400, 0.95}, {"Bayraklı", 38.4630, 27.1700, 2000, 0.90},
    {"Bornova", 38.4670, 27.2150, 2800, 0.85}, {"Buca", 38.3880, 27.1750, 3100, 0.90},
    {"Karabağlar", 38.3860, 27.1250, 2500, 0.95}, {"Gaziemir", 38.3240, 27.1300, 1900, 0.75},
    {"Balçova", 38.3900, 27.0500, 1400, 0.80}, {"Narlıdere", 38.3930, 27.0050, 1500, 0.70},
    {"Çiğli", 38.4960, 27.0700, 2300, 0.80}, {"Mavişehir", 38.4780, 27.0750, 900, 0.90},
    {"Menemen", 38.6050, 27.0720, 1700, 0.65}, {"Ulukent", 38.5450, 27.0440, 800, 0.55},
    {"Egekent", 38.5300, 27.0560, 900, 0.55}, {"Hatay", 38.4000, 27.1100, 1500, 1.00},
    {"Göztepe", 38.3990, 27.0850, 1400, 1.00}, {"Şirinyer", 38.3900, 27.1500, 1400, 0.95},
    {"Sarnıç", 38.3130, 27.1460, 900, 0.50}, {"Pınarbaşı", 38.4300, 27.2450, 1200, 0.60},
    {"Işıkkent", 38.4150, 27.2150, 1000, 0.55}, {"Halkapınar", 38.4430, 27.1700, 900, 0.85},
    {"Eşrefpaşa", 38.4080, 27.1350, 1100, 1.00}, {"Güzelyalı", 38.3995, 27.0760, 900, 1.00},
    {"Harmandalı", 38.5100, 27.1000, 1100, 0.45}, {"Sasalı & Çamaltı", 38.4900, 26.9680, 1300, 0.04},
    {"Fuar İzmir", 38.3340, 27.1120, 900, 0.25}, {"İnciraltı", 38.4070, 27.0380, 1100, 0.08}
};

struct MountainDef { const char* name; double lat, lon; float peak, sx, sz, rotDeg; };
static const int NUM_MOUNTAINS = 12;
static const MountainDef MOUNTAINS[NUM_MOUNTAINS] = {
    {"Yamanlar Dağı", 38.545, 27.185, 1076, 3200, 2600, 20},
    {"Karşıyaka Sırtı", 38.485, 27.145, 520, 2400, 700, 10},
    {"Dumanlıdağ", 38.615, 27.210, 1090, 3600, 3000, -15},
    {"Menemen Yıldıztepe", 38.616, 27.102, 260, 1100, 850, 0},
    {"Balçova Dağı", 38.355, 26.985, 950, 3200, 2200, 25},
    {"Çatalkaya", 38.375, 27.055, 640, 2000, 1500, 10},
    {"Kadifekale", 38.4142, 27.1453, 186, 650, 550, 0},
    {"Belkahve / Nif", 38.425, 27.310, 780, 4500, 3500, -30},
    {"Gaziemir Platosu", 38.310, 27.150, 122, 5200, 4200, 0},
    {"Buca Kaynaklar Tepeleri", 38.372, 27.225, 460, 2800, 2100, -15},
    {"Bornova Evka-4 Sırtları", 38.498, 27.228, 430, 2400, 1800, 15},
    {"Güzelbahçe Kızılbahçe Sırtı", 38.342, 26.915, 690, 2900, 2000, 20}
};

struct BasinDef { const char* name; double lat, lon; float radius, minH, maxH; };
static const int NUM_BASINS = 4;
static const BasinDef BASINS[NUM_BASINS] = {
    {"Gediz-Menemen-Çiğli Ovası", 38.565, 27.025, 8800, 2, 18},
    {"Bornova Ovası", 38.45, 27.19, 3200, 4, 35},
    {"Gaziemir-Havalimanı Platosu", 38.31, 27.15, 3600, 115, 130},
    {"Çamaltı Tuzlası - Kuş Cenneti Deltası", 38.495, 26.955, 5800, 2, 5}
};

// Havalimanlari
static const double LTBJ_LAT = 38.2924, LTBJ_LON = 27.1570;
static const float LTBJ_ELEV = 125.0f, LTBJ_HDG = 165.8f, LTBJ_LEN = 3240.0f, LTBJ_W = 45.0f, LTBJ_SEP = 330.0f;
static const double LTBL_LAT = 38.5130, LTBL_LON = 27.0100;
static const float LTBL_ELEV = 5.0f, LTBL_HDG = 167.0f, LTBL_LEN = 2900.0f, LTBL_W = 45.0f;

// Simge yapilar
static const double FOLKART_LAT = 38.4542, FOLKART_LON = 27.1755;
static const double MISTRAL_LAT = 38.4495, MISTRAL_LON = 27.1795;
static const double EGEPERLA_LAT = 38.4465, EGEPERLA_LON = 27.1752;
static const double SAAT_LAT = 38.41885, SAAT_LON = 27.12868;
static const double KULTURPARK_LAT = 38.4275, KULTURPARK_LON = 27.1450;
static const double INCIRALTI_LAT = 38.4080, INCIRALTI_LON = 27.0290;
static const double MAVISEHIR_LAT = 38.478, MAVISEHIR_LON = 27.075;
static const double AOSB_LAT = 38.490, AOSB_LON = 27.038;
static const double ATASAN_LAT = 38.5089, ATASAN_LON = 27.0564;
static const double ESBAS_LAT = 38.3500, ESBAS_LON = 27.1300;
static const double KADIFEKALE_LAT = 38.4142, KADIFEKALE_LON = 27.1453;
static const double ASANSOR_LAT = 38.4088, ASANSOR_LON = 27.1175;
static const double CUMHURIYET_LAT = 38.4305, CUMHURIYET_LON = 27.1345;
static const double GUNDOGDU_LAT = 38.4352, GUNDOGDU_LON = 27.1382;
static const double HILTON_LAT = 38.4258, HILTON_LON = 27.1385;
static const double KSK_ANIT_LAT = 38.4552, KSK_ANIT_LON = 27.1190;

// heading (derece, kuzeyden saat yonu) -> dunya yon vektoru (x,z)
static inline Vec2 headingDir(float hdgDeg) { float h = hdgDeg * DEG2RAD; return Vec2(std::sin(h), -std::cos(h)); }

// =====================================================================================
//  BOLUM 3 : GOMULU 5x7 BITMAP FONT (+ Turkce karakterler)
// =====================================================================================
static const unsigned char FONT5x7[96][5] = {
    {0x00,0x00,0x00,0x00,0x00},{0x00,0x00,0x5F,0x00,0x00},{0x00,0x07,0x00,0x07,0x00},{0x14,0x7F,0x14,0x7F,0x14},
    {0x24,0x2A,0x7F,0x2A,0x12},{0x23,0x13,0x08,0x64,0x62},{0x36,0x49,0x55,0x22,0x50},{0x00,0x05,0x03,0x00,0x00},
    {0x00,0x1C,0x22,0x41,0x00},{0x00,0x41,0x22,0x1C,0x00},{0x08,0x2A,0x1C,0x2A,0x08},{0x08,0x08,0x3E,0x08,0x08},
    {0x00,0x50,0x30,0x00,0x00},{0x08,0x08,0x08,0x08,0x08},{0x00,0x60,0x60,0x00,0x00},{0x20,0x10,0x08,0x04,0x02},
    {0x3E,0x51,0x49,0x45,0x3E},{0x00,0x42,0x7F,0x40,0x00},{0x42,0x61,0x51,0x49,0x46},{0x21,0x41,0x45,0x4B,0x31},
    {0x18,0x14,0x12,0x7F,0x10},{0x27,0x45,0x45,0x45,0x39},{0x3C,0x4A,0x49,0x49,0x30},{0x01,0x71,0x09,0x05,0x03},
    {0x36,0x49,0x49,0x49,0x36},{0x06,0x49,0x49,0x29,0x1E},{0x00,0x36,0x36,0x00,0x00},{0x00,0x56,0x36,0x00,0x00},
    {0x08,0x14,0x22,0x41,0x00},{0x14,0x14,0x14,0x14,0x14},{0x00,0x41,0x22,0x14,0x08},{0x02,0x01,0x51,0x09,0x06},
    {0x32,0x49,0x79,0x41,0x3E},{0x7E,0x11,0x11,0x11,0x7E},{0x7F,0x49,0x49,0x49,0x36},{0x3E,0x41,0x41,0x41,0x22},
    {0x7F,0x41,0x41,0x22,0x1C},{0x7F,0x49,0x49,0x49,0x41},{0x7F,0x09,0x09,0x01,0x01},{0x3E,0x41,0x41,0x51,0x32},
    {0x7F,0x08,0x08,0x08,0x7F},{0x00,0x41,0x7F,0x41,0x00},{0x20,0x40,0x41,0x3F,0x01},{0x7F,0x08,0x14,0x22,0x41},
    {0x7F,0x40,0x40,0x40,0x40},{0x7F,0x02,0x04,0x02,0x7F},{0x7F,0x04,0x08,0x10,0x7F},{0x3E,0x41,0x41,0x41,0x3E},
    {0x7F,0x09,0x09,0x09,0x06},{0x3E,0x41,0x51,0x21,0x5E},{0x7F,0x09,0x19,0x29,0x46},{0x46,0x49,0x49,0x49,0x31},
    {0x01,0x01,0x7F,0x01,0x01},{0x3F,0x40,0x40,0x40,0x3F},{0x1F,0x20,0x40,0x20,0x1F},{0x7F,0x20,0x18,0x20,0x7F},
    {0x63,0x14,0x08,0x14,0x63},{0x03,0x04,0x78,0x04,0x03},{0x61,0x51,0x49,0x45,0x43},{0x00,0x00,0x7F,0x41,0x41},
    {0x02,0x04,0x08,0x10,0x20},{0x41,0x41,0x7F,0x00,0x00},{0x04,0x02,0x01,0x02,0x04},{0x40,0x40,0x40,0x40,0x40},
    {0x00,0x01,0x02,0x04,0x00},{0x20,0x54,0x54,0x54,0x78},{0x7F,0x48,0x44,0x44,0x38},{0x38,0x44,0x44,0x44,0x20},
    {0x38,0x44,0x44,0x48,0x7F},{0x38,0x54,0x54,0x54,0x18},{0x08,0x7E,0x09,0x01,0x02},{0x08,0x14,0x54,0x54,0x3C},
    {0x7F,0x08,0x04,0x04,0x78},{0x00,0x44,0x7D,0x40,0x00},{0x20,0x40,0x44,0x3D,0x00},{0x00,0x7F,0x10,0x28,0x44},
    {0x00,0x41,0x7F,0x40,0x00},{0x7C,0x04,0x18,0x04,0x78},{0x7C,0x08,0x04,0x04,0x78},{0x38,0x44,0x44,0x44,0x38},
    {0x7C,0x14,0x14,0x14,0x08},{0x08,0x14,0x14,0x18,0x7C},{0x7C,0x08,0x04,0x04,0x08},{0x48,0x54,0x54,0x54,0x20},
    {0x04,0x3F,0x44,0x40,0x20},{0x3C,0x40,0x40,0x20,0x7C},{0x1C,0x20,0x40,0x20,0x1C},{0x3C,0x40,0x30,0x40,0x3C},
    {0x44,0x28,0x10,0x28,0x44},{0x0C,0x50,0x50,0x50,0x3C},{0x44,0x64,0x54,0x4C,0x44},{0x00,0x08,0x36,0x41,0x00},
    {0x00,0x00,0x7F,0x00,0x00},{0x00,0x41,0x36,0x08,0x00},{0x02,0x01,0x02,0x04,0x02},{0x7F,0x7F,0x7F,0x7F,0x7F}
};
// Ek glifler: Ç ç Ğ ğ İ ı Ö ö Ş ş Ü ü °  (bit7 = alt cizgi alti / cedilla)
static const int NUM_EXTRA_GLYPHS = 13;
static const unsigned int EXTRA_CP[NUM_EXTRA_GLYPHS] = {
    0x00C7, 0x00E7, 0x011E, 0x011F, 0x0130, 0x0131, 0x00D6, 0x00F6, 0x015E, 0x015F, 0x00DC, 0x00FC, 0x00B0 };
static const unsigned char EXTRA_GLYPHS[NUM_EXTRA_GLYPHS][5] = {
    {0x3E,0x41,0xC1,0x41,0x22}, {0x38,0x44,0xC4,0x44,0x20}, {0x3C,0x43,0x42,0x4B,0x38}, {0x08,0x15,0x54,0x55,0x3C},
    {0x00,0x44,0x7D,0x44,0x00}, {0x00,0x44,0x7C,0x40,0x00}, {0x38,0x45,0x44,0x45,0x38}, {0x30,0x49,0x48,0x49,0x30},
    {0x46,0x49,0xC9,0x49,0x31}, {0x48,0x54,0xD4,0x54,0x20}, {0x3C,0x41,0x40,0x41,0x3C}, {0x3C,0x41,0x40,0x21,0x7C},
    {0x00,0x06,0x09,0x09,0x06} };

// UTF-8 -> glif indeksi listesi
static void utf8ToGlyphs(const char* s, std::vector<int>& out) {
    out.clear();
    const unsigned char* p = (const unsigned char*)s;
    while (*p) {
        unsigned int cp = 0;
        if (*p < 0x80) { cp = *p; p++; }
        else if ((*p & 0xE0) == 0xC0 && p[1]) { cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); p += 2; }
        else if ((*p & 0xF0) == 0xE0 && p[1] && p[2]) { cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); p += 3; }
        else { p++; cp = '?'; }
        int g = -1;
        if (cp >= 32 && cp < 128) g = (int)cp - 32;
        else {
            for (int i = 0; i < NUM_EXTRA_GLYPHS; i++) if (EXTRA_CP[i] == cp) { g = 96 + i; break; }
            if (g < 0) g = '?' - 32;
        }
        out.push_back(g);
    }
}
static inline const unsigned char* glyphData(int g) { return g < 96 ? FONT5x7[g] : EXTRA_GLYPHS[g - 96]; }
static inline bool glyphPixel(int g, int col, int row) { return (glyphData(g)[col] >> row) & 1; }
static const int TOTAL_GLYPHS = 96 + NUM_EXTRA_GLYPHS;

// =====================================================================================
//  BOLUM 4 : 2D / 3D METIN CIZIMI VE 10 ADET 512x512 PROSEDUREL HD DOKU (8x ANISO)
// =====================================================================================
static void drawText2D(float x, float y, float scale, const char* txt,
                       float r = 0.2f, float g = 1.0f, float b = 0.45f, float a = 1.0f) {
    std::vector<int> gls;
    utf8ToGlyphs(txt, gls);
    glDisable(GL_TEXTURE_2D);
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
    float cx = x;
    for (int gi : gls) {
        for (int col = 0; col < 5; col++) {
            unsigned char bits = glyphData(gi)[col];
            if (!bits) continue;
            for (int row = 0; row < 8; row++) {
                if ((bits >> row) & 1) {
                    float px = cx + col * scale;
                    float py = y + row * scale;
                    glVertex2f(px, py);
                    glVertex2f(px + scale, py);
                    glVertex2f(px + scale, py + scale);
                    glVertex2f(px, py + scale);
                }
            }
        }
        cx += 6.0f * scale;
    }
    glEnd();
}

static float textWidth2D(float scale, const char* txt) {
    std::vector<int> gls;
    utf8ToGlyphs(txt, gls);
    return (float)gls.size() * 6.0f * scale;
}

// 3D dunyada istasyon ve tabela yazisi (drawTextWorld)
static void drawTextWorld(const Vec3& center, float yawRad, float pixelSize, const char* txt,
                          float r = 1.0f, float g = 0.95f, float b = 0.75f) {
    std::vector<int> gls;
    utf8ToGlyphs(txt, gls);
    float totalW = (float)gls.size() * 6.0f * pixelSize;
    float startX = -0.5f * totalW;
    float startY = 3.5f * pixelSize;
    Vec3 right(std::cos(yawRad), 0.0f, -std::sin(yawRad));
    Vec3 up(0.0f, 1.0f, 0.0f);
    Vec3 fwd(std::sin(yawRad), 0.0f, std::cos(yawRad));

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    // Arka tabela plakasi
    Vec3 p0 = center - right * (totalW * 0.55f) - up * (4.5f * pixelSize);
    Vec3 p1 = center + right * (totalW * 0.55f) - up * (4.5f * pixelSize);
    Vec3 p2 = center + right * (totalW * 0.55f) + up * (5.0f * pixelSize);
    Vec3 p3 = center - right * (totalW * 0.55f) + up * (5.0f * pixelSize);
    glColor3f(0.06f, 0.14f, 0.34f);
    glBegin(GL_QUADS);
    glVertex3f(p0.x, p0.y, p0.z); glVertex3f(p1.x, p1.y, p1.z);
    glVertex3f(p2.x, p2.y, p2.z); glVertex3f(p3.x, p3.y, p3.z);
    glVertex3f(p3.x, p3.y, p3.z); glVertex3f(p2.x, p2.y, p2.z);
    glVertex3f(p1.x, p1.y, p1.z); glVertex3f(p0.x, p0.y, p0.z);
    glEnd();

    // Iki yuzlu 3D harfler
    glColor3f(r, g, b);
    glBegin(GL_QUADS);
    for (int side = -1; side <= 1; side += 2) {
        Vec3 off = fwd * (0.08f * (float)side);
        Vec3 rDir = right * (float)side;
        float cx = startX;
        for (int gi : gls) {
            for (int col = 0; col < 5; col++) {
                unsigned char bits = glyphData(gi)[col];
                if (!bits) continue;
                for (int row = 0; row < 8; row++) {
                    if ((bits >> row) & 1) {
                        float x0 = cx + col * pixelSize;
                        float y0 = startY - (row + 1) * pixelSize;
                        Vec3 v00 = center + off + rDir * x0 + up * y0;
                        Vec3 v10 = center + off + rDir * (x0 + pixelSize) + up * y0;
                        Vec3 v11 = center + off + rDir * (x0 + pixelSize) + up * (y0 + pixelSize);
                        Vec3 v01 = center + off + rDir * x0 + up * (y0 + pixelSize);
                        glVertex3f(v00.x, v00.y, v00.z);
                        glVertex3f(v10.x, v10.y, v10.z);
                        glVertex3f(v11.x, v11.y, v11.z);
                        glVertex3f(v01.x, v01.y, v01.z);
                    }
                }
            }
            cx += 6.0f * pixelSize;
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

// 12 adet Doku (11x 512x512 Prosedurel HD Doku + 1x 2048x2048 Gercek Overture GIS Arazi Ortusu)
enum {
    TEX_GRASS = 0,
    TEX_ROCK,
    TEX_WATER,
    TEX_ASPHALT,
    TEX_CONCRETE,
    TEX_APT_FACADE,
    TEX_GLASS_TOWER,
    TEX_INDUSTRIAL,
    TEX_ROOF_TILE,
    TEX_BALLAST,
    TEX_NIGHT_WINDOWS,
    TEX_GIS_MAP,
    NUM_TEXTURES = 12
};
static GLuint g_tex[NUM_TEXTURES] = {0};
static GLuint g_texParticle = 0;
static bool   g_glReady = false;

static void buildProceduralTextures() {
    glGenTextures(NUM_TEXTURES, g_tex);
    float maxAniso = 1.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
    float targetAniso = std::min(8.0f, std::max(1.0f, maxAniso));

    const int W = 512, H = 512;
    std::vector<unsigned char> rgb(W * H * 3);

    for (int tid = 0; tid < TEX_GIS_MAP; tid++) {
        parallelFor(H, [&](int y0, int y1) {
            for (int y = y0; y < y1; y++) {
                float v = (float)y / (float)H;
                for (int x = 0; x < W; x++) {
                    float u = (float)x / (float)W;
                    float n1 = fbm2(u * 8.0f, v * 8.0f, 5, 2.0f, 0.5f, 8);
                    float n2 = fbm2(u * 32.0f, v * 32.0f, 4, 2.0f, 0.5f, 32);
                    float r = 128, g = 128, b = 128;

                    if (tid == TEX_GRASS) {
                        // Ege Akdeniz vejetasyon / maki / kizilcam ve zengin zemin dokusu
                        float patch = 0.5f + 0.5f * n1;
                        float micro = 0.5f + 0.5f * n2;
                        float soilMix = clampf(n1 * 0.65f + n2 * 0.35f, 0.0f, 1.0f);
                        float grnR = lerpf(34.0f, 74.0f, patch);
                        float grnG = lerpf(78.0f, 136.0f, patch);
                        float grnB = lerpf(26.0f, 46.0f, patch);
                        r = lerpf(grnR, 114.0f, soilMix * 0.32f) + (micro - 0.5f) * 24.0f;
                        g = lerpf(grnG, 102.0f, soilMix * 0.32f) + (micro - 0.5f) * 28.0f;
                        b = lerpf(grnB, 66.0f,  soilMix * 0.32f) + (micro - 0.5f) * 18.0f;
                    } else if (tid == TEX_ROCK) {
                        // Yamanlar / Catalkaya kalker ve bazalt kaya dokusu (sedimanter tabakali)
                        float rg = ridged2(u * 10.0f, v * 10.0f, 5);
                        float cr = std::sin((u * 14.0f + n1 * 2.2f) * PI_F * 2.0f) * 0.5f + 0.5f;
                        float val = clampf(0.42f * rg + 0.35f * (0.5f + 0.5f * n2) + 0.23f * cr, 0.0f, 1.0f);
                        r = lerpf(86.0f, 172.0f, val);
                        g = lerpf(84.0f, 166.0f, val);
                        b = lerpf(78.0f, 158.0f, val);
                    } else if (tid == TEX_WATER) {
                        // Ege Korfezi ince dalgacikli kostik su dokusu
                        float w1 = std::sin((u * 16.0f + n1 * 2.1f) * 2.0f * PI_F);
                        float w2 = std::cos((v * 22.0f - u * 6.0f + n2 * 1.9f) * 2.0f * PI_F);
                        float ripple = std::sin((u * 40.0f + v * 36.0f + n2 * 3.0f) * 2.0f * PI_F) * 0.5f + 0.5f;
                        float foam = clampf((n2 - 0.46f) * 4.5f, 0.0f, 1.0f);
                        float c = clampf(0.48f + 0.24f * w1 + 0.18f * w2 + 0.10f * ripple, 0.0f, 1.0f);
                        r = lerpf(16.0f, 56.0f, c) + foam * 145.0f;
                        g = lerpf(62.0f, 146.0f, c) + foam * 168.0f;
                        b = lerpf(118.0f, 206.0f, c) + foam * 182.0f;
                    } else if (tid == TEX_ASPHALT) {
                        // Yuksek kontrastli ince agrega asfalt ve lastik izi dokusu
                        float grain = (hash01(x, y) - 0.5f) * 20.0f + n2 * 12.0f;
                        float wear = std::sin(u * 2.0f * PI_F) * 4.0f;
                        float base = 42.0f + grain + wear;
                        r = base; g = base + 1.5f; b = base + 4.0f;
                    } else if (tid == TEX_CONCRETE) {
                        // Akdeniz granit / kordon tas kaldirim derzleri
                        int gx = x % 32, gy = y % 32;
                        float joint = (gx <= 1 || gy <= 1) ? -28.0f : 0.0f;
                        float tileVar = (hash01(x / 32, y / 32) - 0.5f) * 12.0f;
                        float base = 196.0f + n2 * 14.0f + joint + tileVar;
                        r = base + 3.0f; g = base + 1.0f; b = base - 3.0f;
                    } else if (tid == TEX_APT_FACADE) {
                        // Modern Akdeniz apartman cephesi: kat silmeleri, derin balkon golgeleri, reflekte pencereler
                        int wx = x % 64, wy = y % 64;
                        int floorId = y / 64, colId = x / 64;
                        bool isSlabLedge = (wy < 4 || wy >= 61);
                        bool isColumn = (wx < 5 || wx >= 59);
                        bool isBalconyRail = (wy >= 43 && wy <= 56 && wx >= 6 && wx <= 58);
                        bool isBalconyShadow = (wy >= 7 && wy < 14 && wx >= 7 && wx <= 57);
                        bool isWindow = (wy >= 13 && wy <= 42 && wx >= 11 && wx <= 53);
                        bool isFrame = isWindow && (wx < 14 || wx > 50 || wy < 16 || wy > 39 || (wx >= 31 && wx <= 33));
                        bool isAcUnit = (wy >= 45 && wy <= 54 && wx >= 9 && wx <= 20 && ((floorId + colId) % 3 == 0));

                        float plaster = 232.0f + n2 * 9.0f - (isSlabLedge ? 36.0f : 0.0f)
                                      - (isColumn ? 10.0f : 0.0f) - (isBalconyShadow ? 42.0f : 0.0f);
                        r = plaster; g = plaster - 4.0f; b = plaster - 11.0f;
                        if (isWindow) {
                            if (isFrame) {
                                r = 242; g = 244; b = 248;
                            } else {
                                float refl = 0.5f + 0.5f * std::sin((float)wy * 0.16f + (float)wx * 0.06f);
                                float roomTint = hash01(colId * 13 + (wx > 32), floorId * 29);
                                r = lerpf(28.0f, 76.0f, refl) + roomTint * 16.0f;
                                g = lerpf(56.0f, 128.0f, refl) + roomTint * 18.0f;
                                b = lerpf(98.0f, 198.0f, refl) + roomTint * 20.0f;
                            }
                        }
                        if (isBalconyRail) {
                            if (wy <= 45) { r = 195; g = 205; b = 214; }
                            else { r = 135; g = 175; b = 198; }
                        }
                        if (isAcUnit) {
                            r = 236; g = 234; b = 230;
                        }
                    } else if (tid == TEX_GLASS_TOWER) {
                        // Bayrakli / Mavisehir cam giydirme perde duvar: gokyuzu yansimasi + koyu aluminyum karkas
                        int wx = x % 32, wy = y % 32;
                        bool mullion = (wx < 2 || wy < 2);
                        bool spandrel = (wy >= 2 && wy < 7);
                        int cid = x / 32, fid = y / 32;
                        if (mullion) {
                            r = 42; g = 52; b = 64;
                        } else if (spandrel) {
                            r = 95; g = 122; b = 148;
                        } else {
                            float h = hash01(cid * 19, fid * 37);
                            float skyStreak = 0.5f + 0.5f * std::sin((u * 4.0f + v * 2.5f) * PI_F + n1 * 1.6f);
                            if (h > 0.68f) {
                                r = lerpf(168.0f, 226.0f, skyStreak);
                                g = lerpf(208.0f, 242.0f, skyStreak);
                                b = lerpf(236.0f, 255.0f, skyStreak);
                            } else {
                                r = lerpf(26.0f, 78.0f, skyStreak);
                                g = lerpf(74.0f, 152.0f, skyStreak);
                                b = lerpf(130.0f, 222.0f, skyStreak);
                            }
                        }
                    } else if (tid == TEX_INDUSTRIAL) {
                        // Cigli AOSB / Ata Sanayi oluklu sac + cati bandi aydinlatma
                        float corrug = std::sin(u * 128.0f * 2.0f * PI_F) * 16.0f;
                        bool skylight = ((y % 128) >= 22 && (y % 128) <= 44 && (x % 64) >= 8 && (x % 64) <= 56);
                        if (skylight) {
                            r = 185; g = 216; b = 242;
                        } else {
                            r = 136.0f + corrug + n2 * 10.0f;
                            g = 152.0f + corrug + n2 * 10.0f;
                            b = 174.0f + corrug + n2 * 10.0f;
                        }
                    } else if (tid == TEX_ROOF_TILE) {
                        // Ege Marsilya / Alaturka kiremit cati dokusu (derin oluk golgeli sicak terracotta)
                        float waveX = std::sin(u * 64.0f * 2.0f * PI_F) * 26.0f;
                        float rowStep = ((y % 16) < 3) ? -44.0f : (((y % 16) >= 13) ? 12.0f : 0.0f);
                        float tileVar = (hash01(x / 8, y / 16) - 0.5f) * 18.0f;
                        r = 192.0f + waveX + rowStep + tileVar + n1 * 12.0f;
                        g = 76.0f  + waveX * 0.45f + rowStep * 0.45f + tileVar * 0.5f + n1 * 8.0f;
                        b = 44.0f  + waveX * 0.30f + rowStep * 0.30f + tileVar * 0.3f + n1 * 6.0f;
                    } else if (tid == TEX_BALLAST) {
                        // IZBAN kirma tas balast + beton travers
                        float stone = (hash01(x / 2, y / 2) - 0.5f) * 55.0f + n2 * 28.0f;
                        bool sleeper = ((y % 32) >= 8 && (y % 32) <= 20 && x >= 64 && x <= 448);
                        if (sleeper) {
                            r = 114.0f + n2 * 14.0f; g = 110.0f + n2 * 14.0f; b = 106.0f + n2 * 14.0f;
                        } else {
                            r = 118.0f + stone; g = 114.0f + stone; b = 110.0f + stone;
                        }
                    } else if (tid == TEX_NIGHT_WINDOWS) {
                        // Gece parlayan pencere katmani (sadece pencere icleri isikli, duvarlar saf 0,0,0 -> GL_ONE toplamsal)
                        int wx = x % 64, wy = y % 64;
                        int floorId = y / 64, colId = x / 64;
                        bool leftPane = (wx >= 11 && wx <= 29 && wy >= 14 && wy <= 43);
                        bool rightPane = (wx >= 34 && wx <= 52 && wy >= 14 && wy <= 43);
                        r = 0.0f; g = 0.0f; b = 0.0f;
                        if (leftPane || rightPane) {
                            int paneIdx = colId * 2 + (rightPane ? 1 : 0);
                            float h = hash01(paneIdx * 31 + 7, floorId * 53 + 11);
                            if (h > 0.48f) {
                                float tintH = hash01(paneIdx * 17 + 3, floorId * 41 + 9);
                                float shade = 0.72f + 0.28f * (1.0f - std::fabs((wy - 28.5f) / 14.0f));
                                if (tintH < 0.45f) {
                                    // Sicak tungsten apartman isigi
                                    r = 255.0f * shade; g = 198.0f * shade; b = 104.0f * shade;
                                } else if (tintH < 0.75f) {
                                    // Altin / yumusak beyaz ev isigi
                                    r = 255.0f * shade; g = 228.0f * shade; b = 165.0f * shade;
                                } else if (tintH < 0.90f) {
                                    // Soguk beyaz / ofis LED
                                    r = 188.0f * shade; g = 226.0f * shade; b = 255.0f * shade;
                                } else {
                                    // Hafif camgobegi TV/ofis pariltisi
                                    r = 148.0f * shade; g = 198.0f * shade; b = 255.0f * shade;
                                }
                            }
                        }
                    }
                    int idx = (y * W + x) * 3;
                    rgb[idx + 0] = (unsigned char)clampf(r, 0.0f, 255.0f);
                    rgb[idx + 1] = (unsigned char)clampf(g, 0.0f, 255.0f);
                    rgb[idx + 2] = (unsigned char)clampf(b, 0.0f, 255.0f);
                }
            }
        });

        glBindTexture(GL_TEXTURE_2D, g_tex[tid]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, targetAniso);
        gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, W, H, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    }

    // Yumusak duman / alev / patlama partikul sprite dokusu (64x64 RGBA)
    glGenTextures(1, &g_texParticle);
    const int PW = 64, PH = 64;
    std::vector<unsigned char> rgba(PW * PH * 4);
    for (int y = 0; y < PH; y++) {
        float dy = ((float)y + 0.5f) / (float)PH * 2.0f - 1.0f;
        for (int x = 0; x < PW; x++) {
            float dx = ((float)x + 0.5f) / (float)PW * 2.0f - 1.0f;
            float r2 = dx * dx + dy * dy;
            float a = (r2 < 1.0f) ? std::pow(1.0f - r2, 1.85f) : 0.0f;
            int idx = (y * PW + x) * 4;
            rgba[idx + 0] = 255;
            rgba[idx + 1] = 255;
            rgba[idx + 2] = 255;
            rgba[idx + 3] = (unsigned char)clampf(a * 255.0f, 0.0f, 255.0f);
        }
    }
    glBindTexture(GL_TEXTURE_2D, g_texParticle);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, PW, PH, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

    glBindTexture(GL_TEXTURE_2D, 0);
    g_glReady = true;
}

// =====================================================================================
//  BOLUM 5 : CATMULL-ROM SPLINE, RELAXCURVATURE, WORLDMASKGRID (2048x2048) & ARAZI
// =====================================================================================
static Vec2 catmullRom2D(const Vec2& p0, const Vec2& p1, const Vec2& p2, const Vec2& p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return (p1 * 2.0f + (p2 - p0) * t + (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2 +
            (-p0 + p1 * 3.0f - p2 * 3.0f + p3) * t3) * 0.5f;
}

static Vec3 catmullRom3D(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return (p1 * 2.0f + (p2 - p0) * t + (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2 +
            (-p0 + p1 * 3.0f - p2 * 3.0f + p3) * t3) * 0.5f;
}

// Egrilik yumusatmasi (relaxCurvature) - keskin donusleri minRadius yaricapina yumusatir
static void relaxCurvature2D(std::vector<Vec2>& pts, float minRadius, int iterations) {
    int n = (int)pts.size();
    if (n < 3) return;
    std::vector<Vec2> tmp = pts;
    for (int it = 0; it < iterations; it++) {
        for (int i = 1; i < n - 1; i++) {
            Vec2 a = pts[i - 1], b = pts[i], c = pts[i + 1];
            Vec2 ab = b - a, bc = c - b;
            float lab = len2(ab), lbc = len2(bc);
            if (lab < 1e-3f || lbc < 1e-3f) continue;
            Vec2 u1 = ab * (1.0f / lab), u2 = bc * (1.0f / lbc);
            float cosA = clampf(dot2(u1, u2), -1.0f, 1.0f);
            float ang = std::acos(cosA);
            float avgL = 0.5f * (lab + lbc);
            float radius = (ang > 1e-4f) ? (avgL / ang) : 1e9f;
            float w = (radius < minRadius) ? 0.42f : 0.16f;
            Vec2 mid = (a + c) * 0.5f;
            tmp[i] = b * (1.0f - w) + mid * w;
        }
        pts = tmp;
    }
}

static void relaxCurvature3D(std::vector<Vec3>& pts, float minRadius, int iterations) {
    int n = (int)pts.size();
    if (n < 3) return;
    std::vector<Vec3> tmp = pts;
    for (int it = 0; it < iterations; it++) {
        for (int i = 1; i < n - 1; i++) {
            Vec2 a(pts[i - 1].x, pts[i - 1].z);
            Vec2 b(pts[i].x, pts[i].z);
            Vec2 c(pts[i + 1].x, pts[i + 1].z);
            float lab = len2(b - a), lbc = len2(c - b);
            if (lab < 1e-3f || lbc < 1e-3f) continue;
            Vec2 u1 = (b - a) * (1.0f / lab), u2 = (c - b) * (1.0f / lbc);
            float ang = std::acos(clampf(dot2(u1, u2), -1.0f, 1.0f));
            float radius = (ang > 1e-4f) ? (0.5f * (lab + lbc) / ang) : 1e9f;
            float w = (radius < minRadius) ? 0.45f : 0.15f;
            Vec3 mid = (pts[i - 1] + pts[i + 1]) * 0.5f;
            tmp[i].x = lerpf(pts[i].x, mid.x, w);
            tmp[i].z = lerpf(pts[i].z, mid.z, w);
            tmp[i].y = lerpf(pts[i].y, mid.y, 0.25f);
        }
        pts = tmp;
    }
}

// 2D polilinini sabit adimla (orn. 15m) yeniden ornekle
static std::vector<Vec2> resamplePolyline2D(const std::vector<Vec2>& in, float stepM) {
    std::vector<Vec2> out;
    if (in.size() < 2) return in;
    out.push_back(in[0]);
    float acc = 0.0f;
    Vec2 prev = in[0];
    for (size_t i = 1; i < in.size(); i++) {
        Vec2 cur = in[i];
        float segL = len2(cur - prev);
        while (acc + segL >= stepM && segL > 1e-5f) {
            float need = stepM - acc;
            float t = need / segL;
            Vec2 np = prev + (cur - prev) * t;
            out.push_back(np);
            prev = np;
            segL = len2(cur - prev);
            acc = 0.0f;
        }
        acc += segL;
        prev = cur;
    }
    if (len2(out.back() - in.back()) > 1.0f) out.push_back(in.back());
    return out;
}

static std::vector<Vec3> resamplePolyline3D(const std::vector<Vec3>& in, float stepM) {
    std::vector<Vec3> out;
    if (in.size() < 2) return in;
    out.push_back(in[0]);
    float acc = 0.0f;
    Vec3 prev = in[0];
    for (size_t i = 1; i < in.size(); i++) {
        Vec3 cur = in[i];
        float dx = cur.x - prev.x, dz = cur.z - prev.z;
        float segL = std::sqrt(dx * dx + dz * dz);
        while (acc + segL >= stepM && segL > 1e-5f) {
            float need = stepM - acc;
            float t = need / segL;
            Vec3 np = lerpV(prev, cur, t);
            out.push_back(np);
            prev = np;
            dx = cur.x - prev.x; dz = cur.z - prev.z;
            segL = std::sqrt(dx * dx + dz * dz);
            acc = 0.0f;
        }
        acc += segL;
        prev = cur;
    }
    if (len(out.back() - in.back()) > 1.0f) out.push_back(in.back());
    return out;
}

// Koridor segment yapisi (IZBAN, Otoyol, Pist)
struct CorridorSeg {
    Vec2 a, b, tang;
    float elevA, elevB, halfW;
    int type;
};

// Dunya verileri (global)
static std::vector<Vec2> g_coastSpline;      // 15m adimli kiyi egrisi (34 GPS noktasindan)
static std::vector<Vec2> g_coastTangents;    // Her kiyi noktasinda birim teget
static std::vector<Vec2> g_coastInNormals;   // Her kiyi noktasindan karaya dogru puruzsuz birim normal
static std::vector<Vec2> g_seaPolygon;       // Batida (-22000,-16500) ve (-22000,+5000) ile kapali korfez poligonu
static std::vector<Vec3> g_izbanSpline;      // 28 istasyon Catmull-Rom + relaxCurvature(180m) 15m ornekli
static std::vector<Vec2> g_izbanTangents;
static std::vector<float> g_izbanCumDist;
static float g_izbanTotalLen = 0.0f;
static float g_stationSplineS[NUM_STATIONS]; // Her istasyonun spline uzerindeki yay uzunlugu
static Vec3 g_stationPos[NUM_STATIONS];
static Vec2 g_stationTang[NUM_STATIONS];

struct RoadSpline {
    std::string name;
    int type;
    float halfW;
    int lanes;
    std::vector<Vec3> pts;
    std::vector<Vec2> tangs;
    std::vector<float> cumDist;
    float totalLen = 0.0f;
};
static std::vector<RoadSpline> g_roads;
static std::vector<CorridorSeg> g_corridorSegs;

// 2048x2048 O(1) WorldMaskGrid matrisi (~20m cozunurluk)
static const int MASK_N = 2048;
struct WorldMaskCell {
    float signedSeaDist; // denizde < 0, karada > 0
    float corridorDist;  // yol/ray/pist koridor kenarina mesafe (>0 disarida, <=0 koridor icinde)
    float corridorElev;  // en yakin koridorun hedef irtifasi
    float corridorBlend; // 0..1 koridor tesviye agirligi
    float urbanBlend;    // 0..1 kentsel parsel / cadde zemini orani
    float coastTangX, coastTangZ;
    float roadTangX, roadTangZ;
    float elev;          // nihai arazi yuksekligi
};
static std::vector<WorldMaskCell> g_maskGrid(MASK_N * MASK_N);

static inline void worldToGridF(float wx, float wz, float& gx, float& gz) {
    gx = (wx - WORLD_X0) / (WORLD_X1 - WORLD_X0) * (MASK_N - 1);
    gz = (wz - WORLD_Z0) / (WORLD_Z1 - WORLD_Z0) * (MASK_N - 1);
}
static inline void gridToWorld(int ix, int iz, float& wx, float& wz) {
    wx = WORLD_X0 + ((float)ix / (float)(MASK_N - 1)) * (WORLD_X1 - WORLD_X0);
    wz = WORLD_Z0 + ((float)iz / (float)(MASK_N - 1)) * (WORLD_Z1 - WORLD_Z0);
}
static inline const WorldMaskCell& maskCellAt(int ix, int iz) {
    ix = clampi(ix, 0, MASK_N - 1);
    iz = clampi(iz, 0, MASK_N - 1);
    return g_maskGrid[iz * MASK_N + ix];
}
static inline const WorldMaskCell& queryWorldMask(float wx, float wz) {
    float gx, gz;
    worldToGridF(wx, wz, gx, gz);
    return maskCellAt((int)(gx + 0.5f), (int)(gz + 0.5f));
}

// Bilinear enterpolasyonlu yukseklik ve signedSeaDist sorgusu
static float sampleTerrainElev(float wx, float wz) {
    float gx, gz;
    worldToGridF(wx, wz, gx, gz);
    int x0 = clampi((int)std::floor(gx), 0, MASK_N - 2);
    int z0 = clampi((int)std::floor(gz), 0, MASK_N - 2);
    float fx = clampf(gx - x0, 0.0f, 1.0f);
    float fz = clampf(gz - z0, 0.0f, 1.0f);
    const WorldMaskCell& c00 = g_maskGrid[z0 * MASK_N + x0];
    const WorldMaskCell& c10 = g_maskGrid[z0 * MASK_N + x0 + 1];
    const WorldMaskCell& c01 = g_maskGrid[(z0 + 1) * MASK_N + x0];
    const WorldMaskCell& c11 = g_maskGrid[(z0 + 1) * MASK_N + x0 + 1];
    float e0 = lerpf(c00.elev, c10.elev, fx);
    float e1 = lerpf(c01.elev, c11.elev, fx);
    return lerpf(e0, e1, fz);
}

static float sampleSignedSeaDist(float wx, float wz) {
    float gx, gz;
    worldToGridF(wx, wz, gx, gz);
    int x0 = clampi((int)std::floor(gx), 0, MASK_N - 2);
    int z0 = clampi((int)std::floor(gz), 0, MASK_N - 2);
    float fx = clampf(gx - x0, 0.0f, 1.0f);
    float fz = clampf(gz - z0, 0.0f, 1.0f);
    const WorldMaskCell& c00 = g_maskGrid[z0 * MASK_N + x0];
    const WorldMaskCell& c10 = g_maskGrid[z0 * MASK_N + x0 + 1];
    const WorldMaskCell& c01 = g_maskGrid[(z0 + 1) * MASK_N + x0];
    const WorldMaskCell& c11 = g_maskGrid[(z0 + 1) * MASK_N + x0 + 1];
    float d0 = lerpf(c00.signedSeaDist, c10.signedSeaDist, fx);
    float d1 = lerpf(c01.signedSeaDist, c11.signedSeaDist, fx);
    return lerpf(d0, d1, fz);
}

static Vec3 sampleTerrainNormal(float wx, float wz) {
    const float eps = 12.0f;
    float hL = sampleTerrainElev(wx - eps, wz);
    float hR = sampleTerrainElev(wx + eps, wz);
    float hD = sampleTerrainElev(wx, wz - eps);
    float hU = sampleTerrainElev(wx, wz + eps);
    return norm(Vec3(hL - hR, 2.0f * eps, hD - hU));
}

// =====================================================================================
//  GERCEK OVERTURE + OSM + DEM VERI KATMANI (izmir_osm_world.bin v1 & v2)
// =====================================================================================
static const int DEM_N = 1024;
static std::vector<float> g_realDEM(DEM_N * DEM_N, 0.0f);
static bool g_hasRealDEM = false;

enum GisLandClass : uint8_t {
    LC_SEA = 0,
    LC_SALT_WETLAND = 1,
    LC_WATER = 2,
    LC_URBAN = 3,
    LC_INDUSTRIAL = 4,
    LC_PARK = 5,
    LC_FOREST = 6,
    LC_SHRUB = 7,
    LC_CROP = 8,
    LC_BARREN = 9
};
static std::vector<uint8_t> g_gisGrid(MASK_N * MASK_N, LC_SHRUB);
static bool g_hasGisGrid = false;
static std::vector<Vec2> g_realTrees;

static inline float terrainRenderElevCell(const WorldMaskCell& mc) {
    if (mc.signedSeaDist >= 0.0f) return std::max(0.45f, mc.elev);
    if (mc.signedSeaDist > -28.0f) {
        return lerpf(std::max(0.45f, mc.elev), -0.05f, -mc.signedSeaDist / 28.0f);
    }
    float t = clampf((-mc.signedSeaDist - 28.0f) / 36.0f, 0.0f, 1.0f);
    return lerpf(-0.05f, -3.5f, t);
}

static float sampleRenderedTerrainMaxElev(float wx, float wz) {
    float bilinear = sampleTerrainElev(wx, wz);
    if (!g_hasGisGrid) return bilinear;
    float gx, gz;
    worldToGridF(wx, wz, gx, gz);
    const int step = 2;
    int ix0 = clampi((((int)std::floor(gx)) & ~1), 0, MASK_N - 4);
    int iz0 = clampi((((int)std::floor(gz)) & ~1), 0, MASK_N - 4);
    float fx = clampf((gx - (float)ix0) * 0.5f, 0.0f, 1.0f);
    float fz = clampf((gz - (float)iz0) * 0.5f, 0.0f, 1.0f);
    float h00 = terrainRenderElevCell(maskCellAt(ix0,        iz0));
    float h10 = terrainRenderElevCell(maskCellAt(ix0 + step, iz0));
    float h01 = terrainRenderElevCell(maskCellAt(ix0,        iz0 + step));
    float h11 = terrainRenderElevCell(maskCellAt(ix0 + step, iz0 + step));
    float triElev = (fx + fz <= 1.0f)
        ? (h00 + fx * (h10 - h00) + fz * (h01 - h00))
        : (h11 + (1.0f - fx) * (h01 - h11) + (1.0f - fz) * (h10 - h11));
    return std::max(bilinear, triElev);
}

struct V2RailSeg {
    uint8_t rtype;     // 0=izban(standard_gauge), 1=tram, 2=metro(surface)
    uint8_t isBridge;
    std::vector<Vec2> pts;
};
static std::vector<V2RailSeg> g_v2Rails;

struct V2RoadSeg {
    uint8_t tier;      // 0=motorway/trunk, 1=primary, 2=secondary, 3=tertiary, 4=residential/service
    uint8_t isBridge;
    int8_t level;
    float halfW;
    Vec2 a, b;
};
struct V2MajorRoadPoly {
    uint8_t tier;
    uint8_t isBridge;
    int8_t level;
    float halfW;
    std::vector<Vec2> pts;
    std::vector<float> ys;
};
static std::vector<V2MajorRoadPoly> g_v2MajorRoads;
static std::vector<RoadSpline> g_v2DriveRoutes;
static bool g_hasV2Roads = false;

struct OsmSideStreetSeg { Vec2 a, b; };
struct OsmGardenPoly { int gtype; std::vector<Vec2> pts; };

static const float OSM_CHUNK_SZ = 400.0f;
static const int OSM_CK_NX = (int)((WORLD_X1 - WORLD_X0) / OSM_CHUNK_SZ) + 1; // 88
static const int OSM_CK_NZ = (int)((WORLD_Z1 - WORLD_Z0) / OSM_CHUNK_SZ) + 1; // 113
static std::vector<std::vector<OsmSideStreetSeg>> g_chunkSideStreets(OSM_CK_NX * OSM_CK_NZ);
static std::vector<std::vector<V2RoadSeg>> g_chunkV2Roads(OSM_CK_NX * OSM_CK_NZ);
static std::vector<std::vector<OsmGardenPoly>> g_chunkGardens(OSM_CK_NX * OSM_CK_NZ);

struct OsmBuildingRec {
    float x, z, w, d, yaw, h;
    uint8_t kind, r, g, b;
};
static std::vector<OsmBuildingRec> g_osmBuildings;

static inline uint8_t sampleGisClass(float wx, float wz) {
    if (!g_hasGisGrid) return LC_SHRUB;
    float gx, gz;
    worldToGridF(wx, wz, gx, gz);
    int ix = clampi((int)(gx + 0.5f), 0, MASK_N - 1);
    int iz = clampi((int)(gz + 0.5f), 0, MASK_N - 1);
    return g_gisGrid[iz * MASK_N + ix];
}

static std::vector<Vec2> subdividePolylinePreserveCorners(const std::vector<Vec2>& pts, float maxStepM) {
    std::vector<Vec2> out;
    if (pts.size() < 2) return pts;
    out.push_back(pts[0]);
    for (size_t k = 0; k + 1 < pts.size(); k++) {
        Vec2 a = pts[k], b = pts[k + 1];
        float d = len2(b - a);
        if (d < 0.25f) continue;
        int nSub = std::max(1, (int)std::ceil(d / maxStepM));
        for (int s = 1; s <= nSub; s++) {
            float t = (float)s / (float)nSub;
            out.push_back(Vec2(lerpf(a.x, b.x, t), lerpf(a.z, b.z, t)));
        }
    }
    return out;
}

static bool isOnRunwayStrip(Vec2 p) {
    Vec2 ltbj = geoPt(LTBJ_LAT, LTBJ_LON);
    Vec2 d1 = headingDir(LTBJ_HDG);
    Vec2 r1(-d1.z, d1.x);
    for (int s = -1; s <= 1; s += 2) {
        Vec2 rc = ltbj + r1 * ((float)s * LTBJ_SEP * 0.5f);
        Vec2 rel = p - rc;
        if (std::fabs(dot2(rel, d1)) < LTBJ_LEN * 0.52f && std::fabs(dot2(rel, r1)) < 42.0f) return true;
    }
    Vec2 ltbl = geoPt(LTBL_LAT, LTBL_LON);
    Vec2 d2 = headingDir(LTBL_HDG);
    Vec2 r2(-d2.z, d2.x);
    Vec2 rel2 = p - ltbl;
    if (std::fabs(dot2(rel2, d2)) < LTBL_LEN * 0.52f && std::fabs(dot2(rel2, r2)) < 42.0f) return true;
    return false;
}

static void pushRoadSegToV2(uint8_t tier, uint8_t isBridge, int8_t level, float halfW, const std::vector<Vec2>& pts) {
    if (pts.size() < 2) return;
    if (tier <= 2 || isBridge) {
        std::vector<Vec2> subPts = subdividePolylinePreserveCorners(pts, 16.0f);
        if (subPts.size() >= 2) {
            g_v2MajorRoads.push_back({tier, isBridge, level, halfW, std::move(subPts), {}});
        }
        return;
    }
    for (size_t k = 0; k + 1 < pts.size(); k++) {
        Vec2 p0 = pts[k], p1 = pts[k + 1];
        float dx = p1.x - p0.x, dz = p1.z - p0.z;
        float d = std::sqrt(dx * dx + dz * dz);
        if (d < 0.5f) continue;
        int nSub = std::max(1, (int)std::ceil(d / 14.0f));
        for (int s = 0; s < nSub; s++) {
            float t0 = (float)s / (float)nSub;
            float t1 = (float)(s + 1) / (float)nSub;
            Vec2 a(p0.x + dx * t0, p0.z + dz * t0);
            Vec2 b(p0.x + dx * t1, p0.z + dz * t1);
            Vec2 mid(0.5f * (a.x + b.x), 0.5f * (a.z + b.z));
            if (tier >= 4 && (isOnRunwayStrip(a) || isOnRunwayStrip(b) || isOnRunwayStrip(mid))) continue;
            int cx = clampi((int)((mid.x - WORLD_X0) / OSM_CHUNK_SZ), 0, OSM_CK_NX - 1);
            int cz = clampi((int)((mid.z - WORLD_Z0) / OSM_CHUNK_SZ), 0, OSM_CK_NZ - 1);
            g_chunkV2Roads[cz * OSM_CK_NX + cx].push_back({tier, isBridge, level, halfW, a, b});
        }
    }
}

static float sampleOsmDem(float wx, float wz) {
    if (!g_hasRealDEM) return 4.0f;
    float gx = (wx - WORLD_X0) / (WORLD_X1 - WORLD_X0) * (DEM_N - 1);
    float gz = (wz - WORLD_Z0) / (WORLD_Z1 - WORLD_Z0) * (DEM_N - 1);
    int x0 = clampi((int)std::floor(gx), 0, DEM_N - 2);
    int z0 = clampi((int)std::floor(gz), 0, DEM_N - 2);
    float fx = clampf(gx - x0, 0.0f, 1.0f);
    float fz = clampf(gz - z0, 0.0f, 1.0f);
    float e00 = g_realDEM[z0 * DEM_N + x0];
    float e10 = g_realDEM[z0 * DEM_N + x0 + 1];
    float e01 = g_realDEM[(z0 + 1) * DEM_N + x0];
    float e11 = g_realDEM[(z0 + 1) * DEM_N + x0 + 1];
    float e0 = e00 * (1.0f - fx) + e10 * fx;
    float e1 = e01 * (1.0f - fx) + e11 * fx;
    return std::max(2.0f, e0 * (1.0f - fz) + e1 * fz);
}

static bool loadOsmWorldBin(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::printf("[OSM BIN] %s bulunamadi, sentetik harita kullanilacak.\n", path);
        return false;
    }
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "IZMIROSM", 8) != 0) {
        std::fclose(f);
        return false;
    }
    uint32_t ver = 0;
    std::fread(&ver, 4, 1, f);
    float refs[6];
    std::fread(refs, sizeof(float), 6, f);
    uint32_t dw = 0, dh = 0;
    std::fread(&dw, 4, 1, f);
    std::fread(&dh, 4, 1, f);
    if (dw == DEM_N && dh == DEM_N) {
        std::fread(g_realDEM.data(), sizeof(float), DEM_N * DEM_N, f);
        g_hasRealDEM = true;
    } else {
        std::fseek(f, (long)dw * (long)dh * (long)sizeof(float), SEEK_CUR);
    }

    if (ver >= 2) {
        uint32_t mn = 0;
        std::fread(&mn, 4, 1, f);
        if (mn == (uint32_t)MASK_N) {
            std::fread(g_gisGrid.data(), 1, MASK_N * MASK_N, f);
            g_hasGisGrid = true;
        } else {
            std::fseek(f, (long)mn * (long)mn, SEEK_CUR);
        }

        uint32_t nTrees = 0;
        std::fread(&nTrees, 4, 1, f);
        g_realTrees.clear();
        if (nTrees > 0) {
            std::vector<float> tbuf(nTrees * 2);
            std::fread(tbuf.data(), sizeof(float), nTrees * 2, f);
            g_realTrees.reserve(nTrees);
            for (uint32_t i = 0; i < nTrees; i++) {
                g_realTrees.emplace_back(tbuf[2 * i], tbuf[2 * i + 1]);
            }
        }

        uint32_t nRail = 0;
        std::fread(&nRail, 4, 1, f);
        g_v2Rails.clear();
        g_v2Rails.reserve(nRail);
        for (uint32_t i = 0; i < nRail; i++) {
            uint8_t rtype = 0, isBr = 0;
            uint16_t np = 0;
            std::fread(&rtype, 1, 1, f);
            std::fread(&isBr, 1, 1, f);
            std::fread(&np, 2, 1, f);
            std::vector<float> rbuf(np * 2);
            if (np > 0) std::fread(rbuf.data(), sizeof(float), np * 2, f);
            V2RailSeg rs;
            rs.rtype = rtype;
            rs.isBridge = isBr;
            rs.pts.reserve(np);
            for (uint16_t k = 0; k < np; k++) rs.pts.emplace_back(rbuf[2 * k], rbuf[2 * k + 1]);
            if (rs.pts.size() >= 2) {
                rs.pts = subdividePolylinePreserveCorners(rs.pts, 16.0f);
                g_v2Rails.push_back(std::move(rs));
            }
        }

        uint32_t nRoads = 0;
        std::fread(&nRoads, 4, 1, f);
        g_v2MajorRoads.clear();
        g_v2DriveRoutes.clear();
        for (auto& ck : g_chunkV2Roads) ck.clear();
        for (uint32_t i = 0; i < nRoads; i++) {
            uint8_t tier = 4, isBr = 0, hwDm = 25;
            int8_t lvl = 0;
            uint16_t np = 0;
            std::fread(&tier, 1, 1, f);
            std::fread(&isBr, 1, 1, f);
            std::fread(&lvl, 1, 1, f);
            std::fread(&hwDm, 1, 1, f);
            std::fread(&np, 2, 1, f);
            std::vector<float> rbuf(np * 2);
            if (np > 0) std::fread(rbuf.data(), sizeof(float), np * 2, f);
            std::vector<Vec2> pts;
            pts.reserve(np);
            for (uint16_t k = 0; k < np; k++) pts.emplace_back(rbuf[2 * k], rbuf[2 * k + 1]);
            pushRoadSegToV2(tier, isBr, lvl, (float)hwDm * 0.1f, pts);
        }
        g_hasV2Roads = (nRoads > 0);

        uint32_t nBldg = 0;
        std::fread(&nBldg, 4, 1, f);
        g_osmBuildings.resize(nBldg);
        for (uint32_t i = 0; i < nBldg; i++) {
            float bdat[6];
            uint8_t cdat[4];
            std::fread(bdat, sizeof(float), 6, f);
            std::fread(cdat, 1, 4, f);
            g_osmBuildings[i].x = bdat[0];
            g_osmBuildings[i].z = bdat[1];
            g_osmBuildings[i].w = bdat[2];
            g_osmBuildings[i].d = bdat[3];
            g_osmBuildings[i].yaw = bdat[4];
            g_osmBuildings[i].h = bdat[5];
            g_osmBuildings[i].kind = cdat[0];
            g_osmBuildings[i].r = cdat[1];
            g_osmBuildings[i].g = cdat[2];
            g_osmBuildings[i].b = cdat[3];
        }

        std::fclose(f);
        std::printf("[GIS BIN v2] %s yuklendi! DEM=%ux%u, GIS=%ux%u, Agac=%zu, Ray=%zu, Yol=%u (Ana=%zu), Bina=%zu\n",
                    path, dw, dh, mn, mn, g_realTrees.size(), g_v2Rails.size(), nRoads, g_v2MajorRoads.size(), g_osmBuildings.size());
        return true;
    }

    uint32_t nCoast = 0;
    std::fread(&nCoast, 4, 1, f);
    for (uint32_t i = 0; i < nCoast; i++) {
        uint32_t np = 0;
        std::fread(&np, 4, 1, f);
        std::fseek(f, np * 8, SEEK_CUR);
    }

    uint32_t nRail = 0;
    std::fread(&nRail, 4, 1, f);
    for (uint32_t i = 0; i < nRail; i++) {
        uint32_t np = 0;
        std::fread(&np, 4, 1, f);
        std::fseek(f, np * 8, SEEK_CUR);
    }

    uint32_t nSt = 0;
    std::fread(&nSt, 4, 1, f);
    for (uint32_t i = 0; i < nSt; i++) {
        char sname[48];
        float dat[4];
        std::fread(sname, 1, 48, f);
        std::fread(dat, sizeof(float), 4, f);
    }

    uint32_t nMaj = 0;
    std::fread(&nMaj, 4, 1, f);
    for (uint32_t i = 0; i < nMaj; i++) {
        char rname[48];
        uint32_t rt = 0, np = 0;
        float hw = 0.0f;
        std::fread(rname, 1, 48, f);
        std::fread(&rt, 4, 1, f);
        std::fread(&hw, 4, 1, f);
        std::fread(&np, 4, 1, f);
        std::fseek(f, np * 8, SEEK_CUR);
    }

    uint32_t nSide = 0;
    std::fread(&nSide, 4, 1, f);
    for (uint32_t i = 0; i < nSide; i++) {
        uint32_t np = 0;
        std::fread(&np, 4, 1, f);
        std::vector<Vec2> pts(np);
        std::fread(pts.data(), sizeof(float) * 2, np, f);
        for (size_t k = 0; k + 1 < pts.size(); k++) {
            float mx = 0.5f * (pts[k].x + pts[k + 1].x);
            float mz = 0.5f * (pts[k].z + pts[k + 1].z);
            int cx = clampi((int)((mx - WORLD_X0) / OSM_CHUNK_SZ), 0, OSM_CK_NX - 1);
            int cz = clampi((int)((mz - WORLD_Z0) / OSM_CHUNK_SZ), 0, OSM_CK_NZ - 1);
            g_chunkSideStreets[cz * OSM_CK_NX + cx].push_back({pts[k], pts[k + 1]});
        }
    }

    uint32_t nGard = 0;
    std::fread(&nGard, 4, 1, f);
    for (uint32_t i = 0; i < nGard; i++) {
        uint32_t gt = 0, np = 0;
        std::fread(&gt, 4, 1, f);
        std::fread(&np, 4, 1, f);
        std::vector<Vec2> pts(np);
        std::fread(pts.data(), sizeof(float) * 2, np, f);
        if (!pts.empty()) {
            float mx = pts[0].x, mz = pts[0].z;
            int cx = clampi((int)((mx - WORLD_X0) / OSM_CHUNK_SZ), 0, OSM_CK_NX - 1);
            int cz = clampi((int)((mz - WORLD_Z0) / OSM_CHUNK_SZ), 0, OSM_CK_NZ - 1);
            g_chunkGardens[cz * OSM_CK_NX + cx].push_back({(int)gt, pts});
        }
    }

    uint32_t nBldg = 0;
    std::fread(&nBldg, 4, 1, f);
    g_osmBuildings.resize(nBldg);
    for (uint32_t i = 0; i < nBldg; i++) {
        float bdat[6];
        uint8_t cdat[4];
        std::fread(bdat, sizeof(float), 6, f);
        std::fread(cdat, 1, 4, f);
        g_osmBuildings[i].x = bdat[0];
        g_osmBuildings[i].z = bdat[1];
        g_osmBuildings[i].w = bdat[2];
        g_osmBuildings[i].d = bdat[3];
        g_osmBuildings[i].yaw = bdat[4];
        g_osmBuildings[i].h = bdat[5];
        g_osmBuildings[i].kind = cdat[0];
        g_osmBuildings[i].r = cdat[1];
        g_osmBuildings[i].g = cdat[2];
        g_osmBuildings[i].b = cdat[3];
    }

    std::fclose(f);
    std::printf("[OSM BIN] izmir_osm_world.bin yuklendi! DEM=1024x1024, Binalar=%zu, AraSokaklar=%u, Bahceler=%u\n",
                g_osmBuildings.size(), nSide, nGard);
    return true;
}

// Ham arazi yuksekligi (terrainBase): Ovalar + Anizotropik Gauss Daglar + fBm kaya gurultusu
static float computeTerrainBaseRaw(float wx, float wz) {
    if (g_hasRealDEM) {
        return sampleOsmDem(wx, wz);
    }
    // Baz yukseklik + hafif tepelik
    float base = 6.0f + 8.0f * (0.5f + 0.5f * fbm2(wx * 0.00018f + 3.1f, wz * 0.00018f + 7.4f, 4));
    float rockFbm = 0.55f + 0.45f * ridged2(wx * 0.00055f + 1.7f, wz * 0.00055f - 2.3f, 4);
    float detailFbm = fbm2(wx * 0.0014f, wz * 0.0014f, 3) * 14.0f;

    float mtnSum = 0.0f;
    for (int i = 0; i < NUM_MOUNTAINS; i++) {
        Vec2 mc = geoPt(MOUNTAINS[i].lat, MOUNTAINS[i].lon);
        float dx = wx - mc.x, dz = wz - mc.z;
        float rad = MOUNTAINS[i].rotDeg * DEG2RAD;
        float ca = std::cos(rad), sa = std::sin(rad);
        float rx = dx * ca + dz * sa;
        float rz = -dx * sa + dz * ca;
        float d2 = (rx * rx) / (MOUNTAINS[i].sx * MOUNTAINS[i].sx) +
                   (rz * rz) / (MOUNTAINS[i].sz * MOUNTAINS[i].sz);
        if (d2 < 12.0f) {
            float g = std::exp(-0.5f * d2);
            // Kadifekale ve Gaziemir Platosu daha duz tepeli, yuksek daglar kaya fBm dokulu
            float shape = (i == 8) ? std::exp(-0.28f * d2 * d2)
                        : (i == 6) ? (g * (0.88f + 0.12f * rockFbm))
                                   : (g * (0.72f + 0.42f * rockFbm));
            mtnSum += MOUNTAINS[i].peak * shape;
        }
    }
    float h = base + mtnSum + detailFbm * clampf(mtnSum / 180.0f, 0.0f, 1.0f);

    // 3 duz aluvyon canagi (Gediz/Menemen/Cigli Ovasi, Bornova Ovasi, Gaziemir-Havalimani Platosu)
    for (int i = 0; i < NUM_BASINS; i++) {
        Vec2 bc = geoPt(BASINS[i].lat, BASINS[i].lon);
        float dist = len2(Vec2(wx - bc.x, wz - bc.z));
        if (dist < BASINS[i].radius) {
            float t = smooth01(1.0f - dist / BASINS[i].radius);
            float n01 = 0.5f + 0.5f * fbm2(wx * 0.0003f + i * 5.0f, wz * 0.0003f, 3);
            float targetH = lerpf(BASINS[i].minH, BASINS[i].maxH, n01);
            // Gaziemir platosunda 115..130m'ye sabitlerken, Gediz ve Bornova ovalarini alcak duzlukte tutar
            if (i == 2) {
                h = lerpf(h, targetH, t * 0.85f);
            } else {
                h = lerpf(h, std::min(h, targetH), t * 0.88f);
            }
        }
    }
    return std::max(2.0f, h);
}

static void edt1dDouble(const double* f, double* d, int n, double stepM, std::vector<int>& v, std::vector<double>& z) {
    double a = stepM * stepM;
    int k = 0;
    v[0] = 0;
    z[0] = -1e30;
    z[1] = 1e30;
    for (int q = 1; q < n; q++) {
        double rq = f[q] + a * (double)q * (double)q;
        double s = 0.0;
        while (true) {
            int p = v[k];
            double rp = f[p] + a * (double)p * (double)p;
            s = (rq - rp) / (2.0 * a * (double)(q - p));
            if (s > z[k] || k == 0) break;
            k--;
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = 1e30;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < (double)q) k++;
        double dq = (double)q - (double)v[k];
        d[q] = a * dq * dq + f[v[k]];
    }
}

static void computeExactSeaEDT2D() {
    float dx = (WORLD_X1 - WORLD_X0) / (float)(MASK_N - 1);
    float dz = (WORLD_Z1 - WORLD_Z0) / (float)(MASK_N - 1);
    const double INF = 1e11;
    std::vector<double> temp(MASK_N * MASK_N);
    std::vector<double> work(MASK_N * MASK_N);
    std::vector<float> signedOut(MASK_N * MASK_N, 0.0f);

    auto runPass = [&](bool targetSea) {
        parallelFor(MASK_N, [&](int z0, int z1) {
            std::vector<double> f(MASK_N), d(MASK_N), zz(MASK_N + 1);
            std::vector<int> v(MASK_N);
            for (int iz = z0; iz < z1; iz++) {
                const uint8_t* rowG = &g_gisGrid[iz * MASK_N];
                for (int ix = 0; ix < MASK_N; ix++) {
                    bool isHit = targetSea ? (rowG[ix] == LC_SEA) : (rowG[ix] != LC_SEA);
                    f[ix] = isHit ? 0.0 : INF;
                }
                edt1dDouble(f.data(), d.data(), MASK_N, dx, v, zz);
                std::memcpy(&temp[iz * MASK_N], d.data(), MASK_N * sizeof(double));
            }
        });
        parallelFor(MASK_N, [&](int x0, int x1) {
            std::vector<double> f(MASK_N), d(MASK_N), zz(MASK_N + 1);
            std::vector<int> v(MASK_N);
            for (int ix = x0; ix < x1; ix++) {
                for (int iz = 0; iz < MASK_N; iz++) {
                    f[iz] = temp[iz * MASK_N + ix];
                }
                edt1dDouble(f.data(), d.data(), MASK_N, dz, v, zz);
                for (int iz = 0; iz < MASK_N; iz++) {
                    work[iz * MASK_N + ix] = d[iz];
                }
            }
        });
        parallelFor(MASK_N, [&](int z0, int z1) {
            for (int iz = z0; iz < z1; iz++) {
                for (int ix = 0; ix < MASK_N; ix++) {
                    bool isHit = targetSea ? (g_gisGrid[iz * MASK_N + ix] == LC_SEA)
                                           : (g_gisGrid[iz * MASK_N + ix] != LC_SEA);
                    if (isHit) continue;
                    float dist = (float)std::sqrt(std::max(0.0, work[iz * MASK_N + ix]));
                    signedOut[iz * MASK_N + ix] = targetSea ? dist : -dist;
                }
            }
        });
    };

    runPass(true);
    runPass(false);
    for (int i = 0; i < MASK_N * MASK_N; i++) {
        g_maskGrid[i].signedSeaDist = signedOut[i];
    }
}

static bool nudgeInlandBySeaEDT(float& wx, float& wz, float minSeaClearance) {
    if (!g_hasGisGrid) return false;
    for (int it = 0; it < 28; it++) {
        float sd = sampleSignedSeaDist(wx, wz);
        if (sd >= minSeaClearance) return true;
        float gx = sampleSignedSeaDist(wx + 18.0f, wz) - sampleSignedSeaDist(wx - 18.0f, wz);
        float gz = sampleSignedSeaDist(wx, wz + 18.0f) - sampleSignedSeaDist(wx, wz - 18.0f);
        float gm = std::sqrt(gx * gx + gz * gz);
        if (gm < 1e-4f) break;
        float step = clampf((minSeaClearance + 2.0f) - sd, 6.0f, 65.0f);
        wx += (gx / gm) * step;
        wz += (gz / gm) * step;
    }
    if (sampleSignedSeaDist(wx, wz) >= minSeaClearance) return true;
    // Medial-aks veya dar iskele durumu icin izgarada en yakin kara hucresini bul
    float fgx, fgz;
    worldToGridF(wx, wz, fgx, fgz);
    int cx = clampi((int)(fgx + 0.5f), 0, MASK_N - 1);
    int cz = clampi((int)(fgz + 0.5f), 0, MASK_N - 1);
    float bestD2 = 1e18f;
    float bestX = wx, bestZ = wz;
    for (int r = 2; r <= 128; r = (r < 16) ? r + 2 : r * 2) {
        int x0 = std::max(0, cx - r), x1 = std::min(MASK_N - 1, cx + r);
        int z0 = std::max(0, cz - r), z1 = std::min(MASK_N - 1, cz + r);
        for (int iz = z0; iz <= z1; iz++) {
            for (int ix = x0; ix <= x1; ix++) {
                if (g_maskGrid[iz * MASK_N + ix].signedSeaDist >= minSeaClearance + 2.0f) {
                    float px, pz;
                    gridToWorld(ix, iz, px, pz);
                    float d2 = (px - wx) * (px - wx) + (pz - wz) * (pz - wz);
                    if (d2 < bestD2) { bestD2 = d2; bestX = px; bestZ = pz; }
                }
            }
        }
        if (bestD2 < 1e17f) {
            wx = bestX;
            wz = bestZ;
            return true;
        }
    }
    return false;
}

// Kiyi poligonunu, IZBAN spline'ini, otoyollari ve 2048x2048 WorldMaskGrid'i olustur
static void buildGeographyAndWorldMask() {
    if (g_hasGisGrid) {
        computeExactSeaEDT2D();
    }
    // 1. 34 gercek WGS84 GPS kontrol noktasindan Izmir Korfezi kiyi spline'i
    std::vector<Vec2> ctrlPts;
    ctrlPts.reserve(34);
    for (int i = 0; i < 34; i++) ctrlPts.push_back(geoPt(COAST_CTRL[i].lat, COAST_CTRL[i].lon));

    std::vector<Vec2> rawCoast;
    for (int i = 0; i < 33; i++) {
        Vec2 p0 = ctrlPts[std::max(0, i - 1)];
        Vec2 p1 = ctrlPts[i];
        Vec2 p2 = ctrlPts[i + 1];
        Vec2 p3 = ctrlPts[std::min(33, i + 2)];
        float segLen = len2(p2 - p1);
        int steps = std::max(8, (int)(segLen / 12.0f));
        for (int s = 0; s < steps; s++) {
            rawCoast.push_back(catmullRom2D(p0, p1, p2, p3, (float)s / (float)steps));
        }
    }
    rawCoast.push_back(ctrlPts.back());
    relaxCurvature2D(rawCoast, 45.0f, 3);
    g_coastSpline = resamplePolyline2D(rawCoast, 15.0f);

    g_coastTangents.resize(g_coastSpline.size());
    for (size_t i = 0; i < g_coastSpline.size(); i++) {
        size_t i0 = (i > 0) ? i - 1 : 0;
        size_t i1 = (i + 1 < g_coastSpline.size()) ? i + 1 : i;
        g_coastTangents[i] = norm2(g_coastSpline[i1] - g_coastSpline[i0]);
    }
    // Topolojik olarak Korfez kiyi poligonu guneyden doguya ve kuzeyden batiya ilerlediginden,
    // sag-el normali Vec2(-t.z, t.x) daima karaya (iceri) bakar; 7-noktali pencere ile puruzsuzlestirilir
    int coastN = (int)g_coastSpline.size();
    g_coastInNormals.resize(coastN);
    for (int i = 0; i < coastN; i++) {
        Vec2 sum(0.0f, 0.0f);
        for (int k = -3; k <= 3; k++) {
            int j = clampi(i + k, 0, coastN - 1);
            Vec2 t = g_coastTangents[j];
            sum = sum + Vec2(-t.z, t.x);
        }
        g_coastInNormals[i] = norm2(sum);
    }

    // Batida (-22000m, -16500m) ve (-22000m, +5000m) acik deniz koseleriyle su poligonunu kapat
    g_seaPolygon = g_coastSpline;
    g_seaPolygon.push_back(Vec2(SEA_CORNER_NW_X, SEA_CORNER_NW_Z));
    g_seaPolygon.push_back(Vec2(SEA_CORNER_SW_X, SEA_CORNER_SW_Z));

    auto prePushInland = [&](float& wx, float& wz, float minSeaClearance) {
        if (nudgeInlandBySeaEDT(wx, wz, minSeaClearance)) return;
        int crossings = 0;
        int polyN = (int)g_seaPolygon.size();
        for (int i = 0, j = polyN - 1; i < polyN; j = i++) {
            float zi = g_seaPolygon[i].z, zj = g_seaPolygon[j].z;
            if ((zi > wz) != (zj > wz)) {
                float xHit = g_seaPolygon[i].x + (wz - zi) * (g_seaPolygon[j].x - g_seaPolygon[i].x) / (zj - zi);
                if (xHit > wx) crossings++;
            }
        }
        bool inSea = (crossings & 1) != 0;
        int bestI = 0;
        float bestD2 = 1e18f;
        for (int i = 0; i < coastN; i++) {
            float dx = g_coastSpline[i].x - wx, dz = g_coastSpline[i].z - wz;
            float d2 = dx * dx + dz * dz;
            if (d2 < bestD2) { bestD2 = d2; bestI = i; }
        }
        if (inSea || bestD2 < minSeaClearance * minSeaClearance) {
            Vec2 target = g_coastSpline[bestI] + g_coastInNormals[bestI] * (minSeaClearance + 1.5f);
            wx = target.x;
            wz = target.z;
        }
    };

    auto smoothProfileAndClampGrade = [](std::vector<Vec3>& pts, float maxStepDeltaY, int passes, bool pinEnds) {
        size_t n = pts.size();
        if (n < 3) return;
        std::vector<float> tmp(n);
        for (int p = 0; p < passes; p++) {
            for (size_t i = 0; i < n; i++) {
                if (pinEnds && (i == 0 || i == n - 1)) { tmp[i] = pts[i].y; continue; }
                size_t i0 = (i > 0) ? i - 1 : 0;
                size_t i1 = (i + 1 < n) ? i + 1 : i;
                tmp[i] = 0.25f * pts[i0].y + 0.50f * pts[i].y + 0.25f * pts[i1].y;
            }
            for (size_t i = 0; i < n; i++) pts[i].y = tmp[i];
        }
        for (int iter = 0; iter < 4; iter++) {
            for (size_t i = 1; i < n; i++) {
                float d = pts[i].y - pts[i - 1].y;
                if (d > maxStepDeltaY) pts[i].y = pts[i - 1].y + maxStepDeltaY;
                else if (d < -maxStepDeltaY) pts[i].y = pts[i - 1].y - maxStepDeltaY;
            }
            for (size_t i = n - 1; i-- > 0;) {
                float d = pts[i].y - pts[i + 1].y;
                if (d > maxStepDeltaY) pts[i].y = pts[i + 1].y + maxStepDeltaY;
                else if (d < -maxStepDeltaY) pts[i].y = pts[i + 1].y - maxStepDeltaY;
            }
        }
        for (int p = 0; p < 6; p++) {
            for (size_t i = 1; i + 1 < n; i++) {
                tmp[i] = 0.25f * pts[i - 1].y + 0.50f * pts[i].y + 0.25f * pts[i + 1].y;
            }
            for (size_t i = 1; i + 1 < n; i++) pts[i].y = tmp[i];
        }
    };

    // 2. 28 istasyonlu IZBAN hatti: Catmull-Rom Spline + relaxCurvature(minRadius = 180m)
    std::vector<Vec3> stCtrl(NUM_STATIONS);
    for (int i = 0; i < NUM_STATIONS; i++) {
        Vec2 p = geoPt(STATIONS[i].lat, STATIONS[i].lon);
        stCtrl[i] = Vec3(p.x, STATIONS[i].elev, p.z);
    }
    std::vector<Vec3> rawIzban;
    for (int i = 0; i < NUM_STATIONS - 1; i++) {
        Vec3 p0 = stCtrl[std::max(0, i - 1)];
        Vec3 p1 = stCtrl[i];
        Vec3 p2 = stCtrl[i + 1];
        Vec3 p3 = stCtrl[std::min(NUM_STATIONS - 1, i + 2)];
        float d = len2(Vec2(p2.x - p1.x, p2.z - p1.z));
        int steps = std::max(12, (int)(d / 15.0f));
        for (int s = 0; s < steps; s++) {
            rawIzban.push_back(catmullRom3D(p0, p1, p2, p3, (float)s / (float)steps));
        }
    }
    rawIzban.push_back(stCtrl.back());
    // Halkapinar-Alsancak-Hilal ucgen kavisini minRadius = 180m ile yumusat
    relaxCurvature3D(rawIzban, 180.0f, 14);
    g_izbanSpline = resamplePolyline3D(rawIzban, 15.0f);
    for (auto& pt : g_izbanSpline) {
        prePushInland(pt.x, pt.z, 24.0f);
        float e = computeTerrainBaseRaw(pt.x, pt.z);
        pt.y = std::max(pt.y, e + 0.45f);
    }
    smoothProfileAndClampGrade(g_izbanSpline, 0.55f, 36, false);

    g_izbanTangents.resize(g_izbanSpline.size());
    g_izbanCumDist.resize(g_izbanSpline.size(), 0.0f);
    for (size_t i = 0; i < g_izbanSpline.size(); i++) {
        size_t i0 = (i > 0) ? i - 1 : 0;
        size_t i1 = (i + 1 < g_izbanSpline.size()) ? i + 1 : i;
        g_izbanTangents[i] = norm2(Vec2(g_izbanSpline[i1].x - g_izbanSpline[i0].x,
                                        g_izbanSpline[i1].z - g_izbanSpline[i0].z));
        if (i > 0) {
            float ds = len2(Vec2(g_izbanSpline[i].x - g_izbanSpline[i - 1].x,
                                 g_izbanSpline[i].z - g_izbanSpline[i - 1].z));
            g_izbanCumDist[i] = g_izbanCumDist[i - 1] + ds;
        }
    }
    g_izbanTotalLen = g_izbanCumDist.back();

    // Istasyonlarin yumusatilmis spline uzerindeki izdüsümlerini kaydet
    for (int s = 0; s < NUM_STATIONS; s++) {
        Vec2 target = geoPt(STATIONS[s].lat, STATIONS[s].lon);
        float bestD = 1e18f;
        size_t bestIdx = 0;
        for (size_t i = 0; i < g_izbanSpline.size(); i++) {
            float d = len2(Vec2(g_izbanSpline[i].x - target.x, g_izbanSpline[i].z - target.z));
            if (d < bestD) { bestD = d; bestIdx = i; }
        }
        g_stationSplineS[s] = g_izbanCumDist[bestIdx];
        g_stationPos[s] = g_izbanSpline[bestIdx];
        g_stationTang[s] = g_izbanTangents[bestIdx];
    }

    // 3. Otoyol aglari (O-30 Cevreyolu, D550 Anadolu/Altinyol/Yesildere-Akcay, Sahil Bulvarlari)
    std::vector<RoadDef> rdefs = makeRoadDefs();
    g_roads.clear();
    for (size_t rdIdx = 0; rdIdx < rdefs.size(); rdIdx++) {
        const auto& rd = rdefs[rdIdx];
        bool isCoastalBlvd = rd.isCoastal;
        std::vector<Vec3> cpts;
        for (const auto& ll : rd.pts) {
            Vec2 p = geoPt(ll.lat, ll.lon);
            if (isCoastalBlvd) {
                int bestI = 0;
                float bestD = 1e18f;
                for (int ci = 0; ci < coastN; ci++) {
                    float d = len2(g_coastSpline[ci] - p);
                    if (d < bestD) { bestD = d; bestI = ci; }
                }
                if (bestD < 260.0f) {
                    p = g_coastSpline[bestI] + g_coastInNormals[bestI] * 21.0f;
                }
            }
            float e = computeTerrainBaseRaw(p.x, p.z);
            cpts.push_back(Vec3(p.x, e, p.z));
        }
        std::vector<Vec3> rawR;
        int n = (int)cpts.size();
        for (int i = 0; i < n - 1; i++) {
            Vec3 p0 = cpts[std::max(0, i - 1)];
            Vec3 p1 = cpts[i];
            Vec3 p2 = cpts[i + 1];
            Vec3 p3 = cpts[std::min(n - 1, i + 2)];
            float d = len2(Vec2(p2.x - p1.x, p2.z - p1.z));
            int steps = std::max(8, (int)(d / 20.0f));
            for (int s = 0; s < steps; s++) rawR.push_back(catmullRom3D(p0, p1, p2, p3, (float)s / (float)steps));
        }
        rawR.push_back(cpts.back());
        relaxCurvature3D(rawR, isCoastalBlvd ? 85.0f : 140.0f, 6);
        RoadSpline rs;
        rs.name = rd.name; rs.type = rd.type; rs.halfW = rd.halfW; rs.lanes = rd.lanes;
        rs.pts = resamplePolyline3D(rawR, 20.0f);
        for (auto& pt : rs.pts) {
            prePushInland(pt.x, pt.z, 20.0f);
        }
        if (isCoastalBlvd) {
            for (auto& pt : rs.pts) {
                int bestI = 0;
                float bestD = 1e18f;
                for (int ci = 0; ci < coastN; ci++) {
                    float d = len2(g_coastSpline[ci] - Vec2(pt.x, pt.z));
                    if (d < bestD) { bestD = d; bestI = ci; }
                }
                if (bestD < 180.0f) {
                    Vec2 snapped = g_coastSpline[bestI] + g_coastInNormals[bestI] * 21.0f;
                    pt.x = snapped.x;
                    pt.z = snapped.z;
                    pt.y = 2.45f;
                }
            }
        }
        // Tegetleri hesapla ve her 20m noktasinda arazi + yan banket kotunu ornekle
        rs.tangs.resize(rs.pts.size());
        rs.cumDist.resize(rs.pts.size(), 0.0f);
        for (size_t i = 0; i < rs.pts.size(); i++) {
            size_t i0 = (i > 0) ? i - 1 : 0;
            size_t i1 = (i + 1 < rs.pts.size()) ? i + 1 : i;
            rs.tangs[i] = norm2(Vec2(rs.pts[i1].x - rs.pts[i0].x, rs.pts[i1].z - rs.pts[i0].z));
            if (i > 0) {
                rs.cumDist[i] = rs.cumDist[i - 1] + len2(Vec2(rs.pts[i].x - rs.pts[i - 1].x,
                                                              rs.pts[i].z - rs.pts[i - 1].z));
            }
        }
        if (!isCoastalBlvd) {
            for (size_t i = 0; i < rs.pts.size(); i++) {
                Vec3& pt = rs.pts[i];
                Vec2 n(-rs.tangs[i].z, rs.tangs[i].x);
                float w = rs.halfW + 10.0f;
                float eC = computeTerrainBaseRaw(pt.x, pt.z);
                float eL = computeTerrainBaseRaw(pt.x - n.x * w, pt.z - n.z * w);
                float eR = computeTerrainBaseRaw(pt.x + n.x * w, pt.z + n.z * w);
                pt.y = std::max(2.45f, 0.50f * eC + 0.50f * std::max(eL, eR) + 0.45f);
            }
            smoothProfileAndClampGrade(rs.pts, 1.25f, 44, false);
        }
        rs.totalLen = rs.cumDist.back();
        g_roads.push_back(rs);
    }

    // 4. Tum koridor segmentlerini (gercek v2 modunda sadece LTBJ/LTBL pistleri; sentetik modda IZBAN+Otoyol+Pist) topla
    g_corridorSegs.clear();
    if (!g_hasV2Roads) {
        for (size_t i = 0; i + 1 < g_izbanSpline.size(); i += 2) {
            size_t j = std::min(i + 2, g_izbanSpline.size() - 1);
            Vec2 a(g_izbanSpline[i].x, g_izbanSpline[i].z);
            Vec2 b(g_izbanSpline[j].x, g_izbanSpline[j].z);
            g_corridorSegs.push_back({a, b, norm2(b - a), g_izbanSpline[i].y, g_izbanSpline[j].y, 9.0f, CT_RAIL});
        }
        for (const auto& rs : g_roads) {
            for (size_t i = 0; i + 1 < rs.pts.size(); i++) {
                Vec2 a(rs.pts[i].x, rs.pts[i].z);
                Vec2 b(rs.pts[i + 1].x, rs.pts[i + 1].z);
                g_corridorSegs.push_back({a, b, norm2(b - a), rs.pts[i].y, rs.pts[i + 1].y, rs.halfW, rs.type});
            }
        }
    }
    // LTBJ cift paralel pist + apron koridoru
    {
        Vec2 c = geoPt(LTBJ_LAT, LTBJ_LON);
        Vec2 dir = headingDir(LTBJ_HDG);
        Vec2 right(-dir.z, dir.x);
        for (int r = -1; r <= 1; r++) {
            Vec2 rc = c + right * (r * LTBJ_SEP * 0.5f);
            Vec2 a = rc - dir * (LTBJ_LEN * 0.55f);
            Vec2 b = rc + dir * (LTBJ_LEN * 0.55f);
            float hw = (r == 0) ? 240.0f : 90.0f;
            g_corridorSegs.push_back({a, b, dir, LTBJ_ELEV, LTBJ_ELEV, hw, CT_RUNWAY});
        }
    }
    // LTBL Cigli askeri pist koridoru
    {
        Vec2 c = geoPt(LTBL_LAT, LTBL_LON);
        Vec2 dir = headingDir(LTBL_HDG);
        Vec2 a = c - dir * (LTBL_LEN * 0.55f);
        Vec2 b = c + dir * (LTBL_LEN * 0.55f);
        g_corridorSegs.push_back({a, b, dir, LTBL_ELEV, LTBL_ELEV, 160.0f, CT_RUNWAY});
    }

    // 5. Hizli mekânsal kovalar (64x64) ile 2048x2048 WorldMaskGrid'i paralel doldur
    const int BK = 64;
    std::vector<std::vector<int>> coastBuckets(BK * BK);
    std::vector<std::vector<int>> corrBuckets(BK * BK);
    auto bktIdx = [&](float wx, float wz, int& bx, int& bz) {
        bx = clampi((int)((wx - WORLD_X0) / (WORLD_X1 - WORLD_X0) * BK), 0, BK - 1);
        bz = clampi((int)((wz - WORLD_Z0) / (WORLD_Z1 - WORLD_Z0) * BK), 0, BK - 1);
    };
    const float padR = 2600.0f;
    for (int i = 0; i + 1 < (int)g_coastSpline.size(); i++) {
        float x0 = std::min(g_coastSpline[i].x, g_coastSpline[i + 1].x) - padR;
        float x1 = std::max(g_coastSpline[i].x, g_coastSpline[i + 1].x) + padR;
        float z0 = std::min(g_coastSpline[i].z, g_coastSpline[i + 1].z) - padR;
        float z1 = std::max(g_coastSpline[i].z, g_coastSpline[i + 1].z) + padR;
        int bx0, bz0, bx1, bz1;
        bktIdx(x0, z0, bx0, bz0); bktIdx(x1, z1, bx1, bz1);
        for (int z = bz0; z <= bz1; z++)
            for (int x = bx0; x <= bx1; x++)
                coastBuckets[z * BK + x].push_back(i);
    }
    for (int i = 0; i < (int)g_corridorSegs.size(); i++) {
        float x0 = std::min(g_corridorSegs[i].a.x, g_corridorSegs[i].b.x) - padR;
        float x1 = std::max(g_corridorSegs[i].a.x, g_corridorSegs[i].b.x) + padR;
        float z0 = std::min(g_corridorSegs[i].a.z, g_corridorSegs[i].b.z) - padR;
        float z1 = std::max(g_corridorSegs[i].a.z, g_corridorSegs[i].b.z) + padR;
        int bx0, bz0, bx1, bz1;
        bktIdx(x0, z0, bx0, bz0); bktIdx(x1, z1, bx1, bz1);
        for (int z = bz0; z <= bz1; z++)
            for (int x = bx0; x <= bx1; x++)
                corrBuckets[z * BK + x].push_back(i);
    }

    // Ilce merkezlerini ve kent ormani merkezlerini onceden metrik koordinata cevir
    Vec2 distCenters[NUM_DISTRICTS];
    for (int d = 0; d < NUM_DISTRICTS; d++) distCenters[d] = geoPt(DISTRICTS[d].lat, DISTRICTS[d].lon);
    Vec2 kpCenter  = geoPt(KULTURPARK_LAT, KULTURPARK_LON);
    Vec2 incCenter = geoPt(INCIRALTI_LAT, INCIRALTI_LON);
    Vec2 aosbCenter = geoPt(AOSB_LAT, AOSB_LON);

    // 2048x2048 matrisi paralel olarak hesapla
    parallelFor(MASK_N, [&](int iz0, int iz1) {
        std::vector<float> xIntersections;
        xIntersections.reserve(64);
        int polyN = (int)g_seaPolygon.size();

        for (int iz = iz0; iz < iz1; iz++) {
            float dummyX, wz;
            gridToWorld(0, iz, dummyX, wz);

            // Bu Z satiri icin deniz poligonu kesisim X noktalarini bul (sentetik kiyi modu icin)
            xIntersections.clear();
            if (!g_hasGisGrid) {
                for (int i = 0, j = polyN - 1; i < polyN; j = i++) {
                    float zi = g_seaPolygon[i].z, zj = g_seaPolygon[j].z;
                    if ((zi > wz) != (zj > wz)) {
                        float xi = g_seaPolygon[i].x, xj = g_seaPolygon[j].x;
                        float xHit = xi + (wz - zi) * (xj - xi) / (zj - zi);
                        xIntersections.push_back(xHit);
                    }
                }
                std::sort(xIntersections.begin(), xIntersections.end());
            }

            for (int ix = 0; ix < MASK_N; ix++) {
                float wx, wzCell;
                gridToWorld(ix, iz, wx, wzCell);
                int bx, bz;
                bktIdx(wx, wzCell, bx, bz);

                WorldMaskCell& cell = g_maskGrid[iz * MASK_N + ix];
                bool inSea = false;
                float bestCoastD = 3200.0f;
                Vec2 bestCoastT(1.0f, 0.0f);

                // En yakin kiyi tegeti (ve sentetik modda mesafe)
                const auto& cb = coastBuckets[bz * BK + bx];
                for (int si : cb) {
                    float d = segDist2D(wx, wzCell, g_coastSpline[si].x, g_coastSpline[si].z,
                                        g_coastSpline[si + 1].x, g_coastSpline[si + 1].z, nullptr);
                    if (d < bestCoastD) {
                        bestCoastD = d;
                        bestCoastT = g_coastTangents[si];
                    }
                }
                if (cb.empty() && !g_hasGisGrid) {
                    for (int si = 0; si + 1 < (int)g_coastSpline.size(); si += 8) {
                        float d = segDist2D(wx, wzCell, g_coastSpline[si].x, g_coastSpline[si].z,
                                            g_coastSpline[si + 1].x, g_coastSpline[si + 1].z, nullptr);
                        if (d < bestCoastD) { bestCoastD = d; bestCoastT = g_coastTangents[si]; }
                    }
                }

                if (g_hasGisGrid) {
                    float gisD = cell.signedSeaDist;
                    inSea = (gisD < 0.0f);
                    bestCoastD = std::fabs(gisD);
                } else {
                    int crossings = 0;
                    for (float xh : xIntersections) {
                        if (xh > wx) crossings++;
                    }
                    inSea = (crossings & 1) != 0;
                    cell.signedSeaDist = inSea ? -bestCoastD : bestCoastD;
                }

                // En yakin koridor (yol/ray/pist) mesafesi, tegeti ve tesviye yuksekligi
                float bestCorrD = 3200.0f;
                float bestCorrElev = 4.0f;
                float bestCorrBlend = 0.0f;
                Vec2 bestRoadT(0.0f, -1.0f);
                float bestRoadCenterD = 1e9f;

                const auto& rbk = corrBuckets[bz * BK + bx];
                for (int ci : rbk) {
                    const CorridorSeg& cs = g_corridorSegs[ci];
                    float tSeg = 0.0f;
                    float centerD = segDist2D(wx, wzCell, cs.a.x, cs.a.z, cs.b.x, cs.b.z, &tSeg);
                    float edgeD = centerD - cs.halfW;
                    if (edgeD < bestCorrD) {
                        bestCorrD = edgeD;
                    }
                    if (centerD < bestRoadCenterD && cs.type != CT_RUNWAY) {
                        bestRoadCenterD = centerD;
                        bestRoadT = cs.tang;
                    }
                    // Koridor tesviye etki yaricapi (68m arazi izgara hucresini kapsayacak kadar genis banket tesviyesi)
                    float flatR = cs.halfW + 46.0f;
                    float transR = (cs.type == CT_RUNWAY) ? (cs.halfW + 300.0f) : (cs.halfW + 145.0f);
                    if (centerD < transR) {
                        float w = (centerD <= flatR) ? 1.0f : (1.0f - smooth01((centerD - flatR) / (transR - flatR)));
                        float segElev = lerpf(cs.elevA, cs.elevB, tSeg);
                        if (w > bestCorrBlend) {
                            bestCorrBlend = w;
                            bestCorrElev = segElev;
                        }
                    }
                }

                cell.corridorDist = bestCorrD;
                cell.corridorElev = bestCorrElev;
                cell.corridorBlend = bestCorrBlend;
                cell.coastTangX = bestCoastT.x;
                cell.coastTangZ = bestCoastT.z;
                cell.roadTangX = bestRoadT.x;
                cell.roadTangZ = bestRoadT.z;

                // Nihai arazi yuksekligini smooth01 ile sifir hatayla tesviye et
                if (inSea) {
                    cell.elev = 0.0f;
                    cell.urbanBlend = 0.0f;
                } else {
                    float hRaw = computeTerrainBaseRaw(wx, wzCell);
                    // Kiyi gecisi: 0 < signedSeaDist < 34m -> +2.15m rihtim, kordon ve sahil bulvari terasi
                    float hCoast = hRaw;
                    if (bestCoastD < 34.0f) {
                        hCoast = 2.15f;
                    } else if (bestCoastD < 260.0f) {
                        float tC = smooth01((bestCoastD - 34.0f) / 226.0f);
                        hCoast = lerpf(2.15f, std::max(2.3f, hRaw), tC);
                    }
                    // Pist (ve sentetik modda otoyol/IZBAN) koridorlarini smooth01 ile tesviye et
                    float hFinal = lerpf(hCoast, bestCorrElev, bestCorrBlend);
                    cell.elev = std::max(2.0f, hFinal);

                    float uBlend = 0.0f;
                    if (g_hasGisGrid) {
                        uint8_t lc = g_gisGrid[iz * MASK_N + ix];
                        if (lc == LC_URBAN) uBlend = 0.92f;
                        else if (lc == LC_INDUSTRIAL) uBlend = 0.85f;
                        if (cell.elev > 300.0f) uBlend *= clampf(1.0f - (cell.elev - 300.0f) / 90.0f, 0.0f, 1.0f);
                    } else {
                        for (int d = 0; d < NUM_DISTRICTS; d++) {
                            float r = DISTRICTS[d].radius * 1.35f;
                            float dist = len2(Vec2(wx - distCenters[d].x, wzCell - distCenters[d].z));
                            if (dist < r) {
                                float w = smooth01(1.0f - dist / r) * DISTRICTS[d].density;
                                if (w > uBlend) uBlend = w;
                            }
                        }
                        if (bestCoastD < 1400.0f && wx > -9500.0f && wzCell > -9000.0f) {
                            uBlend = std::max(uBlend, smooth01(1.0f - bestCoastD / 1400.0f) * 0.90f);
                        }
                        if (len2(Vec2(wx - aosbCenter.x, wzCell - aosbCenter.z)) < 1900.0f) {
                            uBlend = std::max(uBlend, 0.85f);
                        }
                        if (std::fabs(wx - kpCenter.x) < 480.0f && std::fabs(wzCell - kpCenter.z) < 620.0f) uBlend = 0.0f;
                        if (std::fabs(wx - incCenter.x) < 1350.0f && std::fabs(wzCell - incCenter.z) < 520.0f) uBlend = 0.0f;
                        if (wx < -10200.0f && wzCell < -3000.0f) uBlend = 0.0f;
                        if (cell.elev > 205.0f) uBlend *= clampf(1.0f - (cell.elev - 205.0f) / 70.0f, 0.0f, 1.0f);
                    }
                    cell.urbanBlend = uBlend;
                }
            }
        }
    });

    auto pushInlandFromSea = [&](float& wx, float& wz, float minSeaClearance) {
        if (nudgeInlandBySeaEDT(wx, wz, minSeaClearance)) return;
        float sd = sampleSignedSeaDist(wx, wz);
        if (sd >= minSeaClearance) return;
        int bestI = 0;
        float bestD = 1e18f;
        for (int i = 0; i < coastN; i++) {
            float d = len2(g_coastSpline[i] - Vec2(wx, wz));
            if (d < bestD) { bestD = d; bestI = i; }
        }
        Vec2 target = g_coastSpline[bestI] + g_coastInNormals[bestI] * (minSeaClearance + 1.0f);
        wx = target.x;
        wz = target.z;
    };

    // Otoyol dugumlerini on-tesviyeli yumusak egim profili uzerinde tutup olasi yerel zirvelerin ustune kaldir
    for (auto& rs : g_roads) {
        for (auto& p : rs.pts) {
            pushInlandFromSea(p.x, p.z, 20.0f);
        }
        rs.cumDist[0] = 0.0f;
        for (size_t i = 0; i < rs.pts.size(); i++) {
            size_t i0 = (i > 0) ? i - 1 : 0;
            size_t i1 = (i + 1 < rs.pts.size()) ? i + 1 : i;
            rs.tangs[i] = norm2(Vec2(rs.pts[i1].x - rs.pts[i0].x, rs.pts[i1].z - rs.pts[i0].z));
            if (i > 0) {
                rs.cumDist[i] = rs.cumDist[i - 1] + len2(Vec2(rs.pts[i].x - rs.pts[i - 1].x,
                                                              rs.pts[i].z - rs.pts[i - 1].z));
            }
        }
        rs.totalLen = rs.cumDist.back();

        auto maxRoadTerrY = [&](size_t idx) {
            Vec3 p = rs.pts[idx];
            Vec2 t = rs.tangs[idx];
            Vec2 n(-t.z, t.x);
            float w = rs.halfW + 10.0f;
            float eC = sampleTerrainElev(p.x, p.z);
            float eL = sampleTerrainElev(p.x - n.x * w, p.z - n.z * w);
            float eR = sampleTerrainElev(p.x + n.x * w, p.z + n.z * w);
            return std::max(eC, std::max(eL, eR));
        };
        for (size_t i = 0; i < rs.pts.size(); i++) {
            rs.pts[i].y = std::max(rs.pts[i].y, maxRoadTerrY(i) + 0.45f);
        }
        smoothProfileAndClampGrade(rs.pts, 1.30f, 12, false);
        for (size_t i = 0; i < rs.pts.size(); i++) {
            rs.pts[i].y = std::max(rs.pts[i].y, maxRoadTerrY(i) + 0.38f);
        }
    }

    // IZBAN hattinin denizden en az 24m iceride kalmasini ve on-tesviyeli yumusak egimde kalmasini sagla
    for (size_t i = 0; i < g_izbanSpline.size(); i++) {
        pushInlandFromSea(g_izbanSpline[i].x, g_izbanSpline[i].z, 24.0f);
        Vec2 t = g_izbanTangents[i];
        Vec2 n(-t.z, t.x);
        float eC = sampleTerrainElev(g_izbanSpline[i].x, g_izbanSpline[i].z);
        float eL = sampleTerrainElev(g_izbanSpline[i].x - n.x * 10.0f, g_izbanSpline[i].z - n.z * 10.0f);
        float eR = sampleTerrainElev(g_izbanSpline[i].x + n.x * 10.0f, g_izbanSpline[i].z + n.z * 10.0f);
        g_izbanSpline[i].y = std::max(g_izbanSpline[i].y, std::max(eC, std::max(eL, eR)) + 0.50f);
    }
    smoothProfileAndClampGrade(g_izbanSpline, 0.58f, 12, false);
    for (size_t i = 0; i < g_izbanSpline.size(); i++) {
        Vec2 t = g_izbanTangents[i];
        Vec2 n(-t.z, t.x);
        float eC = sampleTerrainElev(g_izbanSpline[i].x, g_izbanSpline[i].z);
        float eL = sampleTerrainElev(g_izbanSpline[i].x - n.x * 10.0f, g_izbanSpline[i].z - n.z * 10.0f);
        float eR = sampleTerrainElev(g_izbanSpline[i].x + n.x * 10.0f, g_izbanSpline[i].z + n.z * 10.0f);
        g_izbanSpline[i].y = std::max(g_izbanSpline[i].y, std::max(eC, std::max(eL, eR)) + 0.42f);
    }
    for (int s = 0; s < NUM_STATIONS; s++) {
        float bestD = 1e18f;
        int bestI = 0;
        Vec2 st2(g_stationPos[s].x, g_stationPos[s].z);
        for (int i = 0; i < (int)g_izbanSpline.size(); i++) {
            float d = len2(Vec2(g_izbanSpline[i].x, g_izbanSpline[i].z) - st2);
            if (d < bestD) { bestD = d; bestI = i; }
        }
        g_stationPos[s] = g_izbanSpline[bestI];
        g_stationTang[s] = g_izbanTangents[bestI];
    }

    // V2 kiyi yollarindaki raster kenar tasmasi ile aciktaki iskele/mendirek patikalarini temizle
    if (g_hasV2Roads) {
        for (auto& rp : g_v2MajorRoads) {
            if (rp.isBridge) continue;
            for (auto& p : rp.pts) {
                float sd = sampleSignedSeaDist(p.x, p.z);
                if (sd < 2.0f && sd > -45.0f) {
                    pushInlandFromSea(p.x, p.z, 2.0f);
                }
            }
        }
        for (auto& ckRoads : g_chunkV2Roads) {
            ckRoads.erase(
                std::remove_if(ckRoads.begin(), ckRoads.end(), [&](V2RoadSeg& seg) {
                    if (seg.a.x < WORLD_X0 + 8.0f || seg.a.x > WORLD_X1 - 8.0f ||
                        seg.a.z < WORLD_Z0 + 8.0f || seg.a.z > WORLD_Z1 - 8.0f ||
                        seg.b.x < WORLD_X0 + 8.0f || seg.b.x > WORLD_X1 - 8.0f ||
                        seg.b.z < WORLD_Z0 + 8.0f || seg.b.z > WORLD_Z1 - 8.0f) {
                        return true;
                    }
                    if (seg.isBridge || seg.tier < 4) return false;
                    Vec2 mid = (seg.a + seg.b) * 0.5f;
                    return sampleSignedSeaDist(mid.x, mid.z) < -6.0f &&
                           (sampleSignedSeaDist(seg.a.x, seg.a.z) < -6.0f ||
                            sampleSignedSeaDist(seg.b.x, seg.b.z) < -6.0f);
                }),
                ckRoads.end()
            );
            for (auto& seg : ckRoads) {
                if (seg.isBridge) continue;
                float sdA = sampleSignedSeaDist(seg.a.x, seg.a.z);
                if (sdA < 1.5f && sdA > -40.0f) pushInlandFromSea(seg.a.x, seg.a.z, 1.5f);
                float sdB = sampleSignedSeaDist(seg.b.x, seg.b.z);
                if (sdB < 1.5f && sdB > -40.0f) pushInlandFromSea(seg.b.x, seg.b.z, 1.5f);
            }
        }
    }
}

// 7 Gercek IZDENIZ Vapur Iskelesi tanimlari ve deniz yaklasma koordinatlari
struct PierDef { const char* name; double lat, lon; };
static const int NUM_PIERS = 7;
static const PierDef PIERS[NUM_PIERS] = {
    {"Konak İskele", 38.4155, 27.1250},
    {"Pasaport İskele", 38.4265, 27.1318},
    {"Alsancak İskele", 38.4375, 27.1398},
    {"Karşıyaka İskele", 38.4547, 27.1194},
    {"Bostanlı İskele", 38.4556, 27.0972},
    {"Göztepe İskele", 38.3985, 27.0960},
    {"Üçkuyular İskele", 38.4050, 27.0706}
};
static Vec2 g_pierShorePos[NUM_PIERS];
static Vec2 g_pierSeaPos[NUM_PIERS];

static void computePierSeaPositions() {
    int n = (int)g_coastSpline.size();
    for (int p = 0; p < NUM_PIERS; p++) {
        Vec2 target = geoPt(PIERS[p].lat, PIERS[p].lon);
        g_pierShorePos[p] = target;
        if (g_hasGisGrid) {
            Vec2 shorePos = target;
            for (int it = 0; it < 36; it++) {
                float sd = sampleSignedSeaDist(shorePos.x, shorePos.z);
                if (sd >= 24.0f && sd <= 34.0f) break;
                float gx = sampleSignedSeaDist(shorePos.x + 16.0f, shorePos.z) - sampleSignedSeaDist(shorePos.x - 16.0f, shorePos.z);
                float gz = sampleSignedSeaDist(shorePos.x, shorePos.z + 16.0f) - sampleSignedSeaDist(shorePos.x, shorePos.z - 16.0f);
                float gm = std::sqrt(gx * gx + gz * gz);
                if (gm < 1e-4f) break;
                float delta = clampf(28.0f - sd, -65.0f, 65.0f);
                shorePos.x += (gx / gm) * delta;
                shorePos.z += (gz / gm) * delta;
            }
            g_pierShorePos[p] = shorePos;

            Vec2 seaPos = shorePos;
            bool ok = false;
            for (int it = 0; it < 36; it++) {
                float sd = sampleSignedSeaDist(seaPos.x, seaPos.z);
                if (sd <= -140.0f) { ok = true; break; }
                float gx = sampleSignedSeaDist(seaPos.x + 18.0f, seaPos.z) - sampleSignedSeaDist(seaPos.x - 18.0f, seaPos.z);
                float gz = sampleSignedSeaDist(seaPos.x, seaPos.z + 18.0f) - sampleSignedSeaDist(seaPos.x, seaPos.z - 18.0f);
                float gm = std::sqrt(gx * gx + gz * gz);
                if (gm < 1e-4f) break;
                float step = clampf(sd + 145.0f, 10.0f, 75.0f);
                seaPos.x -= (gx / gm) * step;
                seaPos.z -= (gz / gm) * step;
            }
            if (ok) {
                Vec2 seaDir = norm2(seaPos - shorePos);
                Vec2 dockPos = shorePos + seaDir * 135.0f;
                g_pierSeaPos[p] = (sampleSignedSeaDist(dockPos.x, dockPos.z) <= -45.0f) ? dockPos : seaPos;
                continue;
            }
        }
        int bestI = 0;
        float bestD = 1e18f;
        for (int i = 0; i < n; i++) {
            float d = len2(g_coastSpline[i] - target);
            if (d < bestD) { bestD = d; bestI = i; }
        }
        Vec2 base = g_coastSpline[bestI];
        Vec2 seaDir = g_coastInNormals[bestI] * -1.0f;
        g_pierSeaPos[p] = base + seaDir * 95.0f;
    }
}

// Bulunulan ilce adini dondurur (HUD icin)
static const char* queryDistrictName(float wx, float wz) {
    if (sampleSignedSeaDist(wx, wz) < 0.0f) return "İzmir Körfezi";
    float bestScore = 1e18f;
    const char* best = "İzmir Metropol";
    for (int i = 0; i < NUM_DISTRICTS; i++) {
        Vec2 dc = geoPt(DISTRICTS[i].lat, DISTRICTS[i].lon);
        float d = len2(Vec2(wx - dc.x, wz - dc.z)) / DISTRICTS[i].radius;
        if (d < bestScore) { bestScore = d; best = DISTRICTS[i].name; }
    }
    return best;
}
// =====================================================================================
//  BOLUM 6 : 50.000+ CAKISMASIZ BINA, CITYCHUNK DISPLAY LISTS, SIMGE YAPILAR VE TRAFIK
// =====================================================================================
enum {
    BLDG_APT_COAST = 0,
    BLDG_MAVISEHIR_RES = 1,
    BLDG_FACTORY = 2,
    BLDG_URBAN_APT = 3,
    BLDG_TOWER_GLASS = 4,
    BLDG_HILLSIDE_HOUSE = 5,
    BLDG_HISTORIC_KEMERALTI = 6,
    BLDG_MOSQUE = 7,
    BLDG_INSTITUTION = 8,
    BLDG_GREENHOUSE = 9
};

struct BuildingInst {
    float x, y, z;
    float w, d, h;
    float cosY, sinY;
    uint8_t type;
    uint8_t colorVar;
};

struct TreeInst {
    float x, y, z, h, r;
    uint8_t kind; // 0: Kizilcam/Fistikcami, 1: Ege Selvisi, 2: Kordon Palmiyesi, 3: Zeytin/Maki
};

// 400x400m CityChunk Display List yapisi
static const float CHUNK_SIZE = 400.0f;
static const int CHUNK_NX = (int)((WORLD_X1 - WORLD_X0) / CHUNK_SIZE) + 1; // 88
static const int CHUNK_NZ = (int)((WORLD_Z1 - WORLD_Z0) / CHUNK_SIZE) + 1; // 113

struct CityChunk {
    float cx, cy, cz;
    Vec2 uDir{1.0f, 0.0f};
    Vec2 vDir{0.0f, 1.0f};
    std::vector<BuildingInst> bldgs;
    std::vector<TreeInst> trees;
    GLuint dlApt = 0;
    GLuint dlGlass = 0;
    GLuint dlInd = 0;
    GLuint dlRoof = 0;
    GLuint dlTileRoof = 0;
    GLuint dlNearDetails = 0;
    GLuint dlMidTrees = 0;
    GLuint dlSkylineLod = 0;
    GLuint dlNightWindows = 0;
    GLuint dlGround = 0; // Sehir adalari, mahalle caddeleri, ara sokaklar, yaya kaldirimlari ve avlular
    bool compiled = false;
};
static std::vector<CityChunk> g_cityChunks(CHUNK_NX * CHUNK_NZ);
static int g_totalBuildings = 0;
static int g_totalTrees = 0;

// O(1) SpatialHashGrid (50.000+ bina ve simge kule carpismasi icin, 50m hucre)
static const float SH_CELL = 50.0f;
static const int SH_NX = (int)((WORLD_X1 - WORLD_X0) / SH_CELL) + 1; // 701
static const int SH_NZ = (int)((WORLD_Z1 - WORLD_Z0) / SH_CELL) + 1; // 901
static std::vector<std::vector<int>> g_spatialHash(SH_NX * SH_NZ);
static std::vector<BuildingInst> g_allBuildings;

static inline void registerBuildingInSpatialHash(int bIdx) {
    const BuildingInst& b = g_allBuildings[bIdx];
    float rad = 0.55f * std::sqrt(b.w * b.w + b.d * b.d);
    int x0 = clampi((int)((b.x - rad - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int x1 = clampi((int)((b.x + rad - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int z0 = clampi((int)((b.z - rad - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    int z1 = clampi((int)((b.z + rad - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    for (int z = z0; z <= z1; z++) {
        for (int x = x0; x <= x1; x++) {
            g_spatialHash[z * SH_NX + x].push_back(bIdx);
        }
    }
}

// O(1) Bina carpismasi testi (dunya uretimi, silah balistigi ve genel sorgu)
static int findBuildingHitPoint(const Vec3& p, float radius) {
    int x0 = clampi((int)((p.x - radius - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int x1 = clampi((int)((p.x + radius - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int z0 = clampi((int)((p.z - radius - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    int z1 = clampi((int)((p.z + radius - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    for (int gz = z0; gz <= z1; gz++) {
        for (int gx = x0; gx <= x1; gx++) {
            for (int idx : g_spatialHash[gz * SH_NX + gx]) {
                const BuildingInst& b = g_allBuildings[idx];
                if (p.y - radius > b.y + b.h || p.y + radius < b.y) continue;
                float dx = p.x - b.x, dz = p.z - b.z;
                float lx = std::fabs(dx * b.cosY + dz * b.sinY);
                float lz = std::fabs(-dx * b.sinY + dz * b.cosY);
                if (lx <= b.w * 0.5f + radius && lz <= b.d * 0.5f + radius) return idx;
            }
        }
    }
    return -1;
}

static bool checkBuildingCollision(const Vec3& p, float radius) {
    return findBuildingHitPoint(p, radius) >= 0;
}

// Dinamik 100m carpma penceresi: Ucak cevresinde yalnizca 100m icindeki binalar carpilabilir olur,
// 100m otesindeki binalar sadece gorsel olarak cizilir; ucak yaklastikca tekrar aktiflesir.
static const float BLDG_COLLIDE_RADIUS = 100.0f;
static std::vector<int> g_activeColliders100m;
static Vec3 g_colliderAnchorPos(1e9f, 1e9f, 1e9f);
static bool g_collidersDirty = true;
static std::vector<uint32_t> g_colliderVisitStamp;
static uint32_t g_colliderStampGen = 1;

static inline float distSqPointToBuildingOBB(const Vec3& p, const BuildingInst& b) {
    float dx = p.x - b.x, dz = p.z - b.z;
    float lx = dx * b.cosY + dz * b.sinY;
    float lz = -dx * b.sinY + dz * b.cosY;
    float cx = std::max(0.0f, std::fabs(lx) - b.w * 0.5f);
    float cz = std::max(0.0f, std::fabs(lz) - b.d * 0.5f);
    float cy = 0.0f;
    if (p.y < b.y) cy = b.y - p.y;
    else if (p.y > b.y + b.h) cy = p.y - (b.y + b.h);
    return cx * cx + cy * cy + cz * cz;
}

static void refreshActiveColliders100m(const Vec3& acPos) {
    Vec3 dAnchor = acPos - g_colliderAnchorPos;
    if (!g_collidersDirty && dot(dAnchor, dAnchor) < 12.0f * 12.0f) return;
    g_colliderAnchorPos = acPos;
    g_collidersDirty = false;
    g_activeColliders100m.clear();
    if (g_allBuildings.empty()) return;
    if (g_colliderVisitStamp.size() != g_allBuildings.size()) {
        g_colliderVisitStamp.assign(g_allBuildings.size(), 0u);
    }
    g_colliderStampGen++;
    if (g_colliderStampGen == 0u) {
        std::fill(g_colliderVisitStamp.begin(), g_colliderVisitStamp.end(), 0u);
        g_colliderStampGen = 1u;
    }
    const float scanR = BLDG_COLLIDE_RADIUS + 18.0f;
    int x0 = clampi((int)((acPos.x - scanR - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int x1 = clampi((int)((acPos.x + scanR - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int z0 = clampi((int)((acPos.z - scanR - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    int z1 = clampi((int)((acPos.z + scanR - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    const float keepR2 = scanR * scanR;
    for (int gz = z0; gz <= z1; gz++) {
        for (int gx = x0; gx <= x1; gx++) {
            for (int idx : g_spatialHash[gz * SH_NX + gx]) {
                if (g_colliderVisitStamp[idx] == g_colliderStampGen) continue;
                g_colliderVisitStamp[idx] = g_colliderStampGen;
                if (distSqPointToBuildingOBB(acPos, g_allBuildings[idx]) <= keepR2) {
                    g_activeColliders100m.push_back(idx);
                }
            }
        }
    }
}

static bool checkAircraftBuildingCollision100m(const Vec3& testPt, const Vec3& acPos, float radius) {
    const float maxCollR2 = BLDG_COLLIDE_RADIUS * BLDG_COLLIDE_RADIUS;
    for (int idx : g_activeColliders100m) {
        const BuildingInst& b = g_allBuildings[idx];
        if (distSqPointToBuildingOBB(acPos, b) > maxCollR2) continue;
        if (testPt.y - radius > b.y + b.h || testPt.y + radius < b.y) continue;
        float dx = testPt.x - b.x, dz = testPt.z - b.z;
        float lx = std::fabs(dx * b.cosY + dz * b.sinY);
        float lz = std::fabs(-dx * b.sinY + dz * b.cosY);
        if (lx <= b.w * 0.5f + radius && lz <= b.d * 0.5f + radius) return true;
    }
    return false;
}

// 2D Oriented Bounding Box (SAT) + Koridor/Deniz kose kontrolu ile %100 cakismasiz bina yerlestirme testi
static bool canPlaceBuildingOBB(const BuildingInst& cand, float minGap) {
    float hw = cand.w * 0.5f, hd = cand.d * 0.5f;
    Vec2 rA(cand.cosY, cand.sinY), fA(-cand.sinY, cand.cosY);
    // Binanin 4 kosesinin de denizden > 20m ve yol/ray koridorundan > 18m uzakta oldugunu dogrula
    const float sx[4] = {-hw, hw, hw, -hw};
    const float sz[4] = {-hd, -hd, hd, hd};
    for (int c = 0; c < 4; c++) {
        float cx = cand.x + rA.x * sx[c] + fA.x * sz[c];
        float cz = cand.z + rA.z * sx[c] + fA.z * sz[c];
        const WorldMaskCell& mc = queryWorldMask(cx, cz);
        if (mc.signedSeaDist <= 20.0f || mc.corridorDist <= 18.0f) return false;
    }

    float searchR = 0.55f * std::sqrt(cand.w * cand.w + cand.d * cand.d) + minGap + 18.0f;
    int x0 = clampi((int)((cand.x - searchR - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int x1 = clampi((int)((cand.x + searchR - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int z0 = clampi((int)((cand.z - searchR - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    int z1 = clampi((int)((cand.z + searchR - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);

    float hwA = hw + minGap * 0.5f, hdA = hd + minGap * 0.5f;
    for (int gz = z0; gz <= z1; gz++) {
        for (int gx = x0; gx <= x1; gx++) {
            for (int idx : g_spatialHash[gz * SH_NX + gx]) {
                const BuildingInst& b = g_allBuildings[idx];
                float dx = b.x - cand.x, dz = b.z - cand.z;
                float hwB = b.w * 0.5f + minGap * 0.5f;
                float hdB = b.d * 0.5f + minGap * 0.5f;
                Vec2 rB(b.cosY, b.sinY), fB(-b.sinY, b.cosY);

                // Eksen 1: rA
                float proj1 = hwA + hwB * std::fabs(dot2(rA, rB)) + hdB * std::fabs(dot2(rA, fB));
                if (std::fabs(dx * rA.x + dz * rA.z) >= proj1) continue;
                // Eksen 2: fA
                float proj2 = hdA + hwB * std::fabs(dot2(fA, rB)) + hdB * std::fabs(dot2(fA, fB));
                if (std::fabs(dx * fA.x + dz * fA.z) >= proj2) continue;
                // Eksen 3: rB
                float proj3 = hwB + hwA * std::fabs(dot2(rB, rA)) + hdA * std::fabs(dot2(rB, fA));
                if (std::fabs(dx * rB.x + dz * rB.z) >= proj3) continue;
                // Eksen 4: fB
                float proj4 = hdB + hwA * std::fabs(dot2(fB, rA)) + hdA * std::fabs(dot2(fB, fA));
                if (std::fabs(dx * fB.x + dz * fB.z) >= proj4) continue;

                return false; // 2D OBB kesismesi var
            }
        }
    }
    return true;
}

// Yamaclardaki binalarin havada asili kalmamasi icin 4 kose kotunu tarayip subasman temeli ekler
static void applyBuildingFoundationOnSlope(BuildingInst& b) {
    float hw = b.w * 0.5f, hd = b.d * 0.5f;
    Vec2 r(b.cosY, b.sinY), f(-b.sinY, b.cosY);
    float e0 = sampleTerrainElev(b.x - r.x * hw - f.x * hd, b.z - r.z * hw - f.z * hd);
    float e1 = sampleTerrainElev(b.x + r.x * hw - f.x * hd, b.z + r.z * hw - f.z * hd);
    float e2 = sampleTerrainElev(b.x + r.x * hw + f.x * hd, b.z + r.z * hw + f.z * hd);
    float e3 = sampleTerrainElev(b.x - r.x * hw + f.x * hd, b.z - r.z * hw + f.z * hd);
    float minE = std::min(std::min(e0, e1), std::min(e2, e3));
    float maxE = std::max(std::max(e0, e1), std::max(e2, e3));
    float slopeDiff = clampf(maxE - minE, 0.0f, 9.0f);
    b.y = minE - 0.7f;
    b.h += slopeDiff + 0.7f;
}

// Ozel bolge mesafe kontrolleri (Havalimani, Kulturpark, Inciralti Kent Ormani)
static bool isInAirportExclusion(float wx, float wz) {
    Vec2 ltbj = geoPt(LTBJ_LAT, LTBJ_LON);
    Vec2 d1 = headingDir(LTBJ_HDG);
    Vec2 r1(-d1.z, d1.x);
    Vec2 rel1(wx - ltbj.x, wz - ltbj.z);
    if (std::fabs(dot2(rel1, d1)) < 2100.0f && std::fabs(dot2(rel1, r1)) < 750.0f) return true;

    Vec2 ltbl = geoPt(LTBL_LAT, LTBL_LON);
    Vec2 d2 = headingDir(LTBL_HDG);
    Vec2 r2(-d2.z, d2.x);
    Vec2 rel2(wx - ltbl.x, wz - ltbl.z);
    if (std::fabs(dot2(rel2, d2)) < 1800.0f && std::fabs(dot2(rel2, r2)) < 480.0f) return true;
    return false;
}

static bool isInUrbanForest(float wx, float wz) {
    Vec2 kp = geoPt(KULTURPARK_LAT, KULTURPARK_LON);
    if (std::fabs(wx - kp.x) < 480.0f && std::fabs(wz - kp.z) < 620.0f) return true;
    Vec2 inc = geoPt(INCIRALTI_LAT, INCIRALTI_LON);
    if (std::fabs(wx - inc.x) < 1350.0f && std::fabs(wz - inc.z) < 520.0f) return true;
    return false;
}

// 420.000+ Gercek Overture+OSM Binayi veya Yedek Ada/Parsel Duzeniyle Uret
static void generateCityBuildingsAndTrees() {
    g_allBuildings.clear();
    g_allBuildings.reserve(g_osmBuildings.empty() ? 65000 : (g_osmBuildings.size() + 128));
    for (auto& b : g_spatialHash) b.clear();
    for (int i = 0; i < CHUNK_NX * CHUNK_NZ; i++) {
        int cx = i % CHUNK_NX, cz = i / CHUNK_NX;
        CityChunk& ck = g_cityChunks[i];
        ck.cx = WORLD_X0 + (cx + 0.5f) * CHUNK_SIZE;
        ck.cz = WORLD_Z0 + (cz + 0.5f) * CHUNK_SIZE;
        ck.cy = sampleTerrainElev(ck.cx, ck.cz);
        const WorldMaskCell& mc = queryWorldMask(ck.cx, ck.cz);
        if (mc.signedSeaDist < 1800.0f) {
            ck.uDir = norm2(Vec2(mc.coastTangX, mc.coastTangZ));
        } else {
            ck.uDir = norm2(Vec2(mc.roadTangX, mc.roadTangZ));
        }
        if (len2(ck.uDir) < 0.1f) ck.uDir = Vec2(1.0f, 0.0f);
        ck.vDir = Vec2(-ck.uDir.z, ck.uDir.x);
        ck.bldgs.clear();
        ck.trees.clear();
    }

    Vec2 maviC  = geoPt(MAVISEHIR_LAT, MAVISEHIR_LON);
    Vec2 aosbC  = geoPt(AOSB_LAT, AOSB_LON);
    Vec2 ataC   = geoPt(ATASAN_LAT, ATASAN_LON);
    Vec2 esbC   = geoPt(ESBAS_LAT, ESBAS_LON);
    Vec2 bayC   = geoPt(FOLKART_LAT, FOLKART_LON);
    Vec2 mistC  = geoPt(MISTRAL_LAT, MISTRAL_LON);
    Vec2 egepC  = geoPt(EGEPERLA_LAT, EGEPERLA_LON);
    Vec2 saatC  = geoPt(SAAT_LAT, SAAT_LON);
    Vec2 asanC  = geoPt(ASANSOR_LAT, ASANSOR_LON);
    Vec2 cumhC  = geoPt(CUMHURIYET_LAT, CUMHURIYET_LON);
    Vec2 gundC  = geoPt(GUNDOGDU_LAT, GUNDOGDU_LON);
    Vec2 hiltC  = geoPt(HILTON_LAT, HILTON_LON);
    Vec2 kskAC  = geoPt(KSK_ANIT_LAT, KSK_ANIT_LON);
    Vec2 kadifC = geoPt(KADIFEKALE_LAT, KADIFEKALE_LON);
    Vec2 fuarC  = geoPt(38.3320, 27.1140);
    Vec2 tuzlaC = geoPt(38.5050, 26.9380);
    Vec2 portA  = geoPt(38.4422, 27.1430);
    Vec2 portB  = geoPt(38.4460, 27.1585);
    float portMinX = std::min(portA.x, portB.x) - 15.0f, portMaxX = std::max(portA.x, portB.x) + 15.0f;
    float portMinZ = std::min(portA.z, portB.z) - 15.0f, portMaxZ = std::max(portA.z, portB.z) + 15.0f;
    bool realMode = !g_osmBuildings.empty();

    auto isNearLandmark = [&](float wx, float wz) {
        Vec2 p(wx, wz);
        if (realMode) {
            if (len2(p - bayC) < 115.0f || len2(p - mistC) < 95.0f || len2(p - egepC) < 90.0f) return true;
            if (len2(p - saatC) < 55.0f  || len2(p - cumhC) < 75.0f || len2(p - gundC) < 70.0f) return true;
            if (len2(p - asanC) < 60.0f  || len2(p - hiltC) < 85.0f || len2(p - kskAC) < 70.0f) return true;
            if (len2(p - kadifC) < 125.0f) return true;
            return false;
        }
        if (wx >= portMinX && wx <= portMaxX && wz >= portMinZ && wz <= portMaxZ) return true;
        if (len2(p - bayC) < 155.0f || len2(p - mistC) < 135.0f || len2(p - egepC) < 135.0f) return true;
        if (len2(p - saatC) < 175.0f || len2(p - cumhC) < 125.0f || len2(p - gundC) < 115.0f) return true;
        if (len2(p - asanC) < 80.0f  || len2(p - hiltC) < 105.0f || len2(p - kskAC) < 95.0f)  return true;
        if (len2(p - kadifC) < 165.0f || len2(p - fuarC) < 320.0f) return true;
        if ((wx < -10400.0f && wz < -2000.0f) || len2(p - tuzlaC) < 3200.0f) return true;
        for (int i = 0; i < NUM_PIERS; i++) {
            Vec2 pp = geoPt(PIERS[i].lat, PIERS[i].lon);
            if (len2(p - pp) < 75.0f) return true;
        }
        return false;
    };

    // 1. Sentetik yedek kiyi seridi (sadece binary veri yoksa)
    if (!realMode) {
        for (size_t i = 0; i + 1 < g_coastSpline.size(); i += 2) {
            Vec2 cp = g_coastSpline[i];
            if (cp.x < -9800.0f || cp.z < -8200.0f) continue;
            Vec2 tang = g_coastTangents[i];
            Vec2 inNorm = g_coastInNormals[i];

            for (int row = 0; row < 2; row++) {
                float offsetIn = 54.0f + row * 36.0f;
                Vec2 bp = cp + inNorm * offsetIn;
                if (bp.x <= WORLD_X0 + 100 || bp.x >= WORLD_X1 - 100 ||
                    bp.z <= WORLD_Z0 + 100 || bp.z >= WORLD_Z1 - 100) continue;
                const WorldMaskCell& mc = queryWorldMask(bp.x, bp.z);
                if (mc.signedSeaDist <= 46.0f || mc.corridorDist <= 20.0f) continue;
                if (isInUrbanForest(bp.x, bp.z) || isInAirportExclusion(bp.x, bp.z) || isNearLandmark(bp.x, bp.z)) continue;

                uint32_t h = hash2i((int)bp.x, (int)bp.z + row * 97);
                int floors = (row == 0) ? (7 + (int)(h % 3)) : (6 + (int)(h % 3));
                BuildingInst b;
                b.x = bp.x;
                b.z = bp.z;
                b.w = 22.0f + (float)((h >> 4) & 3);
                b.d = 15.0f + (float)((h >> 8) & 3);
                b.h = floors * 3.25f;
                b.cosY = tang.x;
                b.sinY = tang.z;
                b.type = BLDG_APT_COAST;
                b.colorVar = (uint8_t)(h & 7);
                applyBuildingFoundationOnSlope(b);

                if (canPlaceBuildingOBB(b, 1.5f)) {
                    int bIdx = (int)g_allBuildings.size();
                    g_allBuildings.push_back(b);
                    registerBuildingInSpatialHash(bIdx);
                    int ckx = clampi((int)((b.x - WORLD_X0) / CHUNK_SIZE), 0, CHUNK_NX - 1);
                    int ckz = clampi((int)((b.z - WORLD_Z0) / CHUNK_SIZE), 0, CHUNK_NZ - 1);
                    g_cityChunks[ckz * CHUNK_NX + ckx].bldgs.push_back(b);
                }
            }
        }
    }

    // 1B. Gercek Overture + OSM bina verisini (izmir_osm_world.bin) dogrudan konum, aci ve yukseklikleriyle yerlestir
    if (realMode) {
        std::printf("[CITY] %zu gercek Overture+OSM binasi 3D haritaya isleniyor...\n", g_osmBuildings.size());
        for (const auto& ob : g_osmBuildings) {
            if (ob.x <= WORLD_X0 + 80.0f || ob.x >= WORLD_X1 - 80.0f ||
                ob.z <= WORLD_Z0 + 80.0f || ob.z >= WORLD_Z1 - 80.0f) continue;
            const WorldMaskCell& mc = queryWorldMask(ob.x, ob.z);
            if (mc.signedSeaDist <= 1.5f || mc.corridorDist <= 0.5f) continue;
            if (isInAirportExclusion(ob.x, ob.z) || isNearLandmark(ob.x, ob.z)) continue;

            // Yalnizca ayni merkezdeki cift kayitlari (<2m) ele, gercek bitisik nizam bloklari koru
            bool dup = false;
            int hx = clampi((int)((ob.x - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
            int hz = clampi((int)((ob.z - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
            for (int gz = std::max(0, hz - 1); gz <= std::min(SH_NZ - 1, hz + 1) && !dup; gz++) {
                for (int gx = std::max(0, hx - 1); gx <= std::min(SH_NX - 1, hx + 1) && !dup; gx++) {
                    for (int idx : g_spatialHash[gz * SH_NX + gx]) {
                        const BuildingInst& eb = g_allBuildings[idx];
                        float dx = eb.x - ob.x, dz = eb.z - ob.z;
                        if (dx * dx + dz * dz < 4.0f) { dup = true; break; }
                    }
                }
            }
            if (dup) continue;

            BuildingInst b;
            b.x = ob.x;
            b.z = ob.z;
            b.w = ob.w;
            b.d = ob.d;
            b.h = ob.h;
            b.cosY = std::cos(ob.yaw);
            b.sinY = std::sin(ob.yaw);
            uint32_t hv = hash2i((int)ob.x, (int)ob.z);
            b.colorVar = (uint8_t)(((ob.r >> 5) ^ (ob.g >> 4) ^ (ob.b >> 3) ^ (hv & 7)) & 7);

            if (ob.kind == 1) b.type = BLDG_TOWER_GLASS;
            else if (ob.kind == 2) b.type = BLDG_FACTORY;
            else if (ob.kind == 4) b.type = BLDG_MOSQUE;
            else if (ob.kind == 3 || ob.kind == 5 || ob.kind == 6) b.type = BLDG_INSTITUTION;
            else if (ob.kind == 7) b.type = BLDG_HILLSIDE_HOUSE;
            else if (ob.kind == 8) b.type = BLDG_GREENHOUSE;
            else if (len2(Vec2(ob.x, ob.z) - saatC) < 850.0f && ob.h <= 11.5f) b.type = BLDG_HISTORIC_KEMERALTI;
            else if (mc.elev > 55.0f && ob.h <= 11.5f) b.type = BLDG_HILLSIDE_HOUSE;
            else if (mc.signedSeaDist < 120.0f && ob.h >= 16.0f) b.type = BLDG_APT_COAST;
            else b.type = BLDG_URBAN_APT;

            applyBuildingFoundationOnSlope(b);
            int bIdx = (int)g_allBuildings.size();
            g_allBuildings.push_back(b);
            registerBuildingInSpatialHash(bIdx);
            int ckx = clampi((int)((b.x - WORLD_X0) / CHUNK_SIZE), 0, CHUNK_NX - 1);
            int ckz = clampi((int)((b.z - WORLD_Z0) / CHUNK_SIZE), 0, CHUNK_NZ - 1);
            g_cityChunks[ckz * CHUNK_NX + ckx].bldgs.push_back(b);
        }
        std::printf("  Toplam %zu gercek bina 3D sehre yerlestirildi.\n", g_allBuildings.size());
    }

    // 2. Sentetik yedek Ada/Parsel izgarasi (sadece binary veri yoksa calisir)
    const float stepU = 26.5f;
    const float stepV = 23.0f;

    if (!realMode) {
        for (int ckz = 0; ckz < CHUNK_NZ; ckz++) {
            for (int ckx = 0; ckx < CHUNK_NX; ckx++) {
                CityChunk& ck = g_cityChunks[ckz * CHUNK_NX + ckx];
                Vec2 uD = ck.uDir;
                Vec2 vD = ck.vDir;

                for (int iv = -11; iv <= 11; iv++) {
                    if ((iv + 120) % 3 == 0) continue;
                    float vCoord = iv * stepV;

                    for (int iu = -10; iu <= 10; iu++) {
                        if ((iu + 120) % 5 == 0) continue;
                        float uCoord = iu * stepU;

                        float wx = ck.cx + uD.x * uCoord + vD.x * vCoord;
                        float wz = ck.cz + uD.z * uCoord + vD.z * vCoord;

                        if (std::fabs(wx - ck.cx) >= CHUNK_SIZE * 0.49f ||
                            std::fabs(wz - ck.cz) >= CHUNK_SIZE * 0.49f) continue;
                        if (wx <= WORLD_X0 + 160.0f || wx >= WORLD_X1 - 160.0f ||
                            wz <= WORLD_Z0 + 160.0f || wz >= WORLD_Z1 - 160.0f) continue;

                        const WorldMaskCell& mc = queryWorldMask(wx, wz);
                        if (mc.signedSeaDist <= 52.0f || mc.corridorDist <= 20.0f || mc.elev > 220.0f) continue;
                        if (isInAirportExclusion(wx, wz) || isInUrbanForest(wx, wz) || isNearLandmark(wx, wz)) continue;

                        if (mc.elev > 75.0f) {
                            bool isBucaGaziemir = (wx > 200.0f && wx < 5200.0f && wz > -2500.0f && wz < 12500.0f && mc.elev < 135.0f);
                            bool isKadifekale = (len2(Vec2(wx, wz) - kadifC) < 650.0f && mc.elev < 185.0f);
                            bool isBornovaFoothill = (wx > 4800.0f && wx < 8800.0f && wz > -4500.0f && wz < -1000.0f && mc.elev < 85.0f);
                            if (!isBucaGaziemir && !isKadifekale && !isBornovaFoothill) continue;
                        }

                        float density = 0.0f;
                        for (int d = 0; d < NUM_DISTRICTS; d++) {
                            Vec2 dc = geoPt(DISTRICTS[d].lat, DISTRICTS[d].lon);
                            float r = DISTRICTS[d].radius * 1.48f;
                            float dist = len2(Vec2(wx - dc.x, wz - dc.z));
                            if (dist < r) {
                                float w = (1.0f - (dist / r) * (dist / r)) * DISTRICTS[d].density;
                                if (w > density) density = w;
                            }
                        }
                        if (mc.signedSeaDist < 1500.0f && wx > -11500.0f && wz > -12000.0f) {
                            density = std::max(density, 0.88f * (1.0f - mc.signedSeaDist / 1650.0f));
                        }
                        if (mc.corridorDist < 550.0f && mc.elev < 175.0f && wx > -10000.0f) {
                            density = std::max(density, 0.72f * (1.0f - mc.corridorDist / 600.0f));
                        }

                        float dMavi = len2(Vec2(wx, wz) - maviC);
                        float dAosb = len2(Vec2(wx, wz) - aosbC);
                        float dAta  = len2(Vec2(wx, wz) - ataC);
                        float dEsb  = len2(Vec2(wx, wz) - esbC);
                        float dBay  = len2(Vec2(wx, wz) - bayC);
                        if (dAosb < 1850.0f || dAta < 1100.0f || dEsb < 1100.0f) density = std::max(density, 0.85f);
                        if (dMavi < 1150.0f) density = std::max(density, 0.90f);

                        uint32_t hv = hash2i(ckx * 131 + iu + 50, ckz * 197 + iv + 50);
                        float prob = ((hv >> 16) & 1023) * (1.0f / 1023.0f);
                        if (prob > density) continue;

                        BuildingInst b;
                        b.x = wx;
                        b.z = wz;
                        b.cosY = uD.x;
                        b.sinY = uD.z;
                        b.colorVar = (uint8_t)(hv & 7);

                        Vec3 nTerr = sampleTerrainNormal(wx, wz);
                        bool isKadifekaleSlope = (len2(Vec2(wx, wz) - kadifC) < 950.0f && mc.elev > 35.0f);
                        bool isHillSlope = (nTerr.y < 0.952f) || isKadifekaleSlope;
                        bool isKemeralti = (wx >= 160.0f && wx <= 1180.0f && wz >= -240.0f && wz <= 560.0f);

                        if (isKemeralti) {
                            int floors = 1 + (int)(hv % 3);
                            b.w = 13.0f + (float)((hv >> 4) & 3);
                            b.d = 11.0f + (float)((hv >> 8) & 3);
                            b.h = floors * 2.85f + 1.2f;
                            b.type = BLDG_HISTORIC_KEMERALTI;
                        } else if (dMavi < 1050.0f) {
                            if ((iu & 1) != 0 || ((iv & 1) != 0 && (hv % 3 != 0))) continue;
                            int floors = 12 + (int)(hv % 11);
                            b.w = 21.0f + (float)((hv >> 3) & 5);
                            b.d = 17.0f + (float)((hv >> 7) & 5);
                            b.h = floors * 3.15f;
                            b.type = BLDG_MAVISEHIR_RES;
                        } else if (dAosb < 1800.0f || dAta < 1050.0f || dEsb < 1050.0f) {
                            if ((iu & 1) != 0) continue;
                            b.w = 28.0f + (float)((hv >> 4) % 7);
                            b.d = 17.5f + (float)((hv >> 8) % 5);
                            b.h = 9.5f + (float)((hv >> 12) % 6);
                            b.type = BLDG_FACTORY;
                        } else if (dBay < 1100.0f && (hv % 4 == 0)) {
                            int floors = 6 + (int)(hv % 7);
                            b.w = 21.0f + (float)((hv >> 4) & 5);
                            b.d = 18.0f + (float)((hv >> 8) & 5);
                            b.h = floors * 3.4f;
                            b.type = BLDG_URBAN_APT;
                        } else if (mc.signedSeaDist < 240.0f) {
                            int floors = 7 + (int)(hv % 3);
                            b.w = 18.5f + (float)((hv >> 4) & 3);
                            b.d = 13.8f + (float)((hv >> 8) & 3);
                            b.h = floors * 3.25f;
                            b.type = BLDG_APT_COAST;
                        } else if (isHillSlope) {
                            int floors = 2 + (int)(hv % 2);
                            b.w = 12.0f + (float)((hv >> 4) & 3);
                            b.d = 10.5f + (float)((hv >> 8) & 3);
                            b.h = floors * 2.9f;
                            b.type = BLDG_HILLSIDE_HOUSE;
                        } else {
                            int floors = 4 + (int)(hv % 5);
                            b.w = 17.5f + (float)((hv >> 4) & 3);
                            b.d = 13.5f + (float)((hv >> 8) & 3);
                            b.h = floors * 3.1f;
                            b.type = BLDG_URBAN_APT;
                        }
                        applyBuildingFoundationOnSlope(b);

                        if (canPlaceBuildingOBB(b, 1.4f)) {
                            int bIdx = (int)g_allBuildings.size();
                            g_allBuildings.push_back(b);
                            registerBuildingInSpatialHash(bIdx);
                            ck.bldgs.push_back(b);
                        }
                    }
                }
            }
        }

        if ((int)g_allBuildings.size() < 53000) {
            for (int ckz = 0; ckz < CHUNK_NZ && (int)g_allBuildings.size() < 55000; ckz++) {
                for (int ckx = 0; ckx < CHUNK_NX && (int)g_allBuildings.size() < 55000; ckx++) {
                    CityChunk& ck = g_cityChunks[ckz * CHUNK_NX + ckx];
                    if (ck.cx < -10500.0f || ck.cx > 11500.0f || ck.cz < -21500.0f || ck.cz > 13500.0f) continue;
                    Vec2 uD = ck.uDir, vD = ck.vDir;
                    for (int iv = -13; iv <= 13 && (int)g_allBuildings.size() < 55000; iv++) {
                        if ((iv + 120) % 3 == 0) continue;
                        float vCoord = iv * stepV;
                        for (int iu = -12; iu <= 12 && (int)g_allBuildings.size() < 55000; iu++) {
                            if ((iu + 120) % 5 == 0) continue;
                            float uCoord = iu * stepU;
                            float wx = ck.cx + uD.x * uCoord + vD.x * vCoord;
                            float wz = ck.cz + uD.z * uCoord + vD.z * vCoord;
                            if (std::fabs(wx - ck.cx) >= CHUNK_SIZE * 0.49f ||
                                std::fabs(wz - ck.cz) >= CHUNK_SIZE * 0.49f) continue;
                            const WorldMaskCell& mc = queryWorldMask(wx, wz);
                            if (mc.signedSeaDist <= 52.0f || mc.corridorDist <= 20.0f || mc.elev > 210.0f) continue;
                            if (isInAirportExclusion(wx, wz) || isInUrbanForest(wx, wz) || isNearLandmark(wx, wz)) continue;
                            if (mc.elev > 75.0f) {
                                bool isBucaGaziemir = (wx > 200.0f && wx < 5200.0f && wz > -2500.0f && wz < 12500.0f && mc.elev < 135.0f);
                                bool isKadifekale = (len2(Vec2(wx, wz) - kadifC) < 650.0f && mc.elev < 185.0f);
                                bool isBornovaFoothill = (wx > 4800.0f && wx < 8800.0f && wz > -4500.0f && wz < -1000.0f && mc.elev < 85.0f);
                                if (!isBucaGaziemir && !isKadifekale && !isBornovaFoothill) continue;
                            }

                            uint32_t hv = hash2i(ckx * 97 + iu + 31, ckz * 131 + iv + 19);
                            BuildingInst b;
                            b.x = wx;
                            b.z = wz;
                            b.w = 15.0f + (float)(hv & 3);
                            b.d = 12.5f + (float)((hv >> 2) & 3);
                            Vec3 nTerr = sampleTerrainNormal(wx, wz);
                            bool slopeH = (nTerr.y < 0.952f) || (len2(Vec2(wx, wz) - kadifC) < 950.0f && mc.elev > 35.0f);
                            b.h = (slopeH ? (2 + (int)((hv >> 4) % 3)) : (4 + (int)((hv >> 4) % 5))) * 3.05f;
                            b.cosY = uD.x;
                            b.sinY = uD.z;
                            b.type = slopeH ? BLDG_HILLSIDE_HOUSE : BLDG_URBAN_APT;
                            b.colorVar = (uint8_t)(hv & 7);
                            applyBuildingFoundationOnSlope(b);

                            if (canPlaceBuildingOBB(b, 1.3f)) {
                                int bIdx = (int)g_allBuildings.size();
                                g_allBuildings.push_back(b);
                                registerBuildingInSpatialHash(bIdx);
                                ck.bldgs.push_back(b);
                            }
                        }
                    }
                }
            }
        }
    }

    // Simge gokdelenleri de SpatialHashGrid'e kaydet (carpismalar icin)
    {
        Vec2 fk = geoPt(FOLKART_LAT, FOLKART_LON);
        for (int s = -1; s <= 1; s += 2) {
            BuildingInst b{fk.x + s * 42.0f, sampleTerrainElev(fk.x, fk.z), fk.z, 36.0f, 36.0f, 200.0f, 1, 0, BLDG_TOWER_GLASS, 0};
            int idx = (int)g_allBuildings.size();
            g_allBuildings.push_back(b);
            registerBuildingInSpatialHash(idx);
        }
        Vec2 ms = geoPt(MISTRAL_LAT, MISTRAL_LON);
        {
            BuildingInst b{ms.x, sampleTerrainElev(ms.x, ms.z), ms.z, 38.0f, 38.0f, 216.0f, 1, 0, BLDG_TOWER_GLASS, 1};
            int idx = (int)g_allBuildings.size();
            g_allBuildings.push_back(b);
            registerBuildingInSpatialHash(idx);
        }
        Vec2 ep = geoPt(EGEPERLA_LAT, EGEPERLA_LON);
        {
            BuildingInst b{ep.x, sampleTerrainElev(ep.x, ep.z), ep.z, 40.0f, 32.0f, 186.0f, 1, 0, BLDG_TOWER_GLASS, 2};
            int idx = (int)g_allBuildings.size();
            g_allBuildings.push_back(b);
            registerBuildingInSpatialHash(idx);
        }
        Vec2 hl = geoPt(HILTON_LAT, HILTON_LON);
        {
            BuildingInst b{hl.x, sampleTerrainElev(hl.x, hl.z), hl.z, 34.0f, 28.0f, 142.0f, 1, 0, BLDG_TOWER_GLASS, 3};
            int idx = (int)g_allBuildings.size();
            g_allBuildings.push_back(b);
            registerBuildingInSpatialHash(idx);
        }
    }
    g_totalBuildings = (int)g_allBuildings.size();

    // 3. EGE ORMANLARI, KULTURPARK, INCIRALTI, YAMANLAR, BALCOVA/CATALKAYA, BELKAHVE, OVERTURE GERCEK AGACLAR VE PALMIYELER
    //    4 Farkli Ege Agac Turu: 0=Kizilcam/Fistikcami, 1=Ege Selvisi, 2=Palmiye, 3=Zeytin/Maki
    g_totalTrees = 0;
    Rng trng(202635u);
    auto addTree = [&](float tx, float tz, float h, float r, uint8_t kind) {
        if (tx <= WORLD_X0 + 80.0f || tx >= WORLD_X1 - 80.0f ||
            tz <= WORLD_Z0 + 80.0f || tz >= WORLD_Z1 - 80.0f) return;
        const WorldMaskCell& mc = queryWorldMask(tx, tz);
        if (mc.signedSeaDist <= 4.5f) return;
        if (g_hasGisGrid) {
            uint8_t lc = sampleGisClass(tx, tz);
            if (lc == LC_SEA || lc == LC_WATER || lc == LC_SALT_WETLAND) return;
        }
        TreeInst t{tx, mc.elev, tz, h, r, kind};
        int ckx = clampi((int)((tx - WORLD_X0) / CHUNK_SIZE), 0, CHUNK_NX - 1);
        int ckz = clampi((int)((tz - WORLD_Z0) / CHUNK_SIZE), 0, CHUNK_NZ - 1);
        g_cityChunks[ckz * CHUNK_NX + ckx].trees.push_back(t);
        g_totalTrees++;
    };

    if (g_hasGisGrid) {
        // 3A. Overture land.parquet'ten gelen 4.200+ gercek bireysel agac
        for (size_t i = 0; i < g_realTrees.size(); i++) {
            float tx = g_realTrees[i].x, tz = g_realTrees[i].z;
            const WorldMaskCell& mc = queryWorldMask(tx, tz);
            if (mc.signedSeaDist <= 4.5f) continue;
            if (isInAirportExclusion(tx, tz)) continue;
            uint8_t lc = sampleGisClass(tx, tz);
            uint8_t kind = 0;
            if (mc.signedSeaDist < 180.0f && (i % 2 == 0)) kind = 2; // Sahil palmiyesi
            else if (lc == LC_CROP) kind = 3;
            else if (lc == LC_FOREST) kind = (uint8_t)((i % 5 == 0) ? 1 : 0);
            else kind = (uint8_t)(i % 4);
            addTree(tx, tz, trng.range(10.5f, 19.0f), trng.range(4.0f, 6.8f), kind);
        }

        // 3B. Kordon, Karsiyaka, Bostanli ve Guzelyali sahil parklari palmiye dizileri
        for (size_t i = 0; i + 1 < g_coastSpline.size(); i++) {
            Vec2 cp = g_coastSpline[i];
            Vec2 inNorm = g_coastInNormals[i];
            Vec2 tp1 = cp + inNorm * 10.0f;
            const WorldMaskCell& mc = queryWorldMask(tp1.x, tp1.z);
            uint8_t lc = sampleGisClass(tp1.x, tp1.z);
            if (mc.signedSeaDist > 4.5f && (lc == LC_PARK || lc == LC_SHRUB) &&
                !checkBuildingCollision(Vec3(tp1.x, mc.elev + 3.0f, tp1.z), 3.0f)) {
                addTree(tp1.x, tp1.z, 12.5f + (i % 3) * 1.4f, 4.3f, 2);
            }
        }

        // 3C. Gercek 2048x2048 GIS Orman (LC_FOREST) & Tarim/Zeytinlik (LC_CROP) agac katmani
        for (float wz = WORLD_Z0 + 120.0f; wz < WORLD_Z1 - 120.0f; wz += 92.0f) {
            for (float wx = WORLD_X0 + 120.0f; wx < WORLD_X1 - 120.0f; wx += 92.0f) {
                float tx = wx + trng.range(-38.0f, 38.0f);
                float tz = wz + trng.range(-38.0f, 38.0f);
                uint8_t lc = sampleGisClass(tx, tz);
                if (lc != LC_FOREST && lc != LC_CROP) continue;
                if (lc == LC_CROP && trng.uni() > 0.24f) continue;
                const WorldMaskCell& mc = queryWorldMask(tx, tz);
                if (mc.signedSeaDist <= 14.0f || mc.corridorDist <= 22.0f) continue;
                if (isInAirportExclusion(tx, tz)) continue;
                if (checkBuildingCollision(Vec3(tx, mc.elev + 4.0f, tz), 6.0f)) continue;
                uint32_t hv = hash2i((int)tx, (int)tz);
                uint8_t kind = (lc == LC_FOREST) ? ((hv % 5 == 0) ? 1 : 0) : 3;
                addTree(tx, tz, trng.range(10.0f, 20.5f), trng.range(4.2f, 7.2f), kind);
            }
        }

        // 3D. Gercek GIS Park ve Korular (Kulturpark, Inciralti, Bostanli Sahil, Hasanaga, Asik Veysel)
        for (float wz = WORLD_Z0 + 120.0f; wz < WORLD_Z1 - 120.0f; wz += 38.0f) {
            for (float wx = WORLD_X0 + 120.0f; wx < WORLD_X1 - 120.0f; wx += 38.0f) {
                float tx = wx + trng.range(-15.0f, 15.0f);
                float tz = wz + trng.range(-15.0f, 15.0f);
                if (sampleGisClass(tx, tz) != LC_PARK) continue;
                if (trng.uni() > 0.55f) continue;
                const WorldMaskCell& mc = queryWorldMask(tx, tz);
                if (mc.signedSeaDist <= 6.0f || mc.corridorDist <= 12.0f) continue;
                if (isInAirportExclusion(tx, tz)) continue;
                if (checkBuildingCollision(Vec3(tx, mc.elev + 3.0f, tz), 4.5f)) continue;
                uint32_t hv = hash2i((int)tx, (int)tz);
                uint8_t kind = (mc.signedSeaDist < 120.0f && (hv % 3 == 0)) ? 2 : (uint8_t)(hv % 4);
                addTree(tx, tz, trng.range(10.5f, 20.0f), trng.range(4.0f, 6.8f), kind);
            }
        }
    } else {
        // Sentetik agac yerlestirme yedegi (izmir_osm_world.bin yoksa)
        Vec2 kp = geoPt(KULTURPARK_LAT, KULTURPARK_LON);
        for (int i = 0; i < 3200; i++) {
            float tx = kp.x + trng.range(-450.0f, 450.0f);
            float tz = kp.z + trng.range(-590.0f, 590.0f);
            uint8_t kind = (uint8_t)((i % 5 == 0) ? 2 : (i % 4 == 0 ? 3 : (i % 3 == 0 ? 1 : 0)));
            addTree(tx, tz, trng.range(12.0f, 22.0f), trng.range(4.2f, 7.2f), kind);
        }
        Vec2 inc = geoPt(INCIRALTI_LAT, INCIRALTI_LON);
        for (int i = 0; i < 4500; i++) {
            float tx = inc.x + trng.range(-1320.0f, 1320.0f);
            float tz = inc.z + trng.range(-500.0f, 500.0f);
            if (queryWorldMask(tx, tz).corridorDist > 15.0f) {
                uint8_t kind = (uint8_t)((i % 4 == 0) ? 3 : (i % 3));
                addTree(tx, tz, trng.range(10.5f, 19.5f), trng.range(4.0f, 6.8f), kind);
            }
        }
        for (size_t i = 0; i + 1 < g_coastSpline.size(); i++) {
            Vec2 cp = g_coastSpline[i];
            Vec2 inNorm = g_coastInNormals[i];
            Vec2 tp1 = cp + inNorm * 9.5f;
            addTree(tp1.x, tp1.z, 13.0f + (i % 3) * 1.5f, 4.4f, 2);
            Vec2 tp2 = cp + inNorm * 36.5f;
            if (queryWorldMask(tp2.x, tp2.z).corridorDist > 11.0f) {
                addTree(tp2.x, tp2.z, 12.5f, 4.6f, (uint8_t)(i % 3 == 0 ? 2 : (i % 2 == 0 ? 0 : 3)));
            }
        }
        for (size_t rIdx = 1; rIdx < g_roads.size(); rIdx++) {
            const auto& rs = g_roads[rIdx];
            for (size_t i = 2; i + 2 < rs.pts.size(); i += 2) {
                Vec3 p = rs.pts[i];
                Vec2 t = rs.tangs[i];
                Vec2 n(-t.z, t.x);
                for (int s = -1; s <= 1; s += 2) {
                    float tx = p.x + n.x * (s * (rs.halfW + 4.5f));
                    float tz = p.z + n.z * (s * (rs.halfW + 4.5f));
                    const WorldMaskCell& mc = queryWorldMask(tx, tz);
                    if (mc.signedSeaDist > 12.0f && mc.corridorDist > 2.5f &&
                        !checkBuildingCollision(Vec3(tx, mc.elev + 3.0f, tz), 4.0f)) {
                        uint8_t k = (mc.signedSeaDist < 260.0f) ? 2 : (uint8_t)((i + rIdx) % 4 == 0 ? 1 : 3);
                        addTree(tx, tz, 11.5f + (i % 3) * 1.4f, 4.2f, k);
                    }
                }
            }
        }
        struct ForestZone { double lat, lon; float rx, rz; int count; uint8_t mainKind; };
        static const ForestZone FORESTS[] = {
            {38.525, 27.165, 4800.0f, 3800.0f, 6400, 0},
            {38.490, 27.135, 3400.0f, 2200.0f, 3800, 0},
            {38.368, 27.015, 4400.0f, 2800.0f, 5800, 0},
            {38.352, 26.945, 3600.0f, 2400.0f, 3400, 3},
            {38.435, 27.255, 3800.0f, 2900.0f, 4200, 0},
            {38.362, 27.215, 3200.0f, 2500.0f, 3200, 3},
            {38.485, 26.965, 2400.0f, 1800.0f, 2200, 3},
            {38.585, 27.125, 3800.0f, 2600.0f, 3600, 0}
        };
        for (const auto& fz : FORESTS) {
            Vec2 fc = geoPt(fz.lat, fz.lon);
            for (int i = 0; i < fz.count; i++) {
                float a = trng.range(0.0f, 2.0f * PI_F);
                float rad = std::sqrt(trng.uni());
                float tx = fc.x + std::cos(a) * rad * fz.rx;
                float tz = fc.z + std::sin(a) * rad * fz.rz;
                const WorldMaskCell& mc = queryWorldMask(tx, tz);
                if (mc.signedSeaDist <= 28.0f || mc.corridorDist <= 22.0f) continue;
                if (isInAirportExclusion(tx, tz)) continue;
                if (checkBuildingCollision(Vec3(tx, mc.elev + 4.0f, tz), 6.0f)) continue;
                uint8_t k = (i % 5 == 0) ? 1 : ((i % 3 == 0) ? 3 : fz.mainKind);
                addTree(tx, tz, trng.range(10.0f, 21.0f), trng.range(4.2f, 7.4f), k);
            }
        }
    }

    // Simge kule ve yapilarin (Folkart, Mistral, Hilton, Saat Kulesi, Asansor, Liman, vb.)
    // 100m carpisma ve silah/bomba fizigine dahil edilmesi (cift cizim yapmadan)
    auto addColliderOnly = [&](float cx, float baseY, float cz, float w, float d, float h,
                               float cosY, float sinY, uint8_t btype) {
        BuildingInst b;
        b.x = cx; b.y = baseY; b.z = cz;
        b.w = w; b.d = d; b.h = h;
        b.cosY = cosY; b.sinY = sinY;
        b.type = btype;
        b.colorVar = 0;
        int bIdx = (int)g_allBuildings.size();
        g_allBuildings.push_back(b);
        registerBuildingInSpatialHash(bIdx);
    };
    Vec2 fkP = geoPt(FOLKART_LAT, FOLKART_LON);
    float fkY = sampleTerrainElev(fkP.x, fkP.z);
    addColliderOnly(fkP.x - 42.0f, fkY, fkP.z, 32.0f, 32.0f, 200.0f, 1, 0, BLDG_TOWER_GLASS);
    addColliderOnly(fkP.x + 42.0f, fkY, fkP.z, 32.0f, 32.0f, 200.0f, 1, 0, BLDG_TOWER_GLASS);
    Vec2 msP = geoPt(MISTRAL_LAT, MISTRAL_LON);
    float msY = sampleTerrainElev(msP.x, msP.z);
    addColliderOnly(msP.x, msY, msP.z, 34.0f, 28.0f, 216.0f, 1, 0, BLDG_TOWER_GLASS);
    addColliderOnly(msP.x + 46.0f, msY, msP.z + 18.0f, 28.0f, 26.0f, 154.0f, 1, 0, BLDG_TOWER_GLASS);
    Vec2 epP = geoPt(EGEPERLA_LAT, EGEPERLA_LON);
    float epY = sampleTerrainElev(epP.x, epP.z);
    addColliderOnly(epP.x - 28.0f, epY, epP.z, 28.0f, 24.0f, 186.0f, 1, 0, BLDG_TOWER_GLASS);
    addColliderOnly(epP.x + 28.0f, epY, epP.z, 26.0f, 22.0f, 142.0f, 1, 0, BLDG_TOWER_GLASS);
    Vec2 hlP = geoPt(HILTON_LAT, HILTON_LON);
    addColliderOnly(hlP.x, sampleTerrainElev(hlP.x, hlP.z), hlP.z, 34.0f, 26.0f, 142.0f, 1, 0, BLDG_TOWER_GLASS);
    Vec2 skP = geoPt(SAAT_LAT, SAAT_LON);
    float skY = sampleTerrainElev(skP.x, skP.z);
    addColliderOnly(skP.x, skY, skP.z, 8.0f, 8.0f, 34.0f, 1, 0, BLDG_INSTITUTION);
    addColliderOnly(skP.x + 52.0f, skY, skP.z - 12.0f, 16.0f, 16.0f, 16.5f, 1, 0, BLDG_MOSQUE);
    Vec2 kpPier = geoPt(38.4215, 27.1275);
    addColliderOnly(kpPier.x - 35.0f, 2.2f, kpPier.z, 95.0f, 36.0f, 12.7f, 0.88f, -0.47f, BLDG_HISTORIC_KEMERALTI);
    Vec2 asP = geoPt(ASANSOR_LAT, ASANSOR_LON);
    float asY = sampleTerrainElev(asP.x, asP.z);
    addColliderOnly(asP.x, asY - 4.0f, asP.z, 12.0f, 12.0f, 58.0f, 0.92f, 0.38f, BLDG_HISTORIC_KEMERALTI);
    addColliderOnly(asP.x, asY + 51.0f, asP.z + 6.0f, 24.0f, 18.0f, 5.5f, 0.92f, 0.38f, BLDG_HISTORIC_KEMERALTI);
    Vec2 kskA = geoPt(KSK_ANIT_LAT, KSK_ANIT_LON);
    float kskY = sampleTerrainElev(kskA.x, kskA.z);
    addColliderOnly(kskA.x, kskY, kskA.z, 16.0f, 14.0f, 42.0f, 0.95f, 0.31f, BLDG_INSTITUTION);
    Vec2 kultP = geoPt(KULTURPARK_LAT, KULTURPARK_LON);
    addColliderOnly(kultP.x - 120.0f, sampleTerrainElev(kultP.x - 120.0f, kultP.z + 90.0f), kultP.z + 90.0f, 16.0f, 16.0f, 48.0f, 1, 0, BLDG_INSTITUTION);
    Vec2 hisP = geoPt(38.4196, 27.1394);
    addColliderOnly(hisP.x, sampleTerrainElev(hisP.x, hisP.z), hisP.z, 28.0f, 28.0f, 15.0f, 1, 0, BLDG_MOSQUE);
    Vec2 kzP = geoPt(38.4202, 27.1380);
    addColliderOnly(kzP.x, sampleTerrainElev(kzP.x, kzP.z), kzP.z, 54.0f, 44.0f, 10.6f, 1, 0, BLDG_HISTORIC_KEMERALTI);
    Vec2 portP = geoPt(38.4438, 27.1500);
    float portY = sampleTerrainElev(portP.x, portP.z);
    for (int c = 0; c < 4; c++) {
        addColliderOnly(portP.x - 180.0f + c * 135.0f, portY, portP.z + 128.0f, 94.0f, 38.0f, 13.0f, 1, 0, BLDG_FACTORY);
    }
    Vec2 shipP = geoPt(38.4445, 27.1388);
    addColliderOnly(shipP.x, 0.0f, shipP.z, 32.0f, 185.0f, 28.0f, 0.78f, 0.62f, BLDG_INSTITUTION);
}

// =====================================================================================
//  STATIK DUNYA VE CITYCHUNK DISPLAY LIST DERLEYICILERI
// =====================================================================================
static const int WORLD_TILE_N = 8;
static const int NUM_WORLD_TILES = WORLD_TILE_N * WORLD_TILE_N;
static GLuint g_dlTerrainTile[NUM_WORLD_TILES] = {0};
static GLuint g_dlSeaTile[NUM_WORLD_TILES] = {0};
static GLuint g_dlRoadTile[NUM_WORLD_TILES] = {0};
static float g_roadTileRadius[NUM_WORLD_TILES] = {0};
static GLuint g_dlRails = 0;
static GLuint g_dlHorizonApron = 0;
static GLuint g_dlQuayAndPromenade = 0;
static GLuint g_dlLandmarksAndAirports = 0;
static std::vector<Vec3> g_nightStreetLights;
static std::vector<Vec3> g_nightBeaconLights;

static void emitBoxFaces(float x, float y, float z, float w, float d, float h,
                         float cosY, float sinY, float uScale, float vScale, bool capTop = true) {
    float hw = w * 0.5f, hd = d * 0.5f;
    Vec2 r(cosY, sinY), f(-sinY, cosY);
    Vec2 c0 = Vec2(x, z) - r * hw - f * hd;
    Vec2 c1 = Vec2(x, z) + r * hw - f * hd;
    Vec2 c2 = Vec2(x, z) + r * hw + f * hd;
    Vec2 c3 = Vec2(x, z) - r * hw + f * hd;
    float y0 = y, y1 = y + h;
    float uw = w * uScale, ud = d * uScale, vh = h * vScale;

    // On yuz (-f)
    glNormal3f(-f.x, 0.0f, -f.z);
    glTexCoord2f(0, 0);   glVertex3f(c0.x, y0, c0.z);
    glTexCoord2f(uw, 0);  glVertex3f(c1.x, y0, c1.z);
    glTexCoord2f(uw, vh); glVertex3f(c1.x, y1, c1.z);
    glTexCoord2f(0, vh);  glVertex3f(c0.x, y1, c0.z);
    // Sag yuz (+r)
    glNormal3f(r.x, 0.0f, r.z);
    glTexCoord2f(0, 0);   glVertex3f(c1.x, y0, c1.z);
    glTexCoord2f(ud, 0);  glVertex3f(c2.x, y0, c2.z);
    glTexCoord2f(ud, vh); glVertex3f(c2.x, y1, c2.z);
    glTexCoord2f(0, vh);  glVertex3f(c1.x, y1, c1.z);
    // Arka yuz (+f)
    glNormal3f(f.x, 0.0f, f.z);
    glTexCoord2f(0, 0);   glVertex3f(c2.x, y0, c2.z);
    glTexCoord2f(uw, 0);  glVertex3f(c3.x, y0, c3.z);
    glTexCoord2f(uw, vh); glVertex3f(c3.x, y1, c3.z);
    glTexCoord2f(0, vh);  glVertex3f(c2.x, y1, c2.z);
    // Sol yuz (-r)
    glNormal3f(-r.x, 0.0f, -r.z);
    glTexCoord2f(0, 0);   glVertex3f(c3.x, y0, c3.z);
    glTexCoord2f(ud, 0);  glVertex3f(c0.x, y0, c0.z);
    glTexCoord2f(ud, vh); glVertex3f(c0.x, y1, c0.z);
    glTexCoord2f(0, vh);  glVertex3f(c3.x, y1, c3.z);
    // Ust yuz (+Y kapak)
    if (capTop) {
        glNormal3f(0.0f, 1.0f, 0.0f);
        glTexCoord2f(0, 0);   glVertex3f(c0.x, y1, c0.z);
        glTexCoord2f(uw, 0);  glVertex3f(c1.x, y1, c1.z);
        glTexCoord2f(uw, ud); glVertex3f(c2.x, y1, c2.z);
        glTexCoord2f(0, ud);  glVertex3f(c3.x, y1, c3.z);
    }
}

static void emitRoofQuadOrPitch(const BuildingInst& b) {
    float hw = b.w * 0.53f, hd = b.d * 0.53f;
    Vec2 r(b.cosY, b.sinY), f(-b.sinY, b.cosY);
    Vec2 c0 = Vec2(b.x, b.z) - r * hw - f * hd;
    Vec2 c1 = Vec2(b.x, b.z) + r * hw - f * hd;
    Vec2 c2 = Vec2(b.x, b.z) + r * hw + f * hd;
    Vec2 c3 = Vec2(b.x, b.z) - r * hw + f * hd;
    float y1 = b.y + b.h;

    // Sadece Tarihi Kemeraltı ve Kadifekale yamaç evleri geleneksel kiremit kırma çatıya sahiptir
    bool pitchedRoof = (b.type == BLDG_HISTORIC_KEMERALTI) || (b.type == BLDG_HILLSIDE_HOUSE);
    if (pitchedRoof) {
        // 4-Yuzlu Gercek Kiremit Kirma Cati (bina teget acisi b.cosY/b.sinY ile donmus normaller)
        float apexY = y1 + ((b.type == BLDG_HISTORIC_KEMERALTI) ? 2.4f : 2.8f);
        Vec2 ridgeL = Vec2(b.x, b.z) - r * (hw * 0.40f);
        Vec2 ridgeR = Vec2(b.x, b.z) + r * (hw * 0.40f);
        glColor3f(0.84f, 0.38f, 0.22f); // Sicak Ege terracotta kiremit

        // On yamac (-f yonu)
        glNormal3f(-f.x * 0.48f, 0.88f, -f.z * 0.48f);
        glTexCoord2f(0, 0); glVertex3f(c0.x, y1, c0.z);
        glTexCoord2f(2, 0); glVertex3f(c1.x, y1, c1.z);
        glTexCoord2f(2, 1); glVertex3f(ridgeR.x, apexY, ridgeR.z);
        glTexCoord2f(0, 1); glVertex3f(ridgeL.x, apexY, ridgeL.z);

        // Arka yamac (+f yonu)
        glNormal3f(f.x * 0.48f, 0.88f, f.z * 0.48f);
        glTexCoord2f(0, 0); glVertex3f(c2.x, y1, c2.z);
        glTexCoord2f(2, 0); glVertex3f(c3.x, y1, c3.z);
        glTexCoord2f(2, 1); glVertex3f(ridgeL.x, apexY, ridgeL.z);
        glTexCoord2f(0, 1); glVertex3f(ridgeR.x, apexY, ridgeR.z);

        // Sol kirma ucgen yamac (-r yonu)
        glNormal3f(-r.x * 0.48f, 0.88f, -r.z * 0.48f);
        glTexCoord2f(0, 0); glVertex3f(c3.x, y1, c3.z);
        glTexCoord2f(1, 0); glVertex3f(c0.x, y1, c0.z);
        glTexCoord2f(0.5f, 1); glVertex3f(ridgeL.x, apexY, ridgeL.z);
        glTexCoord2f(0.5f, 1); glVertex3f(ridgeL.x, apexY, ridgeL.z);

        // Sag kirma ucgen yamac (+r yonu)
        glNormal3f(r.x * 0.48f, 0.88f, r.z * 0.48f);
        glTexCoord2f(0, 0); glVertex3f(c1.x, y1, c1.z);
        glTexCoord2f(1, 0); glVertex3f(c2.x, y1, c2.z);
        glTexCoord2f(0.5f, 1); glVertex3f(ridgeR.x, apexY, ridgeR.z);
        glTexCoord2f(0.5f, 1); glVertex3f(ridgeR.x, apexY, ridgeR.z);
    } else if (b.type == BLDG_MOSQUE) {
        // Osmanli cami kursun kubbe katmanlari ve minare kulahi
        glColor3f(0.46f, 0.50f, 0.55f);
        float dmW = std::min(b.w, b.d) * 0.52f;
        emitBoxFaces(b.x, b.y + b.h * 0.72f, b.z, dmW, dmW, b.h * 0.22f, b.cosY, b.sinY, 0.05f, 0.05f, true);
        emitBoxFaces(b.x, b.y + b.h * 0.94f, b.z, dmW * 0.55f, dmW * 0.55f, b.h * 0.14f, b.cosY, b.sinY, 0.05f, 0.05f, true);
        if (std::min(b.w, b.d) >= 8.0f) {
            Vec2 mp = Vec2(b.x, b.z) + r * (b.w * 0.42f) - f * (b.d * 0.42f);
            float minTopY = b.y + b.h * 1.15f + 12.0f;
            glColor3f(0.24f, 0.27f, 0.32f);
            emitBoxFaces(mp.x, minTopY, mp.z, 1.15f, 1.15f, 4.2f, b.cosY, b.sinY, 0.1f, 0.1f, true);
        }
    } else if (b.type == BLDG_GREENHOUSE) {
        glColor3f(0.86f, 0.94f, 0.98f);
        glNormal3f(0.0f, 1.0f, 0.0f);
        glTexCoord2f(0, 0); glVertex3f(c0.x, y1 + 0.5f, c0.z);
        glTexCoord2f(2, 0); glVertex3f(c1.x, y1 + 0.5f, c1.z);
        glTexCoord2f(2, 2); glVertex3f(c2.x, y1 + 0.5f, c2.z);
        glTexCoord2f(0, 2); glVertex3f(c3.x, y1 + 0.5f, c3.z);
    } else {
        // Modern Akdeniz Teras Cati (temel cati kapagi; parapet ve Gunisi detayi dlNearDetails'dedir)
        if (b.type == BLDG_FACTORY) glColor3f(0.55f, 0.65f, 0.78f);
        else if (b.type == BLDG_INSTITUTION) glColor3f(0.84f, 0.86f, 0.88f);
        else glColor3f(0.88f, 0.90f, 0.92f);
        glNormal3f(0.0f, 1.0f, 0.0f);
        glTexCoord2f(0, 0); glVertex3f(c0.x, y1, c0.z);
        glTexCoord2f(2, 0); glVertex3f(c1.x, y1, c1.z);
        glTexCoord2f(2, 2); glVertex3f(c2.x, y1, c2.z);
        glTexCoord2f(0, 2); glVertex3f(c3.x, y1, c3.z);
    }
}

static void emitNearRoofDetailAndBalcony(const BuildingInst& b) {
    Vec2 r(b.cosY, b.sinY), f(-b.sinY, b.cosY);
    float y1 = b.y + b.h;
    if (b.type == BLDG_HISTORIC_KEMERALTI) {
        glColor3f(0.52f, 0.32f, 0.20f);
        emitBoxFaces(b.x, b.y + b.h * 0.55f, b.z, b.w + 0.65f, b.d + 0.65f, 0.35f, b.cosY, b.sinY, 0.04f, 0.04f, false);
        return;
    }
    if ((b.type == BLDG_APT_COAST || b.type == BLDG_URBAN_APT) && b.h >= 14.0f && b.w * b.d >= 160.0f) {
        glColor3f(0.96f, 0.95f, 0.92f);
        emitBoxFaces(b.x, b.y + b.h * 0.48f, b.z, b.w + 0.70f, b.d + 0.70f, 0.40f, b.cosY, b.sinY, 0.04f, 0.04f, false);
        emitBoxFaces(b.x, b.y + b.h - 0.55f, b.z, b.w + 0.85f, b.d + 0.85f, 0.48f, b.cosY, b.sinY, 0.04f, 0.04f, false);
    } else if ((b.type == BLDG_MAVISEHIR_RES || b.type == BLDG_TOWER_GLASS) && b.h >= 24.0f) {
        glColor3f(0.92f, 0.95f, 0.99f);
        emitBoxFaces(b.x, b.y + b.h * 0.50f, b.z, b.w + 0.80f, b.d + 0.80f, 0.52f, b.cosY, b.sinY, 0.05f, 0.04f, false);
        emitBoxFaces(b.x, b.y + b.h - 0.70f, b.z, b.w + 1.00f, b.d + 1.00f, 0.62f, b.cosY, b.sinY, 0.05f, 0.04f, false);
    }
    if (b.type != BLDG_FACTORY && b.type != BLDG_INSTITUTION &&
        b.type != BLDG_HISTORIC_KEMERALTI && b.type != BLDG_HILLSIDE_HOUSE &&
        b.type != BLDG_MOSQUE && b.type != BLDG_GREENHOUSE &&
        b.w * b.d >= 150.0f && b.h >= 10.0f) {
        glColor3f(0.88f, 0.88f, 0.86f);
        emitBoxFaces(b.x, y1, b.z, b.w, b.d, 0.72f, b.cosY, b.sinY, 0.04f, 0.04f);
        glColor3f(0.92f, 0.93f, 0.95f);
        emitBoxFaces(b.x, y1, b.z, b.w * 0.30f, b.d * 0.30f, 2.3f, b.cosY, b.sinY, 0.1f, 0.1f);
        float panW = b.w * 0.22f, panD = b.d * 0.18f;
        Vec2 panPos = Vec2(b.x, b.z) + r * (b.w * 0.22f) - f * (b.d * 0.18f);
        glColor3f(0.20f, 0.22f, 0.26f);
        emitBoxFaces(panPos.x, y1 + 0.45f, panPos.z, panW, panD, 0.58f, b.cosY, b.sinY, 0.05f, 0.05f);
        glColor3f(0.86f, 0.89f, 0.92f);
        emitBoxFaces(panPos.x, y1 + 1.08f, panPos.z - f.z * 0.8f, panW * 0.82f, 0.9f, 0.68f, b.cosY, b.sinY, 0.05f, 0.05f);
    }
}

static void compileChunkIfNeeded(CityChunk& ck) {
    if (ck.compiled) return;
    ck.compiled = true;
    int ckIdx = (int)(&ck - g_cityChunks.data());
    bool hasV2R = (ckIdx >= 0 && ckIdx < (int)g_chunkV2Roads.size() && !g_chunkV2Roads[ckIdx].empty());
    bool hasGard = (ckIdx >= 0 && ckIdx < (int)g_chunkGardens.size() && !g_chunkGardens[ckIdx].empty());
    bool hasSide = (!g_hasV2Roads && ckIdx >= 0 && ckIdx < (int)g_chunkSideStreets.size() && !g_chunkSideStreets[ckIdx].empty());
    if (ck.bldgs.empty() && ck.trees.empty() && !hasV2R && !hasGard && !hasSide) return;

    // 0. Sehir Adalari, Mahalle Caddeleri, Ara Sokaklar, Sanayi Sahalari, Kaldirimlar ve Bahceler
    if (!ck.bldgs.empty() || hasV2R || hasGard) {
        ck.dlGround = glGenLists(1);
        glNewList(ck.dlGround, GL_COMPILE);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-2.5f, -2.5f);
        glEnable(GL_TEXTURE_2D);

        // A) Gercek Overture v2 Ara Sokaklari ve Toplayici Caddeler
        if (hasV2R) {
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ASPHALT]);
            glBegin(GL_QUADS);
            for (const auto& seg : g_chunkV2Roads[ckIdx]) {
                Vec2 d = seg.b - seg.a;
                float l = len2(d);
                if (l < 0.8f) continue;
                Vec2 norm(-d.z / l, d.x / l);
                float hw = std::max(1.8f, seg.halfW);
                Vec2 p0 = seg.a - norm * hw, p1 = seg.b - norm * hw;
                Vec2 p2 = seg.b + norm * hw, p3 = seg.a + norm * hw;
                Vec2 mid = (seg.a + seg.b) * 0.5f;
                float dy = (seg.level > 0) ? (0.36f + 3.2f * seg.level) : 0.34f;
                float e0L = sampleRenderedTerrainMaxElev(p0.x, p0.z);
                float e0C = sampleRenderedTerrainMaxElev(seg.a.x, seg.a.z);
                float e0R = sampleRenderedTerrainMaxElev(p3.x, p3.z);
                float e1L = sampleRenderedTerrainMaxElev(p1.x, p1.z);
                float e1C = sampleRenderedTerrainMaxElev(seg.b.x, seg.b.z);
                float e1R = sampleRenderedTerrainMaxElev(p2.x, p2.z);
                float eMC = sampleRenderedTerrainMaxElev(mid.x, mid.z);
                float y0 = std::max(std::max(e0C, std::max(e0L, e0R)), eMC - 0.15f) + dy;
                float y1 = std::max(std::max(e1C, std::max(e1L, e1R)), eMC - 0.15f) + dy;
                if (seg.tier <= 3) glColor3f(0.68f, 0.70f, 0.74f);
                else glColor3f(0.76f, 0.78f, 0.81f);
                glNormal3f(0.0f, 1.0f, 0.0f);
                glTexCoord2f(0.0f, 0.0f); glVertex3f(p0.x, y0, p0.z);
                glTexCoord2f(l * 0.12f, 0.0f); glVertex3f(p1.x, y1, p1.z);
                glTexCoord2f(l * 0.12f, 1.0f); glVertex3f(p2.x, y1, p2.z);
                glTexCoord2f(0.0f, 1.0f); glVertex3f(p3.x, y0, p3.z);
                if (std::fabs(e0L - e0R) > 0.30f || std::fabs(e1L - e1R) > 0.30f || seg.level > 0) {
                    glColor3f(0.45f, 0.44f, 0.42f);
                    glNormal3f(-norm.x, 0.25f, -norm.z);
                    glTexCoord2f(0.0f, 0.0f); glVertex3f(p0.x, y0, p0.z);
                    glTexCoord2f(0.0f, 1.0f); glVertex3f(p0.x, y0 - 1.6f, p0.z);
                    glTexCoord2f(l * 0.12f, 1.0f); glVertex3f(p1.x, y1 - 1.6f, p1.z);
                    glTexCoord2f(l * 0.12f, 0.0f); glVertex3f(p1.x, y1, p1.z);
                    glNormal3f(norm.x, 0.25f, norm.z);
                    glTexCoord2f(0.0f, 0.0f); glVertex3f(p2.x, y1, p2.z);
                    glTexCoord2f(0.0f, 1.0f); glVertex3f(p2.x, y1 - 1.6f, p2.z);
                    glTexCoord2f(l * 0.12f, 1.0f); glVertex3f(p3.x, y0 - 1.6f, p3.z);
                    glTexCoord2f(l * 0.12f, 0.0f); glVertex3f(p3.x, y0, p3.z);
                }
            }
            glEnd();
        } else if (ckIdx >= 0 && ckIdx < (int)g_chunkSideStreets.size() && !g_chunkSideStreets[ckIdx].empty()) {
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ASPHALT]);
            glColor3f(0.82f, 0.83f, 0.85f);
            glBegin(GL_QUADS);
            for (const auto& seg : g_chunkSideStreets[ckIdx]) {
                Vec2 d = seg.b - seg.a;
                float l = len2(d);
                if (l < 1.0f) continue;
                Vec2 norm(-d.z / l, d.x / l);
                float hw = 3.2f; // ~6.4m genislikte gercek mahalle sokagi
                Vec2 p0 = seg.a - norm * hw, p1 = seg.b - norm * hw;
                Vec2 p2 = seg.b + norm * hw, p3 = seg.a + norm * hw;
                float y0 = sampleTerrainElev(seg.a.x, seg.a.z) + 0.16f;
                float y1 = sampleTerrainElev(seg.b.x, seg.b.z) + 0.16f;
                glNormal3f(0.0f, 1.0f, 0.0f);
                glTexCoord2f(0.0f, 0.0f); glVertex3f(p0.x, y0, p0.z);
                glTexCoord2f(2.0f, 0.0f); glVertex3f(p1.x, y1, p1.z);
                glTexCoord2f(2.0f, 1.0f); glVertex3f(p2.x, y1, p2.z);
                glTexCoord2f(0.0f, 1.0f); glVertex3f(p3.x, y0, p3.z);
            }
            glEnd();
        } else if (!g_hasV2Roads) {
            // Sentetik yedek ara sokak koridorlari
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ASPHALT]);
            glColor3f(0.85f, 0.86f, 0.88f);
            glBegin(GL_QUADS);
            for (const auto& b : ck.bldgs) {
                Vec2 r(b.cosY, b.sinY);
                Vec2 f(-b.sinY, b.cosY);
                bool isInd = (b.type == BLDG_FACTORY);
                float halfRoadW = isInd ? 6.2f : 4.6f;
                float halfRoadL = b.w * 0.5f + (isInd ? 9.5f : 4.5f);
                float swPad = isInd ? 4.5f : 2.5f;
                Vec2 stCenter = Vec2(b.x, b.z) - f * (b.d * 0.5f + swPad + halfRoadW);
                Vec2 s0 = stCenter - r * halfRoadL - f * halfRoadW;
                Vec2 s1 = stCenter + r * halfRoadL - f * halfRoadW;
                Vec2 s2 = stCenter + r * halfRoadL + f * halfRoadW;
                Vec2 s3 = stCenter - r * halfRoadL + f * halfRoadW;
                float ySt = b.y + 0.16f;
                glNormal3f(0.0f, 1.0f, 0.0f);
                glTexCoord2f(0.0f, 0.0f); glVertex3f(s0.x, ySt, s0.z);
                glTexCoord2f(2.0f, 0.0f); glVertex3f(s1.x, ySt, s1.z);
                glTexCoord2f(2.0f, 1.5f); glVertex3f(s2.x, ySt, s2.z);
                glTexCoord2f(0.0f, 1.5f); glVertex3f(s3.x, ySt, s3.z);
            }
            glEnd();
        }

        // A2) Gercek OSM Park ve Bahce Poligonlari (v1 uyumluluk)
        if (hasGard) {
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GRASS]);
            for (const auto& g : g_chunkGardens[ckIdx]) {
                if (g.pts.size() < 3) continue;
                if (g.gtype == 0) glColor3f(0.38f, 0.82f, 0.40f);
                else if (g.gtype == 1) glColor3f(0.24f, 0.58f, 0.28f);
                else glColor3f(0.70f, 0.68f, 0.45f);
                glBegin(GL_POLYGON);
                glNormal3f(0.0f, 1.0f, 0.0f);
                for (const auto& pt : g.pts) {
                    float ye = sampleTerrainElev(pt.x, pt.z) + 0.14f;
                    glTexCoord2f(pt.x * 0.05f, pt.z * 0.05f);
                    glVertex3f(pt.x, ye, pt.z);
                }
                glEnd();
            }
        }

        // Sentetik modda bina etrafi beton/bahce tabanlari (v2'de gercek GIS zemin + dar sokaklar korunur)
        if (!g_hasV2Roads && !ck.bldgs.empty()) {
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
            glBegin(GL_QUADS);
            for (const auto& b : ck.bldgs) {
                bool isInd = (b.type == BLDG_FACTORY);
                if (isInd) glColor3f(0.76f, 0.77f, 0.79f);
                else glColor3f(0.92f, 0.92f, 0.90f);
                float pad = isInd ? 9.2f : 4.8f;
                float swW = b.w + pad;
                float swD = b.d + pad;
                Vec2 r(b.cosY, b.sinY);
                Vec2 f(-b.sinY, b.cosY);
                Vec2 c0 = Vec2(b.x, b.z) - r * (swW * 0.5f) - f * (swD * 0.5f);
                Vec2 c1 = Vec2(b.x, b.z) + r * (swW * 0.5f) - f * (swD * 0.5f);
                Vec2 c2 = Vec2(b.x, b.z) + r * (swW * 0.5f) + f * (swD * 0.5f);
                Vec2 c3 = Vec2(b.x, b.z) - r * (swW * 0.5f) + f * (swD * 0.5f);
                float ySw = b.y + 0.28f;
                glNormal3f(0.0f, 1.0f, 0.0f);
                glTexCoord2f(0.0f, 0.0f); glVertex3f(c0.x, ySw, c0.z);
                glTexCoord2f(2.5f, 0.0f); glVertex3f(c1.x, ySw, c1.z);
                glTexCoord2f(2.5f, 2.5f); glVertex3f(c2.x, ySw, c2.z);
                glTexCoord2f(0.0f, 2.5f); glVertex3f(c3.x, ySw, c3.z);
            }
            glEnd();

            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GRASS]);
            glColor3f(0.42f, 0.80f, 0.34f);
            glBegin(GL_QUADS);
            for (const auto& b : ck.bldgs) {
                if (b.type != BLDG_URBAN_APT && b.type != BLDG_APT_COAST && b.type != BLDG_MAVISEHIR_RES) continue;
                Vec2 f(-b.sinY, b.cosY);
                Vec2 r(b.cosY, b.sinY);
                float halfGdnW = b.w * 0.5f + ((b.type == BLDG_MAVISEHIR_RES) ? 8.5f : 3.8f);
                float halfGdnD = (b.type == BLDG_MAVISEHIR_RES) ? 7.5f : 4.2f;
                Vec2 gCenter = Vec2(b.x, b.z) + f * (b.d * 0.5f + 2.5f + halfGdnD);
                Vec2 g0 = gCenter - r * halfGdnW - f * halfGdnD;
                Vec2 g1 = gCenter + r * halfGdnW - f * halfGdnD;
                Vec2 g2 = gCenter + r * halfGdnW + f * halfGdnD;
                Vec2 g3 = gCenter - r * halfGdnW + f * halfGdnD;
                float gy = b.y + 0.20f;
                glNormal3f(0.0f, 1.0f, 0.0f);
                glTexCoord2f(0.0f, 0.0f); glVertex3f(g0.x, gy, g0.z);
                glTexCoord2f(1.5f, 0.0f); glVertex3f(g1.x, gy, g1.z);
                glTexCoord2f(1.5f, 1.5f); glVertex3f(g2.x, gy, g2.z);
                glTexCoord2f(0.0f, 1.5f); glVertex3f(g3.x, gy, g3.z);
            }
            glEnd();
        }

        glDisable(GL_POLYGON_OFFSET_FILL);
        glEndList();
    }

    // 1. Apartman, yamac evi, Kemeralti, cami ve okul/hastane cepheleri (capTop=false -> cati dlRoof/dlTileRoof'da)
    bool hasApt = false, hasGlass = false, hasInd = false, hasFlatRoof = false, hasTileRoof = false;
    bool hasNearDetail = false, hasSkyline = false, hasNight = false;
    for (const auto& b : ck.bldgs) {
        if (b.type == BLDG_FACTORY) hasInd = true;
        else if (b.type == BLDG_MAVISEHIR_RES || b.type == BLDG_TOWER_GLASS) hasGlass = true;
        else hasApt = true;
        bool pitched = (b.type == BLDG_HISTORIC_KEMERALTI) || (b.type == BLDG_HILLSIDE_HOUSE);
        if (pitched) hasTileRoof = true; else hasFlatRoof = true;
        if (b.h >= 9.0f || b.type == BLDG_TOWER_GLASS || b.type == BLDG_MAVISEHIR_RES || b.type == BLDG_MOSQUE) hasSkyline = true;
        if (b.h >= 5.5f && b.type != BLDG_MOSQUE && b.type != BLDG_GREENHOUSE) hasNight = true;
        if (b.type == BLDG_HISTORIC_KEMERALTI || b.h >= 10.0f) hasNearDetail = true;
    }

    if (hasApt) {
        ck.dlApt = glGenLists(1);
        glNewList(ck.dlApt, GL_COMPILE);
        glBegin(GL_QUADS);
        for (const auto& b : ck.bldgs) {
            if (b.type == BLDG_FACTORY || b.type == BLDG_MAVISEHIR_RES || b.type == BLDG_TOWER_GLASS) continue;
            if (b.type == BLDG_MOSQUE) {
                glColor3f(0.95f, 0.91f, 0.82f);
                emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h * 0.72f, b.cosY, b.sinY, 0.04f, 0.04f, false);
                if (std::min(b.w, b.d) >= 8.0f) {
                    Vec2 r(b.cosY, b.sinY), f(-b.sinY, b.cosY);
                    Vec2 mp = Vec2(b.x, b.z) + r * (b.w * 0.42f) - f * (b.d * 0.42f);
                    glColor3f(0.98f, 0.96f, 0.92f);
                    emitBoxFaces(mp.x, b.y, mp.z, 2.3f, 2.3f, b.h * 0.35f, b.cosY, b.sinY, 0.08f, 0.08f, true);
                    emitBoxFaces(mp.x, b.y, mp.z, 1.50f, 1.50f, b.h * 1.15f + 12.0f, b.cosY, b.sinY, 0.08f, 0.08f, false);
                    glColor3f(0.92f, 0.88f, 0.78f);
                    emitBoxFaces(mp.x, b.y + b.h * 1.15f + 7.0f, mp.z, 2.4f, 2.4f, 0.75f, b.cosY, b.sinY, 0.08f, 0.08f, true);
                }
                continue;
            }
            if (b.type == BLDG_INSTITUTION) {
                static const float instTint[4][3] = {
                    {0.93f, 0.90f, 0.86f}, {0.88f, 0.91f, 0.94f}, {0.94f, 0.89f, 0.84f}, {0.90f, 0.92f, 0.90f}
                };
                const float* tc = instTint[b.colorVar & 3];
                glColor3f(tc[0], tc[1], tc[2]);
                emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h, b.cosY, b.sinY, 0.04f, 0.035f, false);
                continue;
            }
            if (b.type == BLDG_GREENHOUSE) {
                glColor3f(0.84f, 0.93f, 0.96f);
                emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h, b.cosY, b.sinY, 0.05f, 0.05f, false);
                continue;
            }
            if (b.type == BLDG_HISTORIC_KEMERALTI) {
                static const float histTint[4][3] = {
                    {0.92f, 0.84f, 0.70f}, {0.88f, 0.76f, 0.62f}, {0.86f, 0.72f, 0.58f}, {0.94f, 0.88f, 0.78f}
                };
                const float* tc = histTint[b.colorVar & 3];
                glColor3f(tc[0], tc[1], tc[2]);
                emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h, b.cosY, b.sinY, 0.04f, 0.035f, false);
                continue;
            }
            static const float tint[8][3] = {
                {0.97f, 0.95f, 0.91f}, {0.95f, 0.89f, 0.82f}, {0.88f, 0.92f, 0.96f}, {0.95f, 0.85f, 0.80f},
                {0.92f, 0.94f, 0.89f}, {0.90f, 0.88f, 0.85f}, {0.98f, 0.92f, 0.84f}, {0.86f, 0.90f, 0.93f}
            };
            const float* tc = tint[b.colorVar & 7];
            glColor3f(tc[0], tc[1], tc[2]);
            emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h, b.cosY, b.sinY, 0.045f, 0.038f, false);
        }
        glEnd();
        glEndList();
    }

    // 2. Mavisehir rezidans ve Bayrakli cam kuleleri
    if (hasGlass) {
        ck.dlGlass = glGenLists(1);
        glNewList(ck.dlGlass, GL_COMPILE);
        glBegin(GL_QUADS);
        for (const auto& b : ck.bldgs) {
            if (b.type != BLDG_MAVISEHIR_RES && b.type != BLDG_TOWER_GLASS) continue;
            if (b.type == BLDG_MAVISEHIR_RES) {
                int cv = b.colorVar & 3;
                if (cv == 0) glColor3f(0.92f, 0.90f, 0.85f);
                else if (cv == 1) glColor3f(0.84f, 0.89f, 0.95f);
                else if (cv == 2) glColor3f(0.95f, 0.88f, 0.78f);
                else glColor3f(0.86f, 0.92f, 0.90f);
            } else {
                glColor3f(0.76f, 0.89f, 0.99f);
            }
            emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h, b.cosY, b.sinY, 0.055f, 0.042f, false);
        }
        glEnd();
        glEndList();
    }

    // 3. Cigli AOSB / Ata Sanayi fabrika holleri
    if (hasInd) {
        ck.dlInd = glGenLists(1);
        glNewList(ck.dlInd, GL_COMPILE);
        glBegin(GL_QUADS);
        for (const auto& b : ck.bldgs) {
            if (b.type != BLDG_FACTORY) continue;
            if (b.colorVar & 1) glColor3f(0.62f, 0.74f, 0.88f);
            else glColor3f(0.76f, 0.79f, 0.83f);
            emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h, b.cosY, b.sinY, 0.035f, 0.05f, false);
        }
        glEnd();
        glEndList();
    }

    // 4A. Modern Akdeniz Teras Catilari (tek glBindTexture ile batchlenmis)
    if (hasFlatRoof) {
        ck.dlRoof = glGenLists(1);
        glNewList(ck.dlRoof, GL_COMPILE);
        glBegin(GL_QUADS);
        for (const auto& b : ck.bldgs) {
            bool pitched = (b.type == BLDG_HISTORIC_KEMERALTI) || (b.type == BLDG_HILLSIDE_HOUSE);
            if (!pitched) emitRoofQuadOrPitch(b);
        }
        glEnd();
        glEndList();
    }

    // 4B. Tarihi Kemeralti ve Kadifekale Kiremit Kirma Catilari
    if (hasTileRoof) {
        ck.dlTileRoof = glGenLists(1);
        glNewList(ck.dlTileRoof, GL_COMPILE);
        glBegin(GL_QUADS);
        for (const auto& b : ck.bldgs) {
            bool pitched = (b.type == BLDG_HISTORIC_KEMERALTI) || (b.type == BLDG_HILLSIDE_HOUSE);
            if (pitched) emitRoofQuadOrPitch(b);
        }
        glEnd();
        glEndList();
    }

    // 4C. Yakin LOD Mimari Detaylar + Katmanli 3D Ege Agaclari (< 3.4km)
    if (hasNearDetail || !ck.trees.empty()) {
        ck.dlNearDetails = glGenLists(1);
        glNewList(ck.dlNearDetails, GL_COMPILE);
        if (hasNearDetail) {
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
            glBegin(GL_QUADS);
            for (const auto& b : ck.bldgs) emitNearRoofDetailAndBalcony(b);
            glEnd();
        }
        if (!ck.trees.empty()) {
            glDisable(GL_TEXTURE_2D);
            glColor3f(0.40f, 0.26f, 0.15f);
            glBegin(GL_QUADS);
            for (const auto& t : ck.trees) {
                float trW = (t.kind == 2) ? 0.38f : 0.52f;
                float trH = (t.kind == 2) ? (t.h * 0.80f) : (t.h * 0.34f);
                glNormal3f(0.0f, 0.2f, 1.0f);
                glVertex3f(t.x - trW, t.y, t.z); glVertex3f(t.x + trW, t.y, t.z);
                glVertex3f(t.x + trW * 0.65f, t.y + trH, t.z); glVertex3f(t.x - trW * 0.65f, t.y + trH, t.z);
                glNormal3f(1.0f, 0.2f, 0.0f);
                glVertex3f(t.x, t.y, t.z - trW); glVertex3f(t.x, t.y, t.z + trW);
                glVertex3f(t.x, t.y + trH, t.z + trW * 0.65f); glVertex3f(t.x, t.y + trH, t.z - trW * 0.65f);
            }
            glEnd();

            glBegin(GL_TRIANGLES);
            for (const auto& t : ck.trees) {
                if (t.kind == 2) {
                    float topY = t.y + t.h * 0.80f;
                    float crownY = t.y + t.h * 0.98f;
                    static const float c6[6] = {1.0f, 0.5f, -0.5f, -1.0f, -0.5f, 0.5f};
                    static const float s6[6] = {0.0f, 0.866f, 0.866f, 0.0f, -0.866f, -0.866f};
                    for (int f = 0; f < 6; f++) {
                        float dx = c6[f], dz = s6[f];
                        float px = -dz * 0.75f, pz = dx * 0.75f;
                        float midX = t.x + dx * (t.r * 0.62f), midZ = t.z + dz * (t.r * 0.62f);
                        float tipX = t.x + dx * (t.r * 1.22f), tipZ = t.z + dz * (t.r * 1.22f);
                        float tipY = topY - t.r * 0.30f;
                        glNormal3f(dx * 0.4f, 0.9f, dz * 0.4f);
                        glColor3f(0.18f, 0.56f, 0.16f);
                        glVertex3f(t.x, topY, t.z); glVertex3f(midX + px, crownY, midZ + pz); glVertex3f(midX - px, crownY, midZ - pz);
                        glColor3f(0.28f, 0.68f, 0.22f);
                        glVertex3f(midX + px, crownY, midZ + pz); glVertex3f(tipX, tipY, tipZ); glVertex3f(midX - px, crownY, midZ - pz);
                    }
                } else if (t.kind == 1) {
                    float y0 = t.y + t.h * 0.18f;
                    float yTop = t.y + t.h * 1.06f;
                    float r = t.r * 0.46f;
                    Vec3 p0(t.x - r, y0, t.z - r), p1(t.x + r, y0, t.z - r);
                    Vec3 p2(t.x + r, y0, t.z + r), p3(t.x - r, y0, t.z + r);
                    Vec3 apex(t.x, yTop, t.z);
                    glColor3f(0.10f, 0.36f, 0.14f);
                    glNormal3f(0.0f, 0.35f, -0.94f); glVertex3f(p0.x, p0.y, p0.z); glVertex3f(p1.x, p1.y, p1.z); glVertex3f(apex.x, apex.y, apex.z);
                    glNormal3f(0.94f, 0.35f, 0.0f);  glVertex3f(p1.x, p1.y, p1.z); glVertex3f(p2.x, p2.y, p2.z); glVertex3f(apex.x, apex.y, apex.z);
                    glColor3f(0.14f, 0.44f, 0.18f);
                    glNormal3f(0.0f, 0.35f, 0.94f);  glVertex3f(p2.x, p2.y, p2.z); glVertex3f(p3.x, p3.y, p3.z); glVertex3f(apex.x, apex.y, apex.z);
                    glNormal3f(-0.94f, 0.35f, 0.0f); glVertex3f(p3.x, p3.y, p3.z); glVertex3f(p0.x, p0.y, p0.z); glVertex3f(apex.x, apex.y, apex.z);
                } else {
                    for (int tier = 0; tier < 2; tier++) {
                        float yBase = t.y + t.h * (tier == 0 ? 0.28f : 0.56f);
                        float yPeak = t.y + t.h * (tier == 0 ? 0.78f : 1.05f);
                        float rad = t.r * (tier == 0 ? 0.95f : 0.72f);
                        if (t.kind == 0) glColor3f(tier == 0 ? 0.14f : 0.20f, tier == 0 ? 0.44f : 0.56f, tier == 0 ? 0.16f : 0.20f);
                        else             glColor3f(tier == 0 ? 0.26f : 0.36f, tier == 0 ? 0.46f : 0.56f, tier == 0 ? 0.22f : 0.28f);
                        Vec3 p0(t.x - rad, yBase, t.z), p1(t.x, yBase, t.z - rad);
                        Vec3 p2(t.x + rad, yBase, t.z), p3(t.x, yBase, t.z + rad);
                        Vec3 apex(t.x, yPeak, t.z);
                        glNormal3f(-0.6f, 0.55f, -0.6f); glVertex3f(p0.x, p0.y, p0.z); glVertex3f(p1.x, p1.y, p1.z); glVertex3f(apex.x, apex.y, apex.z);
                        glNormal3f( 0.6f, 0.55f, -0.6f); glVertex3f(p1.x, p1.y, p1.z); glVertex3f(p2.x, p2.y, p2.z); glVertex3f(apex.x, apex.y, apex.z);
                        glNormal3f( 0.6f, 0.55f,  0.6f); glVertex3f(p2.x, p2.y, p2.z); glVertex3f(p3.x, p3.y, p3.z); glVertex3f(apex.x, apex.y, apex.z);
                        glNormal3f(-0.6f, 0.55f,  0.6f); glVertex3f(p3.x, p3.y, p3.z); glVertex3f(p0.x, p0.y, p0.z); glVertex3f(apex.x, apex.y, apex.z);
                    }
                }
            }
            glEnd();
            glEnable(GL_TEXTURE_2D);
        }
        glEndList();
    }

    // 4D. Orta Mesafe Hafif Agac Silueti (3.4km..5.8km)
    if (!ck.trees.empty()) {
        ck.dlMidTrees = glGenLists(1);
        glNewList(ck.dlMidTrees, GL_COMPILE);
        glBegin(GL_TRIANGLES);
        for (const auto& t : ck.trees) {
            if (t.kind == 2) glColor3f(0.22f, 0.58f, 0.20f);
            else if (t.kind == 1) glColor3f(0.12f, 0.40f, 0.16f);
            else glColor3f(0.18f, 0.48f, 0.18f);
            float yB = t.y + t.h * 0.24f, yT = t.y + t.h;
            glNormal3f(0.0f, 0.7f, 0.7f);
            glVertex3f(t.x - t.r, yB, t.z); glVertex3f(t.x + t.r, yB, t.z); glVertex3f(t.x, yT, t.z);
            glNormal3f(0.7f, 0.7f, 0.0f);
            glVertex3f(t.x, yB, t.z - t.r); glVertex3f(t.x, yB, t.z + t.r); glVertex3f(t.x, yT, t.z);
        }
        glEnd();
        glEndList();
    }

    // 4E. Uzak Ufuk Skyline LOD (7.6km..14.5km: yuksek kuleler, rezidanslar ve camiler)
    if (hasSkyline) {
        ck.dlSkylineLod = glGenLists(1);
        glNewList(ck.dlSkylineLod, GL_COMPILE);
        glBegin(GL_QUADS);
        for (const auto& b : ck.bldgs) {
            if (b.h < 9.0f && b.type != BLDG_TOWER_GLASS && b.type != BLDG_MAVISEHIR_RES && b.type != BLDG_MOSQUE) continue;
            if (b.type == BLDG_TOWER_GLASS) glColor3f(0.54f, 0.70f, 0.86f);
            else if (b.type == BLDG_MAVISEHIR_RES) glColor3f(0.84f, 0.86f, 0.90f);
            else glColor3f(0.86f, 0.84f, 0.80f);
            emitBoxFaces(b.x, b.y, b.z, b.w, b.d, b.h, b.cosY, b.sinY, 0.04f, 0.04f, true);
        }
        glEnd();
        glEndList();
    }

    // 4F. Gece Parlayan Pencere Isiklari Katmani (TEX_NIGHT_WINDOWS ile GL_ONE toplamsal)
    if (hasNight) {
        ck.dlNightWindows = glGenLists(1);
        glNewList(ck.dlNightWindows, GL_COMPILE);
        glBegin(GL_QUADS);
        for (const auto& b : ck.bldgs) {
            if (b.h < 5.5f || b.type == BLDG_MOSQUE || b.type == BLDG_GREENHOUSE) continue;
            if (b.type == BLDG_TOWER_GLASS) {
                glColor3f(0.76f, 0.90f, 1.00f);
            } else if (b.type == BLDG_MAVISEHIR_RES) {
                glColor3f(1.00f, 0.88f, 0.66f);
            } else if (b.type == BLDG_FACTORY) {
                if ((b.colorVar & 3) != 0) continue;
                glColor3f(0.55f, 0.65f, 0.78f);
            } else if (b.type == BLDG_HISTORIC_KEMERALTI) {
                glColor3f(1.00f, 0.76f, 0.42f);
            } else {
                int cv = b.colorVar & 3;
                if (cv == 0) glColor3f(1.00f, 0.84f, 0.56f);
                else if (cv == 1) glColor3f(0.95f, 0.78f, 0.48f);
                else if (cv == 2) glColor3f(0.90f, 0.92f, 1.00f);
                else glColor3f(1.00f, 0.88f, 0.62f);
            }
            emitBoxFaces(b.x, b.y, b.z, b.w + 0.16f, b.d + 0.16f, b.h - 0.1f, b.cosY, b.sinY, 0.052f, 0.044f, false);
        }
        glEnd();
        glEndList();
    }
}

static void buildGisTerrainTexture() {
    if (!g_hasGisGrid) return;
    std::vector<unsigned char> rgb(MASK_N * MASK_N * 3);
    const Vec3 sunHill = norm(Vec3(-0.50f, 0.72f, 0.45f));
    parallelFor(MASK_N, [&](int z0, int z1) {
        for (int iz = z0; iz < z1; iz++) {
            for (int ix = 0; ix < MASK_N; ix++) {
                float wx, wz;
                gridToWorld(ix, iz, wx, wz);
                uint8_t lc = g_gisGrid[iz * MASK_N + ix];
                const WorldMaskCell& mc = g_maskGrid[iz * MASK_N + ix];
                float y = mc.elev;
                Vec3 n = sampleTerrainNormal(wx, wz);
                float slope = 1.0f - clampf(n.y, 0.0f, 1.0f);
                float altT = clampf((y - 160.0f) / 620.0f, 0.0f, 1.0f);
                float rockT = clampf(slope * 2.4f + altT * 0.55f, 0.0f, 1.0f);
                float hill = clampf(0.74f + 0.46f * dot(n, sunHill), 0.54f, 1.18f);

                float n1 = fbm2(wx * 0.0018f + 1.3f, wz * 0.0018f + 4.7f, 3);
                float n2 = fbm2(wx * 0.0110f - 2.1f, wz * 0.0110f + 0.9f, 2);
                float detail = n1 * 14.0f + n2 * 11.0f;

                float r = 100.0f, g = 115.0f, b = 75.0f;
                if (lc == LC_SEA) {
                    r = 44.0f; g = 112.0f; b = 132.0f;
                    hill = 1.0f;
                } else if (lc == LC_SALT_WETLAND) {
                    float s1 = clampf(0.5f + 0.7f * n1, 0.0f, 1.0f);
                    float s2 = clampf(0.5f + 0.7f * n2, 0.0f, 1.0f);
                    r = lerpf(lerpf(182.0f, 228.0f, s1), 94.0f, s2 * 0.55f) + detail * 0.4f;
                    g = lerpf(lerpf(166.0f, 206.0f, s1), 156.0f, s2 * 0.55f) + detail * 0.4f;
                    b = lerpf(lerpf(142.0f, 210.0f, s1), 168.0f, s2 * 0.55f) + detail * 0.4f;
                } else if (lc == LC_WATER) {
                    r = 42.0f; g = 112.0f; b = 144.0f;
                    hill = 1.0f;
                } else if (lc == LC_URBAN) {
                    float blockTex = std::sin(wx * 0.14f) * std::sin(wz * 0.14f) * 6.0f;
                    r = (118.0f + detail * 0.55f + blockTex) * hill;
                    g = (120.0f + detail * 0.55f + blockTex) * hill;
                    b = (124.0f + detail * 0.55f + blockTex) * hill;
                } else if (lc == LC_INDUSTRIAL) {
                    r = (138.0f + detail * 0.50f) * hill;
                    g = (142.0f + detail * 0.50f) * hill;
                    b = (148.0f + detail * 0.50f) * hill;
                } else if (lc == LC_PARK) {
                    r = (68.0f + detail) * hill;
                    g = (150.0f + detail * 1.15f) * hill;
                    b = (60.0f + detail * 0.6f) * hill;
                } else if (lc == LC_FOREST) {
                    r = (lerpf(44.0f, 154.0f, rockT * 0.75f) + detail * 0.8f) * hill;
                    g = (lerpf(92.0f, 148.0f, rockT * 0.75f) + detail * 0.9f) * hill;
                    b = (lerpf(40.0f, 128.0f, rockT * 0.75f) + detail * 0.6f) * hill;
                } else if (lc == LC_CROP) {
                    float fA = clampf(0.5f + 0.65f * n1, 0.0f, 1.0f);
                    float fB = clampf(0.5f + 0.65f * n2, 0.0f, 1.0f);
                    float furrow = std::sin(wx * 0.09f + wz * 0.06f) * 7.0f;
                    r = (lerpf(lerpf(134.0f, 166.0f, fA), 98.0f, fB * 0.58f) + detail * 0.5f + furrow) * hill;
                    g = (lerpf(lerpf(116.0f, 148.0f, fA), 134.0f, fB * 0.58f) + detail * 0.6f + furrow * 0.8f) * hill;
                    b = (lerpf(lerpf(76.0f,  94.0f,  fA), 66.0f,  fB * 0.58f) + detail * 0.4f + furrow * 0.5f) * hill;
                } else if (lc == LC_BARREN) {
                    r = (174.0f + detail) * hill; g = (164.0f + detail) * hill; b = (142.0f + detail) * hill;
                } else {
                    r = (lerpf(100.0f, 170.0f, rockT) + detail) * hill;
                    g = (lerpf(124.0f, 162.0f, rockT) + detail) * hill;
                    b = (lerpf(68.0f,  140.0f, rockT) + detail * 0.8f) * hill;
                }

                int idx = (iz * MASK_N + ix) * 3;
                rgb[idx + 0] = (unsigned char)clampf(r, 0.0f, 255.0f);
                rgb[idx + 1] = (unsigned char)clampf(g, 0.0f, 255.0f);
                rgb[idx + 2] = (unsigned char)clampf(b, 0.0f, 255.0f);
            }
        }
    });

    float maxAniso = 1.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
    float targetAniso = std::min(8.0f, std::max(1.0f, maxAniso));
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GIS_MAP]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, targetAniso);
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, MASK_N, MASK_N, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Arazi (Land) 8x8 Spatial Tile Display List'leri + Dis Ufuk Ovasi/Deniz Etegi
static void buildTerrainDisplayList() {
    if (g_hasGisGrid) {
        buildGisTerrainTexture();
    }
    const int step = g_hasGisGrid ? 2 : 4;
    const int alignMask = ~(step - 1);
    const float invN = 1.0f / (float)MASK_N;

    for (int tz = 0; tz < WORLD_TILE_N; tz++) {
        int iz0 = ((tz * (MASK_N - step)) / WORLD_TILE_N) & alignMask;
        int iz1 = (tz + 1 == WORLD_TILE_N) ? (MASK_N - step) : (((tz + 1) * (MASK_N - step)) / WORLD_TILE_N) & alignMask;
        for (int tx = 0; tx < WORLD_TILE_N; tx++) {
            int ix0 = ((tx * (MASK_N - step)) / WORLD_TILE_N) & alignMask;
            int ix1 = (tx + 1 == WORLD_TILE_N) ? (MASK_N - step) : (((tx + 1) * (MASK_N - step)) / WORLD_TILE_N) & alignMask;
            int tidx = tz * WORLD_TILE_N + tx;
            g_dlTerrainTile[tidx] = glGenLists(1);
            glNewList(g_dlTerrainTile[tidx], GL_COMPILE);
            glEnable(GL_TEXTURE_2D);
            if (g_hasGisGrid) {
                glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GIS_MAP]);
                for (int iz = iz0; iz < iz1; iz += step) {
                    glBegin(GL_TRIANGLE_STRIP);
                    for (int ix = ix0; ix <= ix1; ix += step) {
                        for (int dz = 0; dz <= step; dz += step) {
                            int jz = std::min(MASK_N - 1, iz + dz);
                            float wx, wz;
                            gridToWorld(ix, jz, wx, wz);
                            const WorldMaskCell& mc = maskCellAt(ix, jz);
                            float y = terrainRenderElevCell(mc);
                            Vec3 n = sampleTerrainNormal(wx, wz);
                            glNormal3f(n.x, n.y, n.z);
                            glColor3f(1.0f, 1.0f, 1.0f);
                            glTexCoord2f(((float)ix + 0.5f) * invN, ((float)jz + 0.5f) * invN);
                            glVertex3f(wx, y, wz);
                        }
                    }
                    glEnd();
                }
            } else {
                glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GRASS]);
                for (int iz = iz0; iz < iz1; iz += step) {
                    glBegin(GL_TRIANGLE_STRIP);
                    for (int ix = ix0; ix <= ix1; ix += step) {
                        for (int dz = 0; dz <= step; dz += step) {
                            int jz = std::min(MASK_N - 1, iz + dz);
                            float wx, wz;
                            gridToWorld(ix, jz, wx, wz);
                            const WorldMaskCell& mc = maskCellAt(ix, jz);
                            float y = terrainRenderElevCell(mc);
                            Vec3 n = sampleTerrainNormal(wx, wz);
                            glNormal3f(n.x, n.y, n.z);
                            float slope = 1.0f - clampf(n.y, 0.0f, 1.0f);
                            float altT = clampf((y - 75.0f) / 600.0f, 0.0f, 1.0f);
                            float rockT = clampf(slope * 2.8f + altT * 0.70f, 0.0f, 1.0f);
                            float uT = clampf(mc.urbanBlend * (1.0f - rockT * 0.85f), 0.0f, 0.88f);
                            float natR = lerpf(0.72f, 1.55f, rockT);
                            float natG = lerpf(0.98f, 1.45f, rockT);
                            float natB = lerpf(0.60f, 1.30f, rockT);
                            if (wx < -11800.0f && wz < -4200.0f && y < 7.0f) {
                                natR = 1.24f; natG = 1.18f; natB = 0.88f;
                            } else if (y < 22.0f && wx < -2500.0f) {
                                natR = 1.15f; natG = 1.12f; natB = 0.68f;
                            }
                            glColor3f(lerpf(natR, 0.38f, uT), lerpf(natG, 0.40f, uT), lerpf(natB, 0.44f, uT));
                            glTexCoord2f(wx * 0.0020f, wz * 0.0020f);
                            glVertex3f(wx, y, wz);
                        }
                    }
                    glEnd();
                }
            }
            glEndList();
        }
    }

    // Dunya sinirlari otesinde sis icine eriyen uzak ufuk eteği (haritanin ani kesilmesini onler)
    g_dlHorizonApron = glGenLists(1);
    glNewList(g_dlHorizonApron, GL_COMPILE);
    glDisable(GL_TEXTURE_2D);
    glNormal3f(0.0f, 1.0f, 0.0f);
    const float ext = 28000.0f;
    const float yAp = -0.35f;
    glBegin(GL_QUADS);
    // Bati seridi (Sasali / Ege denizi ufku)
    glColor3f(0.28f, 0.40f, 0.36f);
    glVertex3f(WORLD_X0 - ext, yAp, WORLD_Z0 - ext); glVertex3f(WORLD_X0, yAp, WORLD_Z0 - ext);
    glVertex3f(WORLD_X0, yAp, WORLD_Z1 + ext);       glVertex3f(WORLD_X0 - ext, yAp, WORLD_Z1 + ext);
    // Dogu seridi
    glColor3f(0.38f, 0.44f, 0.30f);
    glVertex3f(WORLD_X1, yAp, WORLD_Z0 - ext);       glVertex3f(WORLD_X1 + ext, yAp, WORLD_Z0 - ext);
    glVertex3f(WORLD_X1 + ext, yAp, WORLD_Z1 + ext); glVertex3f(WORLD_X1, yAp, WORLD_Z1 + ext);
    // Kuzey seridi
    glVertex3f(WORLD_X0, yAp, WORLD_Z0 - ext); glVertex3f(WORLD_X1, yAp, WORLD_Z0 - ext);
    glVertex3f(WORLD_X1, yAp, WORLD_Z0);       glVertex3f(WORLD_X0, yAp, WORLD_Z0);
    // Guney seridi
    glVertex3f(WORLD_X0, yAp, WORLD_Z1);       glVertex3f(WORLD_X1, yAp, WORLD_Z1);
    glVertex3f(WORLD_X1, yAp, WORLD_Z1 + ext); glVertex3f(WORLD_X0, yAp, WORLD_Z1 + ext);
    glEnd();
    glEndList();
}

// Izmir Korfezi Canli Deniz 8x8 Tile Display List'leri (Y = 0.0m)
static void buildSeaDisplayList() {
    const int seaStep = g_hasGisGrid ? 2 : 4;
    const int alignMask = ~(seaStep - 1);
    for (int tz = 0; tz < WORLD_TILE_N; tz++) {
        int iz0 = ((tz * (MASK_N - seaStep)) / WORLD_TILE_N) & alignMask;
        int iz1 = (tz + 1 == WORLD_TILE_N) ? (MASK_N - seaStep) : (((tz + 1) * (MASK_N - seaStep)) / WORLD_TILE_N) & alignMask;
        for (int tx = 0; tx < WORLD_TILE_N; tx++) {
            int ix0 = ((tx * (MASK_N - seaStep)) / WORLD_TILE_N) & alignMask;
            int ix1 = (tx + 1 == WORLD_TILE_N) ? (MASK_N - seaStep) : (((tx + 1) * (MASK_N - seaStep)) / WORLD_TILE_N) & alignMask;
            int tidx = tz * WORLD_TILE_N + tx;
            bool hasSea = false;
            if (g_hasGisGrid) {
                for (int iz = iz0; iz < iz1 && !hasSea; iz += seaStep) {
                    for (int ix = ix0; ix < ix1; ix += seaStep) {
                        if (maskCellAt(ix, iz).signedSeaDist <= 4.0f ||
                            maskCellAt(ix + seaStep, iz).signedSeaDist <= 4.0f ||
                            maskCellAt(ix, iz + seaStep).signedSeaDist <= 4.0f ||
                            maskCellAt(ix + seaStep, iz + seaStep).signedSeaDist <= 4.0f) {
                            hasSea = true;
                            break;
                        }
                    }
                }
            } else {
                hasSea = true;
            }
            if (!hasSea) {
                g_dlSeaTile[tidx] = 0;
                continue;
            }
            g_dlSeaTile[tidx] = glGenLists(1);
            glNewList(g_dlSeaTile[tidx], GL_COMPILE);
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_WATER]);
            glNormal3f(0.0f, 1.0f, 0.0f);
            for (int iz = iz0; iz < iz1; iz += seaStep) {
                if (g_hasGisGrid) {
                    glBegin(GL_QUADS);
                    for (int ix = ix0; ix < ix1; ix += seaStep) {
                        const WorldMaskCell& c00 = maskCellAt(ix, iz);
                        const WorldMaskCell& c01 = maskCellAt(ix, iz + seaStep);
                        const WorldMaskCell& c11 = maskCellAt(ix + seaStep, iz + seaStep);
                        const WorldMaskCell& c10 = maskCellAt(ix + seaStep, iz);
                        if (c00.signedSeaDist > 4.0f && c01.signedSeaDist > 4.0f &&
                            c11.signedSeaDist > 4.0f && c10.signedSeaDist > 4.0f) {
                            continue;
                        }
                        int xs[4] = {ix, ix, ix + seaStep, ix + seaStep};
                        int zs[4] = {iz, iz + seaStep, iz + seaStep, iz};
                        const WorldMaskCell* cs[4] = {&c00, &c01, &c11, &c10};
                        for (int k = 0; k < 4; k++) {
                            float wx, wz;
                            gridToWorld(xs[k], zs[k], wx, wz);
                            float depthFactor = clampf(-cs[k]->signedSeaDist / 550.0f, 0.0f, 1.0f);
                            glColor3f(lerpf(0.10f, 0.03f, depthFactor),
                                      lerpf(0.60f, 0.22f, depthFactor),
                                      lerpf(0.76f, 0.50f, depthFactor));
                            glTexCoord2f(wx * 0.0012f, wz * 0.0012f);
                            glVertex3f(wx, 0.0f, wz);
                        }
                    }
                    glEnd();
                } else {
                    glBegin(GL_TRIANGLE_STRIP);
                    for (int ix = ix0; ix <= ix1; ix += seaStep) {
                        for (int dz = 0; dz <= seaStep; dz += seaStep) {
                            int jz = std::min(MASK_N - 1, iz + dz);
                            float wx, wz;
                            gridToWorld(ix, jz, wx, wz);
                            const WorldMaskCell& mc = maskCellAt(ix, jz);
                            float depthFactor = clampf(-mc.signedSeaDist / 550.0f, 0.0f, 1.0f);
                            glColor3f(lerpf(0.10f, 0.03f, depthFactor),
                                      lerpf(0.60f, 0.22f, depthFactor),
                                      lerpf(0.76f, 0.50f, depthFactor));
                            glTexCoord2f(wx * 0.0012f, wz * 0.0012f);
                            glVertex3f(wx, 0.0f, wz);
                        }
                    }
                    glEnd();
                }
            }
            glEndList();
        }
    }
}

static void buildTerrainAndSeaDisplayList() {
    buildTerrainDisplayList();
    buildSeaDisplayList();
}

// Kiyi sinirinda Y = 0.0m -> Y = +2.20m 3D beton rihtim duvari,
// yuruyus kordonu, cim/palmiye bandi, sahil bulvari ve 7 vapur iskelesi
static void buildQuayAndPromenadeDisplayList() {
    g_dlQuayAndPromenade = glGenLists(1);
    glNewList(g_dlQuayAndPromenade, GL_COMPILE);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);

    int n = (int)g_coastSpline.size();

    if (!g_hasGisGrid) {
        // 1. 3D Beton Rihtim Duvari (Y = -0.4m .. Y = +2.22m)
        glColor3f(0.80f, 0.81f, 0.82f);
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i < n; i++) {
            Vec2 p = g_coastSpline[i];
            Vec2 seaN = g_coastInNormals[i] * -1.0f;
            glNormal3f(seaN.x, 0.1f, seaN.z);
            glTexCoord2f(i * 0.25f, 0.0f); glVertex3f(p.x, -0.4f, p.z);
            glTexCoord2f(i * 0.25f, 1.0f); glVertex3f(p.x, 2.22f, p.z);
        }
        glEnd();

        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.0f, -1.0f);

        // 2. Yuruyus Kordonu (0..7.5m granit/beton kordon + 7.5..11.5m Kordon cim & palmiye kusagi)
        glColor3f(0.86f, 0.84f, 0.80f);
        glNormal3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i < n; i++) {
            Vec2 p0 = g_coastSpline[i];
            Vec2 p1 = p0 + g_coastInNormals[i] * 7.5f;
            glTexCoord2f(i * 0.2f, 0.0f); glVertex3f(p0.x, 2.22f, p0.z);
            glTexCoord2f(i * 0.2f, 1.0f); glVertex3f(p1.x, 2.24f, p1.z);
        }
        glEnd();

        glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GRASS]);
        glColor3f(0.30f, 0.68f, 0.32f);
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i < n; i++) {
            Vec2 p1 = g_coastSpline[i] + g_coastInNormals[i] * 7.5f;
            Vec2 p2 = g_coastSpline[i] + g_coastInNormals[i] * 11.5f;
            glTexCoord2f(i * 0.2f, 0.0f); glVertex3f(p1.x, 2.24f, p1.z);
            glTexCoord2f(i * 0.2f, 1.0f); glVertex3f(p2.x, 2.25f, p2.z);
        }
        glEnd();

        // 3. Kiyi Sahil Bulvari ve Ic Yaya Esplanadi (11.5m..44.0m)
        glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
        glColor3f(0.76f, 0.76f, 0.74f);
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i < n; i++) {
            Vec2 p2 = g_coastSpline[i] + g_coastInNormals[i] * 11.5f;
            Vec2 p3 = g_coastSpline[i] + g_coastInNormals[i] * 44.0f;
            float y2 = std::max(2.24f, sampleTerrainElev(p2.x, p2.z) + 0.14f);
            float y3 = std::max(2.24f, sampleTerrainElev(p3.x, p3.z) + 0.14f);
            glTexCoord2f(i * 0.25f, 0.0f); glVertex3f(p2.x, y2, p2.z);
            glTexCoord2f(i * 0.25f, 2.0f); glVertex3f(p3.x, y3, p3.z);
        }
        glEnd();
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    g_nightStreetLights.clear();
    for (int i = 0; i < n; i += 2) {
        Vec2 lp = g_coastSpline[i] + g_coastInNormals[i] * 5.0f;
        float ly = std::max(2.6f, sampleTerrainElev(lp.x, lp.z)) + 4.8f;
        g_nightStreetLights.emplace_back(lp.x, ly, lp.z);
    }

    // 4. 7 Gercek IZDENIZ Vapur Iskelesi (Konak, Pasaport, Alsancak, Karsiyaka, Bostanli, Goztepe, Uckuyular)
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
    for (int p = 0; p < NUM_PIERS; p++) {
        Vec2 target = geoPt(PIERS[p].lat, PIERS[p].lon);
        Vec2 base, seaDir, tang;
        if (g_hasGisGrid) {
            Vec2 shoreBase = g_pierShorePos[p];
            Vec2 seaApproach = g_pierSeaPos[p];
            seaDir = norm2(seaApproach - shoreBase);
            if (len2(seaDir) < 0.1f) seaDir = norm2(seaApproach - target);
            if (len2(seaDir) < 0.1f) seaDir = Vec2(-1.0f, 0.0f);
            tang = Vec2(-seaDir.z, seaDir.x);
            base = shoreBase + seaDir * 2.0f;
        } else {
            int bestI = 0;
            float bestD = 1e18f;
            for (int i = 0; i < n; i++) {
                float d = len2(g_coastSpline[i] - target);
                if (d < bestD) { bestD = d; bestI = i; }
            }
            base = g_coastSpline[bestI] - g_coastInNormals[bestI] * -4.0f;
            seaDir = g_coastInNormals[bestI] * -1.0f;
            tang = g_coastTangents[bestI];
            g_pierSeaPos[p] = base + seaDir * 96.0f;
        }
        Vec2 tip = base + seaDir * 94.0f;
        Vec2 pierMid = (base + tip) * 0.5f;
        float deckY = std::max(2.25f, sampleTerrainElev(base.x, base.z) + 0.25f);

        glBegin(GL_QUADS);
        // Solid anchored concrete/stone pier platform down below water level (-1.2m..deckY)
        glColor3f(0.62f, 0.63f, 0.64f);
        emitBoxFaces(pierMid.x, -1.2f, pierMid.z, 16.0f, 94.0f, deckY + 1.2f, tang.x, tang.z, 0.05f, 0.05f);
        // Iskele binasi + cam bekleme salonu + lacivert cati kanopisi
        Vec2 mid = base + seaDir * 48.0f;
        glColor3f(0.84f, 0.86f, 0.89f);
        emitBoxFaces(mid.x, deckY, mid.z, 13.0f, 20.0f, 4.6f, tang.x, tang.z, 0.08f, 0.1f);
        glColor3f(0.14f, 0.34f, 0.64f);
        emitBoxFaces(mid.x, deckY + 4.6f, mid.z, 14.5f, 22.0f, 1.2f, tang.x, tang.z, 0.08f, 0.1f);
        glEnd();
        g_nightStreetLights.emplace_back(tip.x, deckY + 3.5f, tip.z);

        float yaw = std::atan2(seaDir.x, seaDir.z);
        drawTextWorld(Vec3(mid.x, deckY + 9.2f, mid.z), yaw, 0.35f, PIERS[p].name, 0.95f, 0.98f, 1.0f);
    }

    glEndList();
}

// 28 Istasyonlu Cift Hatli IZBAN + Otoyol Aglari Display List
static void computeV2PolyElevs(const std::vector<Vec2>& pts, bool isBridge, int level,
                               float baseOff, float halfW, std::vector<float>& ys) {
    size_t m = pts.size();
    ys.resize(m);
    if (m == 0) return;
    float sampleHW = std::max(2.0f, halfW + 0.5f);
    auto groundMaxAt = [&](size_t i) -> float {
        Vec2 p = pts[i];
        Vec2 d(0.0f, 0.0f);
        if (m >= 2) {
            if (i == 0) d = pts[1] - pts[0];
            else if (i + 1 == m) d = pts[m - 1] - pts[m - 2];
            else d = pts[i + 1] - pts[i - 1];
        }
        float l = len2(d);
        float eC = sampleRenderedTerrainMaxElev(p.x, p.z);
        if (l < 1e-4f) return eC;
        Vec2 n(-d.z / l, d.x / l);
        float eL = sampleRenderedTerrainMaxElev(p.x - n.x * sampleHW, p.z - n.z * sampleHW);
        float eR = sampleRenderedTerrainMaxElev(p.x + n.x * sampleHW, p.z + n.z * sampleHW);
        return std::max(eC, std::max(eL, eR));
    };
    if (!isBridge) {
        float lvlOff = (level > 0) ? (0.44f + 3.2f * level) : baseOff;
        for (size_t i = 0; i < m; i++) {
            ys[i] = groundMaxAt(i) + lvlOff;
        }
        for (int pass = 0; pass < 2; pass++) {
            for (size_t i = 1; i + 1 < m; i++) {
                ys[i] = std::max(ys[i], 0.5f * (ys[i - 1] + ys[i + 1]) - 0.10f);
            }
        }
        return;
    }
    float y0 = groundMaxAt(0) + 0.42f;
    float y1 = groundMaxAt(m - 1) + 0.42f;
    float arch = (level > 0) ? 6.2f : 1.8f;
    for (size_t i = 0; i < m; i++) {
        float t = (m > 1) ? ((float)i / (float)(m - 1)) : 0.0f;
        float yc = lerpf(y0, y1, t) + std::sin(t * PI_F) * arch;
        float te = groundMaxAt(i) + 0.52f;
        ys[i] = std::max(yc, te);
    }
}

static void ensureV2MajorRoadElevs() {
    for (auto& rp : g_v2MajorRoads) {
        if (rp.pts.size() >= 2 && rp.ys.size() != rp.pts.size()) {
            computeV2PolyElevs(rp.pts, rp.isBridge != 0, rp.level, 0.38f, rp.halfW, rp.ys);
        }
    }
}

static void buildIzbanAndRoadsDisplayList() {
    if (g_hasV2Roads) {
        ensureV2MajorRoadElevs();

        // 1. V2 Ana Yollari 8x8 spatial tile'lara dagit (culling ile yuksek FPS) + gece sokak lambalari topla
        std::vector<std::vector<size_t>> tileRoadIdx(NUM_WORLD_TILES);
        const float tileW = (WORLD_X1 - WORLD_X0) / (float)WORLD_TILE_N;
        const float tileH = (WORLD_Z1 - WORLD_Z0) / (float)WORLD_TILE_N;
        for (size_t ri = 0; ri < g_v2MajorRoads.size(); ri++) {
            const auto& rp = g_v2MajorRoads[ri];
            if (rp.pts.size() < 2 || rp.ys.size() != rp.pts.size()) continue;
            Vec2 mid = rp.pts[rp.pts.size() / 2];
            int tx = clampi((int)((mid.x - WORLD_X0) / tileW), 0, WORLD_TILE_N - 1);
            int tz = clampi((int)((mid.z - WORLD_Z0) / tileH), 0, WORLD_TILE_N - 1);
            int tidx = tz * WORLD_TILE_N + tx;
            tileRoadIdx[tidx].push_back(ri);
            float tcx = WORLD_X0 + (tx + 0.5f) * tileW;
            float tcz = WORLD_Z0 + (tz + 0.5f) * tileH;
            for (const auto& p : rp.pts) {
                float d = std::sqrt((p.x - tcx) * (p.x - tcx) + (p.z - tcz) * (p.z - tcz));
                if (d + rp.halfW > g_roadTileRadius[tidx]) g_roadTileRadius[tidx] = d + rp.halfW;
            }
            if (rp.tier <= 1) {
                for (size_t k = 0; k < rp.pts.size(); k += 3) {
                    g_nightStreetLights.emplace_back(rp.pts[k].x, rp.ys[k] + 4.5f, rp.pts[k].z);
                }
            }
        }

        for (int tidx = 0; tidx < NUM_WORLD_TILES; tidx++) {
            if (tileRoadIdx[tidx].empty()) {
                g_dlRoadTile[tidx] = 0;
                continue;
            }
            if (g_roadTileRadius[tidx] < 3600.0f) g_roadTileRadius[tidx] = 3600.0f;
            g_dlRoadTile[tidx] = glGenLists(1);
            glNewList(g_dlRoadTile[tidx], GL_COMPILE);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(-2.5f, -2.5f);

            // 1A. Viyaduk ve Kopru Beton Ayaklari + Kenar Parapetleri
            glDisable(GL_TEXTURE_2D);
            glBegin(GL_QUADS);
            for (size_t ri : tileRoadIdx[tidx]) {
                const auto& rp = g_v2MajorRoads[ri];
                if (!rp.isBridge) continue;
                const std::vector<float>& ys = rp.ys;
                float hw = rp.halfW + 0.8f;
                for (size_t i = 0; i + 1 < rp.pts.size(); i++) {
                    Vec2 d = rp.pts[i + 1] - rp.pts[i];
                    float l = len2(d);
                    if (l < 0.5f) continue;
                    Vec2 n(-d.z / l, d.x / l);
                    Vec2 p0 = rp.pts[i], p1 = rp.pts[i + 1];
                    float yA = ys[i], yB = ys[i + 1];
                    glColor3f(0.76f, 0.77f, 0.79f);
                    for (int s = -1; s <= 1; s += 2) {
                        Vec2 eA = p0 + n * (s * hw), eB = p1 + n * (s * hw);
                        glNormal3f(n.x * s, 0.0f, n.z * s);
                        glVertex3f(eA.x, yA - 1.6f, eA.z);
                        glVertex3f(eB.x, yB - 1.6f, eB.z);
                        glVertex3f(eB.x, yB + 0.65f, eB.z);
                        glVertex3f(eA.x, yA + 0.65f, eA.z);
                    }
                    if ((i % 3 == 0)) {
                        Vec2 mid = (p0 + p1) * 0.5f;
                        float my = 0.5f * (yA + yB);
                        float ty = sampleTerrainElev(mid.x, mid.z);
                        if (my - ty > 2.2f) {
                            glColor3f(0.70f, 0.71f, 0.73f);
                            emitBoxFaces(mid.x, ty - 1.0f, mid.z, std::min(3.2f, rp.halfW * 0.65f), 2.4f, (my - ty) + 0.4f, d.x / l, d.z / l, 0.1f, 0.1f, false);
                        }
                    }
                }
            }
            glEnd();

            // 1B. Gercek Otoyollar, Cevreyolu, Bulvarlar ve Ana Arterler (Asfalt)
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ASPHALT]);
            glBegin(GL_QUADS);
            for (size_t ri : tileRoadIdx[tidx]) {
                const auto& rp = g_v2MajorRoads[ri];
                const std::vector<float>& ys = rp.ys;
                if (rp.tier == 0) glColor3f(0.24f, 0.26f, 0.29f);
                else if (rp.tier == 1) glColor3f(0.32f, 0.34f, 0.38f);
                else glColor3f(0.42f, 0.44f, 0.48f);

                for (size_t i = 0; i + 1 < rp.pts.size(); i++) {
                    Vec2 p0 = rp.pts[i], p1 = rp.pts[i + 1];
                    Vec2 d = p1 - p0;
                    float l = len2(d);
                    if (l < 0.5f) continue;
                    Vec2 n(-d.z / l, d.x / l);
                    Vec2 aL = p0 - n * rp.halfW, aR = p0 + n * rp.halfW;
                    Vec2 bL = p1 - n * rp.halfW, bR = p1 + n * rp.halfW;
                    glNormal3f(0.0f, 1.0f, 0.0f);
                    glTexCoord2f(0.0f, 0.0f); glVertex3f(aL.x, ys[i], aL.z);
                    glTexCoord2f(0.0f, l * 0.08f); glVertex3f(bL.x, ys[i + 1], bL.z);
                    glTexCoord2f(1.0f, l * 0.08f); glVertex3f(bR.x, ys[i + 1], bR.z);
                    glTexCoord2f(1.0f, 0.0f); glVertex3f(aR.x, ys[i], aR.z);
                }
            }
            glEnd();

            // 1B-2. Yuzey yollari icin beton/toprak dolgu kenar etekleri
            glDisable(GL_TEXTURE_2D);
            glColor3f(0.34f, 0.33f, 0.31f);
            glBegin(GL_QUADS);
            for (size_t ri : tileRoadIdx[tidx]) {
                const auto& rp = g_v2MajorRoads[ri];
                if (rp.isBridge) continue;
                const std::vector<float>& ys = rp.ys;
                for (size_t i = 0; i + 1 < rp.pts.size(); i++) {
                    Vec2 p0 = rp.pts[i], p1 = rp.pts[i + 1];
                    Vec2 d = p1 - p0;
                    float l = len2(d);
                    if (l < 0.5f) continue;
                    Vec2 n(-d.z / l, d.x / l);
                    Vec2 aL = p0 - n * rp.halfW, aR = p0 + n * rp.halfW;
                    Vec2 bL = p1 - n * rp.halfW, bR = p1 + n * rp.halfW;
                    glNormal3f(-n.x, 0.25f, -n.z);
                    glVertex3f(aL.x, ys[i], aL.z);
                    glVertex3f(aL.x, ys[i] - 2.5f, aL.z);
                    glVertex3f(bL.x, ys[i + 1] - 2.5f, bL.z);
                    glVertex3f(bL.x, ys[i + 1], bL.z);

                    glNormal3f(n.x, 0.25f, n.z);
                    glVertex3f(bR.x, ys[i + 1], bR.z);
                    glVertex3f(bR.x, ys[i + 1] - 2.5f, bR.z);
                    glVertex3f(aR.x, ys[i] - 2.5f, aR.z);
                    glVertex3f(aR.x, ys[i], aR.z);
                }
            }
            glEnd();

            // 1C. Otoyol/Bulvar Sari Orta Refuj ve Beyaz Kenar Cizgileri (tier 0 ve 1)
            glBegin(GL_QUADS);
            for (size_t ri : tileRoadIdx[tidx]) {
                const auto& rp = g_v2MajorRoads[ri];
                if (rp.tier > 1) continue;
                const std::vector<float>& ys = rp.ys;
                for (size_t i = 0; i + 1 < rp.pts.size(); i++) {
                    Vec2 p0 = rp.pts[i], p1 = rp.pts[i + 1];
                    Vec2 d = p1 - p0;
                    float l = len2(d);
                    if (l < 1.0f) continue;
                    Vec2 n(-d.z / l, d.x / l);
                    glColor3f(0.95f, 0.82f, 0.22f);
                    glNormal3f(0.0f, 1.0f, 0.0f);
                    glVertex3f(p0.x - n.x * 0.35f, ys[i] + 0.06f, p0.z - n.z * 0.35f);
                    glVertex3f(p1.x - n.x * 0.35f, ys[i + 1] + 0.06f, p1.z - n.z * 0.35f);
                    glVertex3f(p1.x + n.x * 0.35f, ys[i + 1] + 0.06f, p1.z + n.z * 0.35f);
                    glVertex3f(p0.x + n.x * 0.35f, ys[i] + 0.06f, p0.z + n.z * 0.35f);
                }
            }
            glEnd();

            glDisable(GL_POLYGON_OFFSET_FILL);
            glEndList();
        }
    }

    // 2. Rayli Sistemler + 28 IZBAN Istasyonu (+ sentetik mod yollari)
    g_dlRails = glGenLists(1);
    glNewList(g_dlRails, GL_COMPILE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-2.5f, -2.5f);

    if (g_hasV2Roads) {
        // Gercek Rayli Sistemler (IZBAN standart hat + Izmir Tramvay + M1 Yuzey Metro; tuneller haric)
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_tex[TEX_BALLAST]);
        glBegin(GL_QUADS);
        for (const auto& rs : g_v2Rails) {
            if (rs.pts.size() < 2) continue;
            float hw = (rs.rtype == 0) ? 3.8f : 2.5f;
            std::vector<float> ys;
            computeV2PolyElevs(rs.pts, rs.isBridge != 0, rs.isBridge ? 1 : 0, 0.38f, hw, ys);
            if (rs.rtype == 1) glColor3f(0.45f, 0.62f, 0.42f);
            else glColor3f(0.74f, 0.72f, 0.69f);

            for (size_t i = 0; i + 1 < rs.pts.size(); i++) {
                Vec2 p0 = rs.pts[i], p1 = rs.pts[i + 1];
                Vec2 d = p1 - p0;
                float l = len2(d);
                if (l < 0.5f) continue;
                Vec2 n(-d.z / l, d.x / l);
                Vec2 aL = p0 - n * hw, aR = p0 + n * hw;
                Vec2 bL = p1 - n * hw, bR = p1 + n * hw;
                glNormal3f(0.0f, 1.0f, 0.0f);
                glTexCoord2f(0.0f, 0.0f); glVertex3f(aL.x, ys[i], aL.z);
                glTexCoord2f(0.0f, l * 0.14f); glVertex3f(bL.x, ys[i + 1], bL.z);
                glTexCoord2f(1.0f, l * 0.14f); glVertex3f(bR.x, ys[i + 1], bR.z);
                glTexCoord2f(1.0f, 0.0f); glVertex3f(aR.x, ys[i], aR.z);
            }
        }
        glEnd();
        glDisable(GL_POLYGON_OFFSET_FILL);

        glDisable(GL_TEXTURE_2D);
        glColor3f(0.86f, 0.90f, 0.95f);
        glBegin(GL_QUADS);
        for (const auto& rs : g_v2Rails) {
            if (rs.pts.size() < 2) continue;
            float hw = (rs.rtype == 0) ? 3.8f : 2.5f;
            std::vector<float> ys;
            computeV2PolyElevs(rs.pts, rs.isBridge != 0, rs.isBridge ? 1 : 0, 0.38f, hw, ys);
            for (size_t i = 0; i + 1 < rs.pts.size(); i++) {
                Vec2 p0 = rs.pts[i], p1 = rs.pts[i + 1];
                Vec2 d = p1 - p0;
                float l = len2(d);
                if (l < 0.5f) continue;
                Vec2 n(-d.z / l, d.x / l);
                for (int s = -1; s <= 1; s += 2) {
                    float ro = s * 0.72f;
                    Vec2 cA = p0 + n * ro, cB = p1 + n * ro;
                    glVertex3f(cA.x - n.x * 0.11f, ys[i] + 0.14f, cA.z - n.z * 0.11f);
                    glVertex3f(cB.x - n.x * 0.11f, ys[i + 1] + 0.14f, cB.z - n.z * 0.11f);
                    glVertex3f(cB.x + n.x * 0.11f, ys[i + 1] + 0.14f, cB.z + n.z * 0.11f);
                    glVertex3f(cA.x + n.x * 0.11f, ys[i] + 0.14f, cA.z + n.z * 0.11f);
                }
            }
        }
        glEnd();
    } else {
        // 1. Sentetik Otoyollar ve Ana Bulvarlar (binary veri yoksa)
        for (const auto& rs : g_roads) {
            glDisable(GL_TEXTURE_2D);
            glColor3f(0.58f, 0.57f, 0.54f);
            for (int side = -1; side <= 1; side += 2) {
                glBegin(GL_QUAD_STRIP);
                for (size_t i = 0; i < rs.pts.size(); i++) {
                    Vec3 p = rs.pts[i];
                    Vec2 t = rs.tangs[i];
                    Vec2 n(-t.z, t.x);
                    Vec2 edgeTop = Vec2(p.x, p.z) + n * (side * rs.halfW);
                    Vec2 edgeBot = Vec2(p.x, p.z) + n * (side * (rs.halfW + 6.5f));
                    glNormal3f(n.x * side, 0.45f, n.z * side);
                    glVertex3f(edgeTop.x, p.y, edgeTop.z);
                    glVertex3f(edgeBot.x, p.y - 8.5f, edgeBot.z);
                }
                glEnd();
            }

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ASPHALT]);
            glColor3f(0.24f, 0.26f, 0.29f);
            glNormal3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_QUAD_STRIP);
            for (size_t i = 0; i < rs.pts.size(); i++) {
                Vec3 p = rs.pts[i];
                Vec2 t = rs.tangs[i];
                Vec2 n(-t.z, t.x);
                Vec2 pL = Vec2(p.x, p.z) - n * rs.halfW;
                Vec2 pR = Vec2(p.x, p.z) + n * rs.halfW;
                glTexCoord2f(0.0f, rs.cumDist[i] * 0.04f); glVertex3f(pL.x, p.y, pL.z);
                glTexCoord2f(1.0f, rs.cumDist[i] * 0.04f); glVertex3f(pR.x, p.y, pR.z);
            }
            glEnd();

            glDisable(GL_TEXTURE_2D);
            glColor3f(0.95f, 0.82f, 0.22f);
            glBegin(GL_QUAD_STRIP);
            for (size_t i = 0; i < rs.pts.size(); i++) {
                Vec3 p = rs.pts[i];
                Vec2 t = rs.tangs[i];
                Vec2 n(-t.z, t.x);
                glVertex3f(p.x - n.x * 0.50f, p.y + 0.10f, p.z - n.z * 0.50f);
                glVertex3f(p.x + n.x * 0.50f, p.y + 0.10f, p.z + n.z * 0.50f);
            }
            glEnd();

            glColor3f(0.92f, 0.93f, 0.95f);
            for (int side = -1; side <= 1; side += 2) {
                glBegin(GL_QUAD_STRIP);
                for (size_t i = 0; i < rs.pts.size(); i++) {
                    Vec3 p = rs.pts[i];
                    Vec2 t = rs.tangs[i];
                    Vec2 n(-t.z, t.x);
                    float off = side * (rs.halfW - 0.65f);
                    glVertex3f(p.x + n.x * (off - 0.20f), p.y + 0.09f, p.z + n.z * (off - 0.20f));
                    glVertex3f(p.x + n.x * (off + 0.20f), p.y + 0.09f, p.z + n.z * (off + 0.20f));
                }
                glEnd();
            }
        }

        glDisable(GL_TEXTURE_2D);
        glColor3f(0.56f, 0.54f, 0.50f);
        for (int side = -1; side <= 1; side += 2) {
            glBegin(GL_QUAD_STRIP);
            for (size_t i = 0; i < g_izbanSpline.size(); i++) {
                Vec3 p = g_izbanSpline[i];
                Vec2 t = g_izbanTangents[i];
                Vec2 n(-t.z, t.x);
                Vec2 topP = Vec2(p.x, p.z) + n * (side * 6.2f);
                Vec2 botP = Vec2(p.x, p.z) + n * (side * 9.8f);
                glNormal3f(n.x * side, 0.45f, n.z * side);
                glVertex3f(topP.x, p.y + 0.14f, topP.z);
                glVertex3f(botP.x, p.y - 6.0f, botP.z);
            }
            glEnd();
        }

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_tex[TEX_BALLAST]);
        glColor3f(0.75f, 0.73f, 0.70f);
        glNormal3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_QUAD_STRIP);
        for (size_t i = 0; i < g_izbanSpline.size(); i++) {
            Vec3 p = g_izbanSpline[i];
            Vec2 t = g_izbanTangents[i];
            Vec2 n(-t.z, t.x);
            Vec2 bL = Vec2(p.x, p.z) - n * 6.2f;
            Vec2 bR = Vec2(p.x, p.z) + n * 6.2f;
            glTexCoord2f(0.0f, g_izbanCumDist[i] * 0.12f); glVertex3f(bL.x, p.y + 0.15f, bL.z);
            glTexCoord2f(1.0f, g_izbanCumDist[i] * 0.12f); glVertex3f(bR.x, p.y + 0.15f, bR.z);
        }
        glEnd();
        glDisable(GL_POLYGON_OFFSET_FILL);

        glDisable(GL_TEXTURE_2D);
        glColor3f(0.86f, 0.89f, 0.94f);
        const float railOffsets[4] = {-2.92f, -1.48f, 1.48f, 2.92f};
        for (int r = 0; r < 4; r++) {
            float ro = railOffsets[r];
            glBegin(GL_QUAD_STRIP);
            for (size_t i = 0; i < g_izbanSpline.size(); i++) {
                Vec3 p = g_izbanSpline[i];
                Vec2 t = g_izbanTangents[i];
                Vec2 n(-t.z, t.x);
                Vec2 rc = Vec2(p.x, p.z) + n * ro;
                glVertex3f(rc.x - n.x * 0.11f, p.y + 0.32f, rc.z - n.z * 0.11f);
                glVertex3f(rc.x + n.x * 0.11f, p.y + 0.32f, rc.z + n.z * 0.11f);
            }
            glEnd();
        }

        glColor3f(0.55f, 0.60f, 0.66f);
        glBegin(GL_QUADS);
        for (size_t i = 0; i < g_izbanSpline.size(); i += 3) {
            Vec3 p = g_izbanSpline[i];
            Vec2 t = g_izbanTangents[i];
            Vec2 n(-t.z, t.x);
            for (int s = -1; s <= 1; s += 2) {
                Vec2 mp = Vec2(p.x, p.z) + n * (s * 5.4f);
                emitBoxFaces(mp.x, p.y, mp.z, 0.35f, 0.35f, 7.2f, t.x, t.z, 1.0f, 1.0f);
            }
            emitBoxFaces(p.x, p.y + 6.9f, p.z, 11.0f, 0.28f, 0.28f, t.x, t.z, 1.0f, 1.0f);
        }
        glEnd();
    }

    // 3. 28 Gercek IZBAN Istasyonu: 160m Peronlar, Camli Gar Binasi, Ust Gecit ve Cift Yonlu 3D Tabela
    for (int s = 0; s < NUM_STATIONS; s++) {
        Vec3 sp = g_stationPos[s];
        Vec2 tang = g_stationTang[s];
        Vec2 normV(-tang.z, tang.x);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
        glBegin(GL_QUADS);
        for (int side = -1; side <= 1; side += 2) {
            Vec2 pc = Vec2(sp.x, sp.z) + normV * (side * 8.6f);
            // 160m uzunlugunda cift 3D yolcu peronu
            glColor3f(0.82f, 0.83f, 0.85f);
            emitBoxFaces(pc.x, sp.y, pc.z, 4.8f, 160.0f, 1.15f, normV.x, normV.z, 0.1f, 0.1f);
            // Kavisli mavi peron kanopisi
            glColor3f(0.12f, 0.38f, 0.78f);
            emitBoxFaces(pc.x, sp.y + 4.6f, pc.z, 5.2f, 115.0f, 0.40f, normV.x, normV.z, 0.1f, 0.1f);
            // Sari guvenlik seridi
            glColor3f(1.0f, 0.88f, 0.12f);
            emitBoxFaces(pc.x - normV.x * (side * 2.15f), sp.y + 1.16f, pc.z - normV.z * (side * 2.15f),
                         0.35f, 160.0f, 0.06f, normV.x, normV.z, 0.1f, 0.1f);
        }
        // Yaya ust gecidi + merdiven ve asansor kuleleri
        glColor3f(0.88f, 0.90f, 0.94f);
        emitBoxFaces(sp.x, sp.y + 7.8f, sp.z, 24.0f, 5.2f, 3.6f, normV.x, normV.z, 0.1f, 0.1f);
        for (int side = -1; side <= 1; side += 2) {
            Vec2 tc = Vec2(sp.x, sp.z) + normV * (side * 11.2f);
            emitBoxFaces(tc.x, sp.y, tc.z, 4.5f, 6.5f, 11.2f, normV.x, normV.z, 0.1f, 0.1f);
        }
        // Modern 2-katli camli IZBAN Istasyon Gar Binasi (bilet holu & turnikeler)
        Vec2 termC = Vec2(sp.x, sp.z) + normV * 19.5f;
        glColor3f(0.94f, 0.95f, 0.98f);
        emitBoxFaces(termC.x, sp.y, termC.z, 16.0f, 44.0f, 11.5f, normV.x, normV.z, 0.1f, 0.1f);
        glColor3f(0.14f, 0.42f, 0.82f);
        emitBoxFaces(termC.x, sp.y + 11.5f, termC.z, 17.0f, 46.0f, 1.2f, normV.x, normV.z, 0.1f, 0.1f);
        glEnd();

        // 3D Istasyon Isim Tabelasi (cift yonlu dev levha)
        float yaw = std::atan2(tang.x, tang.z);
        char stTitle[64];
        std::snprintf(stTitle, sizeof(stTitle), "[%d/28] %s", s + 1, STATIONS[s].name);
        drawTextWorld(Vec3(sp.x, sp.y + 14.8f, sp.z), yaw, 0.82f, stTitle, 1.0f, 0.96f, 0.40f);
        drawTextWorld(Vec3(sp.x, sp.y + 14.8f, sp.z), yaw + PI_F, 0.82f, stTitle, 1.0f, 0.96f, 0.40f);
    }

    glEndList();
}
struct LandmarkGlowTower {
    float cx, baseY, cz, baseW, baseD, height, twistDeg;
    float r, g, b;
};
static std::vector<LandmarkGlowTower> g_landmarkGlowTowers;

// Burgu (twist) yapan lofted kule yardimcisi (Folkart Towers twist = 45 derece, Mistral 216m)
static void drawLoftedTwistedTower(float cx, float baseY, float cz, float baseW, float baseD,
                                   float height, float totalTwistDeg, int slices,
                                   float rCol, float gCol, float bCol) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GLASS_TOWER]);
    for (int s = 0; s < slices; s++) {
        float t0 = (float)s / (float)slices;
        float t1 = (float)(s + 1) / (float)slices;
        float y0 = baseY + t0 * height;
        float y1 = baseY + t1 * height;
        float a0 = t0 * totalTwistDeg * DEG2RAD;
        float a1 = t1 * totalTwistDeg * DEG2RAD;
        float scale0 = 1.0f - 0.14f * t0;
        float scale1 = 1.0f - 0.14f * t1;
        float hw0 = baseW * 0.5f * scale0, hd0 = baseD * 0.5f * scale0;
        float hw1 = baseW * 0.5f * scale1, hd1 = baseD * 0.5f * scale1;

        auto corner = [&](float hw, float hd, float ang, int idx) {
            float lx = (idx == 0 || idx == 3) ? -hw : hw;
            float lz = (idx == 0 || idx == 1) ? -hd : hd;
            float ca = std::cos(ang), sa = std::sin(ang);
            return Vec3(cx + lx * ca - lz * sa, 0.0f, cz + lx * sa + lz * ca);
        };
        glColor3f(rCol, gCol, bCol);
        glBegin(GL_QUADS);
        for (int side = 0; side < 4; side++) {
            int next = (side + 1) & 3;
            Vec3 p00 = corner(hw0, hd0, a0, side); p00.y = y0;
            Vec3 p10 = corner(hw0, hd0, a0, next); p10.y = y0;
            Vec3 p11 = corner(hw1, hd1, a1, next); p11.y = y1;
            Vec3 p01 = corner(hw1, hd1, a1, side); p01.y = y1;
            Vec3 n = norm(cross(p01 - p00, p10 - p00)); // Disa donuk yuzey normali
            glNormal3f(n.x, n.y, n.z);
            glTexCoord2f(0.0f, t0 * 8.0f); glVertex3f(p00.x, p00.y, p00.z);
            glTexCoord2f(2.0f, t0 * 8.0f); glVertex3f(p10.x, p10.y, p10.z);
            glTexCoord2f(2.0f, t1 * 8.0f); glVertex3f(p11.x, p11.y, p11.z);
            glTexCoord2f(0.0f, t1 * 8.0f); glVertex3f(p01.x, p01.y, p01.z);
        }
        glEnd();

        // Mimari kat kusaklari
        glDisable(GL_TEXTURE_2D);
        glColor3f(0.92f, 0.95f, 0.98f);
        glBegin(GL_LINE_LOOP);
        for (int c = 0; c < 4; c++) {
            Vec3 pc = corner(hw1 * 1.01f, hd1 * 1.01f, a1, c);
            glVertex3f(pc.x, y1, pc.z);
        }
        glEnd();
        glEnable(GL_TEXTURE_2D);
    }
    // Cam tac parapet + celik zirve anteni + havacilik ikaz feneri
    float topScale = 0.86f;
    float topY = baseY + height;
    glDisable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glColor3f(0.88f, 0.92f, 0.97f);
    emitBoxFaces(cx, topY, cz, baseW * topScale, baseD * topScale, 2.2f,
                 std::cos(totalTwistDeg * DEG2RAD), std::sin(totalTwistDeg * DEG2RAD), 1, 1);
    glColor3f(0.75f, 0.78f, 0.84f);
    emitBoxFaces(cx, topY + 2.2f, cz, 1.4f, 1.4f, 12.0f, 1, 0, 1, 1);
    glEnd();
    glEnable(GL_TEXTURE_2D);
    g_nightBeaconLights.emplace_back(cx, topY + 14.5f, cz);
    g_landmarkGlowTowers.push_back({cx, baseY, cz, baseW * 1.005f, baseD * 1.005f, height, totalTwistDeg, rCol, gCol, bCol});
}

// Pist + glPolygonOffset esik cizgileri + PAPI + ALS isiklari yardimcisi
static void drawRunwayWithMarkings(Vec2 center, float elev, float hdgDeg, float length, float width, bool isDualLabel, int rwySide) {
    Vec2 dir = headingDir(hdgDeg);
    Vec2 right(-dir.z, dir.x);
    float hl = length * 0.5f, hw = width * 0.5f;

    // Ana asfalt pist yuzeyi
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ASPHALT]);
    glColor3f(0.24f, 0.25f, 0.28f);
    glNormal3f(0.0f, 1.0f, 0.0f);
    Vec2 c0 = center - dir * hl - right * hw;
    Vec2 c1 = center - dir * hl + right * hw;
    Vec2 c2 = center + dir * hl + right * hw;
    Vec2 c3 = center + dir * hl - right * hw;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);   glVertex3f(c0.x, elev + 0.12f, c0.z);
    glTexCoord2f(2.0f, 0.0f);   glVertex3f(c1.x, elev + 0.12f, c1.z);
    glTexCoord2f(2.0f, 40.0f);  glVertex3f(c2.x, elev + 0.12f, c2.z);
    glTexCoord2f(0.0f, 40.0f);  glVertex3f(c3.x, elev + 0.12f, c3.z);
    glEnd();

    // glPolygonOffset ile Z-fighting olmadan pist esik cizgileri (threshold piano keys),
    // merkez kesikli cizgiler ve touchdown zone isaretleri
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-2.0f, -2.0f);
    glColor3f(0.95f, 0.95f, 0.95f);
    glBegin(GL_QUADS);
    float yMark = elev + 0.16f;
    // Esik (Threshold) zebra cubuklari (her iki ucta 10'ar serit)
    for (int endSign = -1; endSign <= 1; endSign += 2) {
        Vec2 endBase = center + dir * (endSign * (hl - 32.0f));
        for (int b = -5; b <= 5; b++) {
            if (b == 0) continue;
            Vec2 bc = endBase + right * (b * 3.4f);
            Vec2 p0 = bc - dir * 18.0f - right * 1.1f;
            Vec2 p1 = bc - dir * 18.0f + right * 1.1f;
            Vec2 p2 = bc + dir * 18.0f + right * 1.1f;
            Vec2 p3 = bc + dir * 18.0f - right * 1.1f;
            glVertex3f(p0.x, yMark, p0.z); glVertex3f(p1.x, yMark, p1.z);
            glVertex3f(p2.x, yMark, p2.z); glVertex3f(p3.x, yMark, p3.z);
        }
        // Touchdown hedef bloklari
        Vec2 tdBase = center + dir * (endSign * (hl - 280.0f));
        for (int s = -1; s <= 1; s += 2) {
            Vec2 tc = tdBase + right * (s * 9.5f);
            Vec2 p0 = tc - dir * 24.0f - right * 2.2f;
            Vec2 p1 = tc - dir * 24.0f + right * 2.2f;
            Vec2 p2 = tc + dir * 24.0f + right * 2.2f;
            Vec2 p3 = tc + dir * 24.0f - right * 2.2f;
            glVertex3f(p0.x, yMark, p0.z); glVertex3f(p1.x, yMark, p1.z);
            glVertex3f(p2.x, yMark, p2.z); glVertex3f(p3.x, yMark, p3.z);
        }
    }
    // Merkez kesikli cizgiler (her 60 metrede 30m beyaz serit)
    for (float s = -hl + 110.0f; s < hl - 110.0f; s += 60.0f) {
        Vec2 mc = center + dir * s;
        Vec2 p0 = mc - dir * 15.0f - right * 0.6f;
        Vec2 p1 = mc - dir * 15.0f + right * 0.6f;
        Vec2 p2 = mc + dir * 15.0f + right * 0.6f;
        Vec2 p3 = mc + dir * 15.0f - right * 0.6f;
        glVertex3f(p0.x, yMark, p0.z); glVertex3f(p1.x, yMark, p1.z);
        glVertex3f(p2.x, yMark, p2.z); glVertex3f(p3.x, yMark, p3.z);
    }
    glEnd();
    glDisable(GL_POLYGON_OFFSET_FILL);

    // PAPI (4'lu Kirmizi/Beyaz yaklasma egim isiklari) ve ALS (Approach Lighting System)
    glDisable(GL_LIGHTING);
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (int endSign = -1; endSign <= 1; endSign += 2) {
        // PAPI 4'lu bar (2 beyaz, 2 kirmizi)
        Vec2 papiBase = center + dir * (endSign * (hl - 310.0f)) - right * (endSign * (hw + 12.0f));
        for (int k = 0; k < 4; k++) {
            if (k < 2) glColor3f(1.0f, 0.98f, 0.92f);
            else glColor3f(1.0f, 0.15f, 0.12f);
            Vec2 lp = papiBase - right * (endSign * k * 6.0f);
            glVertex3f(lp.x, elev + 1.2f, lp.z);
        }
        // ALS yaklasma isik dizisi (pist basindan disari 450m boyunca)
        for (int a = 1; a <= 15; a++) {
            Vec2 ac = center + dir * (endSign * (hl + a * 30.0f));
            glColor3f(1.0f, 0.96f, 0.80f);
            for (int w = -2; w <= 2; w++) {
                Vec2 p = ac + right * (w * 3.0f);
                glVertex3f(p.x, elev + 0.8f, p.z);
            }
            if (a == 5 || a == 10) {
                glColor3f(0.15f, 1.0f, 0.35f);
                for (int w = -6; w <= 6; w++) {
                    Vec2 p = ac + right * (w * 2.5f);
                    glVertex3f(p.x, elev + 0.8f, p.z);
                }
            }
        }
    }
    // Pist kenar isiklari
    for (float s = -hl; s <= hl; s += 50.0f) {
        for (int side = -1; side <= 1; side += 2) {
            Vec2 ep = center + dir * s + right * (side * (hw + 1.2f));
            glColor3f(1.0f, 0.95f, 0.78f);
            glVertex3f(ep.x, elev + 0.65f, ep.z);
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);

    // Pist baslarina 3D pist numarasi (16R/34L, 16L/34R veya 17/35)
    const char* lblNorth = isDualLabel ? (rwySide < 0 ? "16R" : "16L") : "17";
    const char* lblSouth = isDualLabel ? (rwySide < 0 ? "34L" : "34R") : "35";
    Vec2 nPos = center - dir * (hl - 90.0f);
    Vec2 sPos = center + dir * (hl - 90.0f);
    float yawS = std::atan2(-dir.x, -dir.z);
    float yawN = std::atan2(dir.x, dir.z);
    drawTextWorld(Vec3(sPos.x, elev + 2.8f, sPos.z), yawS, 0.65f, lblSouth, 1.0f, 0.92f, 0.35f);
    drawTextWorld(Vec3(nPos.x, elev + 2.8f, nPos.z), yawN, 0.65f, lblNorth, 1.0f, 0.92f, 0.35f);
}

static void buildLandmarksAndAirportsDisplayList() {
    g_nightBeaconLights.clear();
    g_landmarkGlowTowers.clear();
    g_dlLandmarksAndAirports = glGenLists(1);
    glNewList(g_dlLandmarksAndAirports, GL_COMPILE);

    // 1. BAYRAKLI GOKDELENLER BOLGESI:
    //    - Ikiz Folkart Towers (38.4542, 27.1755): 200m yukseklik, twist = 45 derece burgu
    Vec2 fk = geoPt(FOLKART_LAT, FOLKART_LON);
    float fkY = sampleTerrainElev(fk.x, fk.z);
    drawLoftedTwistedTower(fk.x - 42.0f, fkY, fk.z, 32.0f, 32.0f, 200.0f, 45.0f, 28, 0.55f, 0.80f, 0.98f);
    drawLoftedTwistedTower(fk.x + 42.0f, fkY, fk.z, 32.0f, 32.0f, 200.0f, -45.0f, 28, 0.55f, 0.80f, 0.98f);
    drawTextWorld(Vec3(fk.x, fkY + 208.0f, fk.z), 0.0f, 0.75f, "FOLKART TOWERS 200M", 0.55f, 0.92f, 1.0f);

    //    - Mistral Izmir (38.4495, 27.1795, 216m)
    Vec2 ms = geoPt(MISTRAL_LAT, MISTRAL_LON);
    float msY = sampleTerrainElev(ms.x, ms.z);
    drawLoftedTwistedTower(ms.x, msY, ms.z, 34.0f, 28.0f, 216.0f, 22.0f, 26, 0.68f, 0.85f, 0.96f);
    drawLoftedTwistedTower(ms.x + 46.0f, msY, ms.z + 18.0f, 28.0f, 26.0f, 154.0f, -18.0f, 20, 0.72f, 0.88f, 0.98f);
    drawTextWorld(Vec3(ms.x, msY + 224.0f, ms.z), 0.0f, 0.75f, "MISTRAL IZMIR 216M", 0.75f, 0.95f, 1.0f);

    //    - Ege Perla (38.4455, 27.1715)
    Vec2 ep = geoPt(EGEPERLA_LAT, EGEPERLA_LON);
    float epY = sampleTerrainElev(ep.x, ep.z);
    drawLoftedTwistedTower(ep.x - 28.0f, epY, ep.z, 28.0f, 24.0f, 186.0f, 12.0f, 20, 0.82f, 0.86f, 0.92f);
    drawLoftedTwistedTower(ep.x + 28.0f, epY, ep.z, 26.0f, 22.0f, 142.0f, -12.0f, 18, 0.82f, 0.86f, 0.92f);
    drawTextWorld(Vec3(ep.x, epY + 194.0f, ep.z), 0.0f, 0.65f, "EGE PERLA", 0.95f, 0.90f, 0.75f);

    // 2. KONAK MEYDANI (38.41885, 27.12868): Sekizgen Kaideli Detayli Izmir Saat Kulesi, Yali Camii ve Konak Pier
    Vec2 sk = geoPt(SAAT_LAT, SAAT_LON);
    float skY = sampleTerrainElev(sk.x, sk.z);
    glDisable(GL_TEXTURE_2D);
    // Genis Konak Meydani traverten/mermer zemin plakasi
    glColor3f(0.86f, 0.84f, 0.79f);
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.0f, 1.0f, 0.0f);
    glVertex3f(sk.x + 15.0f, skY + 0.16f, sk.z);
    for (int i = 0; i <= 16; i++) {
        float a = i * (2.0f * PI_F / 16.0f);
        float px = sk.x + 15.0f + std::cos(a) * 88.0f;
        float pz = sk.z + std::sin(a) * 72.0f;
        glVertex3f(px, sampleTerrainElev(px, pz) + 0.16f, pz);
    }
    glEnd();
    // Mermer meydan platformu + sekizgen 3 kademeli kaide
    for (int step = 0; step < 3; step++) {
        float r0 = 9.5f - step * 2.2f;
        float y0 = skY + step * 1.1f, y1 = y0 + 1.1f;
        glColor3f(0.93f, 0.89f, 0.82f);
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i <= 8; i++) {
            float a = i * (2.0f * PI_F / 8.0f);
            float ca = std::cos(a), sa = std::sin(a);
            glNormal3f(ca, 0.2f, sa);
            glVertex3f(sk.x + ca * r0, y0, sk.z + sa * r0);
            glVertex3f(sk.x + ca * r0, y1, sk.z + sa * r0);
        }
        glEnd();
    }
    // Sekizgen govde + balkon + saat kadranlari + altin kubbe
    float towerLevels[5][2] = {{3.3f, 3.6f}, {11.0f, 2.8f}, {19.0f, 2.4f}, {24.5f, 2.9f}, {28.5f, 1.9f}};
    for (int lv = 0; lv < 4; lv++) {
        float y0 = skY + towerLevels[lv][0], y1 = skY + towerLevels[lv + 1][0];
        float r0 = towerLevels[lv][1], r1 = towerLevels[lv + 1][1];
        glColor3f((lv == 2) ? 0.86f : 0.94f, (lv == 2) ? 0.78f : 0.89f, (lv == 2) ? 0.66f : 0.79f);
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i <= 8; i++) {
            float a = i * (2.0f * PI_F / 8.0f);
            float ca = std::cos(a), sa = std::sin(a);
            glNormal3f(ca, 0.1f, sa);
            glVertex3f(sk.x + ca * r0, y0, sk.z + sa * r0);
            glVertex3f(sk.x + ca * r1, y1, sk.z + sa * r1);
        }
        glEnd();
    }
    // Kubbe + altin alem diregi
    glColor3f(0.92f, 0.78f, 0.32f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(sk.x, skY + 33.8f, sk.z);
    for (int i = 0; i <= 8; i++) {
        float a = i * (2.0f * PI_F / 8.0f);
        glVertex3f(sk.x + std::cos(a) * 2.0f, skY + 28.5f, sk.z + std::sin(a) * 2.0f);
    }
    glEnd();
    // Saat Kulesi 4 mermer kemer sutu + 4 isikli saat kadrani + alem diregi
    glBegin(GL_QUADS);
    glColor3f(0.95f, 0.91f, 0.84f);
    for (int f = 0; f < 4; f++) {
        float a = f * (PI_F * 0.5f) + PI_F * 0.25f;
        emitBoxFaces(sk.x + std::cos(a) * 4.2f, skY + 3.0f, sk.z + std::sin(a) * 4.2f, 0.85f, 0.85f, 4.8f, 1, 0, 1, 1);
    }
    for (int d = 0; d < 4; d++) {
        float a = d * (PI_F * 0.5f);
        glColor3f(0.98f, 0.95f, 0.78f);
        emitBoxFaces(sk.x + std::cos(a) * 2.55f, skY + 21.8f, sk.z + std::sin(a) * 2.55f,
                     1.45f, 0.25f, 1.45f, std::sin(a), -std::cos(a), 1, 1);
    }
    glColor3f(0.95f, 0.82f, 0.32f);
    emitBoxFaces(sk.x, skY + 33.5f, sk.z, 0.35f, 0.35f, 3.2f, 1, 0, 1, 1);
    glEnd();
    // Konak Yali Camii (Saat Kulesinin 55m dogusunda sekizgen cinili cami + minare)
    glBegin(GL_QUADS);
    glColor3f(0.92f, 0.88f, 0.80f);
    emitBoxFaces(sk.x + 52.0f, skY, sk.z - 12.0f, 16.0f, 16.0f, 11.0f, 1, 0, 1, 1);
    glColor3f(0.28f, 0.62f, 0.72f);
    emitBoxFaces(sk.x + 52.0f, skY + 11.0f, sk.z - 12.0f, 11.0f, 11.0f, 5.5f, 1, 0, 1, 1);
    glColor3f(0.94f, 0.91f, 0.85f);
    emitBoxFaces(sk.x + 61.0f, skY, sk.z - 19.0f, 2.2f, 2.2f, 27.0f, 1, 0, 1, 1);
    // Tarihi Konak Pier (Gustave Eiffel atolyesi celik-tas gumruk binasi, denize uzanan iskele)
    Vec2 kPier = geoPt(38.4215, 27.1275);
    glColor3f(0.76f, 0.72f, 0.66f);
    emitBoxFaces(kPier.x - 35.0f, 2.2f, kPier.z, 95.0f, 36.0f, 9.5f, 0.88f, -0.47f, 1, 1);
    glColor3f(0.68f, 0.34f, 0.24f);
    emitBoxFaces(kPier.x - 35.0f, 11.7f, kPier.z, 92.0f, 28.0f, 3.2f, 0.88f, -0.47f, 1, 1);
    glEnd();
    drawTextWorld(Vec3(sk.x, skY + 38.0f, sk.z), 0.0f, 0.65f, "KONAK SAAT KULESI", 1.0f, 0.92f, 0.55f);

    // 2B. TARIHI ASANSOR (Karatas 38.4086, 27.1175): 58m Kirmizi Tugla Kule + Yamac Terasi
    Vec2 asP = geoPt(ASANSOR_LAT, ASANSOR_LON);
    float asY = sampleTerrainElev(asP.x, asP.z);
    glBegin(GL_QUADS);
    glColor3f(0.76f, 0.32f, 0.24f);
    emitBoxFaces(asP.x, asY - 4.0f, asP.z, 12.0f, 12.0f, 58.0f, 0.92f, 0.38f, 1, 1);
    glColor3f(0.92f, 0.86f, 0.76f);
    emitBoxFaces(asP.x, asY + 51.0f, asP.z + 6.0f, 24.0f, 18.0f, 5.5f, 0.92f, 0.38f, 1, 1);
    glEnd();
    drawTextWorld(Vec3(asP.x, asY + 64.0f, asP.z), 0.0f, 0.55f, "TARIHI ASANSOR", 1.0f, 0.82f, 0.62f);

    // 2C. CUMHURIYET MEYDANI, GUNDOGDU MEYDANI, HILTON IZMIR, KULTURPARK PARASUT KULESI VE KARSIYAKA ANITI
    Vec2 cmP = geoPt(CUMHURIYET_LAT, CUMHURIYET_LON);
    float cmY = sampleTerrainElev(cmP.x, cmP.z);
    Vec2 gdP = geoPt(GUNDOGDU_LAT, GUNDOGDU_LON);
    float gdY = sampleTerrainElev(gdP.x, gdP.z);
    Vec2 hlP = geoPt(HILTON_LAT, HILTON_LON);
    float hlY = sampleTerrainElev(hlP.x, hlP.z);
    Vec2 kpP = geoPt(KULTURPARK_LAT, KULTURPARK_LON);
    float kpY = sampleTerrainElev(kpP.x, kpP.z);
    Vec2 kskA = geoPt(KSK_ANIT_LAT, KSK_ANIT_LON);
    float kskY = sampleTerrainElev(kskA.x, kskA.z);

    drawLoftedTwistedTower(hlP.x, hlY, hlP.z, 34.0f, 26.0f, 142.0f, 0.0f, 18, 0.76f, 0.82f, 0.90f);
    drawTextWorld(Vec3(hlP.x, hlY + 150.0f, hlP.z), 0.0f, 0.60f, "HILTON IZMIR 142M", 0.85f, 0.92f, 1.0f);

    glDisable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    // Cumhuriyet Meydani Ataturk Aniti kaidesi
    glColor3f(0.88f, 0.84f, 0.78f);
    emitBoxFaces(cmP.x, cmY, cmP.z, 18.0f, 18.0f, 8.5f, 0.85f, 0.52f, 1, 1);
    glColor3f(0.32f, 0.28f, 0.22f);
    emitBoxFaces(cmP.x, cmY + 8.5f, cmP.z, 6.5f, 10.5f, 9.5f, 0.85f, 0.52f, 1, 1);
    // Gundogdu Meydani Cumhuriyet Agaci Heykeli
    glColor3f(0.85f, 0.42f, 0.22f);
    emitBoxFaces(gdP.x, gdY, gdP.z, 8.0f, 8.0f, 22.0f, 0.7f, 0.7f, 1, 1);
    // Kulturpark Parasut Kulesi (48m Art-Deco beyaz kule ve 4 atlama terasi)
    glColor3f(0.96f, 0.96f, 0.98f);
    emitBoxFaces(kpP.x - 120.0f, kpY, kpP.z + 90.0f, 9.0f, 9.0f, 48.0f, 1, 0, 1, 1);
    glColor3f(0.18f, 0.52f, 0.88f);
    emitBoxFaces(kpP.x - 120.0f, kpY + 38.0f, kpP.z + 90.0f, 16.0f, 16.0f, 2.2f, 1, 0, 1, 1);
    // Karsiyaka Aniti (42m ikiz beyaz beton yelken anit)
    glColor3f(0.95f, 0.96f, 0.98f);
    emitBoxFaces(kskA.x - 5.5f, kskY, kskA.z, 4.5f, 14.0f, 42.0f, 0.95f, 0.31f, 1, 1);
    emitBoxFaces(kskA.x + 5.5f, kskY, kskA.z, 4.5f, 14.0f, 38.0f, 0.95f, -0.31f, 1, 1);
    glEnd();
    drawTextWorld(Vec3(kskA.x, kskY + 48.0f, kskA.z), 0.0f, 0.55f, "KARSIYAKA ANITI", 0.90f, 0.98f, 1.0f);

    // 3. ALSANCAK LIMANI: Konteyner Terminali Beton Sahasi, Vincler, Konteyner Bloklari ve Kruvaziyer Gemisi
    Vec2 portP = geoPt(38.4438, 27.1500);
    float portY = sampleTerrainElev(portP.x, portP.z);
    glBegin(GL_QUADS);
    // Genis Alsancak Limani beton konteyner istif sahasi
    glColor3f(0.68f, 0.70f, 0.73f);
    emitBoxFaces(portP.x + 10.0f, portY - 0.1f, portP.z + 55.0f, 720.0f, 220.0f, 0.32f, 1, 0, 1, 1);
    for (int c = 0; c < 6; c++) {
        float cx = portP.x - 240.0f + c * 95.0f;
        float cz = portP.z;
        // Kirmizi/Beyaz Gantry konteyner vinci
        glColor3f(0.88f, 0.18f, 0.16f);
        emitBoxFaces(cx - 8.0f, portY, cz - 8.0f, 2.8f, 2.8f, 48.0f, 1, 0, 1, 1);
        emitBoxFaces(cx + 8.0f, portY, cz - 8.0f, 2.8f, 2.8f, 48.0f, 1, 0, 1, 1);
        emitBoxFaces(cx - 8.0f, portY, cz + 8.0f, 2.8f, 2.8f, 48.0f, 1, 0, 1, 1);
        emitBoxFaces(cx + 8.0f, portY, cz + 8.0f, 2.8f, 2.8f, 48.0f, 1, 0, 1, 1);
        glColor3f(0.95f, 0.95f, 0.95f);
        emitBoxFaces(cx, portY + 44.0f, cz - 18.0f, 18.0f, 82.0f, 4.2f, 1, 0, 1, 1);
        // Renkli konteyner istifleri (on ve arka istif sahasi)
        for (int yardZ = 0; yardZ < 2; yardZ++) {
            for (int rowC = 0; rowC < 4; rowC++) {
                for (int k = 0; k < 4; k++) {
                    int cMod = (k + rowC + yardZ + c) % 3;
                    if (cMod == 0) glColor3f(0.16f, 0.42f, 0.78f);
                    else if (cMod == 1) glColor3f(0.82f, 0.34f, 0.14f);
                    else glColor3f(0.18f, 0.62f, 0.36f);
                    emitBoxFaces(cx - 18.0f + rowC * 18.0f, portY + 0.25f + k * 3.2f,
                                 cz + 28.0f + yardZ * 48.0f, 12.0f, 34.0f, 3.0f, 1, 0, 1, 1);
                }
            }
        }
        // Liman lojistik antrepo holleri
        if (c < 4) {
            glColor3f(0.76f, 0.80f, 0.85f);
            emitBoxFaces(portP.x - 180.0f + c * 135.0f, portY, cz + 128.0f, 92.0f, 36.0f, 11.5f, 1, 0, 1, 1);
            glColor3f(0.26f, 0.44f, 0.68f);
            emitBoxFaces(portP.x - 180.0f + c * 135.0f, portY + 11.5f, cz + 128.0f, 94.0f, 38.0f, 1.4f, 1, 0, 1, 1);
        }
    }
    // Alsancak Kruvaziyer Iskelesi aciginda denizde demirli 240m Kruvaziyer Gemisi (Sivri pruva + cam balkon guverteleri + baca)
    Vec2 shipP = geoPt(38.4445, 27.1388);
    {
        Vec2 sR(0.78f, 0.62f), sF(-0.62f, 0.78f);
        // Alt celik tekne govdesi ve konik sivri pruva (bow)
        glColor3f(0.14f, 0.22f, 0.38f);
        emitBoxFaces(shipP.x, 0.0f, shipP.z, 30.0f, 175.0f, 8.5f, sR.x, sR.z, 1, 1);
        Vec2 bowBase = shipP + sF * 87.5f, bowTip = shipP + sF * 124.0f;
        Vec2 bL = bowBase - sR * 15.0f, bR = bowBase + sR * 15.0f;
        glColor3f(0.16f, 0.26f, 0.44f);
        glNormal3f(-sR.x * 0.8f, 0.1f, sF.x * 0.4f);
        glVertex3f(bL.x, 0.0f, bL.z); glVertex3f(bowTip.x, 0.0f, bowTip.z);
        glVertex3f(bowTip.x, 10.5f, bowTip.z); glVertex3f(bL.x, 8.5f, bL.z);
        glNormal3f(sR.x * 0.8f, 0.1f, sF.z * 0.4f);
        glVertex3f(bowTip.x, 0.0f, bowTip.z); glVertex3f(bR.x, 0.0f, bR.z);
        glVertex3f(bR.x, 8.5f, bR.z); glVertex3f(bowTip.x, 10.5f, bowTip.z);
        glNormal3f(0.0f, 1.0f, 0.0f);
        glColor3f(0.86f, 0.89f, 0.93f);
        glVertex3f(bL.x, 8.5f, bL.z); glVertex3f(bowTip.x, 10.5f, bowTip.z);
        glVertex3f(bowTip.x, 10.5f, bowTip.z); glVertex3f(bR.x, 8.5f, bR.z);
        // Ust beyaz yolcu guverteleri + cam seritler
        glColor3f(0.96f, 0.97f, 0.99f);
        emitBoxFaces(shipP.x, 8.5f, shipP.z, 31.0f, 185.0f, 8.5f, sR.x, sR.z, 1, 1);
        glColor3f(0.24f, 0.46f, 0.72f);
        emitBoxFaces(shipP.x, 11.2f, shipP.z, 31.4f, 172.0f, 3.2f, sR.x, sR.z, 1, 1);
        glColor3f(0.92f, 0.95f, 0.98f);
        emitBoxFaces(shipP.x - sF.x * 8.0f, 17.0f, shipP.z - sF.z * 8.0f, 25.0f, 152.0f, 11.0f, sR.x, sR.z, 1, 1);
        // Kopru ustu + baca
        glColor3f(0.97f, 0.98f, 1.00f);
        emitBoxFaces(shipP.x + sF.x * 64.0f, 28.0f, shipP.z + sF.z * 64.0f, 32.0f, 16.0f, 4.5f, sR.x, sR.z, 1, 1);
        glColor3f(0.14f, 0.38f, 0.78f);
        emitBoxFaces(shipP.x - sF.x * 16.0f, 28.0f, shipP.z - sF.z * 16.0f, 9.0f, 20.0f, 13.0f, sR.x, sR.z, 1, 1);
        glColor3f(0.88f, 0.22f, 0.18f);
        emitBoxFaces(shipP.x - sF.x * 18.0f, 41.0f, shipP.z - sF.z * 18.0f, 7.5f, 14.0f, 2.5f, sR.x, sR.z, 1, 1);
    }
    glEnd();
    drawTextWorld(Vec3(portP.x, portY + 62.0f, portP.z), 0.0f, 0.7f, "ALSANCAK LIMANI", 1.0f, 0.95f, 0.75f);

    // 4. ADNAN MENDERES HAVALIMANI (LTBJ: 38.2924, 27.1570, rakim 125m, yon 165.8/345.8,
    //    3240x45m cift paralel 16R/34L ve 16L/34R pistleri, cam catili terminal ve ATC kulesi)
    Vec2 ltbj = geoPt(LTBJ_LAT, LTBJ_LON);
    Vec2 ltbjDir = headingDir(LTBJ_HDG);
    Vec2 ltbjRight(-ltbjDir.z, ltbjDir.x);
    // Taksi yollari ve genis apron betonu
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
    glColor3f(0.68f, 0.70f, 0.73f);
    glBegin(GL_QUADS);
    emitBoxFaces(ltbj.x - ltbjRight.x * 420.0f, LTBJ_ELEV - 0.2f, ltbj.z - ltbjRight.z * 420.0f,
                 320.0f, 1450.0f, 0.28f, ltbjRight.x, ltbjRight.z, 0.02f, 0.02f);
    glEnd();
    // 16R/34L ( Bati pisti ) ve 16L/34R ( Dogu pisti )
    drawRunwayWithMarkings(ltbj - ltbjRight * (LTBJ_SEP * 0.5f), LTBJ_ELEV, LTBJ_HDG, LTBJ_LEN, LTBJ_W, true, -1);
    drawRunwayWithMarkings(ltbj + ltbjRight * (LTBJ_SEP * 0.5f), LTBJ_ELEV, LTBJ_HDG, LTBJ_LEN, LTBJ_W, true, +1);

    // Cam catili modern LTBJ Terminal binasi + korukler + ATC Kulesi
    Vec2 termC = ltbj - ltbjRight * 480.0f;
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GLASS_TOWER]);
    glBegin(GL_QUADS);
    glColor3f(0.76f, 0.88f, 0.98f);
    emitBoxFaces(termC.x, LTBJ_ELEV, termC.z, 130.0f, 680.0f, 24.0f, ltbjRight.x, ltbjRight.z, 0.03f, 0.05f);
    // Kemerli cam cati
    glColor3f(0.58f, 0.82f, 0.96f);
    emitBoxFaces(termC.x, LTBJ_ELEV + 24.0f, termC.z, 110.0f, 640.0f, 6.5f, ltbjRight.x, ltbjRight.z, 0.04f, 0.04f);
    // ATC Kulesi (65m)
    Vec2 atcC = termC - ltbjRight * 140.0f + ltbjDir * 220.0f;
    glColor3f(0.90f, 0.92f, 0.95f);
    emitBoxFaces(atcC.x, LTBJ_ELEV, atcC.z, 12.0f, 12.0f, 52.0f, 1, 0, 0.1f, 0.1f);
    glColor3f(0.35f, 0.72f, 0.95f);
    emitBoxFaces(atcC.x, LTBJ_ELEV + 52.0f, atcC.z, 20.0f, 20.0f, 9.0f, 1, 0, 0.1f, 0.1f);
    glEnd();
    drawTextWorld(Vec3(termC.x, LTBJ_ELEV + 44.0f, termC.z), 0.0f, 0.85f,
                  "ADNAN MENDERES HAVALIMANI LTBJ", 0.55f, 0.96f, 1.0f);

    // 5. CIGLI 2. ANA JET USSU (LTBL: 38.5130, 27.0100, rakim 5m, yon 167/347, 2900x45m askeri pist, HAS siginaklar ve hangarlar)
    Vec2 ltbl = geoPt(LTBL_LAT, LTBL_LON);
    Vec2 ltblDir = headingDir(LTBL_HDG);
    Vec2 ltblRight(-ltblDir.z, ltblDir.x);
    drawRunwayWithMarkings(ltbl, LTBL_ELEV, LTBL_HDG, LTBL_LEN, LTBL_W, false, 0);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_INDUSTRIAL]);
    glBegin(GL_QUADS);
    // 14 adet HAS (Hardened Aircraft Shelter) askeri korugan + 4 buyuk bakim hangari
    for (int h = -6; h <= 7; h++) {
        Vec2 hc = ltbl + ltblRight * 165.0f + ltblDir * (h * 140.0f);
        glColor3f(0.54f, 0.58f, 0.48f);
        emitBoxFaces(hc.x, LTBL_ELEV, hc.z, 28.0f, 36.0f, 9.5f, ltblRight.x, ltblRight.z, 0.05f, 0.05f);
    }
    for (int h = -2; h <= 1; h++) {
        Vec2 hc = ltbl - ltblRight * 175.0f + ltblDir * (h * 190.0f);
        glColor3f(0.66f, 0.72f, 0.78f);
        emitBoxFaces(hc.x, LTBL_ELEV, hc.z, 55.0f, 75.0f, 16.0f, ltblRight.x, ltblRight.z, 0.04f, 0.04f);
    }
    glEnd();
    drawTextWorld(Vec3(ltbl.x, LTBL_ELEV + 38.0f, ltbl.z), 0.0f, 0.80f,
                  "CIGLI 2. ANA JET USSU LTBL", 0.75f, 1.0f, 0.65f);

    // 6. KADIFEKALE SURLARI VE 5 TARIHI BURCU (Pagos 38.4142, 27.1453, 186m)
    Vec2 kk = geoPt(KADIFEKALE_LAT, KADIFEKALE_LON);
    float kkY = sampleTerrainElev(kk.x, kk.z);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ROCK]);
    glColor3f(0.76f, 0.70f, 0.60f);
    glBegin(GL_QUADS);
    // Dis kale beden duvarlari
    emitBoxFaces(kk.x, kkY - 4.0f, kk.z - 52.0f, 165.0f, 8.0f, 16.0f, 0.92f, 0.39f, 0.05f, 0.05f);
    emitBoxFaces(kk.x, kkY - 4.0f, kk.z + 52.0f, 165.0f, 8.0f, 16.0f, 0.92f, 0.39f, 0.05f, 0.05f);
    emitBoxFaces(kk.x - 78.0f, kkY - 4.0f, kk.z, 8.0f, 108.0f, 16.0f, 0.92f, 0.39f, 0.05f, 0.05f);
    emitBoxFaces(kk.x + 78.0f, kkY - 4.0f, kk.z, 8.0f, 108.0f, 16.0f, 0.92f, 0.39f, 0.05f, 0.05f);
    // 5 adet kale burcu
    static const float bastions[5][2] = {{-80.0f, -54.0f}, {0.0f, -56.0f}, {80.0f, -54.0f}, {-80.0f, 54.0f}, {80.0f, 54.0f}};
    for (int b = 0; b < 5; b++) {
        float bx = kk.x + bastions[b][0], bz = kk.z + bastions[b][1];
        float by = sampleTerrainElev(bx, bz);
        glColor3f(0.72f, 0.66f, 0.56f);
        emitBoxFaces(bx, by - 4.0f, bz, 18.0f, 18.0f, 22.0f, 0.92f, 0.39f, 0.06f, 0.06f);
    }
    // 35m Beyaz Bayrak Diregi ve Buyuk Turk Bayragi
    glColor3f(0.96f, 0.96f, 0.98f);
    emitBoxFaces(kk.x, kkY, kk.z, 0.8f, 0.8f, 35.0f, 1, 0, 1, 1);
    glColor3f(0.88f, 0.12f, 0.12f);
    emitBoxFaces(kk.x + 8.0f, kkY + 24.0f, kk.z, 16.0f, 0.25f, 10.0f, 1, 0, 1, 1);
    glEnd();
    drawTextWorld(Vec3(kk.x, kkY + 38.0f, kk.z), 0.0f, 0.65f, "KADIFEKALE 186M", 1.0f, 0.88f, 0.55f);

    // 7. TARIHI KEMERALTI CARSISI: HISAR CAMII VE KIZLARAGASI HANI
    Vec2 hisP = geoPt(38.4196, 27.1394);
    float hisY = sampleTerrainElev(hisP.x, hisP.z);
    Vec2 kzP = geoPt(38.4202, 27.1380);
    float kzY = sampleTerrainElev(kzP.x, kzP.z);
    glBegin(GL_QUADS);
    glColor3f(0.90f, 0.86f, 0.78f);
    emitBoxFaces(hisP.x, hisY, hisP.z, 28.0f, 28.0f, 14.0f, 1, 0, 1, 1);
    glColor3f(0.94f, 0.92f, 0.86f);
    emitBoxFaces(hisP.x + 16.0f, hisY, hisP.z - 16.0f, 2.8f, 2.8f, 32.0f, 1, 0, 1, 1);
    glColor3f(0.84f, 0.74f, 0.62f);
    emitBoxFaces(kzP.x, kzY, kzP.z, 54.0f, 44.0f, 9.2f, 1, 0, 1, 1);
    glColor3f(0.72f, 0.32f, 0.22f);
    emitBoxFaces(kzP.x, kzY + 9.2f, kzP.z, 56.0f, 46.0f, 1.4f, 1, 0, 1, 1);
    glEnd();
    drawTextWorld(Vec3(hisP.x, hisY + 36.0f, hisP.z), 0.0f, 0.60f, "HISAR CAMII & KEMERALTI", 1.0f, 0.95f, 0.65f);

    // 8. BOSTANLI GUNBATIMI TERASI (38.4620, 27.0870)
    Vec2 btP = geoPt(38.4620, 27.0870);
    float btY = sampleTerrainElev(btP.x, btP.z);
    glBegin(GL_QUADS);
    for (int t = 0; t < 5; t++) {
        glColor3f(0.68f + t * 0.03f, 0.48f + t * 0.02f, 0.30f);
        emitBoxFaces(btP.x, btY + t * 0.65f, btP.z, 45.0f - t * 4.0f, 32.0f, 0.65f, 0.88f, 0.47f, 1, 1);
    }
    glEnd();
    drawTextWorld(Vec3(btP.x, btY + 8.5f, btP.z), 0.0f, 0.55f, "BOSTANLI GUNBATIMI TERASI", 1.0f, 0.88f, 0.55f);

    // 9. CAMALTI TUZLASI TUZ TAVALARI VE GEDIZ DELTASI KUS CENNETI (38.508, 26.950)
    Vec2 tuzBase = geoPt(38.5080, 26.9560);
    if (!g_hasGisGrid) {
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.5f, -1.5f);
        glBegin(GL_QUADS);
        glNormal3f(0.0f, 1.0f, 0.0f);
        for (int row = -4; row <= 5; row++) {
            for (int col = -5; col <= 4; col++) {
                float px = tuzBase.x + col * 360.0f - row * 110.0f;
                float pz = tuzBase.z + row * 290.0f + col * 60.0f;
                if (sampleSignedSeaDist(px, pz) < 60.0f) continue;
                float py = sampleTerrainElev(px, pz) + 0.16f;
                glColor3f(0.74f, 0.68f, 0.56f);
                float dw = 172.0f, dd = 136.0f;
                glVertex3f(px - dw, py, pz - dd);
                glVertex3f(px + dw, py, pz - dd);
                glVertex3f(px + dw, py, pz + dd);
                glVertex3f(px - dw, py, pz + dd);

                int mode = (std::abs(row * 7 + col * 11)) % 4;
                if (mode == 0) glColor3f(0.95f, 0.96f, 0.97f);
                else if (mode == 1) glColor3f(0.86f, 0.70f, 0.72f);
                else if (mode == 2) glColor3f(0.28f, 0.62f, 0.68f);
                else glColor3f(0.72f, 0.80f, 0.78f);
                float pw = 156.0f, pd = 122.0f;
                glVertex3f(px - pw, py + 0.06f, pz - pd);
                glVertex3f(px + pw, py + 0.06f, pz - pd);
                glVertex3f(px + pw, py + 0.06f, pz + pd);
                glVertex3f(px - pw, py + 0.06f, pz + pd);
            }
        }
        glEnd();
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    float tuzY = sampleTerrainElev(tuzBase.x, tuzBase.z);
    drawTextWorld(Vec3(tuzBase.x, tuzY + 34.0f, tuzBase.z), 0.0f, 0.75f,
                  "CAMALTI TUZLASI & GEDIZ DELTASI", 0.95f, 0.98f, 0.80f);

    // 10. FUAR IZMIR (GAZIEMIR ULUSLARARASI FUAR ALANI 38.3320, 27.1140)
    Vec2 fuarP = geoPt(38.3320, 27.1140);
    float fuarY = sampleTerrainElev(fuarP.x, fuarP.z);
    if (g_osmBuildings.empty()) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
        glBegin(GL_QUADS);
        glColor3f(0.78f, 0.80f, 0.82f);
        emitBoxFaces(fuarP.x, fuarY - 0.1f, fuarP.z, 360.0f, 240.0f, 0.4f, 0.96f, 0.28f, 0.04f, 0.04f);
        for (int h = -1; h <= 1; h++) {
            glColor3f(0.92f, 0.94f, 0.97f);
            emitBoxFaces(fuarP.x + h * 105.0f, fuarY, fuarP.z, 84.0f, 165.0f, 16.5f, 0.96f, 0.28f, 0.05f, 0.05f);
            glColor3f(0.24f, 0.56f, 0.86f);
            emitBoxFaces(fuarP.x + h * 105.0f, fuarY + 16.5f, fuarP.z, 88.0f, 170.0f, 1.8f, 0.96f, 0.28f, 0.05f, 0.05f);
        }
        glEnd();
    }
    drawTextWorld(Vec3(fuarP.x, fuarY + 26.0f, fuarP.z), 0.0f, 0.65f, "FUAR IZMIR", 0.65f, 0.92f, 1.0f);

    glEndList();
}

// =====================================================================================
//  CANLI TRAFIK: 8 ADET 3-VAGONLU IZBAN TRENI, 6 ADET KORFEZ VAPURU, OTOYOL ARACLARI
// =====================================================================================
struct IzbanTrain {
    float s;          // Spline uzerindeki konum (0..g_izbanTotalLen)
    float speed;      // m/s
    int dir;          // +1 kuzeyden guneye, -1 guneyden kuzeye
    float dwellTimer; // Istasyonda duraklama sayaci
    int lastStation;
};
static const int NUM_IZBAN_TRAINS = 8;
static IzbanTrain g_trains[NUM_IZBAN_TRAINS];

struct FerryBoat {
    int routeA, routeB;
    float t;          // 0..1 iskeleler arasi
    float speed;
    int dir;
    float dwell;
    Vec2 pos;
    Vec2 fwd;
};
static const int NUM_FERRIES = 6;
static FerryBoat g_ferries[NUM_FERRIES];

struct RoadVehicle {
    int roadIdx;
    float s;
    float speed;
    int dir;
    int lane;
    uint8_t kind; // 0 otomobil, 1 ESHOT kirmizi otobus, 2 tir
};
static std::vector<RoadVehicle> g_vehicles;

static void sampleIzbanAtS(float s, Vec3& posOut, Vec2& tangOut) {
    s = clampf(s, 0.0f, g_izbanTotalLen);
    auto it = std::lower_bound(g_izbanCumDist.begin(), g_izbanCumDist.end(), s);
    size_t idx = (it == g_izbanCumDist.end()) ? (g_izbanSpline.size() - 1) : (size_t)(it - g_izbanCumDist.begin());
    if (idx == 0) {
        posOut = g_izbanSpline[0];
        tangOut = g_izbanTangents[0];
        return;
    }
    float s0 = g_izbanCumDist[idx - 1], s1 = g_izbanCumDist[idx];
    float t = (s1 > s0 + 1e-4f) ? ((s - s0) / (s1 - s0)) : 0.0f;
    posOut = lerpV(g_izbanSpline[idx - 1], g_izbanSpline[idx], t);
    tangOut = norm2(g_izbanTangents[idx - 1] * (1.0f - t) + g_izbanTangents[idx] * t);
}

static inline const std::vector<RoadSpline>& activeTrafficRoutes() {
    return (g_hasV2Roads && !g_v2DriveRoutes.empty()) ? g_v2DriveRoutes : g_roads;
}

static void buildV2DriveRoutes() {
    ensureV2MajorRoadElevs();
    g_v2DriveRoutes.clear();
    const int nMaj = (int)g_v2MajorRoads.size();
    if (nMaj == 0) return;

    struct PieceInfo {
        int majIdx;
        float length;
        Vec2 tStart, tEnd;
    };
    std::vector<PieceInfo> pieces;
    pieces.reserve(nMaj);
    for (int i = 0; i < nMaj; i++) {
        const auto& rp = g_v2MajorRoads[i];
        if (rp.tier > 2 || rp.pts.size() < 2 || rp.ys.size() != rp.pts.size()) continue;
        float L = 0.0f;
        for (size_t k = 0; k + 1 < rp.pts.size(); k++) L += len2(rp.pts[k + 1] - rp.pts[k]);
        if (L < 12.0f) continue;
        Vec2 t0 = norm2(rp.pts[1] - rp.pts[0]);
        Vec2 t1 = norm2(rp.pts.back() - rp.pts[rp.pts.size() - 2]);
        if (len2(t0) < 0.1f || len2(t1) < 0.1f) continue;
        pieces.push_back({i, L, t0, t1});
    }
    if (pieces.empty()) return;

    std::vector<int> order(pieces.size());
    for (size_t i = 0; i < pieces.size(); i++) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return pieces[a].length > pieces[b].length;
    });

    const int NB = OSM_CK_NX * OSM_CK_NZ;
    std::vector<std::vector<int>> epBuckets(NB);
    auto bucketIdx = [&](float wx, float wz) -> int {
        int cx = clampi((int)((wx - WORLD_X0) / OSM_CHUNK_SZ), 0, OSM_CK_NX - 1);
        int cz = clampi((int)((wz - WORLD_Z0) / OSM_CHUNK_SZ), 0, OSM_CK_NZ - 1);
        return cz * OSM_CK_NX + cx;
    };
    for (int pi = 0; pi < (int)pieces.size(); pi++) {
        const auto& rp = g_v2MajorRoads[pieces[pi].majIdx];
        int b0 = bucketIdx(rp.pts.front().x, rp.pts.front().z);
        int b1 = bucketIdx(rp.pts.back().x, rp.pts.back().z);
        epBuckets[b0].push_back(pi);
        if (b1 != b0) epBuckets[b1].push_back(pi);
    }

    std::vector<uint8_t> used(pieces.size(), 0);
    auto findBestNext = [&](Vec2 endPt, float endY, Vec2 endTang, uint8_t tier,
                            bool& outReversed) -> int {
        int cx0 = clampi((int)((endPt.x - WORLD_X0) / OSM_CHUNK_SZ), 0, OSM_CK_NX - 1);
        int cz0 = clampi((int)((endPt.z - WORLD_Z0) / OSM_CHUNK_SZ), 0, OSM_CK_NZ - 1);
        int bestPi = -1;
        float bestScore = -1e9f;
        bool bestRev = false;
        const float maxD2 = 4.5f * 4.5f;
        for (int dz = -1; dz <= 1; dz++) {
            int cz = cz0 + dz;
            if (cz < 0 || cz >= OSM_CK_NZ) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int cx = cx0 + dx;
                if (cx < 0 || cx >= OSM_CK_NX) continue;
                for (int pi : epBuckets[cz * OSM_CK_NX + cx]) {
                    if (used[pi]) continue;
                    const auto& rp = g_v2MajorRoads[pieces[pi].majIdx];
                    if (rp.tier != tier) continue;
                    {
                        Vec2 pS = rp.pts.front();
                        float d2 = (pS.x - endPt.x) * (pS.x - endPt.x) + (pS.z - endPt.z) * (pS.z - endPt.z);
                        float dy = std::fabs(rp.ys.front() - endY);
                        if (d2 <= maxD2 && dy <= 2.2f) {
                            float dp = dot2(endTang, pieces[pi].tStart);
                            if (dp > 0.45f) {
                                float score = dp - 0.05f * d2;
                                if (score > bestScore) { bestScore = score; bestPi = pi; bestRev = false; }
                            }
                        }
                    }
                    {
                        Vec2 pE = rp.pts.back();
                        float d2 = (pE.x - endPt.x) * (pE.x - endPt.x) + (pE.z - endPt.z) * (pE.z - endPt.z);
                        float dy = std::fabs(rp.ys.back() - endY);
                        if (d2 <= maxD2 && dy <= 2.2f) {
                            float dp = dot2(endTang, pieces[pi].tEnd * -1.0f);
                            if (dp > 0.45f) {
                                float score = dp - 0.05f * d2;
                                if (score > bestScore) { bestScore = score; bestPi = pi; bestRev = true; }
                            }
                        }
                    }
                }
            }
        }
        outReversed = bestRev;
        return bestPi;
    };

    auto appendPiece = [](std::vector<Vec3>& dst, const V2MajorRoadPoly& rp, bool rev) {
        size_t n = rp.pts.size();
        for (size_t k = 0; k < n; k++) {
            size_t idx = rev ? (n - 1 - k) : k;
            Vec3 np(rp.pts[idx].x, rp.ys[idx], rp.pts[idx].z);
            if (!dst.empty()) {
                float dx = np.x - dst.back().x, dz = np.z - dst.back().z;
                if (dx * dx + dz * dz < 0.25f) continue;
            }
            dst.push_back(np);
        }
    };

    for (int seedPi : order) {
        if (used[seedPi]) continue;
        used[seedPi] = 1;
        const auto& seedRp = g_v2MajorRoads[pieces[seedPi].majIdx];
        uint8_t tier = seedRp.tier;
        float minHW = seedRp.halfW;

        std::vector<Vec3> chain;
        appendPiece(chain, seedRp, false);

        for (int hop = 0; hop < 400 && chain.size() >= 2; hop++) {
            Vec2 endPt(chain.back().x, chain.back().z);
            Vec2 prevPt(chain[chain.size() - 2].x, chain[chain.size() - 2].z);
            bool rev = false;
            int nxt = findBestNext(endPt, chain.back().y, norm2(endPt - prevPt), tier, rev);
            if (nxt < 0) break;
            used[nxt] = 1;
            const auto& nxtRp = g_v2MajorRoads[pieces[nxt].majIdx];
            minHW = std::min(minHW, nxtRp.halfW);
            appendPiece(chain, nxtRp, rev);
        }

        std::reverse(chain.begin(), chain.end());
        for (int hop = 0; hop < 400 && chain.size() >= 2; hop++) {
            Vec2 endPt(chain.back().x, chain.back().z);
            Vec2 prevPt(chain[chain.size() - 2].x, chain[chain.size() - 2].z);
            bool rev = false;
            int nxt = findBestNext(endPt, chain.back().y, norm2(endPt - prevPt), tier, rev);
            if (nxt < 0) break;
            used[nxt] = 1;
            const auto& nxtRp = g_v2MajorRoads[pieces[nxt].majIdx];
            minHW = std::min(minHW, nxtRp.halfW);
            appendPiece(chain, nxtRp, rev);
        }
        std::reverse(chain.begin(), chain.end());

        if (chain.size() < 2) continue;
        float cum = 0.0f;
        std::vector<float> cumD(chain.size(), 0.0f);
        std::vector<Vec2> tangs(chain.size(), Vec2(1.0f, 0.0f));
        for (size_t i = 0; i < chain.size(); i++) {
            if (i > 0) {
                float dx = chain[i].x - chain[i - 1].x;
                float dz = chain[i].z - chain[i - 1].z;
                cum += std::sqrt(dx * dx + dz * dz);
                cumD[i] = cum;
            }
            Vec2 a = (i == 0) ? Vec2(chain[0].x, chain[0].z) : Vec2(chain[i - 1].x, chain[i - 1].z);
            Vec2 b = (i + 1 == chain.size()) ? Vec2(chain.back().x, chain.back().z) : Vec2(chain[i + 1].x, chain[i + 1].z);
            Vec2 td = norm2(b - a);
            tangs[i] = (len2(td) > 0.1f) ? td : Vec2(1.0f, 0.0f);
        }
        if (cum < 250.0f) continue;

        RoadSpline rs;
        rs.name = (tier == 0) ? "v2-otoyol" : ((tier == 1) ? "v2-bulvar" : "v2-ana-arter");
        rs.type = tier;
        rs.halfW = std::max(3.6f, minHW);
        rs.lanes = (rs.halfW >= 8.0f) ? 2 : 1;
        rs.pts = std::move(chain);
        rs.tangs = std::move(tangs);
        rs.cumDist = std::move(cumD);
        rs.totalLen = cum;
        g_v2DriveRoutes.push_back(std::move(rs));
    }
}

static void sampleTrafficVehiclePose(const RoadSpline& rs, const RoadVehicle& v, Vec3& outPos, Vec2& outRight) {
    float s = clampf(v.s, 0.0f, rs.totalLen);
    auto it = std::lower_bound(rs.cumDist.begin(), rs.cumDist.end(), s);
    size_t idx = (it == rs.cumDist.end()) ? (rs.pts.size() - 1) : (size_t)(it - rs.cumDist.begin());
    Vec3 p = rs.pts[idx];
    Vec2 tang = rs.tangs[idx];
    if (idx > 0 && idx < rs.pts.size()) {
        float s0 = rs.cumDist[idx - 1], s1 = rs.cumDist[idx];
        float span = s1 - s0;
        if (span > 1e-4f) {
            float u = clampf((s - s0) / span, 0.0f, 1.0f);
            p = lerpV(rs.pts[idx - 1], rs.pts[idx], u);
            Vec2 td = norm2(Vec2(rs.pts[idx].x - rs.pts[idx - 1].x, rs.pts[idx].z - rs.pts[idx - 1].z));
            if (len2(td) > 0.1f) tang = td;
        }
    }
    Vec2 t = tang * (float)v.dir;
    Vec2 n(-t.z, t.x); // Surucu sag seridi
    float laneSpacing = (rs.lanes > 1) ? std::min(3.2f, std::max(1.8f, rs.halfW - 2.8f)) : 0.0f;
    float laneBase = (rs.halfW >= 7.5f) ? 2.4f : 1.7f;
    float laneOff = clampf(laneBase + (float)v.lane * laneSpacing, 1.2f, std::max(1.4f, rs.halfW - 1.4f));
    outPos = Vec3(p.x + n.x * laneOff, p.y + 0.12f, p.z + n.z * laneOff);
    outRight = n;
}

static void initDynamicTraffic() {
    // 8 adet 3-vagonlu IZBAN treni (hat boyunca dagitilmis, 4'u guney yonlu, 4'u kuzey yonlu)
    for (int i = 0; i < NUM_IZBAN_TRAINS; i++) {
        g_trains[i].s = ((float)i + 0.5f) / (float)NUM_IZBAN_TRAINS * g_izbanTotalLen;
        g_trains[i].dir = (i & 1) ? -1 : 1;
        g_trains[i].speed = 24.0f;
        g_trains[i].dwellTimer = 0.0f;
        g_trains[i].lastStation = -1;
    }

    // 6 adet canli IZDENIZ Korfez vapuru:
    // Konak(0), Pasaport(1), Alsancak(2), Karsiyaka(3), Bostanli(4), Goztepe(5), Uckuyular(6)
    static const int routes[NUM_FERRIES][2] = {
        {0, 3}, // Konak <-> Karsiyaka
        {0, 4}, // Konak <-> Bostanli
        {2, 3}, // Alsancak <-> Karsiyaka
        {1, 4}, // Pasaport <-> Bostanli
        {6, 4}, // Uckuyular <-> Bostanli (Arabalı Vapur)
        {5, 0}  // Goztepe <-> Konak
    };
    for (int i = 0; i < NUM_FERRIES; i++) {
        g_ferries[i].routeA = routes[i][0];
        g_ferries[i].routeB = routes[i][1];
        g_ferries[i].t = 0.15f + 0.14f * i;
        g_ferries[i].dir = (i & 1) ? -1 : 1;
        g_ferries[i].speed = 11.5f;
        g_ferries[i].dwell = 0.0f;
        Vec2 a = g_pierSeaPos[g_ferries[i].routeA];
        Vec2 b = g_pierSeaPos[g_ferries[i].routeB];
        g_ferries[i].pos = a + (b - a) * g_ferries[i].t;
        g_ferries[i].fwd = norm2(b - a);
    }

    // Otoyol araclari (cift yonlu akan trafik)
    g_v2DriveRoutes.clear();
    if (g_hasV2Roads) buildV2DriveRoutes();
    const std::vector<RoadSpline>& driveRoutes = activeTrafficRoutes();
    bool usingV2Routes = (g_hasV2Roads && !g_v2DriveRoutes.empty());
    g_vehicles.clear();
    Rng vrng(998877u);
    float densityDiv = usingV2Routes ? 240.0f : 180.0f;
    int minPerRoute = usingV2Routes ? 2 : 30;
    int maxPerRoute = usingV2Routes ? 45 : 200;
    for (int r = 0; r < (int)driveRoutes.size(); r++) {
        if (driveRoutes[r].totalLen < 20.0f || driveRoutes[r].pts.size() < 2) continue;
        int count = clampi((int)(driveRoutes[r].totalLen / densityDiv), minPerRoute, maxPerRoute);
        for (int k = 0; k < count; k++) {
            RoadVehicle v;
            v.roadIdx = r;
            v.s = vrng.range(0.0f, driveRoutes[r].totalLen);
            v.dir = (k & 1) ? -1 : 1;
            v.lane = k % std::max(1, driveRoutes[r].lanes);
            v.kind = (uint8_t)(k % 7 == 0 ? 1 : (k % 11 == 0 ? 2 : 0));
            v.speed = (v.kind == 0) ? vrng.range(24.0f, 34.0f) : vrng.range(18.0f, 24.0f);
            g_vehicles.push_back(v);
        }
    }
}

static void updateDynamicTraffic(float dt) {
    // 1. 8 IZBAN trenini guncelle (istasyonlarda 4 sn duraklayarak calisir)
    for (int i = 0; i < NUM_IZBAN_TRAINS; i++) {
        IzbanTrain& tr = g_trains[i];
        if (tr.dwellTimer > 0.0f) {
            tr.dwellTimer -= dt;
            tr.speed = 0.0f;
            continue;
        }
        tr.speed = std::min(28.0f, tr.speed + 9.5f * dt);
        tr.s += tr.dir * tr.speed * dt;
        if (tr.s >= g_izbanTotalLen - 20.0f) { tr.s = g_izbanTotalLen - 20.0f; tr.dir = -1; tr.dwellTimer = 3.5f; }
        if (tr.s <= 20.0f) { tr.s = 20.0f; tr.dir = 1; tr.dwellTimer = 3.5f; }

        for (int st = 0; st < NUM_STATIONS; st++) {
            if (st != tr.lastStation && std::fabs(tr.s - g_stationSplineS[st]) < 16.0f) {
                tr.s = g_stationSplineS[st];
                tr.lastStation = st;
                tr.dwellTimer = 3.5f;
                tr.speed = 0.0f;
                break;
            }
        }
    }

    // 2. 6 Korfez vapurunu guncelle
    for (int i = 0; i < NUM_FERRIES; i++) {
        FerryBoat& fb = g_ferries[i];
        if (fb.dwell > 0.0f) { fb.dwell -= dt; continue; }
        Vec2 a = g_pierSeaPos[fb.routeA];
        Vec2 b = g_pierSeaPos[fb.routeB];
        float dist = std::max(100.0f, len2(b - a));
        fb.t += (fb.dir * fb.speed * dt) / dist;
        if (fb.t >= 1.0f) { fb.t = 1.0f; fb.dir = -1; fb.dwell = 4.0f; }
        if (fb.t <= 0.0f) { fb.t = 0.0f; fb.dir = 1;  fb.dwell = 4.0f; }
        fb.pos = a + (b - a) * fb.t;
        fb.fwd = norm2((b - a) * (float)fb.dir);
    }

    // 3. Otoyol araclarini guncelle
    const std::vector<RoadSpline>& routes = activeTrafficRoutes();
    for (auto& v : g_vehicles) {
        if (v.roadIdx < 0 || v.roadIdx >= (int)routes.size()) continue;
        const RoadSpline& rs = routes[v.roadIdx];
        v.s += v.dir * v.speed * dt;
        if (v.s > rs.totalLen) v.s -= rs.totalLen;
        if (v.s < 0.0f) v.s += rs.totalLen;
    }
}

static void renderDynamicTrafficAndBeacons(float simTime, int timeOfDay, const Vec3& eye, const Vec2& camFwd2D) {
    auto isVisible2D = [&](float wx, float wz, float maxDist, float minCos) -> bool {
        float dx = wx - eye.x, dz = wz - eye.z;
        float d2 = dx * dx + dz * dz;
        if (d2 > maxDist * maxDist) return false;
        if (d2 > 750.0f * 750.0f) {
            float d = std::sqrt(d2);
            if ((dx * camFwd2D.x + dz * camFwd2D.z) < d * minCos) return false;
        }
        return true;
    };

    // 1. 28 IZBAN Istasyonu: Zemin-Kule arasi Dikey Isik Sutunu + 75m Donen 3D Elmas + 3D Etiket (mesafe + gorus konisi budamali)
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    for (int s = 0; s < NUM_STATIONS; s++) {
        Vec3 sp = g_stationPos[s];
        if (!isVisible2D(sp.x, sp.z, 13000.0f, 0.20f)) continue;
        float bw = 2.2f;
        glColor4f(0.10f, 0.90f, 1.0f, 0.28f);
        glVertex3f(sp.x - bw, sp.y + 2.0f, sp.z);
        glVertex3f(sp.x + bw, sp.y + 2.0f, sp.z);
        glColor4f(0.10f, 0.95f, 1.0f, 0.85f);
        glVertex3f(sp.x + bw, sp.y + 75.0f, sp.z);
        glVertex3f(sp.x - bw, sp.y + 75.0f, sp.z);

        glVertex3f(sp.x, sp.y + 2.0f, sp.z - bw);
        glVertex3f(sp.x, sp.y + 2.0f, sp.z + bw);
        glVertex3f(sp.x, sp.y + 75.0f, sp.z + bw);
        glVertex3f(sp.x, sp.y + 75.0f, sp.z - bw);
    }
    glEnd();
    glDisable(GL_BLEND);

    float rotA = simTime * 2.4f;
    float ca = std::cos(rotA), sa = std::sin(rotA);
    glBegin(GL_TRIANGLES);
    for (int s = 0; s < NUM_STATIONS; s++) {
        Vec3 sp = g_stationPos[s];
        if (!isVisible2D(sp.x, sp.z, 11000.0f, 0.20f)) continue;
        Vec3 c = sp + Vec3(0.0f, 75.0f + std::sin(simTime * 3.0f + s) * 2.5f, 0.0f);
        float r = 7.5f, h = 12.0f;
        Vec3 top = c + Vec3(0, h, 0), bot = c - Vec3(0, h, 0);
        Vec3 p0(c.x + ca * r, c.y, c.z + sa * r);
        Vec3 p1(c.x - sa * r, c.y, c.z + ca * r);
        Vec3 p2(c.x - ca * r, c.y, c.z - sa * r);
        Vec3 p3(c.x + sa * r, c.y, c.z - ca * r);
        Vec3 ring[5] = {p0, p1, p2, p3, p0};
        for (int k = 0; k < 4; k++) {
            glColor3f(0.15f, 0.92f, 1.0f);
            glVertex3f(top.x, top.y, top.z); glVertex3f(ring[k].x, ring[k].y, ring[k].z); glVertex3f(ring[k + 1].x, ring[k + 1].y, ring[k + 1].z);
            glColor3f(0.08f, 0.58f, 0.98f);
            glVertex3f(bot.x, bot.y, bot.z); glVertex3f(ring[k + 1].x, ring[k + 1].y, ring[k + 1].z); glVertex3f(ring[k].x, ring[k].y, ring[k].z);
        }
        for (int k = 0; k < 4; k++) {
            glColor3f(1.0f, 0.95f, 0.35f);
            glVertex3f(c.x, c.y + 4.5f, c.z);
            glVertex3f(c.x + (ring[k].x - c.x) * 0.45f, c.y, c.z + (ring[k].z - c.z) * 0.45f);
            glVertex3f(c.x + (ring[k + 1].x - c.x) * 0.45f, c.y, c.z + (ring[k + 1].z - c.z) * 0.45f);
        }
    }
    glEnd();

    for (int s = 0; s < NUM_STATIONS; s++) {
        Vec3 sp = g_stationPos[s];
        if (!isVisible2D(sp.x, sp.z, 6200.0f, 0.25f)) continue;
        char bTxt[64];
        std::snprintf(bTxt, sizeof(bTxt), "[%d/28] %s", s + 1, STATIONS[s].name);
        drawTextWorld(Vec3(sp.x, sp.y + 90.0f, sp.z), 0.0f, 0.85f, bTxt, 0.25f, 1.0f, 1.0f);
    }
    glEnable(GL_LIGHTING);

    // 2. 8 Adet 3-Vagonlu Mavi-Beyaz-Kirmizi 3D IZBAN Treni
    glBegin(GL_QUADS);
    for (int i = 0; i < NUM_IZBAN_TRAINS; i++) {
        const IzbanTrain& tr = g_trains[i];
        float trackLat = (tr.dir > 0) ? 2.2f : -2.2f;
        for (int car = -1; car <= 1; car++) {
            float carS = clampf(tr.s + car * 23.5f, 0.0f, g_izbanTotalLen);
            Vec3 cp; Vec2 ct;
            sampleIzbanAtS(carS, cp, ct);
            if (!isVisible2D(cp.x, cp.z, 5500.0f, 0.20f)) continue;
            Vec2 cn(-ct.z, ct.x);
            float cx = cp.x + cn.x * trackLat;
            float cz = cp.z + cn.z * trackLat;
            float cy = cp.y + 0.45f;
            glColor3f(0.10f, 0.34f, 0.78f);
            emitBoxFaces(cx, cy, cz, 3.0f, 22.0f, 1.2f, cn.x, cn.z, 1, 1);
            glColor3f(0.95f, 0.96f, 0.98f);
            emitBoxFaces(cx, cy + 1.2f, cz, 3.0f, 22.0f, 2.1f, cn.x, cn.z, 1, 1);
            glColor3f(0.88f, 0.14f, 0.16f);
            emitBoxFaces(cx, cy + 1.1f, cz, 3.06f, 22.1f, 0.32f, cn.x, cn.z, 1, 1);
        }
    }
    glEnd();

    // 3. 6 Adet Canli 3D IZDENIZ Katamaran / Vapur + Sivri Pruva + Kopuklu V-Su Izi
    for (int i = 0; i < NUM_FERRIES; i++) {
        const FerryBoat& fb = g_ferries[i];
        if (!isVisible2D(fb.pos.x, fb.pos.z, 12000.0f, 0.15f)) continue;
        Vec2 f = fb.fwd;
        Vec2 r(-f.z, f.x);
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        Vec2 stern = fb.pos - f * 20.0f;
        Vec2 wL = stern - f * 120.0f - r * 34.0f;
        Vec2 wR = stern - f * 120.0f + r * 34.0f;
        glBegin(GL_TRIANGLES);
        glColor4f(0.92f, 0.97f, 1.0f, 0.70f);
        glVertex3f(stern.x, 0.18f, stern.z);
        glColor4f(0.85f, 0.95f, 1.0f, 0.0f);
        glVertex3f(wL.x, 0.18f, wL.z);
        glVertex3f(wR.x, 0.18f, wR.z);
        glEnd();
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);

        // Sivri pruvali 3D Katamaran / Vapur Govdesi
        glBegin(GL_QUADS);
        glColor3f(0.10f, 0.28f, 0.64f);
        emitBoxFaces(fb.pos.x, 0.0f, fb.pos.z, 11.4f, 34.0f, 2.8f, r.x, r.z, 1, 1);
        Vec2 bowBase = fb.pos + f * 17.0f, bowTip = fb.pos + f * 25.0f;
        Vec2 bL = bowBase - r * 5.7f, bR = bowBase + r * 5.7f;
        glColor3f(0.12f, 0.34f, 0.72f);
        glVertex3f(bL.x, 0.0f, bL.z); glVertex3f(bowTip.x, 0.0f, bowTip.z);
        glVertex3f(bowTip.x, 3.2f, bowTip.z); glVertex3f(bL.x, 2.8f, bL.z);
        glVertex3f(bowTip.x, 0.0f, bowTip.z); glVertex3f(bR.x, 0.0f, bR.z);
        glVertex3f(bR.x, 2.8f, bR.z); glVertex3f(bowTip.x, 3.2f, bowTip.z);
        glColor3f(0.96f, 0.97f, 0.99f);
        emitBoxFaces(fb.pos.x, 2.8f, fb.pos.z, 10.4f, 30.0f, 3.8f, r.x, r.z, 1, 1);
        glColor3f(timeOfDay >= 1 ? 1.0f : 0.18f, timeOfDay >= 1 ? 0.88f : 0.44f, timeOfDay >= 1 ? 0.50f : 0.76f);
        emitBoxFaces(fb.pos.x, 3.7f, fb.pos.z, 10.6f, 24.0f, 1.5f, r.x, r.z, 1, 1);
        glColor3f(0.88f, 0.20f, 0.16f);
        emitBoxFaces(fb.pos.x - f.x * 4.0f, 6.6f, fb.pos.z - f.z * 4.0f, 4.2f, 8.5f, 2.4f, r.x, r.z, 1, 1);
        glEnd();
    }

    // 4. Akan Cift Yonlu 3D Otoyol Arac Trafigi (2.6km gorus konisi icindekiler tek geciste cizilir)
    struct VisVeh { Vec3 pos; Vec2 n; uint8_t kind; int dir; };
    static std::vector<VisVeh> visVehicles;
    visVehicles.clear();
    const std::vector<RoadSpline>& routes = activeTrafficRoutes();
    for (const auto& v : g_vehicles) {
        if (v.roadIdx < 0 || v.roadIdx >= (int)routes.size()) continue;
        const RoadSpline& rs = routes[v.roadIdx];
        Vec3 vp; Vec2 n;
        sampleTrafficVehiclePose(rs, v, vp, n);
        if (!isVisible2D(vp.x, vp.z, 2600.0f, 0.15f)) continue;
        visVehicles.push_back({vp, n, v.kind, v.dir});
    }

    glBegin(GL_QUADS);
    for (const auto& vv : visVehicles) {
        if (vv.kind == 1) {
            glColor3f(0.88f, 0.14f, 0.14f);
            emitBoxFaces(vv.pos.x, vv.pos.y, vv.pos.z, 2.6f, 12.0f, 3.2f, vv.n.x, vv.n.z, 1, 1);
        } else if (vv.kind == 2) {
            glColor3f(0.24f, 0.52f, 0.84f);
            emitBoxFaces(vv.pos.x, vv.pos.y, vv.pos.z, 2.6f, 14.5f, 3.8f, vv.n.x, vv.n.z, 1, 1);
        } else {
            glColor3f(0.82f, 0.84f, 0.88f);
            emitBoxFaces(vv.pos.x, vv.pos.y, vv.pos.z, 2.0f, 4.6f, 1.6f, vv.n.x, vv.n.z, 1, 1);
        }
    }
    glEnd();

    // Gunbatimi / Gece modunda arac farlari ve stop lambalari
    if (timeOfDay >= 1 && !visVehicles.empty()) {
        glDisable(GL_LIGHTING);
        glPointSize(3.8f);
        glBegin(GL_POINTS);
        for (const auto& vv : visVehicles) {
            if (vv.dir > 0) glColor3f(1.0f, 0.96f, 0.80f);
            else glColor3f(1.0f, 0.18f, 0.12f);
            glVertex3f(vv.pos.x, vv.pos.y + 1.15f, vv.pos.z);
        }
        glEnd();
        glEnable(GL_LIGHTING);
    }
}
// =====================================================================================
//  BOLUM 7 : 3 DETAYLI KESIT ORGULU (LOFTED) UCAK MODELI VE 240 Hz 6-DOF UCUS FIZIGI
// =====================================================================================
enum { AC_JET = 0, AC_CESSNA = 1, AC_GLIDER = 2, NUM_AIRCRAFT = 3 };

struct AircraftSpec {
    const char* name;
    float mass;          // kg
    float wingArea;      // m^2
    float wingSpan;      // m
    float maxThrust;     // N (normal tam gaz)
    float abThrust;      // N (Afterburner ile)
    float cd0;           // parazit surukleme
    float clSlope;       // dCL / dAlpha
    float clMax;
    float stallAlphaDeg;
    float rollRateMax;
    float pitchRateMax;
    float yawRateMax;
};

static const AircraftSpec AC_SPECS[NUM_AIRCRAFT] = {
    {"TÜRK YILDIZLARI F-5 SÜPERSONİK JET",          7400.0f, 26.0f, 10.2f, 56000.0f, 94000.0f, 0.021f, 4.9f, 1.55f, 18.5f, 2.2f, 1.45f, 0.85f},
    {"MSFS CESSNA 172 SKYHAWK SP (SİVİL UÇUŞ)",     1150.0f, 16.8f, 11.2f,  7800.0f, 11200.0f, 0.025f, 5.6f, 1.72f, 16.5f, 1.85f, 1.25f, 0.82f},
    {"DG-1000 AKROBASİ PLANÖRÜ (20M KANAT)",         620.0f, 17.5f, 20.0f,  1800.0f,  2600.0f, 0.012f, 6.1f, 1.75f, 14.5f, 1.35f, 0.95f, 0.65f}
};

enum {
    PT_SMOKE_RIBBON = 0,
    PT_FLAME        = 1,
    PT_BLACK_SMOKE  = 2,
    PT_SPARK        = 3,
    PT_WATER_SPRAY  = 4
};

struct SmokePoint {
    Vec3 pos;
    Vec3 vel;
    float life = 1.0f;
    float maxLife = 1.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f, size = 1.0f;
    float growthRate = 1.4f;
    uint8_t ptype = PT_SMOKE_RIBBON;
};

struct ExplosionEffect {
    Vec3 pos;
    Vec3 surfNorm{0, 1, 0};
    float age = 0.0f;
    float maxAge = 1.4f;
    float maxRadius = 24.0f;
    bool hitBuilding = false;
};

struct BulletProj {
    Vec3 pos;
    Vec3 prevPos;
    Vec3 vel;
    float life = 2.6f;
};

struct BombProj {
    Vec3 pos;
    Vec3 prevPos;
    Vec3 vel;
    float life = 14.0f;
    float trailAccum = 0.0f;
};

struct FireEmitterNode {
    Vec3 pos;
    Vec3 outwardNorm{0, 1, 0};
    float scale = 8.0f;
    float phase = 0.0f;
    bool isRoof = true;
};

struct ActiveFireZone {
    Vec3 center;
    float radius = 16.0f;
    float buildingScale = 12.0f;
    float timeLeft = 30.0f;
    float maxDuration = 30.0f;
    float emitAccum = 0.0f;
    int primaryBldgIdx = -1;
    std::vector<FireEmitterNode> nodes;
};

struct AircraftState {
    int type = AC_CESSNA; // Varsayilan: Microsoft Flight Simulator tarzı Cessna 172 Skyhawk SP
    Vec3 pos;
    Vec3 vel;
    // Gram-Schmidt ortonormal yonelim matrisi (Gimbal Lock'suz: right=+X, up=+Y, fwd=+Z ileri)
    Vec3 right{1, 0, 0};
    Vec3 up{0, 1, 0};
    Vec3 fwd{0, 0, -1};
    Vec3 omega{0, 0, 0}; // govde eksenlerinde acisal hiz (pitch=x, yaw=y, roll=z)

    float throttle = 0.0f;    // 0..1.25 (>0.82 Afterburner)
    float pitchInput = 0.0f;  // -1..+1
    float rollInput = 0.0f;   // -1..+1
    float yawInput = 0.0f;    // -1..+1
    float brakeInput = 0.0f;  // 0..1
    bool gearDown = true;
    float gearAnim = 1.0f;    // 0..1
    int flapStage = 0;        // 0, 1 (15 deg), 2 (30 deg)
    float flapDeg = 0.0f;
    bool smokeOn = false;
    bool droneMode = false;
    bool autopilot = false;
    int autopilotWaypoint = 0;
    bool crashed = false;
    bool onGround = true;

    // Telemetri
    float airspeedMs = 0.0f;
    float alphaDeg = 0.0f;
    float betaDeg = 0.0f;
    float gLoad = 1.0f;
    float vsiMs = 0.0f;
    float aglM = 0.0f;
    float ridgeLiftMs = 0.0f;
    float propAngle = 0.0f;
    float wheelDeflect[3] = {0.0f, 0.0f, 0.0f}; // burun, sol ana, sag ana oleo sıkışması
};
static AircraftState g_ac;
static std::vector<SmokePoint> g_particles;
static std::vector<ExplosionEffect> g_explosions;
static std::vector<BulletProj> g_bullets;
static std::vector<BombProj> g_bombs;
static std::vector<ActiveFireZone> g_activeFires;
static float g_smokeEmitTimer = 0.0f;
static const size_t MAX_PARTICLES = 2400;
static const size_t MAX_ACTIVE_FIRES = 16;
static const float  BOMB_FIRE_DURATION = 30.0f;
static size_t g_particleOverwriteCursor = 0;

// Kamera sarsintisi (ekran titremesi) ve silah/HUD nisan durumu
static float g_camShakeMag = 0.0f;
static float g_camShakeTimer = 0.0f;
static float g_camShakeDur = 1.0f;
static float g_camShakePhase = 0.0f;
static float g_lastShakeSimTime = 0.0f;

static Vec3 g_lastCamEye(0, 100, 0);
static Vec3 g_lastCamFwd(0, 0, -1);
static Vec3 g_lastCamRight(1, 0, 0);
static Vec3 g_lastCamUp(0, 1, 0);
static int  g_lastWinW = 1280, g_lastWinH = 720;

static float g_gunFireCd = 0.0f;
static float g_bombDropCd = 0.0f;
static bool  g_gunTriggerEdge = false;
static bool  g_bombTriggerEdge = false;
static bool  g_gunBarrelToggle = false;
static uint32_t g_weaponVfxSeq = 1;

static bool  g_gunAimValid = false;
static bool  g_gunAimHitBldg = false;
static Vec3  g_gunAimPos(0, 0, 0);
static float g_gunAimDist = 0.0f;

static bool  g_ccipValid = false;
static bool  g_ccipHitBldg = false;
static int   g_ccipHitBldgIdx = -1;
static Vec3  g_ccipPos(0, 0, 0);
static Vec3  g_ccipNorm(0, 1, 0);
static float g_ccipFallTime = 0.0f;

static void pushParticle(const SmokePoint& sp) {
    if (g_particles.size() < MAX_PARTICLES) {
        g_particles.push_back(sp);
    } else if (sp.ptype != PT_SMOKE_RIBBON) {
        g_particles[g_particleOverwriteCursor % MAX_PARTICLES] = sp;
        g_particleOverwriteCursor++;
    }
}

static void triggerCameraShake(float magnitude, float duration) {
    if (duration <= 0.0f || magnitude <= 0.0f) return;
    if (g_camShakeTimer > 0.0f) {
        float remFrac = clampf(g_camShakeTimer / std::max(0.01f, g_camShakeDur), 0.0f, 1.0f);
        float curActive = g_camShakeMag * remFrac;
        g_camShakeMag = std::max(curActive, magnitude);
    } else {
        g_camShakeMag = magnitude;
    }
    g_camShakeDur = std::max(g_camShakeTimer, duration);
    g_camShakeTimer = g_camShakeDur;
}

static bool projectWorldToHUD(const Vec3& wp, float& outX, float& outY) {
    Vec3 rel = wp - g_lastCamEye;
    float cz = dot(rel, g_lastCamFwd);
    if (cz <= 1.5f) return false;
    float cxCam = dot(rel, g_lastCamRight);
    float cyCam = dot(rel, g_lastCamUp);
    float tanHalfY = std::tan(31.0f * DEG2RAD);
    float aspect = (float)g_lastWinW / (float)std::max(1, g_lastWinH);
    float tanHalfX = tanHalfY * aspect;
    float ndcX = cxCam / (cz * tanHalfX);
    float ndcY = cyCam / (cz * tanHalfY);
    outX = g_lastWinW * 0.5f * (1.0f + ndcX);
    outY = g_lastWinH * 0.5f * (1.0f - ndcY);
    return (outX >= -80.0f && outX <= g_lastWinW + 80.0f &&
            outY >= -80.0f && outY <= g_lastWinH + 80.0f);
}

static inline Vec3 bldgLocalToWorld(const BuildingInst& b, float lx, float lyOffset, float lz) {
    return Vec3(
        b.x + lx * b.cosY - lz * b.sinY,
        b.y + lyOffset,
        b.z + lx * b.sinY + lz * b.cosY
    );
}

static inline Vec3 bldgSurfaceNormalAt(const BuildingInst& b, const Vec3& hitPt) {
    if (hitPt.y >= b.y + b.h - 2.2f) return Vec3(0.0f, 1.0f, 0.0f);
    float dx = hitPt.x - b.x, dz = hitPt.z - b.z;
    float lx = dx * b.cosY + dz * b.sinY;
    float lz = -dx * b.sinY + dz * b.cosY;
    float hw = std::max(0.5f, b.w * 0.5f);
    float hd = std::max(0.5f, b.d * 0.5f);
    float nlx = 0.0f, nlz = 0.0f;
    if (std::fabs(lx) / hw >= std::fabs(lz) / hd) {
        nlx = (lx >= 0.0f) ? 1.0f : -1.0f;
    } else {
        nlz = (lz >= 0.0f) ? 1.0f : -1.0f;
    }
    return Vec3(nlx * b.cosY - nlz * b.sinY, 0.0f, nlx * b.sinY + nlz * b.cosY);
}

// Tarama/Ray testi: Mermi, bomba ve HUD aim icin tum haritadaki binalara (OBB) ve araziye/denize carpisma testi
static bool traceWeaponSegment(const Vec3& p0, const Vec3& p1, float hitRadius,
                               Vec3& outPos, Vec3& outNorm, int& outBldgIdx, bool& outHitWater) {
    Vec3 delta = p1 - p0;
    float segLen = len(delta);
    if (segLen < 1e-4f) return false;
    int steps = std::max(1, (int)std::ceil(segLen / 1.5f));
    float prevT = 0.0f;
    for (int i = 1; i <= steps; i++) {
        float t = (float)i / (float)steps;
        Vec3 pt = p0 + delta * t;
        int bIdx = findBuildingHitPoint(pt, hitRadius);
        float terrY = sampleTerrainElev(pt.x, pt.z);
        bool hitB = (bIdx >= 0);
        bool hitT = (pt.y <= terrY + 0.15f);
        if (hitB || hitT) {
            float lo = prevT, hi = t;
            int bestB = bIdx;
            for (int it = 0; it < 5; it++) {
                float mid = 0.5f * (lo + hi);
                Vec3 mp = p0 + delta * mid;
                int mb = findBuildingHitPoint(mp, hitRadius);
                float mt = sampleTerrainElev(mp.x, mp.z);
                if (mb >= 0 || mp.y <= mt + 0.15f) {
                    hi = mid;
                    if (mb >= 0) bestB = mb;
                } else {
                    lo = mid;
                }
            }
            outPos = p0 + delta * hi;
            if (bestB >= 0) {
                const BuildingInst& b = g_allBuildings[bestB];
                outBldgIdx = bestB;
                outHitWater = false;
                outNorm = bldgSurfaceNormalAt(b, outPos);
                if (outNorm.y > 0.5f) outPos.y = b.y + b.h + 0.05f;
            } else {
                outBldgIdx = -1;
                float sd = sampleSignedSeaDist(outPos.x, outPos.z);
                outHitWater = (sd < -3.0f && terrY < 1.2f);
                outPos.y = outHitWater ? 0.05f : (sampleTerrainElev(outPos.x, outPos.z) + 0.08f);
                outNorm = outHitWater ? Vec3(0, 1, 0) : sampleTerrainNormal(outPos.x, outPos.z);
            }
            return true;
        }
        prevT = t;
    }
    return false;
}

static bool traceWeaponRay(const Vec3& origin, const Vec3& dirNorm, float maxDist, float hitRadius,
                           Vec3& outPos, Vec3& outNorm, int& outBldgIdx, bool& outHitWater) {
    float step = 6.0f;
    float tPrev = 0.0f;
    for (float t = step; t <= maxDist; t += step) {
        Vec3 pPrev = origin + dirNorm * tPrev;
        Vec3 pCur  = origin + dirNorm * t;
        if (traceWeaponSegment(pPrev, pCur, hitRadius, outPos, outNorm, outBldgIdx, outHitWater)) {
            return true;
        }
        tPrev = t;
    }
    outPos = origin + dirNorm * maxDist;
    outNorm = Vec3(0, 1, 0);
    outBldgIdx = -1;
    outHitWater = false;
    return false;
}

static void spawnExplosionAt(const Vec3& p, const Vec3& surfNorm, float scale,
                             bool doShake, bool hitBuilding, bool hitWater) {
    g_weaponVfxSeq += 131u;
    Rng r(g_weaponVfxSeq ^ (uint32_t)(std::fabs(p.x * 7.0f + p.z * 19.0f) + 1.0f));

    if (hitWater) {
        int splashCnt = (int)clampf(scale * 2.2f, 18.0f, 85.0f);
        for (int i = 0; i < splashCnt; i++) {
            float a = r.range(0.0f, 2.0f * PI_F);
            float hr = r.range(0.0f, scale * 0.42f);
            SmokePoint sp;
            sp.pos = p + Vec3(std::cos(a) * hr, r.range(0.2f, 1.5f), std::sin(a) * hr);
            sp.vel = Vec3(std::cos(a) * r.range(2.0f, scale * 0.5f),
                          r.range(scale * 0.55f, scale * 1.45f),
                          std::sin(a) * r.range(2.0f, scale * 0.5f));
            sp.maxLife = r.range(1.1f, 2.5f);
            sp.life = sp.maxLife;
            sp.r = r.range(0.84f, 0.96f);
            sp.g = r.range(0.92f, 0.99f);
            sp.b = 1.00f;
            sp.size = r.range(scale * 0.16f, scale * 0.34f);
            sp.growthRate = scale * 0.25f;
            sp.ptype = PT_WATER_SPRAY;
            pushParticle(sp);
        }
    } else {
        ExplosionEffect ex;
        ex.pos = p + surfNorm * 0.6f;
        ex.surfNorm = surfNorm;
        ex.age = 0.0f;
        ex.maxAge = clampf(0.85f + scale * 0.028f, 0.75f, 1.9f);
        ex.maxRadius = scale;
        ex.hitBuilding = hitBuilding;
        if (g_explosions.size() < 48) g_explosions.push_back(ex);

        int flameCnt = (int)clampf(scale * 1.7f, 12.0f, 68.0f);
        for (int i = 0; i < flameCnt; i++) {
            float a = r.range(0.0f, 2.0f * PI_F);
            float spRad = r.range(0.0f, scale * 0.36f);
            SmokePoint sp;
            sp.pos = p + Vec3(std::cos(a) * spRad, r.range(0.0f, scale * 0.22f), std::sin(a) * spRad)
                   + surfNorm * r.range(0.2f, scale * 0.16f);
            sp.vel = Vec3(std::cos(a) * r.range(2.0f, scale * 0.75f),
                          r.range(4.0f, scale * 0.95f),
                          std::sin(a) * r.range(2.0f, scale * 0.75f)) + surfNorm * (scale * 0.25f);
            sp.maxLife = r.range(0.45f, 1.35f);
            sp.life = sp.maxLife;
            float hot = r.uni();
            sp.r = 1.0f;
            sp.g = lerpf(0.32f, 0.92f, hot);
            sp.b = lerpf(0.04f, 0.42f, hot * hot);
            sp.size = r.range(scale * 0.22f, scale * 0.48f);
            sp.growthRate = scale * 0.28f;
            sp.ptype = PT_FLAME;
            pushParticle(sp);
        }

        int smokeCnt = (int)clampf(scale * 1.3f, 8.0f, 52.0f);
        for (int i = 0; i < smokeCnt; i++) {
            float a = r.range(0.0f, 2.0f * PI_F);
            float spRad = r.range(0.0f, scale * 0.40f);
            SmokePoint sp;
            sp.pos = p + Vec3(std::cos(a) * spRad, r.range(0.5f, scale * 0.35f), std::sin(a) * spRad);
            sp.vel = Vec3(std::cos(a) * r.range(1.5f, scale * 0.45f),
                          r.range(5.0f, scale * 0.85f),
                          std::sin(a) * r.range(1.5f, scale * 0.45f));
            sp.maxLife = r.range(2.4f, 5.2f);
            sp.life = sp.maxLife;
            float soot = r.range(0.09f, 0.22f);
            sp.r = soot; sp.g = soot; sp.b = soot + 0.02f;
            sp.size = r.range(scale * 0.28f, scale * 0.56f);
            sp.growthRate = scale * 0.24f;
            sp.ptype = PT_BLACK_SMOKE;
            pushParticle(sp);
        }

        int sparkCnt = (int)clampf(scale * 1.8f, 12.0f, 70.0f);
        for (int i = 0; i < sparkCnt; i++) {
            float a = r.range(0.0f, 2.0f * PI_F);
            float spd = r.range(scale * 0.5f, scale * 2.1f);
            SmokePoint sp;
            sp.pos = p + surfNorm * 0.8f;
            sp.vel = Vec3(std::cos(a) * spd, r.range(scale * 0.35f, scale * 1.65f), std::sin(a) * spd)
                   + surfNorm * (scale * 0.35f);
            sp.maxLife = r.range(0.75f, 2.1f);
            sp.life = sp.maxLife;
            sp.r = 1.0f;
            sp.g = r.range(0.45f, 0.88f);
            sp.b = r.range(0.08f, 0.28f);
            sp.size = 4.2f;
            sp.growthRate = 0.0f;
            sp.ptype = PT_SPARK;
            pushParticle(sp);
        }
    }

    if (doShake) {
        float camDist = len(p - g_lastCamEye);
        float mag = clampf((scale * 0.085f) * (240.0f / (camDist + 85.0f)), 0.12f, 2.8f);
        triggerCameraShake(mag, clampf(0.55f + scale * 0.024f, 0.55f, 1.65f));
    }
}

static void spawnBombFireZone(const Vec3& impactPos, int hitBldgIdx) {
    ActiveFireZone zone;
    zone.center = impactPos;
    zone.timeLeft = BOMB_FIRE_DURATION;
    zone.maxDuration = BOMB_FIRE_DURATION;
    zone.emitAccum = 0.0f;
    zone.primaryBldgIdx = hitBldgIdx;

    // Patlama yakinindaki tum binalari (OBB) topla ki ates gercek bina sekillerine ve kat seviyesine yayilsin
    std::vector<int> affectedBldgs;
    if (hitBldgIdx >= 0 && hitBldgIdx < (int)g_allBuildings.size()) {
        affectedBldgs.push_back(hitBldgIdx);
    }
    float scanR = 30.0f;
    int x0 = clampi((int)((impactPos.x - scanR - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int x1 = clampi((int)((impactPos.x + scanR - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
    int z0 = clampi((int)((impactPos.z - scanR - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    int z1 = clampi((int)((impactPos.z + scanR - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
    for (int gz = z0; gz <= z1; gz++) {
        for (int gx = x0; gx <= x1; gx++) {
            for (int idx : g_spatialHash[gz * SH_NX + gx]) {
                if ((int)affectedBldgs.size() >= 4) break;
                bool dup = false;
                for (int existing : affectedBldgs) if (existing == idx) { dup = true; break; }
                if (dup) continue;
                const BuildingInst& cand = g_allBuildings[idx];
                float dx = impactPos.x - cand.x, dz = impactPos.z - cand.z;
                float lx = dx * cand.cosY + dz * cand.sinY;
                float lz = -dx * cand.sinY + dz * cand.cosY;
                float cx = std::max(0.0f, std::fabs(lx) - cand.w * 0.5f);
                float cz = std::max(0.0f, std::fabs(lz) - cand.d * 0.5f);
                if (cx * cx + cz * cz <= 24.0f * 24.0f) {
                    affectedBldgs.push_back(idx);
                }
            }
        }
    }

    if (!affectedBldgs.empty()) {
        if (zone.primaryBldgIdx < 0) zone.primaryBldgIdx = affectedBldgs[0];
        float maxScale = 10.0f;
        float maxSpanR = 14.0f;
        Rng rng((uint32_t)(std::fabs(impactPos.x * 11.0f + impactPos.z * 29.0f) + 7.0f));

        for (size_t bi = 0; bi < affectedBldgs.size(); bi++) {
            const BuildingInst& b = g_allBuildings[affectedBldgs[bi]];
            float bLevel = clampf(std::max(b.h * 0.65f, 0.55f * (b.w + b.d)), 7.5f, 34.0f);
            if (bi == 0 || bLevel > maxScale) maxScale = bLevel;
            float radFoot = 0.5f * std::sqrt(b.w * b.w + b.d * b.d);
            maxSpanR = std::max(maxSpanR, radFoot + len(Vec3(b.x - impactPos.x, 0.0f, b.z - impactPos.z)));

            float hw = std::max(2.0f, b.w * 0.5f);
            float hd = std::max(2.0f, b.d * 0.5f);
            float nodeScale = clampf(0.45f * std::min(b.w, b.d) + b.h * 0.16f, 4.2f, 15.0f);

            // 1. Cati izgarasi (binanin gercek en/boy ve yonelim acisina tam uyan ates odaklar)
            int nx = clampi((int)std::round(b.w / 6.8f), 2, 5);
            int nz = clampi((int)std::round(b.d / 6.8f), 2, 5);
            for (int ix = 0; ix < nx; ix++) {
                float u = (nx == 1) ? 0.0f : (-0.78f + 1.56f * (float)ix / (float)(nx - 1));
                for (int iz = 0; iz < nz; iz++) {
                    float v = (nz == 1) ? 0.0f : (-0.78f + 1.56f * (float)iz / (float)(nz - 1));
                    FireEmitterNode fn;
                    fn.pos = bldgLocalToWorld(b, u * hw, b.h + 0.35f, v * hd);
                    fn.outwardNorm = Vec3(0, 1, 0);
                    fn.scale = nodeScale * rng.range(0.85f, 1.25f);
                    fn.phase = rng.range(0.0f, 6.28f);
                    fn.isRoof = true;
                    zone.nodes.push_back(fn);
                }
            }

            // 2. Cephe & pencere kat atesleri (yapi yuksekligine ve isabet katina gore 4 duvarda)
            float hitLocalY = (bi == 0) ? clampf(impactPos.y - b.y, b.h * 0.25f, b.h - 1.5f)
                                        : (b.h * 0.72f);
            int sidePts = clampi((int)std::round(0.22f * std::max(b.w, b.d)), 2, 4);
            int tierCnt = (b.h > 36.0f) ? 3 : 2;
            for (int side = 0; side < 4; side++) {
                float nlx = (side == 0) ? 1.0f : (side == 1 ? -1.0f : 0.0f);
                float nlz = (side == 2) ? 1.0f : (side == 3 ? -1.0f : 0.0f);
                Vec3 wNorm(nlx * b.cosY - nlz * b.sinY, 0.15f, nlx * b.sinY + nlz * b.cosY);
                for (int sp = 0; sp < sidePts; sp++) {
                    float tAlong = (sidePts == 1) ? 0.0f : (-0.70f + 1.40f * (float)sp / (float)(sidePts - 1));
                    float lx = (nlx != 0.0f) ? (nlx * (hw + 0.65f)) : (tAlong * hw);
                    float lz = (nlz != 0.0f) ? (nlz * (hd + 0.65f)) : (tAlong * hd);
                    for (int tier = 0; tier < tierCnt; tier++) {
                        float yOff = (tier == 0) ? hitLocalY
                                                 : clampf(hitLocalY - tier * 5.2f, 2.0f, b.h - 2.0f);
                        if (tier > 0 && std::fabs(yOff - hitLocalY) < 2.2f) continue;
                        FireEmitterNode fn;
                        fn.pos = bldgLocalToWorld(b, lx, yOff, lz);
                        fn.outwardNorm = norm(wNorm);
                        fn.scale = nodeScale * rng.range(0.68f, 1.05f);
                        fn.phase = rng.range(0.0f, 6.28f);
                        fn.isRoof = false;
                        zone.nodes.push_back(fn);
                    }
                }
            }
        }
        zone.radius = clampf(maxSpanR, 12.0f, 48.0f);
        zone.buildingScale = maxScale;
    } else {
        // Acik arazi / Tepe uzerinde yere yapisan genis yangin cemberi
        zone.radius = 18.0f;
        zone.buildingScale = 14.0f;
        Rng rng((uint32_t)(std::fabs(impactPos.x * 17.0f + impactPos.z * 31.0f) + 3.0f));
        for (int i = 0; i < 18; i++) {
            float a = (float)i * (2.0f * PI_F / 18.0f) + rng.range(-0.15f, 0.15f);
            float rad = (i < 4) ? rng.range(0.0f, 4.5f) : rng.range(5.0f, 16.0f);
            float gx = impactPos.x + std::cos(a) * rad;
            float gz = impactPos.z + std::sin(a) * rad;
            FireEmitterNode fn;
            fn.pos = Vec3(gx, sampleTerrainElev(gx, gz) + 0.35f, gz);
            fn.outwardNorm = Vec3(0, 1, 0);
            fn.scale = rng.range(4.8f, 8.8f);
            fn.phase = rng.range(0.0f, 6.28f);
            fn.isRoof = true;
            zone.nodes.push_back(fn);
        }
    }

    if (g_activeFires.size() >= MAX_ACTIVE_FIRES) {
        g_activeFires.erase(g_activeFires.begin());
    }
    g_activeFires.push_back(std::move(zone));
}

static void computeWeaponAiming() {
    if (g_ac.crashed) {
        g_gunAimValid = false;
        g_ccipValid = false;
        return;
    }
    // 1. Mermi / Top Nisangahi (Boresight ray trace - tam bina OBB + arazi)
    Vec3 gunOrigin = g_ac.pos + g_ac.fwd * 4.5f;
    Vec3 dummyNorm;
    int hitB = -1;
    bool hitW = false;
    bool hit = traceWeaponRay(gunOrigin, norm(g_ac.fwd), 2000.0f, 0.6f,
                              g_gunAimPos, dummyNorm, hitB, hitW);
    g_gunAimValid = true;
    g_gunAimHitBldg = (hit && hitB >= 0);
    g_gunAimDist = len(g_gunAimPos - gunOrigin);

    // 2. Bomba CCIP (Surekli Hesaplanan Vurus Noktasi - gercek yercekimi ve bina fizigiyle)
    Vec3 bPos = g_ac.pos - g_ac.up * 1.4f + g_ac.fwd * 1.2f;
    Vec3 bVel = g_ac.vel + g_ac.fwd * 4.0f - g_ac.up * 3.0f;
    const float simDt = 0.08f;
    g_ccipValid = false;
    g_ccipHitBldg = false;
    g_ccipHitBldgIdx = -1;
    for (int step = 0; step < 120; step++) {
        Vec3 nextVel = bVel + Vec3(0.0f, -9.80665f, 0.0f) * simDt;
        nextVel -= nextVel * (0.0012f * len(nextVel) * simDt);
        Vec3 nextPos = bPos + (bVel + nextVel) * (0.5f * simDt);
        Vec3 hitPos, hitN;
        int bIdx = -1;
        bool wHit = false;
        if (traceWeaponSegment(bPos, nextPos, 1.2f, hitPos, hitN, bIdx, wHit)) {
            g_ccipValid = true;
            g_ccipPos = hitPos;
            g_ccipNorm = hitN;
            g_ccipHitBldgIdx = bIdx;
            g_ccipHitBldg = (bIdx >= 0);
            g_ccipFallTime = (step + 1) * simDt;
            break;
        }
        bPos = nextPos;
        bVel = nextVel;
        if (bPos.y < -20.0f) break;
    }
}

static void fireBulletStream() {
    if (g_ac.crashed) return;
    g_gunBarrelToggle = !g_gunBarrelToggle;
    float sideOff = g_gunBarrelToggle ? 0.95f : -0.95f;
    Vec3 muzzle = g_ac.pos + g_ac.fwd * 4.6f + g_ac.right * sideOff - g_ac.up * 0.25f;
    Vec3 dir = norm(g_ac.fwd);
    BulletProj bp;
    bp.pos = muzzle;
    bp.prevPos = muzzle - dir * 2.5f;
    bp.vel = g_ac.vel + dir * 740.0f;
    bp.life = 2.6f;
    g_bullets.push_back(bp);

    // Hafif namlu ates/duman parlama partikulu ve mikro titresim
    SmokePoint flash;
    flash.pos = muzzle + dir * 1.2f;
    flash.vel = g_ac.vel + dir * 25.0f;
    flash.maxLife = 0.09f;
    flash.life = 0.09f;
    flash.r = 1.0f; flash.g = 0.85f; flash.b = 0.35f;
    flash.size = 1.6f;
    flash.growthRate = 3.0f;
    flash.ptype = PT_FLAME;
    pushParticle(flash);
    if (g_camShakeTimer < 0.06f) triggerCameraShake(0.06f, 0.07f);
}

static void dropBomb() {
    if (g_ac.crashed) return;
    BombProj bm;
    bm.pos = g_ac.pos - g_ac.up * 1.4f + g_ac.fwd * 1.2f;
    bm.prevPos = bm.pos;
    bm.vel = g_ac.vel + g_ac.fwd * 4.0f - g_ac.up * 3.0f;
    bm.life = 14.0f;
    bm.trailAccum = 0.0f;
    g_bombs.push_back(bm);
    triggerCameraShake(0.18f, 0.14f);
}

static void updateWeaponsAndFires(float dt) {
    if (g_gunFireCd > 0.0f) g_gunFireCd = std::max(0.0f, g_gunFireCd - dt);
    if (g_bombDropCd > 0.0f) g_bombDropCd = std::max(0.0f, g_bombDropCd - dt);

    // Mermiler (swept bina OBB + arazi/deniz carpismasi)
    for (size_t i = 0; i < g_bullets.size();) {
        BulletProj& b = g_bullets[i];
        b.life -= dt;
        b.prevPos = b.pos;
        b.vel.y -= 4.5f * dt;
        b.pos += b.vel * dt;
        Vec3 hitPos, hitN;
        int bIdx = -1;
        bool hitWater = false;
        if (traceWeaponSegment(b.prevPos, b.pos, 0.65f, hitPos, hitN, bIdx, hitWater)) {
            spawnExplosionAt(hitPos, hitN, bIdx >= 0 ? 7.2f : 5.2f, false, (bIdx >= 0), hitWater);
            g_bullets[i] = g_bullets.back();
            g_bullets.pop_back();
            continue;
        }
        if (b.life <= 0.0f || b.pos.y < -15.0f) {
            g_bullets[i] = g_bullets.back();
            g_bullets.pop_back();
        } else {
            i++;
        }
    }

    // Bombalar (gercek yercekimi + surukleme + swept bina OBB fizigi)
    for (size_t i = 0; i < g_bombs.size();) {
        BombProj& bm = g_bombs[i];
        bm.life -= dt;
        bm.prevPos = bm.pos;
        Vec3 nextVel = bm.vel + Vec3(0.0f, -9.80665f, 0.0f) * dt;
        nextVel -= nextVel * (0.0012f * len(nextVel) * dt);
        bm.pos += (bm.vel + nextVel) * (0.5f * dt);
        bm.vel = nextVel;

        bm.trailAccum += dt;
        if (bm.trailAccum >= 0.035f) {
            bm.trailAccum = 0.0f;
            SmokePoint sp;
            sp.pos = bm.pos;
            sp.vel = Vec3(0, 0.6f, 0);
            sp.maxLife = 0.65f;
            sp.life = 0.65f;
            sp.r = 0.85f; sp.g = 0.86f; sp.b = 0.88f;
            sp.size = 1.2f;
            sp.growthRate = 1.8f;
            sp.ptype = PT_BLACK_SMOKE;
            pushParticle(sp);
        }

        Vec3 hitPos, hitN;
        int bIdx = -1;
        bool hitWater = false;
        if (traceWeaponSegment(bm.prevPos, bm.pos, 1.2f, hitPos, hitN, bIdx, hitWater)) {
            float expScale = 34.0f;
            if (bIdx >= 0 && bIdx < (int)g_allBuildings.size()) {
                const BuildingInst& hb = g_allBuildings[bIdx];
                expScale = clampf(std::max(hb.h * 0.65f, 0.5f * (hb.w + hb.d)) + 16.0f, 24.0f, 52.0f);
            }
            spawnExplosionAt(hitPos, hitN, expScale, true, (bIdx >= 0), hitWater);
            if (!hitWater) {
                spawnBombFireZone(hitPos, bIdx);
            }
            g_bombs[i] = g_bombs.back();
            g_bombs.pop_back();
            continue;
        }
        if (bm.life <= 0.0f || bm.pos.y < -20.0f) {
            g_bombs[i] = g_bombs.back();
            g_bombs.pop_back();
        } else {
            i++;
        }
    }

    // Patlama sok dalgasi zamanlayicilari
    for (size_t i = 0; i < g_explosions.size();) {
        g_explosions[i].age += dt;
        if (g_explosions[i].age >= g_explosions[i].maxAge) {
            g_explosions[i] = g_explosions.back();
            g_explosions.pop_back();
        } else {
            i++;
        }
    }

    // 30 saniyelik bina sekline dagilan yanginlar
    for (size_t i = 0; i < g_activeFires.size();) {
        ActiveFireZone& z = g_activeFires[i];
        z.timeLeft -= dt;
        if (z.timeLeft <= 0.0f) {
            g_activeFires[i] = g_activeFires.back();
            g_activeFires.pop_back();
            continue;
        }
        float lifeFrac = clampf(z.timeLeft / std::max(1.0f, z.maxDuration), 0.0f, 1.0f);
        float intensity = (lifeFrac > 0.14f) ? 1.0f : (lifeFrac / 0.14f);
        z.emitAccum += dt;
        if (z.emitAccum >= 0.075f && !z.nodes.empty()) {
            z.emitAccum = 0.0f;
            g_weaponVfxSeq += 37u;
            Rng r(g_weaponVfxSeq + (uint32_t)(i * 97));
            int bursts = clampi((int)z.nodes.size() / 4, 2, 7);
            for (int b = 0; b < bursts; b++) {
                const FireEmitterNode& fn = z.nodes[(size_t)r.range(0.0f, (float)z.nodes.size() - 0.01f)];
                float s = fn.scale * (0.45f + 0.55f * intensity);

                SmokePoint fl;
                fl.pos = fn.pos + Vec3(r.range(-1.2f, 1.2f), r.range(0.2f, s * 0.35f), r.range(-1.2f, 1.2f));
                fl.vel = fn.outwardNorm * r.range(0.8f, 3.2f)
                       + Vec3(r.range(-1.8f, 1.8f), r.range(4.5f, 4.5f + s * 1.15f), r.range(-1.8f, 1.8f));
                fl.maxLife = r.range(0.45f, 0.95f);
                fl.life = fl.maxLife;
                fl.r = 1.0f;
                fl.g = r.range(0.35f, 0.85f);
                fl.b = r.range(0.04f, 0.20f);
                fl.size = r.range(s * 0.42f, s * 0.78f);
                fl.growthRate = s * 0.35f;
                fl.ptype = PT_FLAME;
                pushParticle(fl);

                if (b % 2 == 0) {
                    SmokePoint sm;
                    sm.pos = fn.pos + Vec3(r.range(-1.5f, 1.5f), s * 0.55f, r.range(-1.5f, 1.5f));
                    sm.vel = Vec3(r.range(-1.6f, 1.6f), r.range(5.5f, 9.5f + s * 0.75f), r.range(-1.6f, 1.6f));
                    sm.maxLife = r.range(2.2f, 4.4f);
                    sm.life = sm.maxLife;
                    float soot = r.range(0.10f, 0.22f);
                    sm.r = soot; sm.g = soot; sm.b = soot + 0.01f;
                    sm.size = r.range(s * 0.52f, s * 0.95f);
                    sm.growthRate = s * 0.45f;
                    sm.ptype = PT_BLACK_SMOKE;
                    pushParticle(sm);
                }
            }
        }
        i++;
    }
}

static inline void drawBillboardQuad3D(const Vec3& c, const Vec3& rVec, const Vec3& uVec, float halfSz) {
    Vec3 rx = rVec * halfSz;
    Vec3 uy = uVec * halfSz;
    glTexCoord2f(0.0f, 0.0f); glVertex3f(c.x - rx.x - uy.x, c.y - rx.y - uy.y, c.z - rx.z - uy.z);
    glTexCoord2f(1.0f, 0.0f); glVertex3f(c.x + rx.x - uy.x, c.y + rx.y - uy.y, c.z + rx.z - uy.z);
    glTexCoord2f(1.0f, 1.0f); glVertex3f(c.x + rx.x + uy.x, c.y + rx.y + uy.y, c.z + rx.z + uy.z);
    glTexCoord2f(0.0f, 1.0f); glVertex3f(c.x - rx.x + uy.x, c.y - rx.y + uy.y, c.z - rx.z + uy.z);
}

static void renderExplosions3D(const Vec3& camRight, const Vec3& camUp) {
    if (g_explosions.empty()) return;
    bool hadFog = (glIsEnabled(GL_FOG) != GL_FALSE);
    glDisable(GL_FOG);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texParticle);

    glBegin(GL_QUADS);
    for (const auto& ex : g_explosions) {
        float t = clampf(ex.age / ex.maxAge, 0.0f, 1.0f);
        float coreSz = ex.maxRadius * (0.28f + 0.95f * std::sqrt(t));
        float glowA = (1.0f - t) * (1.0f - t);
        if (t < 0.35f) {
            float flashA = 1.0f - (t / 0.35f);
            glColor4f(1.0f * flashA, 0.96f * flashA, 0.78f * flashA, 1.0f);
            drawBillboardQuad3D(ex.pos, camRight, camUp, coreSz * 0.85f);
        }
        glColor4f(1.0f * glowA, 0.46f * glowA, 0.08f * glowA, 1.0f);
        drawBillboardQuad3D(ex.pos + Vec3(0.0f, coreSz * 0.25f, 0.0f), camRight, camUp, coreSz * 1.35f);
    }
    glEnd();

    // Genisleyen 3D yer/cephe sok dalgasi halkasi
    glDisable(GL_TEXTURE_2D);
    for (const auto& ex : g_explosions) {
        float t = clampf(ex.age / ex.maxAge, 0.0f, 1.0f);
        if (t > 0.72f) continue;
        float rOuter = ex.maxRadius * (0.25f + 1.95f * t);
        float rInner = rOuter * 0.72f;
        float a = (1.0f - t / 0.72f) * 0.85f;
        Vec3 n = norm(ex.surfNorm);
        Vec3 ref = (std::fabs(n.y) < 0.85f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        Vec3 u = norm(cross(ref, n));
        Vec3 v = norm(cross(n, u));
        glBegin(GL_QUAD_STRIP);
        for (int s = 0; s <= 24; s++) {
            float ang = (float)s * (2.0f * PI_F / 24.0f);
            Vec3 rad = u * std::cos(ang) + v * std::sin(ang);
            Vec3 pi = ex.pos + rad * rInner;
            Vec3 po = ex.pos + rad * rOuter;
            glColor4f(1.0f * a, 0.62f * a, 0.16f * a, a);
            glVertex3f(pi.x, pi.y, pi.z);
            glColor4f(0.25f * a, 0.06f * a, 0.0f, 0.0f);
            glVertex3f(po.x, po.y, po.z);
        }
        glEnd();
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glEnable(GL_LIGHTING);
    if (hadFog) glEnable(GL_FOG);
}

static void renderWeaponsAndFires3D(float simTime, const Vec3& camRight, const Vec3& camUp, const Vec3& camEye) {
    // 1. CCIP 3D hedef halkasi (yuzeye yerlesik, bomba dusus noktasi gostergesi)
    if (g_ccipValid && !g_ac.crashed) {
        glDisable(GL_CULL_FACE);
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_TEXTURE_2D);
        float pulse = 0.72f + 0.28f * std::sin(simTime * 7.0f);
        Vec3 n = norm(g_ccipNorm);
        Vec3 ref = (std::fabs(n.y) < 0.85f) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        Vec3 u = norm(cross(ref, n));
        Vec3 v = norm(cross(n, u));
        Vec3 c = g_ccipPos + n * 0.55f;
        float ringR = clampf(len(c - camEye) * 0.015f, 4.5f, 22.0f);
        glLineWidth(2.2f);
        glBegin(GL_LINE_LOOP);
        glColor4f(g_ccipHitBldg ? 1.0f : 1.0f, g_ccipHitBldg ? 0.24f : 0.78f, g_ccipHitBldg ? 0.15f : 0.18f, 0.85f * pulse);
        for (int s = 0; s < 24; s++) {
            float a = (float)s * (2.0f * PI_F / 24.0f);
            Vec3 p = c + (u * std::cos(a) + v * std::sin(a)) * ringR;
            glVertex3f(p.x, p.y, p.z);
        }
        glEnd();
        glBegin(GL_LINES);
        glVertex3f(c.x - u.x * ringR * 1.3f, c.y - u.y * ringR * 1.3f, c.z - u.z * ringR * 1.3f);
        glVertex3f(c.x + u.x * ringR * 1.3f, c.y + u.y * ringR * 1.3f, c.z + u.z * ringR * 1.3f);
        glVertex3f(c.x - v.x * ringR * 1.3f, c.y - v.y * ringR * 1.3f, c.z - v.z * ringR * 1.3f);
        glVertex3f(c.x + v.x * ringR * 1.3f, c.y + v.y * ringR * 1.3f, c.z + v.z * ringR * 1.3f);
        glEnd();
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }

    // 2. Ucaktan ateslenen Tracer Mermiler
    if (!g_bullets.empty()) {
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glLineWidth(2.6f);
        glBegin(GL_LINES);
        for (const auto& b : g_bullets) {
            Vec3 dir = norm(b.vel);
            Vec3 tail = b.pos - dir * 16.0f;
            glColor3f(1.0f, 0.92f, 0.55f);
            glVertex3f(b.pos.x, b.pos.y, b.pos.z);
            glColor3f(1.0f, 0.36f, 0.05f);
            glVertex3f(tail.x, tail.y, tail.z);
        }
        glEnd();
        glLineWidth(1.0f);
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
    }

    // 3. Havadaki 3D Aerodinamik Bombalar (kuyruk kanatcikli)
    if (!g_bombs.empty()) {
        glDisable(GL_CULL_FACE);
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_LIGHTING);
        for (const auto& bm : g_bombs) {
            Vec3 f = (len(bm.vel) > 1.0f) ? norm(bm.vel) : Vec3(0, -1, 0);
            Vec3 r = norm(cross(f, std::fabs(f.y) < 0.9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0)));
            Vec3 u = norm(cross(r, f));
            glPushMatrix();
            float m[16] = {
                r.x, r.y, r.z, 0.0f,
                u.x, u.y, u.z, 0.0f,
                f.x, f.y, f.z, 0.0f,
                bm.pos.x, bm.pos.y, bm.pos.z, 1.0f
            };
            glMultMatrixf(m);
            glBegin(GL_QUADS);
            glColor3f(0.18f, 0.22f, 0.16f);
            // Govde ve konik burun + sari halka
            emitBoxFaces(0.0f, 0.0f, 0.0f, 0.42f, 0.42f, 1.85f, 1, 1, 1, 1);
            glColor3f(0.95f, 0.78f, 0.12f);
            emitBoxFaces(0.0f, 0.0f, 0.72f, 0.44f, 0.44f, 0.28f, 1, 1, 1, 1);
            glColor3f(0.14f, 0.16f, 0.14f);
            emitBoxFaces(0.0f, 0.0f, 1.25f, 0.28f, 0.28f, 0.52f, 1, 1, 1, 1);
            emitBoxFaces(0.0f, 0.0f, -1.15f, 0.95f, 0.08f, 0.45f, 1, 1, 1, 1);
            emitBoxFaces(0.0f, 0.0f, -1.15f, 0.08f, 0.95f, 0.45f, 1, 1, 1, 1);
            glEnd();
            glPopMatrix();
        }
    }

    // 4. 30 Saniyelik Bina Sekline Dagilmis Aktif Alev Perdeleri (additive sprite billboardlari)
    if (!g_activeFires.empty()) {
        bool hadFog = (glIsEnabled(GL_FOG) != GL_FALSE);
        glDisable(GL_FOG);
        glDisable(GL_CULL_FACE);
        glDisable(GL_LIGHTING);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_texParticle);

        glBegin(GL_QUADS);
        for (const auto& z : g_activeFires) {
            if (len(z.center - camEye) > 15000.0f) continue;
            float lifeFrac = clampf(z.timeLeft / std::max(1.0f, z.maxDuration), 0.0f, 1.0f);
            float intensity = (lifeFrac > 0.14f) ? 1.0f : (lifeFrac / 0.14f);
            for (const auto& fn : z.nodes) {
                float flick = 0.78f + 0.28f * std::sin(simTime * 13.5f + fn.phase)
                                    + 0.14f * std::cos(simTime * 21.0f + fn.phase * 1.7f);
                float sz = fn.scale * intensity * clampf(flick, 0.55f, 1.25f);
                if (sz < 0.8f) continue;
                Vec3 flamePos = fn.pos + Vec3(0.0f, sz * 0.52f, 0.0f) + fn.outwardNorm * (fn.isRoof ? 0.0f : 0.45f);
                glColor4f(1.0f * intensity, 0.44f * intensity, 0.06f * intensity, 1.0f);
                drawBillboardQuad3D(flamePos, camRight, camUp, sz * 0.92f);
                // Icteki parlak sari-beyaz cekirdek
                glColor4f(1.0f * intensity, 0.88f * intensity, 0.34f * intensity, 1.0f);
                drawBillboardQuad3D(fn.pos + Vec3(0.0f, sz * 0.28f, 0.0f), camRight, camUp, sz * 0.48f);
            }
        }
        glEnd();

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        glEnable(GL_LIGHTING);
        if (hadFog) glEnable(GL_FOG);
    }
}

static void precompileNearChunks(const Vec3& focusPos, float radius);

// Gram-Schmidt ortonormalizasyon (Gimbal Lock'suz 6-DOF yonelim matrisi)
static void orthonormalizeBasis(Vec3& r, Vec3& u, Vec3& f) {
    f = norm(f);
    u = u - f * dot(u, f);
    if (len(u) < 1e-5f) u = Vec3(0, 1, 0);
    u = norm(u);
    r = norm(cross(f, u));
    u = norm(cross(r, f));
}

static void setOrientationFromYawPitchRoll(float yawDeg, float pitchDeg, float rollDeg) {
    float y = yawDeg * DEG2RAD, p = pitchDeg * DEG2RAD, rl = rollDeg * DEG2RAD;
    // yawDeg: kuzeyden saat yonunde pusula acisi (0 = Kuzey -Z, 90 = Dogu +X)
    // rl > 0: saga yatis (sag kanat asagi -u0, ust eksen saga +r0)
    Vec3 f0(std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p));
    Vec3 r0(std::cos(y), 0.0f, std::sin(y));
    Vec3 u0 = cross(r0, f0);
    Vec3 r = r0 * std::cos(rl) - u0 * std::sin(rl);
    Vec3 u = u0 * std::cos(rl) + r0 * std::sin(rl);
    g_ac.fwd = f0;
    g_ac.right = r;
    g_ac.up = u;
    orthonormalizeBasis(g_ac.right, g_ac.up, g_ac.fwd);
}

// 12+ Enine Kesit Orgulu (Lofted) Govde Cizim Yardimcisi
struct LoftSec {
    float z;       // burundan kuyruga (+Z ileri, -Z geri)
    float rx, ry;  // yatay ve dikey yaricap
    float yOff;    // merkez yukseklik kaymasi
    float topArch; // ust sirt/kanopi kubbeligi
};

static void drawLoftedFuselage(const LoftSec* secs, int nSec, int segsAround,
                               int canopyStart, int canopyEnd,
                               float rBody, float gBody, float bBody,
                               float rAccent, float gAccent, float bAccent) {
    const int M = segsAround;
    for (int i = 0; i < nSec - 1; i++) {
        const LoftSec& s0 = secs[i];
        const LoftSec& s1 = secs[i + 1];
        bool isCanopySeg = (i >= canopyStart && i < canopyEnd);
        float dz = std::max(0.05f, s0.z - s1.z);
        float drAvg = 0.5f * ((s1.rx - s0.rx) + (s1.ry - s0.ry)) / dz;
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= M; j++) {
            float ang = (float)j / (float)M * 2.0f * PI_F;
            float ca = std::cos(ang), sa = std::sin(ang);

            auto evalPt = [&](const LoftSec& s) {
                float yMul = (sa > 0.0f) ? (s.ry * s.topArch) : (s.ry * 0.72f);
                return Vec3(ca * s.rx, s.yOff + sa * yMul, s.z);
            };
            Vec3 p0 = evalPt(s0);
            Vec3 p1 = evalPt(s1);
            Vec3 n = norm(Vec3(ca, sa * 0.90f, drAvg));
            glNormal3f(n.x, n.y, n.z);

            if (isCanopySeg && sa > 0.25f) {
                float tint = 0.75f + 0.25f * sa;
                glColor3f(0.10f * tint, 0.28f * tint, 0.52f * tint);
            } else if (std::fabs(sa) < 0.22f) {
                glColor3f(rAccent, gAccent, bAccent);
            } else {
                glColor3f(rBody, gBody, bBody);
            }
            glVertex3f(p0.x, p0.y, p0.z);
            glVertex3f(p1.x, p1.y, p1.z);
        }
        glEnd();
    }
}

static void drawWheelCylinder3D(float cx, float cy, float cz, float rad, float halfW) {
    const int segs = 12;
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= segs; i++) {
        float a = (float)i / (float)segs * 2.0f * PI_F;
        float ca = std::cos(a), sa = std::sin(a);
        glNormal3f(0.0f, sa, ca);
        glColor3f(0.12f, 0.13f, 0.14f);
        glVertex3f(cx - halfW, cy + sa * rad, cz + ca * rad);
        glVertex3f(cx + halfW, cy + sa * rad, cz + ca * rad);
    }
    glEnd();
    for (int side = -1; side <= 1; side += 2) {
        glBegin(GL_TRIANGLE_FAN);
        glNormal3f((float)side, 0.0f, 0.0f);
        glColor3f(0.76f, 0.79f, 0.84f);
        glVertex3f(cx + side * halfW, cy, cz);
        for (int i = 0; i <= segs; i++) {
            float a = (float)i / (float)segs * 2.0f * PI_F;
            glColor3f(0.14f, 0.15f, 0.16f);
            glVertex3f(cx + side * halfW, cy + std::sin(a) * rad, cz + std::cos(a) * rad);
        }
        glEnd();
    }
}

// 3 Noktali Oleo-Pnomatik Inis Takimi Cizimi (3D silindirik lastikler ve jantlar)
static void drawLandingGear3D(float gearAnim, const float wheelDeflect[3], float wingSpan) {
    if (gearAnim <= 0.02f) return;
    glDisable(GL_TEXTURE_2D);
    float ext = gearAnim;
    float spread = std::min(2.2f, wingSpan * 0.18f);
    Vec3 mounts[3] = {
        Vec3(0.0f, -0.35f, 2.50f),      // Burun dikmesi
        Vec3(-spread, -0.35f, -1.25f),  // Sol ana dikme
        Vec3(+spread, -0.35f, -1.25f)   // Sag ana dikme
    };
    for (int w = 0; w < 3; w++) {
        float strutLen = (1.13f - clampf(wheelDeflect[w], 0.0f, 0.45f)) * ext;
        Vec3 top = mounts[w];
        Vec3 bot = top - Vec3(0.0f, strutLen, 0.0f);
        glColor3f(0.80f, 0.84f, 0.88f);
        glBegin(GL_QUADS);
        emitBoxFaces(top.x, bot.y, top.z, 0.11f, 0.11f, strutLen, 1, 0, 1, 1);
        glEnd();
        drawWheelCylinder3D(bot.x, bot.y - 0.14f, bot.z, (w == 0) ? 0.22f : 0.27f, 0.10f);
    }
}

// Simetrik Tapered Airfoil Stabilizer / Kanat Panel Yardimcisi
static void emitTaperedAirfoilPair(float x0, float y0, float z0F, float z0B, float t0,
                                   float x1, float y1, float z1F, float z1B, float t1,
                                   float rCol, float gCol, float bCol) {
    for (int side = -1; side <= 1; side += 2) {
        float s = (float)side;
        float rx0 = s * x0, rx1 = s * x1;
        glColor3f(rCol, gCol, bCol);
        glNormal3f(0.0f, 1.0f, 0.0f);
        glVertex3f(rx0, y0 + t0 * 0.5f, z0F);
        glVertex3f(rx1, y1 + t1 * 0.5f, z1F);
        glVertex3f(rx1, y1 + t1 * 0.5f, z1B);
        glVertex3f(rx0, y0 + t0 * 0.5f, z0B);

        glNormal3f(0.0f, -1.0f, 0.0f);
        glVertex3f(rx0, y0 - t0 * 0.5f, z0F);
        glVertex3f(rx0, y0 - t0 * 0.5f, z0B);
        glVertex3f(rx1, y1 - t1 * 0.5f, z1B);
        glVertex3f(rx1, y1 - t1 * 0.5f, z1F);

        glNormal3f(0.0f, 0.15f, 0.98f);
        glVertex3f(rx0, y0 - t0 * 0.5f, z0F);
        glVertex3f(rx1, y1 - t1 * 0.5f, z1F);
        glVertex3f(rx1, y1 + t1 * 0.5f, z1F);
        glVertex3f(rx0, y0 + t0 * 0.5f, z0F);

        glNormal3f(0.0f, 0.0f, -1.0f);
        glVertex3f(rx0, y0 - t0 * 0.5f, z0B);
        glVertex3f(rx0, y0 + t0 * 0.5f, z0B);
        glVertex3f(rx1, y1 + t1 * 0.5f, z1B);
        glVertex3f(rx1, y1 - t1 * 0.5f, z1B);

        glNormal3f(s, 0.0f, 0.0f);
        glVertex3f(rx1, y1 - t1 * 0.5f, z1F);
        glVertex3f(rx1, y1 - t1 * 0.5f, z1B);
        glVertex3f(rx1, y1 + t1 * 0.5f, z1B);
        glVertex3f(rx1, y1 + t1 * 0.5f, z1F);
    }
}

// 3 Detayli Kesit Orgulu (Lofted) + Parlak Kompozit/Metal Spekularli Ucak Modeli Render
static void renderAircraftModel(const AircraftState& ac, float simTime) {
    glPushMatrix();
    float m[16] = {
        ac.right.x, ac.right.y, ac.right.z, 0.0f,
        ac.up.x,    ac.up.y,    ac.up.z,    0.0f,
        ac.fwd.x,   ac.fwd.y,   ac.fwd.z,   0.0f,
        ac.pos.x,   ac.pos.y,   ac.pos.z,   1.0f
    };
    glMultMatrixf(m);
    glDisable(GL_TEXTURE_2D);

    // Ucakta parlak metalik/kompozit gunes yansimasi (specular highlight)
    float acSpec[4] = {0.85f, 0.88f, 0.94f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, acSpec);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 64.0f);

    float aileronAng = ac.rollInput * 22.0f * DEG2RAD;
    float elevAng    = ac.pitchInput * 20.0f * DEG2RAD;
    float rudderAng  = ac.yawInput * 24.0f * DEG2RAD;
    float flapAng    = ac.flapDeg * DEG2RAD;

    if (ac.type == AC_JET) {
        // 1. TURK YILDIZLARI / SOLOTURK SUPERSONIK JET (26 segmentli parlak govde + LERX + kanat ucu fuzeleri)
        static const LoftSec jetSecs[14] = {
            { 7.80f, 0.03f, 0.03f, -0.10f, 1.0f},
            { 6.80f, 0.24f, 0.22f, -0.08f, 1.0f},
            { 5.40f, 0.52f, 0.48f, -0.04f, 1.1f},
            { 4.10f, 0.68f, 0.62f,  0.04f, 1.55f},
            { 2.60f, 0.74f, 0.68f,  0.08f, 1.78f},
            { 1.10f, 0.82f, 0.66f,  0.04f, 1.35f},
            {-0.20f, 1.08f, 0.68f,  0.00f, 1.10f},
            {-1.80f, 1.15f, 0.70f,  0.00f, 1.05f},
            {-3.40f, 1.10f, 0.68f,  0.00f, 1.05f},
            {-4.80f, 0.98f, 0.64f,  0.00f, 1.02f},
            {-6.00f, 0.84f, 0.58f,  0.00f, 1.00f},
            {-6.80f, 0.68f, 0.52f,  0.00f, 1.00f},
            {-7.35f, 0.54f, 0.48f,  0.00f, 1.00f},
            {-7.65f, 0.44f, 0.42f,  0.00f, 1.00f}
        };
        drawLoftedFuselage(jetSecs, 14, 24, 3, 6, 0.95f, 0.96f, 0.98f, 0.88f, 0.12f, 0.14f);

        // Arka Titan Egzoz Nozulu Halkasi ve Kapali Turbin Diski
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= 20; j++) {
            float a = (float)j / 20.0f * 2.0f * PI_F;
            float ca = std::cos(a), sa = std::sin(a);
            glNormal3f(ca, sa, -0.25f);
            glColor3f(0.24f, 0.25f, 0.28f);
            glVertex3f(ca * 0.54f, sa * 0.48f, -7.30f);
            glColor3f(0.14f, 0.15f, 0.18f);
            glVertex3f(ca * 0.43f, sa * 0.41f, -7.68f);
        }
        glEnd();
        glBegin(GL_TRIANGLE_FAN);
        glNormal3f(0.0f, 0.0f, -1.0f);
        float glow = clampf(ac.throttle, 0.15f, 1.0f);
        glColor3f(0.35f + 0.65f * glow, 0.22f + 0.45f * glow, 0.14f + 0.20f * glow);
        glVertex3f(0.0f, 0.0f, -7.67f);
        for (int j = 0; j <= 20; j++) {
            float a = (float)j / 20.0f * 2.0f * PI_F;
            glColor3f(0.12f, 0.13f, 0.15f);
            glVertex3f(std::cos(a) * 0.43f, std::sin(a) * 0.41f, -7.67f);
        }
        glEnd();

        // Govde alti merkez harici yakit tanki + LERX + Kanatlar + Dikey kuyruk
        glBegin(GL_QUADS);
        glColor3f(0.92f, 0.93f, 0.96f);
        emitBoxFaces(0.0f, -0.58f, -1.35f, 0.44f, 3.80f, 0.42f, 1, 0, 1, 1);
        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            glColor3f(0.92f, 0.93f, 0.96f);
            emitBoxFaces(s * 0.88f, -0.22f, -0.80f, 0.42f, 2.80f, 0.48f, 1, 0, 1, 1);
            glNormal3f(0.0f, 0.0f, 1.0f);
            glColor3f(0.10f, 0.12f, 0.15f);
            glVertex3f(s * 0.68f, -0.20f, 0.62f); glVertex3f(s * 1.08f, -0.20f, 0.62f);
            glVertex3f(s * 1.08f,  0.24f, 0.62f); glVertex3f(s * 0.68f,  0.24f, 0.62f);

            // LERX (Leading-Edge Root Extension)
            glNormal3f(0.0f, 1.0f, 0.0f);
            glColor3f(0.94f, 0.95f, 0.98f);
            glVertex3f(s * 0.75f, 0.05f, 2.55f); glVertex3f(s * 1.25f, 0.04f, 1.05f);
            glVertex3f(s * 0.92f, 0.04f, 0.85f); glVertex3f(s * 0.72f, 0.05f, 1.45f);

            // Ana kanat ust/alt yuzey (Kirmizi-Beyaz Turk Yildizlari deseni)
            glNormal3f(0.0f, 1.0f, 0.0f);
            glColor3f(0.88f, 0.12f, 0.14f);
            glVertex3f(s * 0.92f,  0.04f,  0.85f); glVertex3f(s * 5.10f, -0.02f, -2.40f);
            glVertex3f(s * 5.10f, -0.02f, -3.65f); glVertex3f(s * 0.92f,  0.04f, -3.90f);

            glColor3f(0.97f, 0.98f, 1.00f);
            glVertex3f(s * 1.35f,  0.055f, -0.45f); glVertex3f(s * 4.45f,  0.005f, -2.55f);
            glVertex3f(s * 4.45f,  0.005f, -3.15f); glVertex3f(s * 1.35f,  0.055f, -1.85f);

            glNormal3f(0.0f, -1.0f, 0.0f);
            glColor3f(0.84f, 0.11f, 0.13f);
            glVertex3f(s * 0.92f, -0.05f,  0.85f); glVertex3f(s * 0.92f, -0.05f, -3.90f);
            glVertex3f(s * 5.10f, -0.08f, -3.65f); glVertex3f(s * 5.10f, -0.08f, -2.40f);

            glNormal3f(s * 0.6f, 0.1f, 0.8f);
            glColor3f(0.95f, 0.96f, 0.98f);
            glVertex3f(s * 0.92f, -0.05f,  0.85f); glVertex3f(s * 5.10f, -0.08f, -2.40f);
            glVertex3f(s * 5.10f, -0.02f, -2.40f); glVertex3f(s * 0.92f,  0.04f,  0.85f);

            // Kanat ucu fuze rayi + beyaz duman/muhimmat podu
            glColor3f(0.95f, 0.95f, 0.96f);
            emitBoxFaces(s * 5.15f, -0.06f, -2.85f, 0.14f, 2.25f, 0.18f, 1, 0, 1, 1);
            glColor3f(0.88f, 0.14f, 0.16f);
            emitBoxFaces(s * 5.18f, -0.05f, -2.55f, 0.12f, 2.65f, 0.12f, 1, 0, 1, 1);

            float dyAil = std::sin(aileronAng * s) * 0.28f;
            glNormal3f(0.0f, 1.0f, 0.0f);
            glColor3f(0.96f, 0.96f, 0.96f);
            glVertex3f(s * 3.20f, 0.0f, -3.72f); glVertex3f(s * 4.95f, -0.05f, -3.62f);
            glVertex3f(s * 4.95f, -0.05f + dyAil, -4.15f); glVertex3f(s * 3.20f, 0.0f + dyAil, -4.25f);

            float dyFlap = -std::sin(flapAng) * 0.35f;
            glColor3f(0.82f, 0.14f, 0.16f);
            glVertex3f(s * 1.05f, 0.0f, -3.85f); glVertex3f(s * 3.10f, 0.0f, -3.72f);
            glVertex3f(s * 3.10f, dyFlap, -4.32f); glVertex3f(s * 1.05f, dyFlap, -4.45f);

            float dyElev = std::sin(elevAng) * 0.36f;
            glColor3f(0.24f, 0.26f, 0.30f);
            glVertex3f(s * 0.72f, -0.05f, -5.40f); glVertex3f(s * 2.85f, -0.08f, -6.45f);
            glVertex3f(s * 2.85f, -0.08f + dyElev, -7.25f); glVertex3f(s * 0.72f, -0.05f + dyElev, -7.10f);
        }

        glColor3f(0.88f, 0.12f, 0.14f);
        emitBoxFaces(0.0f, 0.52f, -2.40f, 0.28f, 3.60f, 0.32f, 1, 0, 1, 1);
        float bW = 0.11f, tW = 0.05f;
        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            glNormal3f(s, 0.05f, 0.0f);
            glColor3f(0.88f, 0.12f, 0.14f);
            glVertex3f(s * bW, 0.45f, -3.50f); glVertex3f(s * tW, 2.85f, -6.25f);
            glVertex3f(s * tW, 2.85f, -7.20f); glVertex3f(s * bW, 0.45f, -7.10f);
            glColor3f(0.97f, 0.98f, 1.00f);
            glVertex3f(s * (bW * 0.82f + 0.004f), 1.25f, -4.60f); glVertex3f(s * (tW * 1.15f + 0.004f), 2.35f, -5.85f);
            glVertex3f(s * (tW * 1.15f + 0.004f), 2.35f, -6.95f); glVertex3f(s * (bW * 0.82f + 0.004f), 1.25f, -6.85f);
        }
        glNormal3f(0.0f, 0.65f, 0.75f);
        glColor3f(0.96f, 0.97f, 0.99f);
        glVertex3f(-bW, 0.45f, -3.50f); glVertex3f( bW, 0.45f, -3.50f);
        glVertex3f( tW, 2.85f, -6.25f); glVertex3f(-tW, 2.85f, -6.25f);
        glNormal3f(0.0f, 0.0f, -1.0f);
        glColor3f(0.88f, 0.12f, 0.14f);
        glVertex3f(-bW, 0.45f, -7.10f); glVertex3f( bW, 0.45f, -7.10f);
        glVertex3f( tW, 2.85f, -7.20f); glVertex3f(-tW, 2.85f, -7.20f);
        glNormal3f(0.0f, 1.0f, 0.0f);
        glColor3f(0.96f, 0.97f, 0.99f);
        glVertex3f(-tW, 2.85f, -6.25f); glVertex3f( tW, 2.85f, -6.25f);
        glVertex3f( tW, 2.85f, -7.20f); glVertex3f(-tW, 2.85f, -7.20f);

        float dxRud = -std::sin(rudderAng) * 0.38f;
        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            glNormal3f(s, 0.0f, -0.2f);
            glColor3f(0.96f, 0.96f, 0.98f);
            glVertex3f(s * (bW * 0.7f), 0.55f, -7.10f); glVertex3f(s * (tW * 0.8f), 2.75f, -7.20f);
            glVertex3f(dxRud + s * 0.025f, 2.75f, -7.75f); glVertex3f(dxRud + s * 0.025f, 0.55f, -7.65f);
        }
        glNormal3f(0.0f, 0.0f, -1.0f);
        glColor3f(0.92f, 0.14f, 0.16f);
        glVertex3f(dxRud - 0.03f, 0.55f, -7.65f); glVertex3f(dxRud + 0.03f, 0.55f, -7.65f);
        glVertex3f(dxRud + 0.03f, 2.75f, -7.75f); glVertex3f(dxRud - 0.03f, 2.75f, -7.75f);

        if (ac.brakeInput > 0.05f) {
            float brkH = ac.brakeInput * 0.65f;
            glColor3f(0.32f, 0.34f, 0.38f);
            glVertex3f(-0.38f, 0.62f, -3.20f); glVertex3f( 0.38f, 0.62f, -3.20f);
            glVertex3f( 0.38f, 0.62f + brkH, -4.35f); glVertex3f(-0.38f, 0.62f + brkH, -4.35f);
        }
        glEnd();

        if (ac.throttle > 0.80f && !ac.crashed) {
            float abPower = clampf((ac.throttle - 0.80f) / 0.45f, 0.0f, 1.0f);
            float flameLen = 2.5f + abPower * 4.8f + std::sin(simTime * 65.0f) * 0.35f;
            glDisable(GL_LIGHTING);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            glBegin(GL_TRIANGLE_FAN);
            glColor4f(0.45f, 0.82f, 1.0f, 0.92f);
            glVertex3f(0.0f, 0.0f, -7.60f);
            for (int k = 0; k <= 16; k++) {
                float a = k * (2.0f * PI_F / 16.0f);
                glColor4f(1.0f, 0.48f, 0.12f, 0.05f);
                glVertex3f(std::cos(a) * 0.48f, std::sin(a) * 0.48f, -7.60f - flameLen);
            }
            glEnd();
            for (int r = 1; r <= 5; r++) {
                float rz = -7.60f - (flameLen * 0.17f * r);
                float rad = (0.38f - r * 0.05f) * (0.85f + 0.15f * std::sin(simTime * 80.0f + r));
                glBegin(GL_TRIANGLE_FAN);
                glColor4f(1.0f, 0.95f, 0.75f, 0.88f * abPower);
                glVertex3f(0.0f, 0.0f, rz);
                for (int k = 0; k <= 12; k++) {
                    float a = k * (2.0f * PI_F / 12.0f);
                    glColor4f(0.25f, 0.72f, 1.0f, 0.25f * abPower);
                    glVertex3f(std::cos(a) * rad, std::sin(a) * rad, rz - 0.35f);
                }
                glEnd();
            }
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDisable(GL_BLEND);
            glEnable(GL_LIGHTING);
        }
        drawLandingGear3D(ac.gearAnim, ac.wheelDeflect, AC_SPECS[ac.type].wingSpan);
    } else if (ac.type == AC_CESSNA) {
        // 2. GELISMIS CESSNA 172 SKYHAWK SP (26 radial segment, gercek kanat & yatay stabilizator profili, 3D tekerlekler)
        static const LoftSec cessnaSecs[14] = {
            { 4.15f, 0.04f, 0.04f, -0.06f, 1.0f},
            { 3.72f, 0.28f, 0.26f, -0.05f, 1.0f},
            { 3.55f, 0.56f, 0.48f, -0.04f, 1.02f},
            { 2.35f, 0.66f, 0.56f,  0.00f, 1.08f},
            { 1.35f, 0.72f, 0.82f,  0.18f, 1.38f},
            { 0.15f, 0.72f, 0.82f,  0.18f, 1.38f},
            {-0.95f, 0.64f, 0.70f,  0.12f, 1.22f},
            {-1.85f, 0.52f, 0.54f,  0.04f, 1.06f},
            {-2.85f, 0.40f, 0.44f,  0.02f, 1.02f},
            {-3.85f, 0.30f, 0.34f,  0.01f, 1.00f},
            {-4.65f, 0.21f, 0.26f,  0.00f, 1.00f},
            {-5.25f, 0.14f, 0.20f,  0.00f, 1.00f},
            {-5.60f, 0.09f, 0.15f,  0.00f, 1.00f},
            {-5.82f, 0.04f, 0.10f,  0.00f, 1.00f}
        };
        drawLoftedFuselage(cessnaSecs, 14, 26, 3, 7, 0.97f, 0.98f, 1.00f, 0.10f, 0.32f, 0.72f);

        glBegin(GL_QUADS);
        // VHF sirt anteni + alt egzoz borusu
        glColor3f(0.86f, 0.88f, 0.92f);
        emitBoxFaces(0.0f, 0.95f, -0.55f, 0.03f, 0.14f, 0.42f, 1, 0, 1, 1);
        glColor3f(0.32f, 0.30f, 0.28f);
        emitBoxFaces(0.18f, -0.46f, 2.25f, 0.08f, 0.65f, 0.08f, 1, 0, 1, 1);

        glNormal3f(0.0f, 0.0f, 1.0f);
        glColor3f(0.12f, 0.14f, 0.18f);
        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            glVertex3f(s * 0.14f, -0.18f, 3.58f); glVertex3f(s * 0.42f, -0.18f, 3.58f);
            glVertex3f(s * 0.42f, -0.02f, 3.58f); glVertex3f(s * 0.14f, -0.02f, 3.58f);

            glNormal3f(s, 0.05f, 0.0f);
            glColor3f(0.88f, 0.70f, 0.22f);
            glVertex3f(s * 0.67f, -0.05f,  2.20f); glVertex3f(s * 0.73f,  0.02f,  0.10f);
            glVertex3f(s * 0.73f,  0.10f,  0.10f); glVertex3f(s * 0.67f,  0.03f,  2.20f);

            glVertex3f(s * 0.73f,  0.02f,  0.10f); glVertex3f(s * 0.31f,  0.04f, -3.85f);
            glVertex3f(s * 0.31f,  0.11f, -3.85f); glVertex3f(s * 0.73f,  0.10f,  0.10f);

            glColor3f(0.14f, 0.32f, 0.56f);
            glVertex3f(s * 0.725f, 0.26f,  1.25f); glVertex3f(s * 0.725f, 0.26f, -0.85f);
            glVertex3f(s * 0.680f, 0.68f, -0.75f); glVertex3f(s * 0.680f, 0.68f,  1.15f);
        }

        // Orta kanat kabin tavan birlesimi
        glColor3f(0.97f, 0.98f, 1.00f);
        emitBoxFaces(0.0f, 0.84f, 0.55f, 1.50f, 1.52f, 0.15f, 1, 0, 1, 1);

        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            float xRoot = s * 0.72f, xTip = s * 5.62f;
            float yRootU = 0.98f, yRootL = 0.83f;
            float yTipU  = 1.18f, yTipL  = 1.07f;
            float zRootF = 1.30f, zRootB = -0.22f;
            float zTipF  = 1.12f, zTipB  = -0.05f;

            glNormal3f(-s * 0.04f, 0.99f, 0.0f);
            glColor3f(0.97f, 0.98f, 1.00f);
            glVertex3f(xRoot, yRootU, zRootF);
            glVertex3f(xTip * 0.84f, lerpf(yRootU, yTipU, 0.84f), lerpf(zRootF, zTipF, 0.84f));
            glVertex3f(xTip * 0.84f, lerpf(yRootU, yTipU, 0.84f), lerpf(zRootB, zTipB, 0.84f));
            glVertex3f(xRoot, yRootU, zRootB);

            glColor3f(0.10f, 0.32f, 0.72f);
            glVertex3f(xTip * 0.84f, lerpf(yRootU, yTipU, 0.84f), lerpf(zRootF, zTipF, 0.84f));
            glVertex3f(xTip, yTipU, zTipF);
            glVertex3f(xTip, yTipU, zTipB);
            glVertex3f(xTip * 0.84f, lerpf(yRootU, yTipU, 0.84f), lerpf(zRootB, zTipB, 0.84f));

            glNormal3f(0.0f, -1.0f, 0.0f);
            glColor3f(0.93f, 0.95f, 0.98f);
            glVertex3f(xRoot, yRootL, zRootF); glVertex3f(xRoot, yRootL, zRootB);
            glVertex3f(xTip,  yTipL,  zTipB); glVertex3f(xTip,  yTipL,  zTipF);

            glNormal3f(0.0f, 0.15f, 0.98f);
            glColor3f(0.98f, 0.99f, 1.00f);
            glVertex3f(xRoot, yRootL, zRootF); glVertex3f(xTip,  yTipL,  zTipF);
            glVertex3f(xTip,  yTipU,  zTipF); glVertex3f(xRoot, yRootU, zRootF);

            glNormal3f(0.0f, 0.0f, -1.0f);
            glColor3f(0.90f, 0.92f, 0.96f);
            glVertex3f(xRoot, yRootL, zRootB); glVertex3f(xRoot, yRootU, zRootB);
            glVertex3f(xTip,  yTipU,  zTipB); glVertex3f(xTip,  yTipL,  zTipB);

            glNormal3f(s, 0.0f, 0.0f);
            glColor3f(0.10f, 0.32f, 0.72f);
            glVertex3f(xTip, yTipU, zTipF);
            glVertex3f(xTip + s * 0.16f, yTipL - 0.05f, zTipF - 0.10f);
            glVertex3f(xTip + s * 0.16f, yTipL - 0.05f, zTipB + 0.05f);
            glVertex3f(xTip, yTipU, zTipB);

            // Wing struts + cuffs
            glNormal3f(s * 0.5f, 0.5f, 0.5f);
            glColor3f(0.94f, 0.96f, 0.99f);
            glVertex3f(s * 0.66f, -0.22f, 0.62f); glVertex3f(s * 0.66f, -0.22f, 0.44f);
            glVertex3f(s * 3.45f,  1.02f, 0.48f); glVertex3f(s * 3.45f,  1.02f, 0.66f);

            glNormal3f(-s * 0.5f, -0.5f, -0.5f);
            glVertex3f(s * 0.66f, -0.28f, 0.62f); glVertex3f(s * 3.45f,  0.96f, 0.66f);
            glVertex3f(s * 3.45f,  0.96f, 0.48f); glVertex3f(s * 0.66f, -0.28f, 0.44f);

            float dyAil = std::sin(aileronAng * s) * 0.24f;
            float yA0 = lerpf(yRootU, yTipU, 0.58f), yA1 = lerpf(yRootU, yTipU, 0.95f);
            float zA0 = lerpf(zRootB, zTipB, 0.58f), zA1 = lerpf(zRootB, zTipB, 0.95f);
            glNormal3f(0.0f, 1.0f, 0.0f);
            glColor3f(0.92f, 0.94f, 0.98f);
            glVertex3f(s * 3.25f, yA0 - 0.04f, zA0); glVertex3f(s * 5.35f, yA1 - 0.04f, zA1);
            glVertex3f(s * 5.35f, yA1 - 0.04f + dyAil, zA1 - 0.38f); glVertex3f(s * 3.25f, yA0 - 0.04f + dyAil, zA0 - 0.38f);

            float dyFlap = -std::sin(flapAng) * 0.32f;
            float yF0 = lerpf(yRootU, yTipU, 0.14f), yF1 = lerpf(yRootU, yTipU, 0.55f);
            float zF0 = lerpf(zRootB, zTipB, 0.14f), zF1 = lerpf(zRootB, zTipB, 0.55f);
            glColor3f(0.14f, 0.38f, 0.78f);
            glVertex3f(s * 0.78f, yF0 - 0.05f, zF0); glVertex3f(s * 3.15f, yF1 - 0.05f, zF1);
            glVertex3f(s * 3.15f, yF1 - 0.05f + dyFlap, zF1 - 0.42f); glVertex3f(s * 0.78f, yF0 - 0.05f + dyFlap, zF0 - 0.42f);
        }

        // Gercek Geriye Daralan Yatay Stabilizator + Hareketli Elevator
        emitTaperedAirfoilPair(0.18f, 0.06f, -4.55f, -5.32f, 0.12f,
                               1.85f, 0.06f, -4.88f, -5.48f, 0.06f, 0.96f, 0.97f, 0.99f);
        float dyElev = std::sin(elevAng) * 0.28f;
        glNormal3f(0.0f, 1.0f, 0.0f);
        glColor3f(0.12f, 0.34f, 0.74f);
        glVertex3f(-1.82f, 0.09f, -5.47f); glVertex3f( 1.82f, 0.09f, -5.47f);
        glVertex3f( 1.72f, 0.09f + dyElev, -5.92f); glVertex3f(-1.72f, 0.09f + dyElev, -5.92f);

        // Swept Dikey Kuyruk + Kavisli Dorsal Fin + Rudder
        float vbW = 0.09f, vtW = 0.045f;
        glNormal3f(0.0f, 0.6f, 0.8f);
        glColor3f(0.96f, 0.97f, 0.99f);
        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            // Sirt yuzgeci fillet
            glNormal3f(s, 0.1f, 0.0f);
            glVertex3f(s * 0.08f, 0.18f, -2.95f); glVertex3f(s * 0.09f, 0.56f, -4.35f);
            glVertex3f(s * 0.09f, 0.12f, -4.35f); glVertex3f(s * 0.08f, 0.12f, -2.95f);

            glNormal3f(s, 0.04f, 0.0f);
            glColor3f(0.96f, 0.97f, 0.99f);
            glVertex3f(s * vbW, 0.12f, -4.15f); glVertex3f(s * vtW, 1.84f, -5.15f);
            glVertex3f(s * vtW, 1.84f, -5.75f); glVertex3f(s * vbW, 0.12f, -5.65f);

            glColor3f(0.10f, 0.32f, 0.72f);
            glVertex3f(s * (vbW + 0.004f), 0.55f, -4.45f); glVertex3f(s * (vtW + 0.004f), 1.55f, -5.05f);
            glVertex3f(s * (vtW + 0.004f), 1.55f, -5.68f); glVertex3f(s * (vbW + 0.004f), 0.55f, -5.58f);

            glColor3f(0.88f, 0.70f, 0.22f);
            glVertex3f(s * (vbW + 0.006f), 0.42f, -4.38f); glVertex3f(s * (vtW + 0.006f), 1.40f, -4.98f);
            glVertex3f(s * (vtW + 0.006f), 1.48f, -5.02f); glVertex3f(s * (vbW + 0.006f), 0.50f, -4.42f);
        }
        glNormal3f(0.0f, 0.55f, 0.83f);
        glColor3f(0.97f, 0.98f, 1.00f);
        glVertex3f(-vbW, 0.12f, -4.15f); glVertex3f( vbW, 0.12f, -4.15f);
        glVertex3f( vtW, 1.84f, -5.15f); glVertex3f(-vtW, 1.84f, -5.15f);

        glNormal3f(0.0f, 0.0f, -1.0f);
        glColor3f(0.10f, 0.32f, 0.72f);
        glVertex3f(-vbW, 0.12f, -5.65f); glVertex3f( vbW, 0.12f, -5.65f);
        glVertex3f( vtW, 1.84f, -5.75f); glVertex3f(-vtW, 1.84f, -5.75f);

        float dxRud = -std::sin(rudderAng) * 0.32f;
        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            glNormal3f(s, 0.0f, -0.2f);
            glColor3f(0.96f, 0.97f, 0.99f);
            glVertex3f(s * (vbW * 0.75f), 0.16f, -5.65f); glVertex3f(s * (vtW * 0.75f), 1.80f, -5.75f);
            glVertex3f(dxRud + s * 0.02f, 1.80f, -6.18f); glVertex3f(dxRud + s * 0.02f, 0.16f, -6.08f);
        }

        if (ac.gearAnim > 0.02f) {
            float ext = ac.gearAnim;
            float spread = 1.48f;
            float yBelly = -0.34f;
            for (int side = -1; side <= 1; side += 2) {
                float s = (float)side;
                float yWheel = yBelly - (1.02f - clampf(ac.wheelDeflect[side < 0 ? 1 : 2], 0.0f, 0.40f)) * ext;
                glColor3f(0.92f, 0.94f, 0.97f);
                glVertex3f(s * 0.45f, yBelly, -0.35f); glVertex3f(s * 0.45f, yBelly, -0.55f);
                glVertex3f(s * spread, yWheel + 0.15f, -0.58f); glVertex3f(s * spread, yWheel + 0.15f, -0.38f);
                glColor3f(0.97f, 0.98f, 1.00f);
                emitBoxFaces(s * spread, yWheel - 0.08f, -0.48f, 0.28f, 0.88f, 0.34f, 1, 0, 1, 1);
                glColor3f(0.10f, 0.32f, 0.72f);
                emitBoxFaces(s * spread, yWheel + 0.04f, -0.48f, 0.29f, 0.90f, 0.09f, 1, 0, 1, 1);
            }
            float yNoseW = yBelly - (1.02f - clampf(ac.wheelDeflect[0], 0.0f, 0.40f)) * ext;
            glColor3f(0.97f, 0.98f, 1.00f);
            emitBoxFaces(0.0f, yNoseW - 0.08f, 2.50f, 0.24f, 0.74f, 0.30f, 1, 0, 1, 1);
            glEnd();
            for (int side = -1; side <= 1; side += 2) {
                float yWheel = yBelly - (1.02f - clampf(ac.wheelDeflect[side < 0 ? 1 : 2], 0.0f, 0.40f)) * ext;
                drawWheelCylinder3D(side * spread, yWheel - 0.14f, -0.48f, 0.21f, 0.09f);
            }
            drawWheelCylinder3D(0.0f, yNoseW - 0.14f, 2.50f, 0.18f, 0.08f);
        } else {
            glEnd();
        }

        // Kanat ucu Navigasyon Isiklari + Strobe + Kuyruk Beacon
        glDisable(GL_LIGHTING);
        glPointSize(6.5f);
        glBegin(GL_POINTS);
        glColor3f(1.0f, 0.12f, 0.10f); glVertex3f(-5.68f, 1.12f, 0.55f);
        glColor3f(0.10f, 1.0f, 0.22f); glVertex3f( 5.68f, 1.12f, 0.55f);
        if (((int)(simTime * 4.0f)) & 1) {
            glColor3f(1.0f, 0.18f, 0.12f); glVertex3f(0.0f, 1.90f, -5.45f);
            glColor3f(1.0f, 1.0f, 1.0f);   glVertex3f(-5.70f, 1.14f, 0.40f);
            glColor3f(1.0f, 1.0f, 1.0f);   glVertex3f( 5.70f, 1.14f, 0.40f);
        }
        glEnd();
        glEnable(GL_LIGHTING);

        // 3-Palli Tapered McCauley Pervane + Blur Diski
        glPushMatrix();
        glTranslatef(0.0f, -0.05f, 3.85f);
        glPushMatrix();
        glRotatef(ac.propAngle * RAD2DEG, 0.0f, 0.0f, 1.0f);
        glBegin(GL_QUADS);
        for (int b = 0; b < 3; b++) {
            float a = b * (2.0f * PI_F / 3.0f);
            float ca = std::cos(a), sa = std::sin(a);
            float dx = ca, dy = sa, px = -sa, py = ca;
            glColor3f(0.15f, 0.17f, 0.20f);
            glNormal3f(0.0f, 0.0f, 1.0f);
            glVertex3f(dx * 0.16f - px * 0.07f, dy * 0.16f - py * 0.07f, 0.02f);
            glVertex3f(dx * 0.16f + px * 0.07f, dy * 0.16f + py * 0.07f, -0.02f);
            glVertex3f(dx * 1.05f + px * 0.03f, dy * 1.05f + py * 0.03f, 0.01f);
            glVertex3f(dx * 1.05f - px * 0.03f, dy * 1.05f - py * 0.03f, -0.01f);
        }
        glEnd();
        glPopMatrix();

        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(0.85f, 0.90f, 0.95f, 0.20f);
        glVertex3f(0.0f, 0.0f, 0.02f);
        for (int k = 0; k <= 24; k++) {
            float a = k * (2.0f * PI_F / 24.0f);
            glColor4f(0.92f, 0.95f, 0.98f, 0.05f);
            glVertex3f(std::cos(a) * 1.12f, std::sin(a) * 1.12f, 0.02f);
        }
        glEnd();
        glDisable(GL_BLEND);
        glEnable(GL_LIGHTING);
        glPopMatrix();
    } else {
        // 3. DG-1000 PLANOR (20 radial segment, cok kesitli yuksek aciklikli kanat + tapered T-kuyruk)
        static const LoftSec gliderSecs[12] = {
            { 4.20f, 0.04f, 0.04f, -0.06f, 1.0f},
            { 3.50f, 0.24f, 0.22f, -0.04f, 1.1f},
            { 2.40f, 0.38f, 0.36f,  0.02f, 1.45f},
            { 1.10f, 0.42f, 0.40f,  0.04f, 1.55f},
            {-0.20f, 0.38f, 0.36f,  0.02f, 1.30f},
            {-1.40f, 0.28f, 0.28f,  0.00f, 1.05f},
            {-2.60f, 0.20f, 0.20f,  0.00f, 1.00f},
            {-3.60f, 0.15f, 0.15f,  0.00f, 1.00f},
            {-4.40f, 0.12f, 0.12f,  0.00f, 1.00f},
            {-5.00f, 0.10f, 0.10f,  0.00f, 1.00f},
            {-5.40f, 0.08f, 0.08f,  0.00f, 1.00f},
            {-5.70f, 0.05f, 0.06f,  0.00f, 1.00f}
        };
        drawLoftedFuselage(gliderSecs, 12, 20, 2, 5, 0.98f, 0.98f, 1.0f, 0.92f, 0.18f, 0.14f);

        glBegin(GL_QUADS);
        emitTaperedAirfoilPair(0.35f, 0.05f, 0.35f, -0.65f, 0.11f,
                               5.60f, 0.22f, 0.12f, -0.66f, 0.08f, 0.98f, 0.98f, 1.00f);
        emitTaperedAirfoilPair(5.60f, 0.22f, 0.12f, -0.66f, 0.08f,
                               10.0f, 0.40f, -0.15f, -0.68f, 0.05f, 0.98f, 0.98f, 1.00f);
        for (int side = -1; side <= 1; side += 2) {
            float s = (float)side;
            float dyAil = std::sin(aileronAng * s) * 0.24f;
            glNormal3f(0.0f, 1.0f, 0.0f);
            glColor3f(0.90f, 0.16f, 0.14f);
            glVertex3f(s * 6.00f, 0.25f, -0.66f); glVertex3f(s * 9.85f, 0.40f, -0.68f);
            glVertex3f(s * 9.85f, 0.40f + dyAil, -0.95f); glVertex3f(s * 6.00f, 0.25f + dyAil, -0.95f);

            glVertex3f(s * 10.0f, 0.40f, -0.15f); glVertex3f(s * 10.22f, 0.98f, -0.32f);
            glVertex3f(s * 10.22f, 0.98f, -0.66f); glVertex3f(s * 10.0f, 0.40f, -0.68f);
        }
        emitBoxFaces(0.0f, 0.05f, -5.45f, 0.09f, 0.85f, 1.55f, 1, 0, 1, 1);
        emitTaperedAirfoilPair(0.08f, 1.56f, -5.22f, -5.75f, 0.08f,
                               1.42f, 1.56f, -5.35f, -5.72f, 0.04f, 0.96f, 0.97f, 0.99f);
        glEnd();
        drawLandingGear3D(ac.gearAnim, ac.wheelDeflect, AC_SPECS[ac.type].wingSpan);
    }

    float zeroSpec[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, zeroSpec);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0f);
    glPopMatrix();
}

// =====================================================================================
//  ISINLANMA NOKTALARI [1]..[9], [0] VE 240 Hz 6-DOF FIZIK MOTORU
// =====================================================================================
static Vec3 g_camFwd(0, 0, -1);
static Vec3 g_camUp(0, 1, 0);

static void syncCameraToAircraft() {
    g_camFwd = g_ac.fwd;
    g_camUp  = Vec3(0.0f, 1.0f, 0.0f);
}

static void teleportAircraft(int slot) {
    g_ac.crashed = false;
    g_collidersDirty = true;
    g_ac.omega = Vec3(0, 0, 0);
    g_ac.pitchInput = g_ac.rollInput = g_ac.yawInput = g_ac.brakeInput = 0.0f;
    g_ac.autopilot = false;
    g_ac.wheelDeflect[0] = g_ac.wheelDeflect[1] = g_ac.wheelDeflect[2] = 0.08f;
    g_ac.alphaDeg = 0.0f;
    g_ac.betaDeg = 0.0f;
    g_ac.vsiMs = 0.0f;
    g_ac.ridgeLiftMs = 0.0f;
    g_particles.clear();
    g_explosions.clear();
    g_bullets.clear();
    g_bombs.clear();
    g_activeFires.clear();
    g_camShakeMag = 0.0f;
    g_camShakeTimer = 0.0f;

    if (slot == 1) {
        // [1] Adnan Menderes LTBJ 34R Pist Basi (Yerde, Kalkisa Hazir)
        Vec2 ltbj = geoPt(LTBJ_LAT, LTBJ_LON);
        Vec2 dir16 = headingDir(LTBJ_HDG);
        Vec2 right(-dir16.z, dir16.x);
        Vec2 rwy34R = ltbj + right * (LTBJ_SEP * 0.5f) + dir16 * (LTBJ_LEN * 0.44f);
        float groundY = sampleTerrainElev(rwy34R.x, rwy34R.z);
        g_ac.pos = Vec3(rwy34R.x, groundY + 1.40f, rwy34R.z);
        setOrientationFromYawPitchRoll(LTBJ_HDG + 180.0f, 0.0f, 0.0f); // 345.8 derece kuzeybatiya hizali
        g_ac.vel = Vec3(0, 0, 0);
        g_ac.airspeedMs = 0.0f;
        g_ac.aglM = 1.40f;
        g_ac.throttle = 0.0f;
        g_ac.gearDown = true;
        g_ac.gearAnim = 1.0f;
        g_ac.flapStage = 1;
        g_ac.flapDeg = 15.0f;
        g_ac.onGround = true;
        g_ac.gLoad = 1.0f;
    } else if (slot == 6) {
        // [6] Cigli LTBL Pist Basi (Yerde, Kalkisa Hazir)
        Vec2 ltbl = geoPt(LTBL_LAT, LTBL_LON);
        Vec2 dir17 = headingDir(LTBL_HDG);
        Vec2 startP = ltbl + dir17 * (LTBL_LEN * 0.43f);
        float groundY = sampleTerrainElev(startP.x, startP.z);
        g_ac.pos = Vec3(startP.x, groundY + 1.40f, startP.z);
        setOrientationFromYawPitchRoll(LTBL_HDG + 180.0f, 0.0f, 0.0f); // 347 derece
        g_ac.vel = Vec3(0, 0, 0);
        g_ac.airspeedMs = 0.0f;
        g_ac.aglM = 1.40f;
        g_ac.throttle = 0.0f;
        g_ac.gearDown = true;
        g_ac.gearAnim = 1.0f;
        g_ac.flapStage = 1;
        g_ac.flapDeg = 15.0f;
        g_ac.onGround = true;
        g_ac.gLoad = 1.0f;
    } else {
        // Havada baslayan isinlanma noktalari ([2]..[5], [7]..[9], [0])
        double lat = SAAT_LAT, lon = SAAT_LON;
        float alt = 260.0f, hdg = 35.0f;
        switch (slot) {
            case 2: lat = SAAT_LAT;    lon = SAAT_LON;    alt = 220.0f; hdg = 25.0f;  break; // Konak Saat Kulesi
            case 3: lat = 38.4375;     lon = 27.1398;     alt = 230.0f; hdg = 35.0f;  break; // Alsancak Kordon/Liman
            case 4: lat = FOLKART_LAT; lon = FOLKART_LON; alt = 310.0f; hdg = 310.0f; break; // Bayrakli Folkart/Mistral
            case 5: lat = 38.4556;     lon = 27.1080;     alt = 240.0f; hdg = 285.0f; break; // Karsiyaka/Bostanli
            case 7: lat = STATIONS[1].lat; lon = STATIONS[1].lon; alt = 260.0f; hdg = 170.0f; break; // Egekent 2 IZBAN
            case 8: lat = STATIONS[2].lat; lon = STATIONS[2].lon; alt = 260.0f; hdg = 170.0f; break; // Ulukent IZBAN
            case 9: lat = STATIONS[0].lat; lon = STATIONS[0].lon; alt = 280.0f; hdg = 185.0f; break; // Menemen IZBAN
            case 0: lat = KULTURPARK_LAT;  lon = KULTURPARK_LON;  alt = 260.0f; hdg = 330.0f; break; // Bornova/Kulturpark
            default: break;
        }
        Vec2 p = geoPt(lat, lon);
        float terr = std::max(0.0f, sampleTerrainElev(p.x, p.z));
        g_ac.pos = Vec3(p.x, terr + alt, p.z);
        setOrientationFromYawPitchRoll(hdg, 1.8f, 0.0f);
        float cruiseSpd = (g_ac.type == AC_JET) ? 130.0f : (g_ac.type == AC_CESSNA ? 78.0f : 42.0f);
        g_ac.vel = g_ac.fwd * cruiseSpd;
        g_ac.airspeedMs = cruiseSpd;
        g_ac.aglM = alt;
        g_ac.alphaDeg = 1.8f;
        g_ac.throttle = (g_ac.type == AC_GLIDER) ? 0.0f : 0.68f;
        g_ac.gearDown = (g_ac.type == AC_CESSNA);
        g_ac.gearAnim = (g_ac.type == AC_CESSNA) ? 1.0f : 0.0f;
        g_ac.flapStage = 0;
        g_ac.flapDeg = 0.0f;
        g_ac.onGround = false;
        g_ac.gLoad = 1.0f;
    }
    syncCameraToAircraft();
    if (g_glReady) precompileNearChunks(g_ac.pos, 3200.0f);
}

// Yamanlar, Dumanlidag ve Balcova yamaclarinda ridge/termal kaldirma (updraft) hesabi
static float computeRidgeAndThermalLift(const Vec3& pos) {
    float terrY = sampleTerrainElev(pos.x, pos.z);
    float agl = pos.y - terrY;
    if (agl < 2.0f || agl > 1600.0f) return 0.0f;
    Vec3 n = sampleTerrainNormal(pos.x, pos.z);
    float horizSlope = std::sqrt(n.x * n.x + n.z * n.z);
    // Ege Imbat meltemi Korfez'den cevre dag yamaclarina (Kuzeyde Yamanlar'a -Z, Guneyde Balcova'ya +Z, Doguda Belkahve'ye +X) dogru yukselir
    Vec3 imbatDir = norm(Vec3(0.75f, 0.0f, (pos.z < 0.0f) ? -0.66f : 0.66f));
    float windward = std::max(0.0f, -dot(imbatDir, Vec3(n.x, 0.0f, n.z)));
    float ridgeUp = (windward * 14.0f + horizSlope * 9.5f) * std::exp(-agl / 580.0f);
    // Yamanlar, Dumanlidag ve Balcova gunesli kayalik sirtlarinda konvektif termal kolonlar
    float thermal = 0.0f;
    if (terrY > 45.0f) {
        float thNoise = 0.55f + 0.45f * perlin2(pos.x * 0.0012f, pos.z * 0.0012f);
        thermal = thNoise * clampf((terrY - 40.0f) / 220.0f, 0.2f, 1.4f) * 3.2f * std::exp(-agl / 950.0f);
    }
    return ridgeUp + thermal;
}

static void spawnCrashParticles(const Vec3& p) {
    bool hitBldg = checkAircraftBuildingCollision100m(p, g_ac.pos, 2.6f);
    bool hitWater = (!hitBldg && sampleSignedSeaDist(p.x, p.z) < -4.0f && p.y < 3.0f);
    Vec3 surfN(0.0f, 1.0f, 0.0f);
    if (hitBldg) {
        int bIdx = findBuildingHitPoint(p, 2.6f);
        if (bIdx >= 0) surfN = bldgSurfaceNormalAt(g_allBuildings[bIdx], p);
    } else if (!hitWater) {
        surfN = sampleTerrainNormal(p.x, p.z);
    }
    spawnExplosionAt(p, surfN, hitBldg ? 30.0f : 26.0f, true, hitBldg, hitWater);
}

static bool checkAircraftCrashBuildings100m(const AircraftSpec& spec) {
    refreshActiveColliders100m(g_ac.pos);
    if (g_activeColliders100m.empty()) return false;
    Vec3 testPts[4] = {
        g_ac.pos,
        g_ac.pos + g_ac.fwd * 4.2f,
        g_ac.pos - g_ac.right * (spec.wingSpan * 0.45f),
        g_ac.pos + g_ac.right * (spec.wingSpan * 0.45f)
    };
    for (int k = 0; k < 4; k++) {
        if (checkAircraftBuildingCollision100m(testPts[k], g_ac.pos, 1.85f)) {
            g_ac.crashed = true;
            spawnCrashParticles(testPts[k]);
            return true;
        }
    }
    return false;
}

// PHYS_DT = 1.0f/240.0f alt-adimli 6-DOF aerodinamik fizik adimi
static const float PHYS_DT = 1.0f / 240.0f;

static void stepFlightPhysicsSubstep(float dt) {
    const AircraftSpec& spec = AC_SPECS[g_ac.type];

    // Animasyonlar (inis takimi, flap, pervane)
    float targetGear = g_ac.gearDown ? 1.0f : 0.0f;
    g_ac.gearAnim += clampf(targetGear - g_ac.gearAnim, -0.8f * dt, 0.8f * dt);
    float targetFlap = (g_ac.flapStage == 1) ? 15.0f : (g_ac.flapStage == 2 ? 30.0f : 0.0f);
    g_ac.flapDeg += clampf(targetFlap - g_ac.flapDeg, -25.0f * dt, 25.0f * dt);
    g_ac.propAngle = wrapAngle(g_ac.propAngle + (35.0f + g_ac.throttle * 140.0f) * dt);

    if (g_ac.crashed) {
        g_ac.vel *= std::max(0.0f, 1.0f - 3.0f * dt);
        return;
    }

    // [F] Serbest Hizli Drone Modu (binalara carpisma 100m icinde aktif)
    if (g_ac.droneMode) {
        float spd = 380.0f * (0.25f + g_ac.throttle * 1.4f);
        Vec3 wDrone = g_ac.right * (g_ac.pitchInput * 1.6f)
                    + g_ac.up    * (-g_ac.yawInput * 1.4f)
                    + g_ac.fwd   * (g_ac.rollInput * 2.2f);
        g_ac.right += cross(wDrone, g_ac.right) * dt;
        g_ac.up    += cross(wDrone, g_ac.up)    * dt;
        g_ac.fwd   += cross(wDrone, g_ac.fwd)   * dt;
        orthonormalizeBasis(g_ac.right, g_ac.up, g_ac.fwd);
        g_ac.vel = g_ac.fwd * spd;
        g_ac.pos += g_ac.vel * dt;
        g_ac.pos.x = clampf(g_ac.pos.x, WORLD_X0 + 150.0f, WORLD_X1 - 150.0f);
        g_ac.pos.z = clampf(g_ac.pos.z, WORLD_Z0 + 150.0f, WORLD_Z1 - 150.0f);
        float terrY = std::max(0.0f, sampleTerrainElev(g_ac.pos.x, g_ac.pos.z));
        if (g_ac.pos.y < terrY + 0.85f) {
            if (g_ac.vel.y < -14.0f || sampleSignedSeaDist(g_ac.pos.x, g_ac.pos.z) < -5.0f) {
                g_ac.crashed = true;
                g_ac.pos.y = terrY + 0.90f;
                spawnCrashParticles(g_ac.pos);
            } else {
                g_ac.pos.y = terrY + 1.25f;
            }
        }
        g_ac.airspeedMs = spd;
        g_ac.vsiMs = g_ac.vel.y;
        g_ac.aglM = g_ac.pos.y - terrY;
        g_ac.gLoad = 1.0f;
        if (!g_ac.crashed) {
            checkAircraftCrashBuildings100m(spec);
        }
        return;
    }

    // Gercek yatis (bankRad > 0 saga yatis) ve yunuslama (pitchRad > 0 burun yukari) acilari
    float bankRad  = std::atan2(-g_ac.right.y, g_ac.up.y);
    float pitchRad = std::asin(clampf(g_ac.fwd.y, -1.0f, 1.0f));

    // [P] IZBAN ve Korfez Otopilot Turu
    float pCmd = g_ac.pitchInput;
    float rCmd = g_ac.rollInput;
    float yCmd = g_ac.yawInput;
    if (g_ac.autopilot) {
        Vec3 wp = g_stationPos[g_ac.autopilotWaypoint] + Vec3(0.0f, 220.0f, 0.0f);
        Vec3 toWp = wp - g_ac.pos;
        float dist2D = len2(Vec2(toWp.x, toWp.z));
        if (dist2D < 520.0f) {
            g_ac.autopilotWaypoint = (g_ac.autopilotWaypoint + 1) % NUM_STATIONS;
        }
        Vec2 desiredDir = norm2(Vec2(toWp.x, toWp.z));
        Vec2 curDir = norm2(Vec2(g_ac.fwd.x, g_ac.fwd.z));
        float hdgErr = std::atan2(cross2(curDir, desiredDir), clampf(dot2(curDir, desiredDir), -1.0f, 1.0f));
        float targetBank = clampf(hdgErr * 1.85f, -0.92f, 0.92f);
        rCmd = clampf((targetBank - bankRad) * 2.8f, -1.0f, 1.0f);

        float safeAlt = std::max(wp.y, sampleTerrainElev(g_ac.pos.x, g_ac.pos.z) + 150.0f);
        float targetPitch = clampf((safeAlt - g_ac.pos.y) * 0.0042f, -0.20f, 0.26f);
        pCmd = clampf((targetPitch - pitchRad) * 3.4f + std::fabs(bankRad) * 0.26f, -1.0f, 1.0f);
        yCmd = clampf(hdgErr * 0.35f, -0.45f, 0.45f);
        float targetSpd = (g_ac.type == AC_JET) ? 125.0f : (g_ac.type == AC_CESSNA ? 55.0f : 38.0f);
        g_ac.throttle = clampf(0.45f + (targetSpd - g_ac.airspeedMs) * 0.018f, 0.15f, 0.85f);
        g_ac.gearDown = false;
    }

    // Hava akimi ve Ridge/Termal kaldirma (updraft)
    g_ac.ridgeLiftMs = computeRidgeAndThermalLift(g_ac.pos);
    Vec3 airVel = g_ac.vel - Vec3(0.0f, g_ac.ridgeLiftMs, 0.0f);
    float V = len(airVel);
    g_ac.airspeedMs = V;

    // Govde eksenlerinde hiz bilesenleri ve gercek hucum acisi (alpha) + yan kayma (beta)
    float vFwd   = dot(airVel, g_ac.fwd);
    float vUp    = dot(airVel, g_ac.up);
    float vRight = dot(airVel, g_ac.right);

    float alpha = (V > 3.0f) ? std::atan2(-vUp, std::max(1.0f, std::fabs(vFwd))) : 0.0f;
    float beta  = (V > 3.0f) ? std::asin(clampf(vRight / V, -1.0f, 1.0f)) : 0.0f;
    g_ac.alphaDeg = alpha * RAD2DEG;
    g_ac.betaDeg  = beta * RAD2DEG;

    // Yumusak tanh Stall gecisi:
    // Kritik alpha uzerinde pre-stall lineer CL ile post-stall sin(2*alpha) arasinda yumusak tanh gecisi
    float alphaCrit = spec.stallAlphaDeg * DEG2RAD;
    float stallBlend = 0.5f * (1.0f + std::tanh((std::fabs(alpha) - alphaCrit) / (4.2f * DEG2RAD)));
    float flapCL = (g_ac.flapDeg / 30.0f) * 0.42f;
    float cl0 = 0.16f + flapCL;
    float clLinear = clampf(spec.clSlope * alpha + cl0, -spec.clMax, spec.clMax + flapCL);
    float clPostStall = 0.95f * std::sin(2.0f * alpha);
    float CL = lerpf(clLinear, clPostStall, stallBlend);

    // Yer Etkisi (Ground Effect): Kanat acikliginin altindaki irtifada kaldirma artar, induklenen surukleme azalir
    float terrY = std::max(0.0f, sampleTerrainElev(g_ac.pos.x, g_ac.pos.z));
    float agl = std::max(0.1f, g_ac.pos.y - terrY);
    g_ac.aglM = agl;
    float geRatio = agl / spec.wingSpan;
    float groundEffectLift = (geRatio < 1.0f) ? (1.0f + 0.24f * std::pow(1.0f - geRatio, 2.0f)) : 1.0f;
    float groundEffectDrag = (geRatio < 1.0f) ? (0.65f + 0.35f * geRatio) : 1.0f;
    CL *= groundEffectLift;

    float aspect = (spec.wingSpan * spec.wingSpan) / spec.wingArea;
    float cdi = (CL * CL) / (PI_F * aspect * 0.82f) * groundEffectDrag;
    float cdGear  = g_ac.gearAnim * 0.018f;
    float cdFlap  = (g_ac.flapDeg / 30.0f) * 0.024f;
    float cdBrake = g_ac.brakeInput * 0.065f;
    float cdStall = stallBlend * 0.45f;
    float CD = spec.cd0 + cdi + cdGear + cdFlap + cdBrake + cdStall;

    // Dinamik basinc q = 0.5 * rho * V^2
    float rho = 1.225f * std::exp(-std::max(0.0f, g_ac.pos.y) / 9200.0f);
    float qDyn = 0.5f * rho * V * V;

    Vec3 velDir = (V > 1.0f) ? (airVel * (1.0f / V)) : g_ac.fwd;
    Vec3 liftDir = norm(g_ac.up - velDir * dot(g_ac.up, velDir));
    Vec3 liftForce = liftDir * (qDyn * spec.wingArea * CL);
    Vec3 dragForce = -velDir * (qDyn * spec.wingArea * CD);
    Vec3 sideForce = -g_ac.right * (qDyn * spec.wingArea * beta * 0.95f);

    // Itki (W/S Gaz + Afterburner)
    float thrustN = 0.0f;
    if (g_ac.throttle <= 0.80f) {
        thrustN = (g_ac.throttle / 0.80f) * spec.maxThrust;
    } else {
        float abT = (g_ac.throttle - 0.80f) / (1.25f - 0.80f);
        thrustN = lerpf(spec.maxThrust, spec.abThrust, clampf(abT, 0.0f, 1.0f));
    }
    Vec3 thrustForce = g_ac.fwd * thrustN;
    Vec3 gravityForce(0.0f, -9.80665f * spec.mass, 0.0f);

    // 3 Noktali Yayli-Damperli Oleo-Pnomatik Tekerlek Suspansiyonu
    Vec3 gearForce(0, 0, 0);
    float gearYawTorque = 0.0f;
    g_ac.onGround = false;

    float spread = std::min(2.2f, spec.wingSpan * 0.18f);
    Vec3 wheelLocal[3] = {
        Vec3(0.0f, -1.48f, 2.50f),      // Burun tekeri
        Vec3(-spread, -1.48f, -1.25f),  // Sol ana teker
        Vec3(+spread, -1.48f, -1.25f)   // Sag ana teker
    };

    float cgGroundY = sampleTerrainElev(g_ac.pos.x, g_ac.pos.z);
    for (int w = 0; w < 3; w++) {
        g_ac.wheelDeflect[w] = 0.0f;
        if (g_ac.gearAnim < 0.5f) continue;
        Vec3 wp = g_ac.pos + g_ac.right * wheelLocal[w].x + g_ac.up * wheelLocal[w].y + g_ac.fwd * wheelLocal[w].z;
        float groundY = sampleTerrainElev(wp.x, wp.z);
        float pen = groundY - wp.y;
        if (pen > 0.0f) {
            g_ac.onGround = true;
            g_ac.wheelDeflect[w] = clampf(pen, 0.0f, 0.45f);
        }
    }

    if (g_ac.onGround && g_ac.gearAnim >= 0.5f) {
        // CG bazli oleo-pnomatik yay + damper dikey tepki kuvveti (burun rotasyonunda ziplatmaz)
        float cgPen = clampf((cgGroundY + 1.48f) - g_ac.pos.y, 0.0f, 0.55f);
        float kSpring = spec.mass * 122.5f; // cgPen = 0.08m'de tam 1.0G (m*g) agirligi dengeler
        float cDamper = spec.mass * 24.0f;
        float fnTotal = clampf(kSpring * cgPen - cDamper * g_ac.vel.y, 0.0f, spec.mass * 9.80665f * 3.0f);
        gearForce.y += fnTotal;

        // Yanal lastik tutunmasi ve tekerlek freni surtunmesi
        float latV = dot(g_ac.vel, g_ac.right);
        float fwdV = dot(g_ac.vel, g_ac.fwd);
        gearForce -= g_ac.right * clampf(latV * spec.mass * 8.0f, -fnTotal * 0.9f, fnTotal * 0.9f);
        float brakeCoef = (0.012f + g_ac.brakeInput * 0.62f);
        gearForce -= g_ac.fwd * (sgnf(fwdV) * std::min(std::fabs(fwdV) * spec.mass * 4.0f, fnTotal * brakeCoef));

        // Burun tekerlegi direksiyonu: Hem A/D (Sol/Sag Ok, Gamepad Sol Analog) hem Q/E ile yerde donus!
        float steerCmd = clampf(rCmd * 0.85f + yCmd, -1.0f, 1.0f);
        gearYawTorque = (-steerCmd * 1.15f) * clampf(std::fabs(fwdV) / 10.0f, 0.28f, 1.0f);
    }

    // SAS (Stability Augmentation System) + Otomatik Trim + MSFS Koordine Viraj Yasasi
    float ctrlAuth = clampf(V / 24.0f, 0.28f, 1.0f);
    float stallNoseDrop = (!g_ac.onGround) ? (-stallBlend * 0.55f * sgnf(alpha)) : 0.0f;

    // Yatisli virajda irtifa kaybini onleyen otomatik kaldirma telafisi: Lift = Weight / cos(bank)
    float bankCos = std::max(0.50f, std::cos(bankRad));
    if (!g_ac.onGround && std::fabs(bankRad) > 0.03f) {
        liftForce *= (1.0f / bankCos);
    }
    float reqCL = (qDyn > 22.0f)
        ? clampf((spec.mass * 9.80665f) / (qDyn * spec.wingArea * groundEffectLift), -0.15f, spec.clMax * 0.85f)
        : cl0;
    float alphaEq = clampf((reqCL - cl0) / spec.clSlope, -4.5f * DEG2RAD, 9.5f * DEG2RAD);

    float targetPitchRate = 0.0f;
    float targetYawRate   = 0.0f;
    float targetRollRate  = 0.0f;
    float turnRateWorldY  = 0.0f;

    if (g_ac.onGround) {
        // Pist uzerinde kanatlari tam yatay tut (sola/saga yatisi sifirla), kalkis hizinda yumusak rotasyon
        targetRollRate = -bankRad * 12.0f;
        float vRot = (g_ac.type == AC_CESSNA) ? 30.0f : 42.0f;
        float rotAuth = smooth01((V - (vRot - 6.0f)) / 12.0f);
        float autoRotateCmd = (g_ac.throttle > 0.60f && pCmd >= -0.05f)
            ? smooth01((V - vRot) / 9.0f) * 0.78f
            : 0.0f;
        float effPull = std::max(pCmd, autoRotateCmd);
        float desiredGroundPitch = (effPull > 0.0f) ? (effPull * 12.0f * DEG2RAD * rotAuth) : 0.0f;
        targetPitchRate = clampf((desiredGroundPitch - pitchRad) * 4.8f, -1.2f, 1.2f);
        targetYawRate = gearYawTorque - beta * 1.15f;
    } else {
        // MSFS Yatis-Acisi (Bank-Angle) & Koordine Donus Kontrolcusu:
        // Sol/Sag (A/D, Ok tuslari veya Gamepad Sol Analog) uçağı ±52 dereceye kadar yatirir,
        // asla ters (bas asagi) devirmez ve tus birakildiginda kanatlari yumusakca duzeltir!
        float maxBankRad = 52.0f * DEG2RAD;
        float desiredBank = rCmd * maxBankRad;
        float bankGain = (std::fabs(rCmd) > 0.05f) ? 4.4f : 2.2f;
        targetRollRate = clampf((desiredBank - bankRad) * bankGain, -spec.rollRateMax, spec.rollRateMax) * ctrlAuth;

        // Dunya Y ekseni etrafinda puruzsuz koordine donus hizi (saga yatista / D tusunda saga doner)
        turnRateWorldY = -(std::sin(bankRad) * 0.54f + rCmd * 0.24f + yCmd * 0.38f) * ctrlAuth;

        // MSFS Pitch-Attitude Zarf Korumasi:
        // Kullanici yukari/asagi tusu veya analog cubugu cektiginde ucagi guvenli tirmanis/alis acisina
        // (±28 derece sivil, ±45 derece jet) puruzsuzce oturtur, dikey stall'a girmesini engeller!
        float maxPitchRad = (g_ac.type == AC_JET) ? (45.0f * DEG2RAD) : (28.0f * DEG2RAD);
        if (std::fabs(pCmd) > 0.05f) {
            float desiredPitch = pCmd * maxPitchRad;
            targetPitchRate = clampf((desiredPitch - pitchRad) * 3.6f, -spec.pitchRateMax, spec.pitchRateMax) * ctrlAuth + stallNoseDrop;
        } else {
            float pitchTrimFb = (alpha - alphaEq) * 1.55f + (pitchRad - 0.025f) * 0.65f;
            targetPitchRate = -pitchTrimFb + stallNoseDrop;
        }
        targetYawRate = -beta * 1.85f;
    }

    float sasRate = 12.5f;
    g_ac.omega.x += (targetPitchRate - g_ac.omega.x) * std::min(1.0f, sasRate * dt);
    g_ac.omega.y += (targetYawRate   - g_ac.omega.y) * std::min(1.0f, sasRate * dt);
    g_ac.omega.z += (targetRollRate  - g_ac.omega.z) * std::min(1.0f, sasRate * dt);

    // Yonelim matrisini govde acisal hizi + Dunya Y ekseni koordine viraj acisal hiziyla entegre et
    Vec3 wWorld = g_ac.right * g_ac.omega.x
                + g_ac.up    * g_ac.omega.y
                + g_ac.fwd   * g_ac.omega.z
                + Vec3(0.0f, turnRateWorldY, 0.0f);
    g_ac.right += cross(wWorld, g_ac.right) * dt;
    g_ac.up    += cross(wWorld, g_ac.up)    * dt;
    g_ac.fwd   += cross(wWorld, g_ac.fwd)   * dt;
    if (g_ac.onGround) {
        g_ac.omega.z = 0.0f;
        g_ac.right.y = 0.0f;
    }
    orthonormalizeBasis(g_ac.right, g_ac.up, g_ac.fwd);

    // Koordine virajda yatay hiz vektorunu de burun yonuyle birlikte dondur (yan kaymayi sifirlar, aninda doner)
    if (!g_ac.onGround && std::fabs(turnRateWorldY) > 1e-4f) {
        float dYaw = turnRateWorldY * dt;
        float ca = std::cos(dYaw), sa = std::sin(dYaw);
        float vx = g_ac.vel.x, vz = g_ac.vel.z;
        g_ac.vel.x =  vx * ca + vz * sa;
        g_ac.vel.z = -vx * sa + vz * ca;
    }

    // Dogrusal ivme ve konum entegrasyonu
    Vec3 totalForce = thrustForce + liftForce + dragForce + sideForce + gravityForce + gearForce;
    Vec3 accel = totalForce * (1.0f / spec.mass);
    g_ac.vel += accel * dt;
    g_ac.pos += g_ac.vel * dt;
    g_ac.vsiMs = g_ac.vel.y;
    g_ac.gLoad = dot(totalForce - gravityForce, g_ac.up) / (spec.mass * 9.80665f);

    // Dunya sinirlarinda yumusak geri donus
    g_ac.pos.x = clampf(g_ac.pos.x, WORLD_X0 + 150.0f, WORLD_X1 - 150.0f);
    g_ac.pos.z = clampf(g_ac.pos.z, WORLD_Z0 + 150.0f, WORLD_Z1 - 150.0f);

    // Yer / Deniz ve 50.000+ Bina O(1) SpatialHashGrid Carpismasi Testi
    float curTerr = sampleTerrainElev(g_ac.pos.x, g_ac.pos.z);
    if (g_ac.pos.y < curTerr + 0.85f) {
        if (!g_ac.gearDown || g_ac.vel.y < -14.0f || sampleSignedSeaDist(g_ac.pos.x, g_ac.pos.z) < -5.0f) {
            g_ac.crashed = true;
            g_ac.pos.y = curTerr + 0.90f;
            spawnCrashParticles(g_ac.pos);
        } else {
            g_ac.pos.y = curTerr + 0.85f;
            if (g_ac.vel.y < 0.0f) g_ac.vel.y = 0.0f;
        }
    }
    if (!g_ac.crashed) {
        checkAircraftCrashBuildings100m(spec);
    }
}

static void updateParticlesAndSmoke(float dt) {
    // [B] Akrobasi dumani (Turk Yildizlari Kirmizi/Beyaz duman izi)
    if (g_ac.smokeOn && !g_ac.crashed) {
        g_smokeEmitTimer += dt;
        while (g_smokeEmitTimer >= 0.018f) {
            g_smokeEmitTimer -= 0.018f;
            SmokePoint sp;
            sp.pos = g_ac.pos - g_ac.fwd * 7.5f;
            sp.vel = -g_ac.fwd * 6.0f + Vec3(0.0f, 0.8f, 0.0f);
            sp.maxLife = 6.5f;
            sp.life = sp.maxLife;
            bool redRibbon = (((int)(g_particles.size())) & 1) != 0;
            sp.r = 0.98f;
            sp.g = redRibbon ? 0.16f : 0.96f;
            sp.b = redRibbon ? 0.18f : 0.98f;
            sp.size = 3.8f;
            sp.growthRate = 1.4f;
            sp.ptype = PT_SMOKE_RIBBON;
            pushParticle(sp);
        }
    } else if (g_ac.crashed && !(sampleSignedSeaDist(g_ac.pos.x, g_ac.pos.z) < -4.0f && g_ac.pos.y < 3.0f)) {
        // Kaza enkazindan cikan surekli siyah duman + kivilcimli alevler
        g_smokeEmitTimer += dt;
        if (g_smokeEmitTimer >= 0.07f) {
            g_smokeEmitTimer = 0.0f;
            g_weaponVfxSeq += 19u;
            Rng r(g_weaponVfxSeq);
            SmokePoint sm;
            sm.pos = g_ac.pos + Vec3(r.range(-2.0f, 2.0f), 1.2f, r.range(-2.0f, 2.0f));
            sm.vel = Vec3(r.range(-1.5f, 1.5f), r.range(7.0f, 12.0f), r.range(-1.5f, 1.5f));
            sm.maxLife = r.range(2.2f, 3.8f);
            sm.life = sm.maxLife;
            sm.r = 0.14f; sm.g = 0.14f; sm.b = 0.16f;
            sm.size = r.range(4.2f, 6.8f);
            sm.growthRate = 3.2f;
            sm.ptype = PT_BLACK_SMOKE;
            pushParticle(sm);

            SmokePoint fl;
            fl.pos = g_ac.pos + Vec3(r.range(-1.5f, 1.5f), 0.6f, r.range(-1.5f, 1.5f));
            fl.vel = Vec3(r.range(-1.2f, 1.2f), r.range(4.0f, 7.5f), r.range(-1.2f, 1.2f));
            fl.maxLife = r.range(0.35f, 0.70f);
            fl.life = fl.maxLife;
            fl.r = 1.0f; fl.g = r.range(0.38f, 0.78f); fl.b = 0.08f;
            fl.size = r.range(3.2f, 5.2f);
            fl.growthRate = 2.0f;
            fl.ptype = PT_FLAME;
            pushParticle(fl);
        }
    }

    for (size_t i = 0; i < g_particles.size();) {
        SmokePoint& pt = g_particles[i];
        pt.life -= dt;
        if (pt.life <= 0.0f) {
            g_particles[i] = g_particles.back();
            g_particles.pop_back();
            continue;
        }
        if (pt.ptype == PT_SPARK || pt.ptype == PT_WATER_SPRAY) {
            pt.vel.y -= 9.81f * dt;
            pt.vel *= std::max(0.0f, 1.0f - 0.55f * dt);
        } else if (pt.ptype == PT_BLACK_SMOKE || pt.ptype == PT_FLAME) {
            pt.vel.x *= std::max(0.0f, 1.0f - 1.25f * dt);
            pt.vel.z *= std::max(0.0f, 1.0f - 1.25f * dt);
            pt.vel.y += (pt.ptype == PT_FLAME ? 4.8f : 2.6f) * dt;
        }
        pt.pos += pt.vel * dt;
        pt.size += pt.growthRate * dt;
        i++;
    }
}

static void renderParticles(const Vec3& camRight, const Vec3& camUp) {
    if (g_particles.empty()) return;
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);

    // 1. Yumusak duman, akrobasi duman izi ve su sicaği billboardlari (alpha blend)
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_texParticle);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_QUADS);
    for (const auto& p : g_particles) {
        if (p.ptype == PT_FLAME || p.ptype == PT_SPARK) continue;
        float a = clampf(p.life / p.maxLife, 0.0f, 1.0f);
        float alphaMul = (p.ptype == PT_SMOKE_RIBBON) ? 0.65f : 0.72f;
        glColor4f(p.r, p.g, p.b, a * alphaMul);
        drawBillboardQuad3D(p.pos, camRight, camUp, p.size * 0.55f);
    }
    glEnd();

    // 2. Parlak alev partikulleri (yumusak dairesel additive blend, sis kapali)
    bool hadFog = (glIsEnabled(GL_FOG) != GL_FALSE);
    glDisable(GL_FOG);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBegin(GL_QUADS);
    for (const auto& p : g_particles) {
        if (p.ptype != PT_FLAME) continue;
        float a = clampf(p.life / p.maxLife, 0.0f, 1.0f);
        float inten = a * a;
        glColor4f(p.r * inten, p.g * inten, p.b * inten, 1.0f);
        drawBillboardQuad3D(p.pos, camRight, camUp, p.size * 0.52f);
    }
    glEnd();

    // 3. Ucan kivilcim ve sicak enkaz parcalari
    glDisable(GL_TEXTURE_2D);
    glPointSize(4.5f);
    glBegin(GL_POINTS);
    for (const auto& p : g_particles) {
        if (p.ptype != PT_SPARK) continue;
        float a = clampf(p.life / p.maxLife, 0.0f, 1.0f);
        glColor4f(p.r * a, p.g * a, p.b * a, 1.0f);
        glVertex3f(p.pos.x, p.pos.y, p.pos.z);
    }
    glEnd();

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glEnable(GL_LIGHTING);
    if (hadFog) glEnable(GL_FOG);
}
// =====================================================================================
//  BOLUM 8 : AVIYONIK HUD, SAG UST 2D WGS84 RADAR, /dev/input/js0 JOYSTICK VE X11/GLX
// =====================================================================================
enum { CAM_CHASE = 0, CAM_COCKPIT = 1, CAM_WING = 2, CAM_TOWER = 3, NUM_CAMS = 4 };
static int g_camMode = CAM_CHASE;
static int g_timeOfDay = 0; // 0 Gunduz, 1 Kizil Korfez Gunbatimi, 2 Gece Parlayan Sehir ve Pist Isiklari
static int g_radarZoomIdx = 1; // 0: 45km, 1: 15km, 2: 4km
static const float RADAR_RANGES[3] = {22500.0f, 7500.0f, 2000.0f};

static void renderRadar2D(int winW, int winH) {
    float radSize = 230.0f;
    float rx0 = winW - radSize - 18.0f;
    float ry0 = 18.0f;
    float rcx = rx0 + radSize * 0.5f;
    float rcy = ry0 + radSize * 0.5f;
    float halfRange = RADAR_RANGES[g_radarZoomIdx];
    float scale = (radSize * 0.5f) / halfRange;

    // Radar arka plan kutusu
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.02f, 0.08f, 0.14f, 0.88f);
    glBegin(GL_QUADS);
    glVertex2f(rx0, ry0); glVertex2f(rx0 + radSize, ry0);
    glVertex2f(rx0 + radSize, ry0 + radSize); glVertex2f(rx0, ry0 + radSize);
    glEnd();

    // Scissor ile radar cercevesi icinde cizim
    glEnable(GL_SCISSOR_TEST);
    glScissor((int)rx0, winH - (int)(ry0 + radSize), (int)radSize, (int)radSize);

    auto toRad = [&](float wx, float wz, float& sx, float& sy) {
        sx = rcx + (wx - g_ac.pos.x) * scale;
        sy = rcy + (wz - g_ac.pos.z) * scale;
    };

    // 1. WorldMaskGrid ile Gercek Deniz / Kara Kontrasti + Topografya Renklendirmesi
    int gridN = 38;
    float cellPx = radSize / (float)gridN;
    glBegin(GL_QUADS);
    for (int gy = 0; gy < gridN; gy++) {
        for (int gx = 0; gx < gridN; gx++) {
            float sx0 = rx0 + gx * cellPx;
            float sy0 = ry0 + gy * cellPx;
            float wx = g_ac.pos.x + ((sx0 + cellPx * 0.5f) - rcx) / scale;
            float wz = g_ac.pos.z + ((sy0 + cellPx * 0.5f) - rcy) / scale;
            const WorldMaskCell& mc = queryWorldMask(wx, wz);
            if (mc.signedSeaDist < 0.0f) {
                glColor4f(0.04f, 0.24f, 0.46f, 0.78f);
            } else {
                float elev = clampf(mc.elev / 750.0f, 0.0f, 1.0f);
                float urb = mc.urbanBlend;
                float rC = lerpf(0.12f + elev * 0.24f, 0.28f, urb * 0.55f);
                float gC = lerpf(0.28f + elev * 0.12f, 0.30f, urb * 0.55f);
                float bC = lerpf(0.14f + elev * 0.08f, 0.26f, urb * 0.55f);
                glColor4f(rC, gC, bC, 0.78f);
            }
            glVertex2f(sx0, sy0);
            glVertex2f(sx0 + cellPx + 0.5f, sy0);
            glVertex2f(sx0 + cellPx + 0.5f, sy0 + cellPx + 0.5f);
            glVertex2f(sx0, sy0 + cellPx + 0.5f);
        }
    }
    glEnd();

    // Gercek WGS84 Korfez Kiyi Cizgisi
    glLineWidth(1.8f);
    glColor4f(0.28f, 0.88f, 1.0f, 0.95f);
    glBegin(GL_LINE_STRIP);
    for (const auto& cp : g_coastSpline) {
        float sx, sy; toRad(cp.x, cp.z, sx, sy);
        glVertex2f(sx, sy);
    }
    glEnd();
    glLineWidth(1.0f);

    // 2. Otoyol ve Sahil Bulvari aglari
    glColor4f(0.75f, 0.78f, 0.82f, 0.72f);
    for (const auto& rs : g_roads) {
        glBegin(GL_LINE_STRIP);
        for (const auto& p : rs.pts) {
            float sx, sy; toRad(p.x, p.z, sx, sy);
            glVertex2f(sx, sy);
        }
        glEnd();
    }

    // 3. IZBAN Hatti + 28 Istasyon + 8 Canli Tren + 6 Korfez Vapuru
    glLineWidth(1.5f);
    glColor4f(0.22f, 0.70f, 1.0f, 0.95f);
    glBegin(GL_LINE_STRIP);
    for (const auto& p : g_izbanSpline) {
        float sx, sy; toRad(p.x, p.z, sx, sy);
        glVertex2f(sx, sy);
    }
    glEnd();
    glLineWidth(1.0f);

    int nearSt = 0;
    float bestStDist = 1e18f;
    for (int s = 0; s < NUM_STATIONS; s++) {
        float d = len2(Vec2(g_ac.pos.x - g_stationPos[s].x, g_ac.pos.z - g_stationPos[s].z));
        if (d < bestStDist) { bestStDist = d; nearSt = s; }
    }

    // 28 IZBAN istasyonu (sari/cyan elmaslar ve isimleri)
    for (int s = 0; s < NUM_STATIONS; s++) {
        float sx, sy; toRad(g_stationPos[s].x, g_stationPos[s].z, sx, sy);
        if (sx < rx0 + 2.0f || sx > rx0 + radSize - 2.0f || sy < ry0 + 2.0f || sy > ry0 + radSize - 2.0f) continue;
        bool isNearest = (s == nearSt);
        if (isNearest) {
            glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
            glPointSize(7.0f);
        } else {
            glColor4f(1.0f, 0.92f, 0.25f, 0.95f);
            glPointSize(4.5f);
        }
        glBegin(GL_POINTS);
        glVertex2f(sx, sy);
        glEnd();
        if (g_radarZoomIdx >= 1 || isNearest) {
            char stLbl[32];
            std::snprintf(stLbl, sizeof(stLbl), "%d.%s", s + 1, STATIONS[s].name);
            drawText2D(sx + 5.0f, sy - 4.0f, isNearest ? 1.2f : 0.95f, stLbl,
                       isNearest ? 0.1f : 0.95f, isNearest ? 1.0f : 0.92f, isNearest ? 1.0f : 0.55f);
        }
    }

    glPointSize(4.0f);
    glBegin(GL_POINTS);
    // 8 Canli IZBAN treni (parlak kirmizi/beyaz noktalar)
    glColor4f(1.0f, 0.22f, 0.28f, 1.0f);
    for (int i = 0; i < NUM_IZBAN_TRAINS; i++) {
        Vec3 tp; Vec2 tt;
        sampleIzbanAtS(g_trains[i].s, tp, tt);
        float sx, sy; toRad(tp.x, tp.z, sx, sy);
        glVertex2f(sx, sy);
    }
    // 6 Korfez vapuru (camgobegi noktalar)
    glColor4f(0.35f, 1.0f, 0.85f, 1.0f);
    for (int i = 0; i < NUM_FERRIES; i++) {
        float sx, sy; toRad(g_ferries[i].pos.x, g_ferries[i].pos.z, sx, sy);
        glVertex2f(sx, sy);
    }
    glEnd();

    // 4. LTBJ ve LTBL Pistleri + Ana Ilce Etiketleri
    glLineWidth(2.8f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.98f);
    glBegin(GL_LINES);
    {
        Vec2 c1 = geoPt(LTBJ_LAT, LTBJ_LON), d1 = headingDir(LTBJ_HDG), r1(-d1.z, d1.x);
        for (int s = -1; s <= 1; s += 2) {
            Vec2 rc = c1 + r1 * (s * LTBJ_SEP * 0.5f);
            Vec2 a = rc - d1 * (LTBJ_LEN * 0.5f), b = rc + d1 * (LTBJ_LEN * 0.5f);
            float x0, y0, x1, y1; toRad(a.x, a.z, x0, y0); toRad(b.x, b.z, x1, y1);
            glVertex2f(x0, y0); glVertex2f(x1, y1);
        }
        Vec2 c2 = geoPt(LTBL_LAT, LTBL_LON), d2 = headingDir(LTBL_HDG);
        Vec2 a = c2 - d2 * (LTBL_LEN * 0.5f), b = c2 + d2 * (LTBL_LEN * 0.5f);
        float x0, y0, x1, y1; toRad(a.x, a.z, x0, y0); toRad(b.x, b.z, x1, y1);
        glVertex2f(x0, y0); glVertex2f(x1, y1);
    }
    glEnd();
    glLineWidth(1.0f);

    struct RadLabel { const char* txt; double lat, lon; };
    static const RadLabel RLABELS[] = {
        {"KONAK",     38.4188, 27.1287},
        {"ALSANCAK",  38.4380, 27.1420},
        {"KARSIYAKA", 38.4580, 27.1150},
        {"BAYRAKLI",  38.4542, 27.1755},
        {"LTBJ",      LTBJ_LAT, LTBJ_LON},
        {"LTBL",      LTBL_LAT, LTBL_LON}
    };
    for (const auto& rl : RLABELS) {
        Vec2 p = geoPt(rl.lat, rl.lon);
        float sx, sy; toRad(p.x, p.z, sx, sy);
        if (sx > rx0 + 10.0f && sx < rx0 + radSize - 42.0f && sy > ry0 + 18.0f && sy < ry0 + radSize - 18.0f) {
            drawText2D(sx + 4.0f, sy - 4.0f, 1.05f, rl.txt, 0.92f, 0.96f, 0.85f);
        }
    }

    // 5. Ucak sembolu ve pruva (heading) vektoru
    glColor4f(0.25f, 1.0f, 0.45f, 1.0f);
    Vec2 hdg2 = norm2(Vec2(g_ac.fwd.x, g_ac.fwd.z));
    glBegin(GL_LINES);
    glVertex2f(rcx, rcy);
    glVertex2f(rcx + hdg2.x * 26.0f, rcy + hdg2.z * 26.0f);
    glVertex2f(rcx - 6.0f, rcy); glVertex2f(rcx + 6.0f, rcy);
    glVertex2f(rcx, rcy - 6.0f); glVertex2f(rcx, rcy + 6.0f);
    glEnd();

    glDisable(GL_SCISSOR_TEST);

    // Radar cercevesi ve olcek basligi
    glColor4f(0.25f, 0.95f, 0.55f, 0.95f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(rx0, ry0); glVertex2f(rx0 + radSize, ry0);
    glVertex2f(rx0 + radSize, ry0 + radSize); glVertex2f(rx0, ry0 + radSize);
    glEnd();
    const char* zoomLabels[3] = {"RADAR [Z]: 45 KM", "RADAR [Z]: 15 KM", "RADAR [Z]: 4 KM"};
    drawText2D(rx0 + 8.0f, ry0 + 8.0f, 1.5f, zoomLabels[g_radarZoomIdx], 0.35f, 1.0f, 0.65f);
    drawText2D(rx0 + 8.0f, ry0 + radSize - 16.0f, 1.3f, "N ^ KORFEZ + 28 IZBAN + LTBJ/LTBL", 0.75f, 0.95f, 1.0f);
    glDisable(GL_BLEND);
}

struct TourStop {
    const char* district;
    const char* description;
    double lat, lon;
    float alt;
    float hdg;
    float pitch;
};

static const int NUM_TOUR_STOPS = 24;
static const TourStop TOUR_STOPS[NUM_TOUR_STOPS] = {
    {"Adnan Menderes Havalimani (LTBJ)", "3240m Cift Paralel Pist, PAPI Isiklari, Terminal ve ATC Kulesi", 38.2936, 27.1520, 180.0f, 345.8f, -4.0f},
    {"Gaziemir & Fuar Izmir", "Ege Serbest Bolgesi (ESBAS), Fuar Izmir Alani ve Onder Caddesi", 38.3280, 27.1350, 210.0f, 340.0f, -6.0f},
    {"Buca & Sirinyer Hipodromu", "Hipodrom, Ugur Mumcu Caddesi, Menderes Caddesi ve Buca Yamaclari", 38.3880, 27.1650, 240.0f, 325.0f, -8.0f},
    {"Balcova & Catalkaya Dagi", "Balcova Terapi Ormani, Teleferik Zirvesi ve Sakarya Caddesi", 38.3880, 27.0500, 290.0f, 45.0f, -7.0f},
    {"Inciralti Kent Ormani", "Cakalburnu Lagunu, Deniz Muzesi ve Kiyi Palmiyeleri", 38.4100, 27.0350, 180.0f, 60.0f, -8.0f},
    {"Guzelyali & Goztepe Sahili", "Mustafa Kemal Sahil Bulvari, Goztepe Iskelesi ve Yali Bloklari", 38.3995, 27.0850, 160.0f, 55.0f, -6.0f},
    {"Karatas Tarihi Asansor", "51m Tugla Kule, Dario Moreno Sokagi ve Karantina Koridoru", 38.4100, 27.1180, 145.0f, 45.0f, -7.0f},
    {"Konak Meydani & Izmir Saat Kulesi", "Sekizgen Kaideli Saat Kulesi, Yali Camii ve Konak Pier", 38.4188, 27.1287, 125.0f, 35.0f, -8.0f},
    {"Tarihi Kemeralti Carsisi", "Hisar Camii Kubbesi, Kizlaragasi Hani ve Geleneksel Ege Dukkanlari", 38.4180, 27.1370, 135.0f, 40.0f, -10.0f},
    {"Kadifekale & Dev Turk Bayragi", "Pagos Tepesi Antik Surlari ve 35m Gonderde Dalgalanan Turk Bayragi", 38.4142, 27.1453, 260.0f, 315.0f, -9.0f},
    {"Alsancak 1. ve 2. Kordon", "Ataturk Caddesi, Gundogdu Meydani ve Kulturpark Yesil Korusu", 38.4375, 27.1420, 155.0f, 25.0f, -7.0f},
    {"Alsancak Limani & Kruvaziyer", "180m Kruvaziyer Gemisi, Konteyner Vincleri ve Demiryolu Rihtimi", 38.4445, 27.1535, 165.0f, 45.0f, -8.0f},
    {"Halkapinar & Hilal IZBAN Kavsagi", "Cift Hatli Balastli Demiryolu Kavisleri ve Catenary Direkleri", 38.4430, 27.1680, 165.0f, 15.0f, -8.0f},
    {"Bayrakli Folkart & Mistral Kuleleri", "200m Twist Ikiz Folkart, 216m Mistral ve Ege Perla", 38.4520, 27.1750, 280.0f, 330.0f, -10.0f},
    {"Altinyol & Turan Sahil Koridoru", "Kiyi Boyunca Uzanan Kesintisiz D550 Cift Yonlu Ekspresyol", 38.4650, 27.1550, 185.0f, 280.0f, -6.0f},
    {"Karsiyaka Sahil & Karsiyaka Aniti", "Cemal Gursel Sahil Bulvari, Karsiyaka Iskelesi ve KSK Aniti", 38.4560, 27.1180, 155.0f, 290.0f, -6.0f},
    {"Bostanli Sahil & Gunbatimi Terasi", "Kademeli Ahsap Gunbatimi Amfisi ve Hasan Ali Yucel Bulvari", 38.4580, 27.0920, 145.0f, 305.0f, -7.0f},
    {"Mavisehir Flamingo & Rezidanslar", "Caher Dudayev Bulvari, Flamingo Bloklari ve Cicek Parki", 38.4750, 27.0700, 185.0f, 320.0f, -8.0f},
    {"Cigli 2. Ana Jet Ussu (LTBL)", "2900m Askeri Pist, HAS Siginaklari ve Jet Hangarlari", 38.5130, 27.0100, 185.0f, 347.0f, -6.0f},
    {"Cigli AOSB & Ata Sanayi", "Oluklu Sac Celik Fabrika Hangarlari ve Duzenli Sanayi Caddeleri", 38.4950, 27.0450, 185.0f, 350.0f, -7.0f},
    {"Egekent-2 & Ulukent (Gercek Harita)", "Gercek OSM Binalari, Siteler, 3. Cadde, Sevgi Cd, Bahceler ve IZBAN Istasyonu", 38.5610, 27.0450, 155.0f, 350.0f, -7.0f},
    {"Sasali Kus Cenneti & Camalti Tuzlasi", "Tuz Uretim Tavalari, Gediz Deltasi ve Dogal Yasam Korulugu", 38.5100, 26.9400, 250.0f, 20.0f, -8.0f},
    {"Menemen Sehir Merkezi & Gediz Ovasi", "IZBAN Kuzey Uc Istasyonu ve Verimli Aluvyon Ovasi", 38.6050, 27.0700, 220.0f, 180.0f, -8.0f},
    {"Yamanlar Dagi Zirvesi (1076m)", "Kizilcam Orman Denizi, Kalker Kayaliklar ve Ege Korfezi Panoramasi", 38.5450, 27.1850, 1220.0f, 215.0f, -12.0f}
};

static bool  g_tourActive = false;
static int   g_tourIndex = 0;
static float g_tourTimer = 0.0f;

static void renderAvionicsHUD(int winW, int winH, float fps) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winW, winH, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    float cx = winW * 0.5f, cy = winH * 0.5f;
    float pitchDeg = std::asin(clampf(g_ac.fwd.y, -1.0f, 1.0f)) * RAD2DEG;
    float rollRad  = std::atan2(-g_ac.right.y, g_ac.up.y);
    float rollDeg  = rollRad * RAD2DEG;
    float hdgDeg   = std::atan2(g_ac.fwd.x, -g_ac.fwd.z) * RAD2DEG;
    if (hdgDeg < 0.0f) hdgDeg += 360.0f;

    float kmh = g_ac.airspeedMs * 3.6f;
    float knots = g_ac.airspeedMs * 1.94384f;
    float altM = g_ac.pos.y;
    float altFt = altM * 3.28084f;

    // 1. Pitch Ladder (Yunuslama Merdiveni) & Roll Olcegi
    glPushMatrix();
    glTranslatef(cx, cy, 0.0f);
    glRotatef(-rollDeg, 0.0f, 0.0f, 1.0f);
    glColor3f(0.20f, 1.0f, 0.45f);
    for (int p = -40; p <= 40; p += 5) {
        float dy = -(p - pitchDeg) * 6.5f;
        if (std::fabs(dy) > 165.0f) continue;
        float w = (p == 0) ? 95.0f : 42.0f;
        glBegin(GL_LINES);
        glVertex2f(-w, dy); glVertex2f(-14.0f, dy);
        glVertex2f( 14.0f, dy); glVertex2f( w, dy);
        if (p != 0) {
            float tick = (p > 0) ? 7.0f : -7.0f;
            glVertex2f(-w, dy); glVertex2f(-w, dy + tick);
            glVertex2f( w, dy); glVertex2f( w, dy + tick);
        }
        glEnd();
        if (p != 0 && (p % 10 == 0)) {
            char buf[16]; std::snprintf(buf, sizeof(buf), "%d", p);
            drawText2D(w + 6.0f, dy - 5.0f, 1.4f, buf, 0.20f, 1.0f, 0.45f);
        }
    }
    // Ust Roll Olcegi yayi
    glBegin(GL_LINE_STRIP);
    for (int a = -60; a <= 60; a += 5) {
        float ar = (a - 90) * DEG2RAD;
        glVertex2f(std::cos(ar) * 145.0f, std::sin(ar) * 145.0f);
    }
    glEnd();
    glPopMatrix();

    // 1. Silah Nisangahi (renderScene3D icinde guncellenen g_gunAim / g_ccip verisi)

    // Ucak referans nisangahi (Aircraft Bore Sight W-sembolu) + Flight Path Marker (FPM)
    glColor3f(0.25f, 1.0f, 0.50f);
    glBegin(GL_LINE_STRIP);
    glVertex2f(cx - 24, cy); glVertex2f(cx - 10, cy); glVertex2f(cx - 5, cy + 6);
    glVertex2f(cx, cy); glVertex2f(cx + 5, cy + 6); glVertex2f(cx + 10, cy); glVertex2f(cx + 24, cy);
    glEnd();

    float gunScreenX = cx, gunScreenY = cy;
    bool hasGunScreen = (!g_ac.crashed && g_gunAimValid && projectWorldToHUD(g_gunAimPos, gunScreenX, gunScreenY));
    if (hasGunScreen) {
        if (g_gunAimHitBldg) glColor3f(1.0f, 0.30f, 0.16f);
        else                 glColor3f(0.28f, 1.0f, 0.85f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 16; i++) {
            float a = i * (2.0f * PI_F / 16.0f);
            glVertex2f(gunScreenX + std::cos(a) * 15.0f, gunScreenY + std::sin(a) * 15.0f);
        }
        glEnd();
        glBegin(GL_LINES);
        glVertex2f(gunScreenX - 22.0f, gunScreenY); glVertex2f(gunScreenX - 9.0f, gunScreenY);
        glVertex2f(gunScreenX + 9.0f,  gunScreenY); glVertex2f(gunScreenX + 22.0f, gunScreenY);
        glVertex2f(gunScreenX, gunScreenY - 22.0f); glVertex2f(gunScreenX, gunScreenY - 9.0f);
        glVertex2f(gunScreenX, gunScreenY + 9.0f);  glVertex2f(gunScreenX, gunScreenY + 22.0f);
        glEnd();
        glPointSize(3.5f);
        glBegin(GL_POINTS);
        glVertex2f(gunScreenX, gunScreenY);
        glEnd();
        if (g_gunAimHitBldg && g_gunAimDist < 1900.0f) {
            char rngBuf[32];
            std::snprintf(rngBuf, sizeof(rngBuf), "%.0fM", g_gunAimDist);
            drawText2D(gunScreenX + 19.0f, gunScreenY - 5.0f, 1.25f, rngBuf, 1.0f, 0.45f, 0.22f);
        }
    }

    float ccipX = cx, ccipY = cy;
    if (!g_ac.crashed && g_ccipValid && projectWorldToHUD(g_ccipPos, ccipX, ccipY)) {
        // Bomb fall-line (Kesikli dusus cizgisi) + CCIP Vurus Noktasi Pipper'i
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0x0F0F);
        glColor3f(1.0f, 0.82f, 0.18f);
        glBegin(GL_LINES);
        glVertex2f(hasGunScreen ? gunScreenX : cx, hasGunScreen ? gunScreenY : cy);
        glVertex2f(ccipX, ccipY);
        glEnd();
        glDisable(GL_LINE_STIPPLE);

        if (g_ccipHitBldg) glColor3f(1.0f, 0.22f, 0.14f);
        else               glColor3f(1.0f, 0.82f, 0.18f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 16; i++) {
            float a = i * (2.0f * PI_F / 16.0f);
            glVertex2f(ccipX + std::cos(a) * 12.0f, ccipY + std::sin(a) * 12.0f);
        }
        glEnd();
        glBegin(GL_LINE_LOOP);
        glVertex2f(ccipX, ccipY - 17.0f);
        glVertex2f(ccipX + 17.0f, ccipY);
        glVertex2f(ccipX, ccipY + 17.0f);
        glVertex2f(ccipX - 17.0f, ccipY);
        glEnd();
        char ccipBuf[48];
        if (g_ccipHitBldg) std::snprintf(ccipBuf, sizeof(ccipBuf), "BINA %.1fs", g_ccipFallTime);
        else               std::snprintf(ccipBuf, sizeof(ccipBuf), "BOMBA %.1fs", g_ccipFallTime);
        drawText2D(ccipX + 20.0f, ccipY - 5.0f, 1.25f, ccipBuf,
                   1.0f, g_ccipHitBldg ? 0.35f : 0.85f, g_ccipHitBldg ? 0.22f : 0.22f);
    }

    float rawFpmX = clampf(g_ac.betaDeg * 5.0f, -120.0f, 120.0f);
    float rawFpmY = clampf(g_ac.alphaDeg * 6.5f, -120.0f, 120.0f);
    float cr = std::cos(-rollRad), sr = std::sin(-rollRad);
    float fpmX = cx + (rawFpmX * cr - rawFpmY * sr);
    float fpmY = cy + (rawFpmX * sr + rawFpmY * cr);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 12; i++) {
        float a = i * (2.0f * PI_F / 12.0f);
        glVertex2f(fpmX + std::cos(a) * 8.0f, fpmY + std::sin(a) * 8.0f);
    }
    glEnd();
    glBegin(GL_LINES);
    glVertex2f(fpmX - 16, fpmY); glVertex2f(fpmX - 8, fpmY);
    glVertex2f(fpmX + 8, fpmY);  glVertex2f(fpmX + 16, fpmY);
    glVertex2f(fpmX, fpmY - 16); glVertex2f(fpmX, fpmY - 8);
    glEnd();

    // 2. Sol Hiz Seridi (KMH / KT)
    float spdX = cx - 250.0f;
    glColor3f(0.20f, 1.0f, 0.45f);
    glBegin(GL_LINE_STRIP);
    glVertex2f(spdX - 55, cy - 120); glVertex2f(spdX, cy - 120);
    glVertex2f(spdX, cy + 120);      glVertex2f(spdX - 55, cy + 120);
    glEnd();
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%4.0f KMH", kmh);
    drawText2D(spdX - 78, cy - 8, 1.8f, buf, 0.35f, 1.0f, 0.55f);
    std::snprintf(buf, sizeof(buf), "%4.0f KT", knots);
    drawText2D(spdX - 74, cy + 14, 1.5f, buf, 0.25f, 0.95f, 0.50f);

    // 3. Sag Irtifa Seridi (Metre / Feet + AGL + VSI)
    float altX = cx + 250.0f;
    glBegin(GL_LINE_STRIP);
    glVertex2f(altX + 55, cy - 120); glVertex2f(altX, cy - 120);
    glVertex2f(altX, cy + 120);      glVertex2f(altX + 55, cy + 120);
    glEnd();
    std::snprintf(buf, sizeof(buf), "%4.0f M", altM);
    drawText2D(altX + 8, cy - 8, 1.8f, buf, 0.35f, 1.0f, 0.55f);
    std::snprintf(buf, sizeof(buf), "%5.0f FT", altFt);
    drawText2D(altX + 8, cy + 14, 1.5f, buf, 0.25f, 0.95f, 0.50f);
    std::snprintf(buf, sizeof(buf), "AGL: %.0f M", g_ac.aglM);
    drawText2D(altX + 8, cy + 38, 1.4f, buf, 0.65f, 1.0f, 0.65f);
    std::snprintf(buf, sizeof(buf), "VSI: %+.1f M/S", g_ac.vsiMs);
    drawText2D(altX + 8, cy + 58, 1.4f, buf, 0.65f, 1.0f, 0.65f);

    // 4. Ust Pusula Seridi (0..360 Derece)
    glColor3f(0.20f, 1.0f, 0.45f);
    glBegin(GL_LINES);
    glVertex2f(cx - 160, 44); glVertex2f(cx + 160, 44);
    glVertex2f(cx, 44);       glVertex2f(cx, 54);
    glEnd();
    std::snprintf(buf, sizeof(buf), "HDG %03d°", (int)(hdgDeg + 0.5f) % 360);
    drawText2D(cx - 38, 22, 1.8f, buf, 0.35f, 1.0f, 0.55f);

    // 5. En Yakin IZBAN Istasyonu Adi / Mesafesi ve Bulunulan Ilce Adi
    int nearSt = 0;
    float bestStDist = 1e18f;
    for (int s = 0; s < NUM_STATIONS; s++) {
        float d = len2(Vec2(g_ac.pos.x - g_stationPos[s].x, g_ac.pos.z - g_stationPos[s].z));
        if (d < bestStDist) { bestStDist = d; nearSt = s; }
    }
    const char* distName = queryDistrictName(g_ac.pos.x, g_ac.pos.z);
    double curLat = zToLat(g_ac.pos.z), curLon = xToLon(g_ac.pos.x);

    // Sol ust telemetri ve cografi konum paneli
    drawText2D(18, 16, 1.9f, "IZMIR 3D UCUS VE SEHIR SIMULATORU", 0.35f, 1.0f, 0.65f);
    drawText2D(18, 38, 1.5f, AC_SPECS[g_ac.type].name, 1.0f, 0.92f, 0.45f);
    std::snprintf(buf, sizeof(buf), "ILCE: %s  |  GPS: %.4f N, %.4f E", distName, curLat, curLon);
    drawText2D(18, 58, 1.5f, buf, 0.55f, 0.96f, 1.0f);
    float toStHdg = std::atan2(g_stationPos[nearSt].x - g_ac.pos.x, -(g_stationPos[nearSt].z - g_ac.pos.z)) * RAD2DEG;
    if (toStHdg < 0.0f) toStHdg += 360.0f;
    std::snprintf(buf, sizeof(buf), "EN YAKIN İZBAN (%d/28): %s (%.2f KM) -> ROTA: %03.0f°",
                  nearSt + 1, STATIONS[nearSt].name, bestStDist * 0.001f, toStHdg);
    drawText2D(18, 78, 1.5f, buf, 1.0f, 0.95f, 0.55f);

    // G-Metre, AoA, Gaz/Afterburner, Inis Takimi, Flap, Ridge Lift
    std::snprintf(buf, sizeof(buf), "G-LOAD: %+.1f G   AoA: %+.1f°   GAZ: %3.0f%%%s   RIDGE LIFT: %+.1f M/S",
                  g_ac.gLoad, g_ac.alphaDeg, g_ac.throttle * 100.0f,
                  (g_ac.throttle > 0.80f && g_ac.type == AC_JET) ? " [AFTERBURNER]" : "",
                  g_ac.ridgeLiftMs);
    drawText2D(18, 98, 1.5f, buf, 0.35f, 1.0f, 0.55f);

    const char* camNames[4] = {"CHASE", "KOKPIT HUD", "KANAT", "KULE"};
    const char* todNames[3] = {"GUNDUZ", "KIZIL KORFEZ GUNBATIMI", "GECE PARLAYAN SEHIR"};
    float maxFireSec = 0.0f;
    for (const auto& fz : g_activeFires) maxFireSec = std::max(maxFireSec, fz.timeLeft);
    std::snprintf(buf, sizeof(buf), "GEAR[G]: %s  FLAP[H]: %d°  DUMAN[B]: %s  DRONE[F]: %s  KAMERA[C]: %s  VAKIT[T]: %s  YANGIN: %zu (%ds)  FPS: %.0f",
                  g_ac.gearDown ? "ACIK" : "KAPALI", (int)g_ac.flapDeg,
                  g_ac.smokeOn ? "ACIK" : "KAPALI",
                  g_ac.droneMode ? "AKTIF" : "KAPALI", camNames[g_camMode], todNames[g_timeOfDay],
                  g_activeFires.size(), (int)std::ceil(maxFireSec), fps);
    drawText2D(18, winH - 42, 1.35f, buf, 0.80f, 0.95f, 0.85f);

    drawText2D(18, winH - 22, 1.3f,
               "[1-6]Bolge  [J/Enter/RB]Mermi  [M/U/LB]Bomba(30s Yangin)  [F]Drone  [I]Tur  [V]Ucak  [R]Kurtar",
               0.65f, 0.88f, 1.0f);

    if (g_tourActive) {
        char tourBanner[256];
        std::snprintf(tourBanner, sizeof(tourBanner),
                      "OTONOM SEHIR DENETIM TURU [I: Kapat] (%02d/%02d): %s - %s",
                      g_tourIndex + 1, NUM_TOUR_STOPS, TOUR_STOPS[g_tourIndex].district, TOUR_STOPS[g_tourIndex].description);
        float tw = textWidth2D(1.4f, tourBanner);
        drawText2D(cx - tw * 0.5f, 75.0f, 1.4f, tourBanner, 1.0f, 0.85f, 0.20f);
    }

    if (g_ac.crashed) {
        const char* crashMsg = "KAZA TESPIT EDILDI! KURTARMAK ICIN [R] VEYA [1]-[0] TUSUNA BASIN";
        float tw = textWidth2D(2.2f, crashMsg);
        drawText2D(cx - tw * 0.5f, cy - 95.0f, 2.2f, crashMsg, 1.0f, 0.22f, 0.18f);
    } else if (g_ac.onGround && g_ac.airspeedMs < 8.0f && g_ac.throttle < 0.15f) {
        const char* rwyMsg = "PIST BASI HAZIR: [W / RT] GAZ VERIN, [A-D / SOL ANALOG] SOL-SAG DONUS, [V / BACK] UCAK DEGISTIR";
        float tw = textWidth2D(1.45f, rwyMsg);
        drawText2D(cx - tw * 0.5f, cy + 145.0f, 1.45f, rwyMsg, 1.0f, 0.92f, 0.38f);
    }

    // Sag Ust 2D WGS84 Radar
    renderRadar2D(winW, winH);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// =====================================================================================
//  /dev/input/js0 OTOMATIK LINUX GAMEPAD / JOYSTICK DESTEGI (Logitech F710 / Xbox)
// =====================================================================================
static int   g_jsFd = -1;
static float g_jsAxis[8] = {0};      // 0:SolX, 1:SolY, 2:LT(0..1), 3:SagX, 4:SagY, 5:RT(0..1), 6:DPadX, 7:DPadY
static float g_jsTrigInit[8] = {0};
static bool  g_jsTrigHasInit[8] = {false};
static bool  g_jsTrigMoved[8] = {false};
static bool  g_jsBtn[16] = {false};
static float g_jsRetryTimer = 0.0f;

static void resetAndRecoverAircraftInAir() {
    g_ac.crashed = false;
    g_camShakeMag = 0.0f;
    g_camShakeTimer = 0.0f;
    float terr = std::max(0.0f, sampleTerrainElev(g_ac.pos.x, g_ac.pos.z));
    if (g_ac.pos.y < terr + 60.0f) {
        g_ac.pos.y = terr + 220.0f;
    }
    float hdg = std::atan2(g_ac.fwd.x, -g_ac.fwd.z) * RAD2DEG;
    setOrientationFromYawPitchRoll(hdg, 2.0f, 0.0f);
    float cruiseSpd = (g_ac.type == AC_JET) ? 130.0f : (g_ac.type == AC_CESSNA ? 78.0f : 42.0f);
    g_ac.vel = g_ac.fwd * cruiseSpd;
    g_ac.omega = Vec3(0, 0, 0);
    g_ac.gearDown = (g_ac.type == AC_CESSNA);
    g_ac.gearAnim = (g_ac.type == AC_CESSNA) ? 1.0f : 0.0f;
    g_ac.onGround = false;
    syncCameraToAircraft();
}

static void cycleAircraftType() {
    g_ac.type = (g_ac.type + 1) % NUM_AIRCRAFT;
    if (g_ac.type == AC_CESSNA) {
        g_ac.gearDown = true;
        g_ac.gearAnim = 1.0f;
    }
    if (!g_ac.onGround) {
        float targetSpd = (g_ac.type == AC_JET) ? 130.0f : (g_ac.type == AC_CESSNA ? 78.0f : 42.0f);
        g_ac.vel = g_ac.fwd * targetSpd;
        g_ac.omega = Vec3(0, 0, 0);
    }
}

static void pollLinuxJoystick(float dt) {
    if (g_jsFd < 0) {
        g_jsRetryTimer -= dt;
        if (g_jsRetryTimer <= 0.0f) {
            g_jsRetryTimer = 2.0f;
            g_jsFd = open("/dev/input/js0", O_RDONLY | O_NONBLOCK);
            if (g_jsFd >= 0) {
                for (int i = 0; i < 8; i++) {
                    g_jsAxis[i] = 0.0f;
                    g_jsTrigInit[i] = -1.0f;
                    g_jsTrigHasInit[i] = false;
                    g_jsTrigMoved[i] = false;
                }
                for (int i = 0; i < 16; i++) g_jsBtn[i] = false;
            }
        }
        return;
    }
    struct js_event ev;
    while (read(g_jsFd, &ev, sizeof(ev)) == sizeof(ev)) {
        bool isInit = (ev.type & JS_EVENT_INIT) != 0;
        uint8_t t = ev.type & ~JS_EVENT_INIT;
        if (t == JS_EVENT_AXIS && ev.number < 8) {
            float v = clampf((float)ev.value / 32767.0f, -1.0f, 1.0f);
            // Eksen 2 (LT) ve Eksen 5 (RT) tek kutuplu tetik eksenleridir (dinlenme degeri -1.0f)
            if (ev.number == 2 || ev.number == 5) {
                if (isInit || !g_jsTrigHasInit[ev.number]) {
                    g_jsTrigInit[ev.number] = v;
                    g_jsTrigHasInit[ev.number] = true;
                    g_jsAxis[ev.number] = 0.0f;
                    continue;
                }
                if (!g_jsTrigMoved[ev.number]) {
                    if (std::fabs(v - g_jsTrigInit[ev.number]) > 0.12f) {
                        g_jsTrigMoved[ev.number] = true;
                    } else {
                        g_jsAxis[ev.number] = 0.0f;
                        continue;
                    }
                }
                // [-1..+1] araligindaki tetigi [0..1] araligina cevir
                float trig01 = (g_jsTrigInit[ev.number] < -0.5f) ? ((v + 1.0f) * 0.5f) : std::max(0.0f, v);
                g_jsAxis[ev.number] = (trig01 > 0.06f) ? clampf((trig01 - 0.06f) / 0.94f, 0.0f, 1.0f) : 0.0f;
            } else {
                // Analog Cubuklar (0:SolX, 1:SolY, 3:SagX, 4:SagY) ve D-Pad (6:X, 7:Y):
                // Dogrudan 0.12f merkez deadzone ile aninda tepki verir!
                if (std::fabs(v) < 0.12f) {
                    g_jsAxis[ev.number] = 0.0f;
                } else {
                    float signV = (v > 0.0f) ? 1.0f : -1.0f;
                    g_jsAxis[ev.number] = signV * clampf((std::fabs(v) - 0.12f) / 0.88f, 0.0f, 1.0f);
                }
            }
        } else if (t == JS_EVENT_BUTTON && ev.number < 16) {
            bool pressed = (ev.value != 0);
            g_jsBtn[ev.number] = pressed;
            if (!isInit && pressed) {
                // F710 / Xbox tus atamalari:
                // 2 (X): Flap | 3 (Y): Kamera | 4 (LB): Bomba At | 5 (RB): Mermi At | 6 (Back): Ucak | 7 (Start): Kurtar | 8/9: Duman | 10: Gear
                if (ev.number == 2) g_ac.flapStage = (g_ac.flapStage + 1) % 3;
                else if (ev.number == 3) g_camMode = (g_camMode + 1) % NUM_CAMS;
                else if (ev.number == 4) g_bombTriggerEdge = true;
                else if (ev.number == 5) g_gunTriggerEdge = true;
                else if (ev.number == 6) cycleAircraftType();
                else if (ev.number == 7) resetAndRecoverAircraftInAir();
                else if (ev.number == 8 || ev.number == 9) g_ac.smokeOn = !g_ac.smokeOn;
                else if (ev.number == 10) g_ac.gearDown = !g_ac.gearDown;
            }
        }
    }
}

// =====================================================================================
//  EGE IKLIMI, COK-BANTLI ATMOSFER KUBBESI, AY + YILDIZLAR + 3D KUMULUS BULUTLARI
// =====================================================================================
static bool g_headlessTour = false;

static void precompileNearChunks(const Vec3& focusPos, float radius) {
    float r2 = radius * radius;
    for (int i = 0; i < CHUNK_NX * CHUNK_NZ; i++) {
        CityChunk& ck = g_cityChunks[i];
        if (ck.compiled) continue;
        float dx = ck.cx - focusPos.x, dz = ck.cz - focusPos.z;
        if (dx * dx + dz * dz <= r2) {
            compileChunkIfNeeded(ck);
        }
    }
}

static inline Vec3 getCelestialSunMoonDir() {
    if (g_timeOfDay == 2) return norm(Vec3(0.42f, 0.27f, -0.86f));   // Ay: NNE Korfez / Bayrakli ufku uzerinde
    if (g_timeOfDay == 1) return norm(Vec3(-0.88f, 0.19f, 0.12f));  // Gunbatimi: Bati Korfez ufku
    return norm(Vec3(-0.45f, 0.74f, 0.40f));                        // Gunduz gunesi
}

static void renderAtmosphereDome3D(const Vec3& eye, const Vec3& sunDir) {
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_FOG);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_CULL_FACE);

    // 5 irtifa bandi: ufuk alti -> ufuk parlakligi -> alcak gok -> orta gok -> zenit
    const float elevs[5] = {-0.18f, 0.00f, 0.14f, 0.44f, 1.00f};
    float baseCol[5][3];
    if (g_timeOfDay == 2) {
        // Gece Ege sema kubbesi (ufukta sıcak Izmir sehir isik halesi + koyu lacivert zenit)
        const float nCol[5][3] = {
            {0.05f, 0.07f, 0.13f}, {0.09f, 0.11f, 0.21f}, {0.04f, 0.07f, 0.16f},
            {0.02f, 0.04f, 0.10f}, {0.01f, 0.02f, 0.06f}
        };
        std::memcpy(baseCol, nCol, sizeof(baseCol));
    } else if (g_timeOfDay == 1) {
        // Kizil Ege Korfezi gunbatimi kubbesi
        const float sCol[5][3] = {
            {0.76f, 0.40f, 0.28f}, {0.95f, 0.50f, 0.29f}, {0.82f, 0.42f, 0.39f},
            {0.34f, 0.26f, 0.48f}, {0.13f, 0.15f, 0.34f}
        };
        std::memcpy(baseCol, sCol, sizeof(baseCol));
    } else {
        // Durgun Ege mavisi gunduz kubbesi
        const float dCol[5][3] = {
            {0.68f, 0.82f, 0.94f}, {0.72f, 0.86f, 0.96f}, {0.44f, 0.71f, 0.94f},
            {0.24f, 0.52f, 0.86f}, {0.12f, 0.33f, 0.72f}
        };
        std::memcpy(baseCol, dCol, sizeof(baseCol));
    }

    const int SEG = 36;
    const float domeR = 40000.0f;
    Vec2 sunAz = norm2(Vec2(sunDir.x, sunDir.z));

    auto emitSkyVertex = [&](int band, int segIdx) {
        float az = (float)segIdx * (2.0f * PI_F / (float)SEG);
        float ca = std::cos(az), sa = std::sin(az);
        float sinE = elevs[band];
        float cosE = std::sqrt(std::max(0.0f, 1.0f - sinE * sinE));
        float r = baseCol[band][0], g = baseCol[band][1], b = baseCol[band][2];

        // Gunes / Ay azimutu yonunde ufuk ve alcak gokte atmosferik sacilma isiltisi
        float azMatch = std::max(0.0f, ca * sunAz.x + sa * sunAz.z);
        float scatterW = std::pow(azMatch, (g_timeOfDay == 1) ? 2.4f : 4.0f) * std::max(0.0f, 1.0f - std::fabs(sinE) * 2.2f);
        if (g_timeOfDay == 1) {
            r = std::min(1.0f, r + scatterW * 0.24f);
            g = std::min(1.0f, g + scatterW * 0.22f);
            b = std::min(1.0f, b + scatterW * 0.08f);
        } else if (g_timeOfDay == 0) {
            r = std::min(1.0f, r + scatterW * 0.14f);
            g = std::min(1.0f, g + scatterW * 0.11f);
            b = std::min(1.0f, b + scatterW * 0.05f);
        } else {
            r = std::min(1.0f, r + scatterW * 0.05f);
            g = std::min(1.0f, g + scatterW * 0.08f);
            b = std::min(1.0f, b + scatterW * 0.15f);
        }
        glColor3f(r, g, b);
        glVertex3f(eye.x + cosE * ca * domeR, eye.y + sinE * domeR, eye.z + cosE * sa * domeR);
    };

    for (int band = 0; band < 4; band++) {
        glBegin(GL_QUAD_STRIP);
        for (int s = 0; s <= SEG; s++) {
            emitSkyVertex(band, s);
            emitSkyVertex(band + 1, s);
        }
        glEnd();
    }

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FOG);
    glEnable(GL_LIGHTING);
}

static void renderWaterShimmer3D(float simTime, const Vec3& eye, const Vec3& sunDir) {
    Vec2 az = norm2(Vec2(sunDir.x, sunDir.z));
    if (dot2(az, az) < 0.1f) return;
    Vec2 perp(-az.z, az.x);

    const float zeroFog[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float savedFog[4];
    glGetFloatv(GL_FOG_COLOR, savedFog);
    glFogfv(GL_FOG_COLOR, zeroFog);

    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);

    float sR = 0.72f, sG = 0.88f, sB = 0.96f;
    if (g_timeOfDay == 1) { sR = 1.00f; sG = 0.68f; sB = 0.28f; }
    else if (g_timeOfDay == 2) { sR = 0.72f; sG = 0.84f; sB = 1.00f; }

    for (int i = 0; i < 96; i++) {
        float tNorm = (float)i / 96.0f;
        float along = 140.0f + std::pow(tNorm, 1.35f) * 11500.0f;
        float lateralJitter = std::sin(i * 2.37f + simTime * 1.1f) * (14.0f + tNorm * 135.0f)
                            + std::cos(i * 4.13f - simTime * 0.8f) * (8.0f + tNorm * 45.0f);
        Vec2 wc(eye.x + az.x * along + perp.x * lateralJitter,
                eye.z + az.z * along + perp.z * lateralJitter);
        if (wc.x < WORLD_X0 || wc.x > WORLD_X1 || wc.z < WORLD_Z0 || wc.z > WORLD_Z1) continue;
        if (sampleSignedSeaDist(wc.x, wc.z) > -18.0f) continue;

        float shimmer = std::max(0.0f, 0.35f + 0.65f * std::sin(simTime * (3.0f + (i % 5) * 0.8f) + i * 1.71f));
        float distFade = (1.0f - tNorm * 0.55f) * shimmer * shimmer * 0.55f;
        if (distFade < 0.02f) continue;

        float hw = 16.0f + tNorm * 58.0f;
        float hd = 2.6f + tNorm * 5.2f;
        glBegin(GL_TRIANGLE_FAN);
        glColor3f(sR * distFade, sG * distFade, sB * distFade);
        glVertex3f(wc.x, 0.12f, wc.z);
        glColor3f(0.0f, 0.0f, 0.0f);
        for (int k = 0; k <= 8; k++) {
            float a = k * (2.0f * PI_F / 8.0f);
            Vec2 ep = wc + perp * (std::cos(a) * hw) + az * (std::sin(a) * hd);
            glVertex3f(ep.x, 0.12f, ep.z);
        }
        glEnd();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glFogfv(GL_FOG_COLOR, savedFog);
    glEnable(GL_LIGHTING);
}

static void renderSkyAndClimate3D(float simTime, const Vec3& eye, const Vec3& sunDir) {
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    // 1. Gok Cisimleri (Sis kapali iken keskin Ay, kraterli yuzey, tac, 1100+ yildiz veya Gunes)
    glDisable(GL_FOG);
    if (g_timeOfDay == 2) {
        const float rad = 32000.0f;

        // 1A. Samanyolu hafif parlak yildiz kusagi (320 soluk yildiz noktasi)
        glPointSize(1.5f);
        glBegin(GL_POINTS);
        for (int i = 0; i < 820; i++) {
            uint32_t h = hash2i(i * 73 + 19, i * 151 + 43);
            float az = ((h & 2047) / 2047.0f) * 2.0f * PI_F;
            float u  = (((h >> 11) & 1023) / 1023.0f);
            float el = 0.06f + std::pow(u, 0.85f) * 1.40f;
            float cosE = std::cos(el);
            float tw = 0.42f + 0.35f * std::sin(simTime * 1.9f + i * 0.71f);
            float tint = ((h >> 21) & 15) / 15.0f;
            glColor4f(lerpf(0.76f, 0.96f, tint), lerpf(0.84f, 0.94f, tint), lerpf(1.00f, 0.86f, tint), tw);
            glVertex3f(eye.x + cosE * std::cos(az) * rad, eye.y + std::sin(el) * rad, eye.z + cosE * std::sin(az) * rad);
        }
        glEnd();

        // 1B. Orta parlaklikta Ege gece yildizlari (230 adet)
        glPointSize(2.6f);
        glBegin(GL_POINTS);
        for (int i = 0; i < 230; i++) {
            uint32_t h = hash2i(i * 193 + 311, i * 89 + 577);
            float az = ((h & 2047) / 2047.0f) * 2.0f * PI_F;
            float el = 0.08f + (((h >> 11) & 1023) / 1023.0f) * 1.38f;
            float cosE = std::cos(el);
            float tw = 0.68f + 0.32f * std::sin(simTime * 2.9f + i * 1.13f);
            bool warmStar = ((h >> 22) & 3) == 0;
            glColor4f(warmStar ? 1.00f : 0.86f, warmStar ? 0.91f : 0.94f, warmStar ? 0.74f : 1.00f, tw);
            glVertex3f(eye.x + cosE * std::cos(az) * rad, eye.y + std::sin(el) * rad, eye.z + cosE * std::sin(az) * rad);
        }
        glEnd();

        // 1C. Parlak ana takim yildizlari (55 adet)
        glPointSize(3.8f);
        glBegin(GL_POINTS);
        for (int i = 0; i < 55; i++) {
            uint32_t h = hash2i(i * 409 + 701, i * 251 + 919);
            float az = ((h & 2047) / 2047.0f) * 2.0f * PI_F;
            float el = 0.12f + (((h >> 11) & 1023) / 1023.0f) * 1.30f;
            float cosE = std::cos(el);
            float tw = 0.80f + 0.20f * std::sin(simTime * 3.6f + i * 1.7f);
            bool blueGiant = ((h >> 22) & 1) == 0;
            glColor4f(blueGiant ? 0.84f : 1.00f, blueGiant ? 0.93f : 0.92f, blueGiant ? 1.00f : 0.72f, tw);
            glVertex3f(eye.x + cosE * std::cos(az) * rad, eye.y + std::sin(el) * rad, eye.z + cosE * std::sin(az) * rad);
        }
        glEnd();

        // 1D. Gece Dolunay + Kraterli Ay Denizleri (Lunar Maria) + Gumus Atmosferik Tac
        Vec3 moonC = eye + sunDir * 31500.0f;
        Vec3 rightV = norm(cross(sunDir, Vec3(0.0f, 1.0f, 0.0f)));
        Vec3 orthoUp = norm(cross(rightV, sunDir));

        // Dis gumus-mavi corona halesi
        glBlendFunc(GL_ONE, GL_ONE);
        const float haloR = 3100.0f;
        glBegin(GL_TRIANGLE_FAN);
        glColor3f(0.18f, 0.24f, 0.36f);
        glVertex3f(moonC.x, moonC.y, moonC.z);
        for (int i = 0; i <= 32; i++) {
            float a = i * (2.0f * PI_F / 32.0f);
            Vec3 p = moonC + (rightV * std::cos(a) + orthoUp * std::sin(a)) * haloR;
            glColor3f(0.0f, 0.0f, 0.0f);
            glVertex3f(p.x, p.y, p.z);
        }
        glEnd();

        // Ay diski (krater/maria golgeli 2 halkali detay)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        const float moonR = 920.0f;
        auto emitMoonPt = [&](float lx, float ly, float rimScale) {
            float maria = std::sin(lx * 3.6f + 0.5f) * std::cos(ly * 3.2f - 0.3f)
                        + 0.55f * std::cos((lx - ly) * 5.2f + 1.1f);
            float shade = clampf(0.84f + 0.13f * maria, 0.68f, 0.99f) * rimScale;
            // Hafif gibbous faz golgesi (sol alt kenarda yumusak terminator)
            float phase = clampf(0.86f + 0.22f * (lx * 0.72f + ly * 0.45f), 0.56f, 1.04f);
            float br = shade * phase;
            glColor4f(std::min(1.0f, br * 1.00f), std::min(1.0f, br * 0.98f), std::min(1.0f, br * 0.90f), 1.0f);
            Vec3 p = moonC + (rightV * lx + orthoUp * ly) * moonR;
            glVertex3f(p.x, p.y, p.z);
        };
        glBegin(GL_TRIANGLES);
        const int MSEG = 32;
        const float midR = 0.56f;
        for (int i = 0; i < MSEG; i++) {
            float a0 = i * (2.0f * PI_F / MSEG);
            float a1 = (i + 1) * (2.0f * PI_F / MSEG);
            float c0 = std::cos(a0), s0 = std::sin(a0);
            float c1 = std::cos(a1), s1 = std::sin(a1);
            // Ic halka
            emitMoonPt(0.0f, 0.0f, 1.00f);
            emitMoonPt(c0 * midR, s0 * midR, 0.96f);
            emitMoonPt(c1 * midR, s1 * midR, 0.96f);
            // Dis halka
            emitMoonPt(c0 * midR, s0 * midR, 0.96f);
            emitMoonPt(c0, s0, 0.84f);
            emitMoonPt(c1, s1, 0.84f);

            emitMoonPt(c0 * midR, s0 * midR, 0.96f);
            emitMoonPt(c1, s1, 0.84f);
            emitMoonPt(c1 * midR, s1 * midR, 0.96f);
        }
        glEnd();
    } else {
        // Gunes veya Kizil Korfez Gunbatimi diski + atmosferik hale
        Vec3 sunC = eye + sunDir * 31500.0f;
        Vec3 rightV = norm(cross(sunDir, Vec3(0.0f, 1.0f, 0.0f)));
        Vec3 orthoUp = norm(cross(rightV, sunDir));
        float diskR = (g_timeOfDay == 1) ? 2150.0f : 1380.0f;
        glBegin(GL_TRIANGLE_FAN);
        if (g_timeOfDay == 1) glColor4f(1.0f, 0.90f, 0.48f, 0.98f);
        else glColor4f(1.0f, 0.99f, 0.90f, 0.98f);
        glVertex3f(sunC.x, sunC.y, sunC.z);
        for (int i = 0; i <= 28; i++) {
            float a = i * (2.0f * PI_F / 28.0f);
            Vec3 p = sunC + (rightV * std::cos(a) + orthoUp * std::sin(a)) * diskR;
            if (g_timeOfDay == 1) glColor4f(1.0f, 0.36f, 0.12f, 0.0f);
            else glColor4f(1.0f, 0.94f, 0.68f, 0.0f);
            glVertex3f(p.x, p.y, p.z);
        }
        glEnd();
    }

    // 2. Ege Imbat Ruzgariyla Suruklenen Katmanli 3D Kumulus Bulutlari (1680m..2250m irtifa)
    glEnable(GL_FOG);
    float cTopR = 0.99f, cTopG = 0.99f, cTopB = 1.00f, cBotR = 0.78f, cBotG = 0.83f, cBotB = 0.90f, cA = 0.46f;
    if (g_timeOfDay == 1) {
        cTopR = 1.00f; cTopG = 0.76f; cTopB = 0.54f;
        cBotR = 0.76f; cBotG = 0.44f; cBotB = 0.40f; cA = 0.50f;
    } else if (g_timeOfDay == 2) {
        cTopR = 0.22f; cTopG = 0.28f; cTopB = 0.42f;
        cBotR = 0.10f; cBotG = 0.13f; cBotB = 0.22f; cA = 0.32f;
    }

    static const float puffDx[5] = {-0.38f, -0.16f, 0.02f, 0.22f, 0.40f};
    static const float puffDz[5] = { 0.12f, -0.16f, 0.00f, 0.15f,-0.10f};
    static const float puffH[5]  = {36.0f,  52.0f,  68.0f, 54.0f, 34.0f};
    static const float puffS[5]  = {0.42f,  0.54f,  0.64f, 0.52f, 0.40f};

    for (int c = 0; c < 48; c++) {
        uint32_t h = hash2i(c * 41 + 7, c * 97 + 13);
        float baseX = WORLD_X0 + 3000.0f + ((h & 4095) / 4095.0f) * (WORLD_X1 - WORLD_X0 - 6000.0f);
        float baseZ = WORLD_Z0 + 3000.0f + (((h >> 12) & 4095) / 4095.0f) * (WORLD_Z1 - WORLD_Z0 - 6000.0f);
        float driftX = std::fmod(simTime * 9.0f + c * 150.0f, 32000.0f) - 16000.0f;
        float cx = baseX + driftX * 0.12f;
        float cz = baseZ;
        float cy = 1680.0f + (c % 6) * 95.0f;
        float cw = 480.0f + (c % 5) * 110.0f;
        float cd = 360.0f + (c % 4) * 90.0f;
        float ddx = cx - eye.x, ddz = cz - eye.z;
        if (ddx * ddx + ddz * ddz > 24000.0f * 24000.0f) continue;

        for (int p = 0; p < 5; p++) {
            float px = cx + puffDx[p] * cw;
            float pz = cz + puffDz[p] * cd;
            float py = cy + (2 - std::abs(p - 2)) * 12.0f;
            float rx = cw * puffS[p];
            float rz = cd * puffS[p];
            glBegin(GL_TRIANGLE_FAN);
            glColor4f(cTopR, cTopG, cTopB, cA);
            glVertex3f(px, py + puffH[p], pz);
            for (int k = 0; k <= 12; k++) {
                float a = k * (2.0f * PI_F / 12.0f);
                glColor4f(cBotR, cBotG, cBotB, 0.0f);
                glVertex3f(px + std::cos(a) * rx, py, pz + std::sin(a) * rz);
            }
            glEnd();
        }
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
}

// =====================================================================================
//  ANA SAHNE RENDER FONKSIYONU
// =====================================================================================
static void renderScene3D(int winW, int winH, float simTime) {
    glViewport(0, 0, winW, winH);

    // Gündüz / Kızıl Körfez Günbatımı / Gece Ay Işığı & Parlayan Şehir atmosfer renkleri
    Vec3 sunDir = getCelestialSunMoonDir();
    float skyR = 0.66f, skyG = 0.81f, skyB = 0.93f;
    float sunAmb[4]  = {0.42f, 0.44f, 0.49f, 1.0f};
    float sunDiff[4] = {0.98f, 0.95f, 0.88f, 1.0f};
    float sunPos[4]  = {sunDir.x, sunDir.y, sunDir.z, 0.0f};
    float modelAmb[4] = {0.06f, 0.06f, 0.08f, 1.0f};
    if (g_timeOfDay == 1) {
        // Kizil Korfez Gunbatimi
        skyR = 0.88f; skyG = 0.48f; skyB = 0.32f;
        sunAmb[0] = 0.44f; sunAmb[1] = 0.28f; sunAmb[2] = 0.26f;
        sunDiff[0] = 1.00f; sunDiff[1] = 0.56f; sunDiff[2] = 0.30f;
    } else if (g_timeOfDay == 2) {
        // Gece Parlayan Sehir ve Ay Isigi (dusuk baz isik -> parlayan pencere/fener kontrasti)
        skyR = 0.05f; skyG = 0.07f; skyB = 0.15f;
        sunAmb[0] = 0.05f; sunAmb[1] = 0.06f; sunAmb[2] = 0.12f;
        sunDiff[0] = 0.12f; sunDiff[1] = 0.16f; sunDiff[2] = 0.28f;
        modelAmb[0] = 0.0f; modelAmb[1] = 0.0f; modelAmb[2] = 0.0f;
    }

    glClearColor(skyR, skyG, skyB, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (double)winW / (double)std::max(1, winH);
    gluPerspective(62.0, aspect, 1.5, 48000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 4 Kamera Modu ([C]: Chase, Kokpit HUD, Kanat, Kule)
    g_camFwd = norm(lerpV(g_camFwd, g_ac.fwd, 0.14f));
    Vec3 targetCamUp = norm(lerpV(Vec3(0.0f, 1.0f, 0.0f), g_ac.up, 0.32f));
    g_camUp = norm(lerpV(g_camUp, targetCamUp, 0.14f));

    Vec3 eye, center, upVec;
    if (g_camMode == CAM_CHASE) {
        eye = g_ac.pos - g_camFwd * 23.5f + g_camUp * 5.6f;
        float terr = sampleTerrainElev(eye.x, eye.z);
        if (eye.y < terr + 2.2f) eye.y = terr + 2.2f;
        center = g_ac.pos + g_ac.fwd * 14.0f + g_ac.up * 1.2f;
        upVec = g_camUp;
    } else if (g_camMode == CAM_COCKPIT) {
        eye = g_ac.pos + g_ac.up * 1.15f + g_ac.fwd * 2.4f;
        center = eye + g_ac.fwd * 50.0f;
        upVec = g_ac.up;
    } else if (g_camMode == CAM_WING) {
        eye = g_ac.pos + g_ac.right * (AC_SPECS[g_ac.type].wingSpan * 0.62f) + g_ac.up * 1.1f - g_ac.fwd * 1.8f;
        center = g_ac.pos + g_ac.fwd * 6.0f;
        upVec = g_ac.up;
    } else {
        Vec2 sk = geoPt(SAAT_LAT, SAAT_LON);
        eye = Vec3(sk.x, sampleTerrainElev(sk.x, sk.z) + 65.0f, sk.z);
        if (g_ac.pos.z > 8000.0f) {
            Vec2 ltbj = geoPt(LTBJ_LAT, LTBJ_LON);
            eye = Vec3(ltbj.x - 350.0f, LTBJ_ELEV + 68.0f, ltbj.z);
        }
        center = g_ac.pos;
        upVec = Vec3(0, 1, 0);
    }

    // Darbe, patlama ve atis sirasinda kamera sarsintisi (ekran titremesi)
    float shakeDt = clampf(simTime - g_lastShakeSimTime, 0.0f, 0.1f);
    if (simTime < g_lastShakeSimTime) shakeDt = 0.016f;
    g_lastShakeSimTime = simTime;
    if (g_camShakeTimer > 0.0f && shakeDt > 0.0f) {
        g_camShakeTimer = std::max(0.0f, g_camShakeTimer - shakeDt);
        g_camShakePhase += shakeDt * 58.0f;
    }
    Vec3 baseFwd = norm(center - eye);
    Vec3 baseRight = norm(cross(baseFwd, upVec));
    Vec3 baseUp = norm(cross(baseRight, baseFwd));
    if (g_camShakeTimer > 0.0f && g_camShakeDur > 0.0f) {
        float env = clampf(g_camShakeTimer / g_camShakeDur, 0.0f, 1.0f);
        float amp = g_camShakeMag * env * env;
        float sx = std::sin(g_camShakePhase * 1.13f) + 0.5f * std::cos(g_camShakePhase * 2.71f);
        float sy = std::cos(g_camShakePhase * 1.47f) + 0.5f * std::sin(g_camShakePhase * 3.19f);
        float sr = std::sin(g_camShakePhase * 1.91f) * 0.045f * amp;
        Vec3 shakeOff = baseRight * (sx * amp) + baseUp * (sy * amp);
        eye += shakeOff;
        center += shakeOff * 0.65f;
        upVec = norm(baseUp + baseRight * sr);
        baseFwd = norm(center - eye);
        baseRight = norm(cross(baseFwd, upVec));
        baseUp = norm(cross(baseRight, baseFwd));
    }
    g_lastCamEye = eye;
    g_lastCamFwd = baseFwd;
    g_lastCamRight = baseRight;
    g_lastCamUp = baseUp;
    g_lastWinW = winW;
    g_lastWinH = winH;

    gluLookAt(eye.x, eye.y, eye.z, center.x, center.y, center.z, upVec.x, upVec.y, upVec.z);

    // 0. Cok-Bantli Atmosfer Kubbesi (gok gecisi + gunes/ay sacilma isiltisi)
    renderAtmosphereDome3D(eye, sunDir);

    // Perspektif sis (ufuk rengiyle tam uyumlu) ve tek-tarafli hizli aydinlatma
    glEnable(GL_FOG);
    float fogCol[4] = {skyR, skyG, skyB, 1.0f};
    glFogfv(GL_FOG_COLOR, fogCol);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 4800.0f);
    glFogf(GL_FOG_END, 28000.0f);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, modelAmb);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glLightfv(GL_LIGHT0, GL_POSITION, sunPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, sunAmb);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, sunDiff);

    Vec2 camFwd2D = norm2(Vec2(center.x - eye.x, center.z - eye.z));
    const float tileW = (WORLD_X1 - WORLD_X0) / (float)WORLD_TILE_N;
    const float tileH = (WORLD_Z1 - WORLD_Z0) / (float)WORLD_TILE_N;
    const float tileHalfDiag = 0.5f * std::sqrt(tileW * tileW + tileH * tileH) + 180.0f;

    auto tileInFrustum = [&](float tcx, float tcz, float rad, float maxDist) -> bool {
        float dx = tcx - eye.x, dz = tcz - eye.z;
        float d2 = dx * dx + dz * dz;
        float limit = maxDist + rad;
        if (d2 > limit * limit) return false;
        float along = dx * camFwd2D.x + dz * camFwd2D.z;
        if (along < -rad - 300.0f) return false;
        return true;
    };

    // 1. 8x8 Spatial Tile Arazi + Ufuk Etegi + Rihtim + Tile Otoyol + Raylar + Simge Yapilar
    glEnable(GL_TEXTURE_2D);
    if (g_dlHorizonApron) glCallList(g_dlHorizonApron);
    for (int tz = 0; tz < WORLD_TILE_N; tz++) {
        float tcz = WORLD_Z0 + (tz + 0.5f) * tileH;
        for (int tx = 0; tx < WORLD_TILE_N; tx++) {
            int tidx = tz * WORLD_TILE_N + tx;
            if (!g_dlTerrainTile[tidx]) continue;
            float tcx = WORLD_X0 + (tx + 0.5f) * tileW;
            if (tileInFrustum(tcx, tcz, tileHalfDiag, 29000.0f)) {
                glCallList(g_dlTerrainTile[tidx]);
            }
        }
    }
    glCallList(g_dlQuayAndPromenade);
    for (int tz = 0; tz < WORLD_TILE_N; tz++) {
        float tcz = WORLD_Z0 + (tz + 0.5f) * tileH;
        for (int tx = 0; tx < WORLD_TILE_N; tx++) {
            int tidx = tz * WORLD_TILE_N + tx;
            if (!g_dlRoadTile[tidx]) continue;
            float tcx = WORLD_X0 + (tx + 0.5f) * tileW;
            if (tileInFrustum(tcx, tcz, g_roadTileRadius[tidx], 16500.0f)) {
                glCallList(g_dlRoadTile[tidx]);
            }
        }
    }
    if (g_dlRails) glCallList(g_dlRails);
    glCallList(g_dlLandmarksAndAirports);

    // 1B. 8x8 Spatial Tile Ege Korfezi Canli Su Yuzeyi + Gunes/Ay Iltisi (Specular Shimmer)
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glTranslatef(simTime * 0.005f, simTime * 0.003f, 0.0f);
    glMatrixMode(GL_MODELVIEW);
    for (int tz = 0; tz < WORLD_TILE_N; tz++) {
        float tcz = WORLD_Z0 + (tz + 0.5f) * tileH;
        for (int tx = 0; tx < WORLD_TILE_N; tx++) {
            int tidx = tz * WORLD_TILE_N + tx;
            if (!g_dlSeaTile[tidx]) continue;
            float tcx = WORLD_X0 + (tx + 0.5f) * tileW;
            if (tileInFrustum(tcx, tcz, tileHalfDiag, 29000.0f)) {
                glCallList(g_dlSeaTile[tidx]);
            }
        }
    }
    glMatrixMode(GL_TEXTURE);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    renderWaterShimmer3D(simTime, eye, sunDir);

    // 1C. Ay, 1100+ Yildiz, Gunes ve 5-Puf Hacimli 3D Kumulus Bulutlari
    renderSkyAndClimate3D(simTime, eye, sunDir);

    // 2. 50.000+ Bina 400x400m CityChunk Multi-LOD Derleme ve Cizimi
    const float tanHalfH = (float)aspect * std::tan(31.0f * DEG2RAD);
    const float frustumSlope = tanHalfH + 0.16f; // Yuvarlanma (bank) payi dahil yatay gorus konisi
    struct PendingChunk { int idx; float d2; };
    std::vector<PendingChunk> uncompiledVis;
    std::vector<int> visGround, visBuildings, visSkyline, visNearDetails, visMidTrees, visNight;
    visGround.reserve(350);
    visBuildings.reserve(450);
    visSkyline.reserve(300);
    visNearDetails.reserve(160);
    visMidTrees.reserve(220);
    if (g_timeOfDay >= 1) visNight.reserve(550);

    for (int i = 0; i < CHUNK_NX * CHUNK_NZ; i++) {
        CityChunk& ck = g_cityChunks[i];
        bool hasV2R = (i < (int)g_chunkV2Roads.size() && !g_chunkV2Roads[i].empty());
        bool hasGard = (i < (int)g_chunkGardens.size() && !g_chunkGardens[i].empty());
        bool hasSide = (!g_hasV2Roads && i < (int)g_chunkSideStreets.size() && !g_chunkSideStreets[i].empty());
        if (ck.bldgs.empty() && ck.trees.empty() && !hasV2R && !hasGard && !hasSide) continue;

        float dx = ck.cx - eye.x, dz = ck.cz - eye.z;
        float d2 = dx * dx + dz * dz;
        if (d2 > 14500.0f * 14500.0f) continue;

        float along = dx * camFwd2D.x + dz * camFwd2D.z;
        if (along < -380.0f) continue;
        if (d2 > 600.0f * 600.0f && along > 180.0f) {
            float lat = std::fabs(-dx * camFwd2D.z + dz * camFwd2D.x);
            if (lat > along * frustumSlope + 340.0f) continue;
        }

        if (!ck.compiled) {
            uncompiledVis.push_back({i, d2});
            continue;
        }

        if (d2 <= 4200.0f * 4200.0f) visGround.push_back(i);
        if (d2 <= 6800.0f * 6800.0f) {
            visBuildings.push_back(i);
            if (d2 <= 2200.0f * 2200.0f) visNearDetails.push_back(i);
            else if (d2 <= 4200.0f * 4200.0f) visMidTrees.push_back(i);
        } else {
            visSkyline.push_back(i);
        }
        if (g_timeOfDay >= 1 && d2 <= 6800.0f * 6800.0f) visNight.push_back(i);
    }

    if (!uncompiledVis.empty()) {
        std::sort(uncompiledVis.begin(), uncompiledVis.end(),
                  [](const PendingChunk& a, const PendingChunk& b) { return a.d2 < b.d2; });
        size_t budget = g_headlessTour ? uncompiledVis.size() : std::min<size_t>(32u, uncompiledVis.size());
        for (size_t k = 0; k < budget; k++) {
            int idx = uncompiledVis[k].idx;
            compileChunkIfNeeded(g_cityChunks[idx]);
            float d2 = uncompiledVis[k].d2;
            if (d2 <= 4200.0f * 4200.0f) visGround.push_back(idx);
            if (d2 <= 6800.0f * 6800.0f) {
                visBuildings.push_back(idx);
                if (d2 <= 2200.0f * 2200.0f) visNearDetails.push_back(idx);
                else if (d2 <= 4200.0f * 4200.0f) visMidTrees.push_back(idx);
            } else {
                visSkyline.push_back(idx);
            }
            if (g_timeOfDay >= 1 && d2 <= 6800.0f * 6800.0f) visNight.push_back(idx);
        }
    }

    // 2A. Mahalle Caddeleri, Ara Sokaklar, Yaya Kaldirimlari ve Ada Bahceleri (< 4.2 km)
    for (int idx : visGround) {
        if (g_cityChunks[idx].dlGround) glCallList(g_cityChunks[idx].dlGround);
    }

    // 2B. Yakin & Orta Mesafe Ana Bina Govdeleri (< 5.2 km, arka yuzey CW culling)
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    glCullFace(GL_BACK);
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_APT_FACADE]);
    for (int idx : visBuildings) if (g_cityChunks[idx].dlApt) glCallList(g_cityChunks[idx].dlApt);

    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GLASS_TOWER]);
    for (int idx : visBuildings) if (g_cityChunks[idx].dlGlass) glCallList(g_cityChunks[idx].dlGlass);

    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_INDUSTRIAL]);
    for (int idx : visBuildings) if (g_cityChunks[idx].dlInd) glCallList(g_cityChunks[idx].dlInd);

    // 2C. Uzak Ufuk Skyline Kule/Rezidans LOD'lari (7.6km..14.5km)
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_GLASS_TOWER]);
    for (int idx : visSkyline) if (g_cityChunks[idx].dlSkylineLod) glCallList(g_cityChunks[idx].dlSkylineLod);
    glDisable(GL_CULL_FACE);

    // 2D. Catilar (beton teras + kiremit kirma) (< 7.6 km)
    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_CONCRETE]);
    for (int idx : visBuildings) if (g_cityChunks[idx].dlRoof) glCallList(g_cityChunks[idx].dlRoof);

    glBindTexture(GL_TEXTURE_2D, g_tex[TEX_ROOF_TILE]);
    for (int idx : visBuildings) if (g_cityChunks[idx].dlTileRoof) glCallList(g_cityChunks[idx].dlTileRoof);

    // 2E. Yakin Mimari Balkon/Teras Detaylari + Katmanli 3D Agaclar (< 3.4 km) ve Orta Mesafe Agac Silueti (3.4..5.8 km)
    for (int idx : visNearDetails) if (g_cityChunks[idx].dlNearDetails) glCallList(g_cityChunks[idx].dlNearDetails);
    glDisable(GL_TEXTURE_2D);
    for (int idx : visMidTrees) if (g_cityChunks[idx].dlMidTrees) glCallList(g_cityChunks[idx].dlMidTrees);

    // 2F. Gece & Alacakaranlik Sehir Isiklari (Parlayan Bina Pencereleri + Gökdelenler + Kordon/Bulvar Lambalari + Ucak Ikaz Fenerleri)
    if (g_timeOfDay >= 1) {
        const float zeroFog[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        glFogfv(GL_FOG_COLOR, zeroFog);
        glDisable(GL_LIGHTING);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glDepthMask(GL_FALSE);

        // Bina pencere isiklari (dlNightWindows, CW culling + TEX_NIGHT_WINDOWS)
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, g_tex[TEX_NIGHT_WINDOWS]);
        glEnable(GL_CULL_FACE);
        glFrontFace(GL_CW);
        glCullFace(GL_BACK);
        for (int idx : visNight) {
            if (g_cityChunks[idx].dlNightWindows) glCallList(g_cityChunks[idx].dlNightWindows);
        }

        // Bayrakli & Konak simge cam gokdelenleri gece aydinlatmasi (burgu acisiyla uyumlu)
        glDisable(GL_CULL_FACE);
        float towerBright = (g_timeOfDay == 2) ? 0.95f : 0.45f;
        glBegin(GL_QUADS);
        for (const auto& tl : g_landmarkGlowTowers) {
            float dx = tl.cx - eye.x, dz = tl.cz - eye.z;
            if (dx * dx + dz * dz > 16000.0f * 16000.0f) continue;
            if (dx * camFwd2D.x + dz * camFwd2D.z < -tl.baseW * 2.0f) continue;
            glColor3f(tl.r * towerBright, tl.g * towerBright, tl.b * towerBright);
            const int SLICES = 8;
            for (int s = 0; s < SLICES; s++) {
                float t0 = (float)s / (float)SLICES;
                float t1 = (float)(s + 1) / (float)SLICES;
                float y0 = tl.baseY + t0 * tl.height;
                float y1 = tl.baseY + t1 * tl.height;
                float a0 = t0 * tl.twistDeg * DEG2RAD;
                float a1 = t1 * tl.twistDeg * DEG2RAD;
                float hw0 = tl.baseW * 0.5f * (1.0f - 0.14f * t0), hd0 = tl.baseD * 0.5f * (1.0f - 0.14f * t0);
                float hw1 = tl.baseW * 0.5f * (1.0f - 0.14f * t1), hd1 = tl.baseD * 0.5f * (1.0f - 0.14f * t1);
                auto cPt = [&](float hw, float hd, float ang, int idx) {
                    float lx = (idx == 0 || idx == 3) ? -hw : hw;
                    float lz = (idx == 0 || idx == 1) ? -hd : hd;
                    float ca = std::cos(ang), sa = std::sin(ang);
                    return Vec3(tl.cx + lx * ca - lz * sa, 0.0f, tl.cz + lx * sa + lz * ca);
                };
                for (int side = 0; side < 4; side++) {
                    int next = (side + 1) & 3;
                    Vec3 p00 = cPt(hw0, hd0, a0, side); p00.y = y0;
                    Vec3 p10 = cPt(hw0, hd0, a0, next); p10.y = y0;
                    Vec3 p11 = cPt(hw1, hd1, a1, next); p11.y = y1;
                    Vec3 p01 = cPt(hw1, hd1, a1, side); p01.y = y1;
                    glTexCoord2f(0.0f, t0 * 8.0f); glVertex3f(p00.x, p00.y, p00.z);
                    glTexCoord2f(2.0f, t0 * 8.0f); glVertex3f(p10.x, p10.y, p10.z);
                    glTexCoord2f(2.0f, t1 * 8.0f); glVertex3f(p11.x, p11.y, p11.z);
                    glTexCoord2f(0.0f, t1 * 8.0f); glVertex3f(p01.x, p01.y, p01.z);
                }
            }
        }
        glEnd();
        glDisable(GL_TEXTURE_2D);

        // Kordon / Sahil Bulvari ve Ana Arter Sokak Lambalari (mesafeye gore 2 point size kademesi)
        float stBright = (g_timeOfDay == 2) ? 0.92f : 0.40f;
        glPointSize(3.6f);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < g_nightStreetLights.size(); i++) {
            const Vec3& lp = g_nightStreetLights[i];
            float dx = lp.x - eye.x, dz = lp.z - eye.z;
            float d2 = dx * dx + dz * dz;
            if (d2 > 2600.0f * 2600.0f) continue;
            if (d2 > 250.0f * 250.0f && (dx * camFwd2D.x + dz * camFwd2D.z) < -100.0f) continue;
            glColor3f(1.00f * stBright, 0.78f * stBright, 0.38f * stBright);
            glVertex3f(lp.x, lp.y, lp.z);
        }
        glEnd();

        glPointSize(2.0f);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < g_nightStreetLights.size(); i++) {
            const Vec3& lp = g_nightStreetLights[i];
            float dx = lp.x - eye.x, dz = lp.z - eye.z;
            float d2 = dx * dx + dz * dz;
            if (d2 <= 2600.0f * 2600.0f || d2 > 9500.0f * 9500.0f) continue;
            if ((dx * camFwd2D.x + dz * camFwd2D.z) < 0.0f) continue;
            glColor3f(0.86f * stBright, 0.60f * stBright, 0.25f * stBright);
            glVertex3f(lp.x, lp.y, lp.z);
        }
        glEnd();

        // Kule & Kopru & Gokdelen yanip sonen kirmizi havacilik ikaz fenerleri (beacon)
        glPointSize(5.5f);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < g_nightBeaconLights.size(); i++) {
            const Vec3& bp = g_nightBeaconLights[i];
            float dx = bp.x - eye.x, dz = bp.z - eye.z;
            float d2 = dx * dx + dz * dz;
            if (d2 > 18000.0f * 18000.0f) continue;
            float pulse = 0.28f + 0.72f * std::max(0.0f, std::sin(simTime * 3.6f + (float)i * 0.9f));
            glColor3f(1.0f * pulse, 0.14f * pulse, 0.08f * pulse);
            glVertex3f(bp.x, bp.y, bp.z);
        }
        glEnd();

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glFogfv(GL_FOG_COLOR, fogCol);
        glEnable(GL_LIGHTING);
    }

    // 3. Canli IZBAN trenleri (8 adet), Korfez vapurlari (6 adet), otoyol trafigi ve 75m elmas beacon'lar
    renderDynamicTrafficAndBeacons(simTime, g_timeOfDay, eye, camFwd2D);

    // 4. Ucak modeli, gece seyir/strobe fenerleri ve akrobasi/kaza partikulleri
    if (g_camMode != CAM_COCKPIT) {
        renderAircraftModel(g_ac, simTime);
        if (g_timeOfDay >= 1) {
            float halfW = AC_SPECS[g_ac.type].wingSpan * 0.49f;
            Vec3 leftTip  = g_ac.pos - g_ac.right * halfW - g_ac.fwd * 0.5f;
            Vec3 rightTip = g_ac.pos + g_ac.right * halfW - g_ac.fwd * 0.5f;
            Vec3 tailTip  = g_ac.pos - g_ac.fwd * 4.5f + g_ac.up * 1.4f;
            float strobe = std::max(0.0f, std::sin(simTime * 8.5f)) > 0.72f ? 1.0f : 0.15f;
            glDisable(GL_LIGHTING);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glPointSize(5.5f);
            glBegin(GL_POINTS);
            glColor3f(1.0f, 0.15f, 0.12f); glVertex3f(leftTip.x, leftTip.y, leftTip.z);
            glColor3f(0.15f, 1.0f, 0.35f); glVertex3f(rightTip.x, rightTip.y, rightTip.z);
            glColor3f(1.0f * strobe, 0.96f * strobe, 0.88f * strobe); glVertex3f(tailTip.x, tailTip.y, tailTip.z);
            glEnd();
            glDisable(GL_BLEND);
            glEnable(GL_LIGHTING);
        }
    }
    computeWeaponAiming();
    renderWeaponsAndFires3D(simTime, baseRight, baseUp, eye);
    renderExplosions3D(baseRight, baseUp);
    renderParticles(baseRight, baseUp);
}

static void savePPM(const char* path, int w, int h) {
    std::vector<uint8_t> px(w * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; y--) {
        std::fwrite(&px[y * w * 3], 1, w * 3, f);
    }
    std::fclose(f);
}

static std::string resolveOsmWorldBinPath() {
    if (const char* envPath = std::getenv("IZMIR_WORLD_BIN")) {
        if (envPath[0] && access(envPath, R_OK) == 0) return envPath;
    }
    if (access("izmir_osm_world.bin", R_OK) == 0) return "izmir_osm_world.bin";
    char exeBuf[1024] = {0};
    ssize_t n = readlink("/proc/self/exe", exeBuf, sizeof(exeBuf) - 1);
    if (n > 0) {
        exeBuf[n] = '\0';
        std::string exeDir(exeBuf);
        size_t slash = exeDir.find_last_of('/');
        if (slash != std::string::npos) {
            std::string cand = exeDir.substr(0, slash + 1) + "izmir_osm_world.bin";
            if (access(cand.c_str(), R_OK) == 0) return cand;
        }
    }
    if (access("../izmir_osm_world.bin", R_OK) == 0) return "../izmir_osm_world.bin";
    return "izmir_osm_world.bin";
}

static std::string resolveRenderOutDir() {
    const char* envDir = std::getenv("IZMIR_OUT_DIR");
    std::string dir = (envDir && envDir[0]) ? std::string(envDir) : std::string("renders");
    mkdir(dir.c_str(), 0755);
    return dir;
}

static int runHeadlessCityInspectionTour(bool nightAndBenchOnly, bool combatPreview = false) {
    g_headlessTour = true;
    int W = 1280, H = 720;
    auto getPlat = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = getPlat ? getPlat(0x31DD, EGL_DEFAULT_DISPLAY, nullptr) : eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint maj, min;
    if (!eglInitialize(dpy, &maj, &min)) {
        std::fprintf(stderr, "EGL initialize failed!\n");
        return 1;
    }
    eglBindAPI(EGL_OPENGL_API);
    EGLint cfgAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    EGLConfig cfg; EGLint num;
    eglChooseConfig(dpy, cfgAttribs, &cfg, 1, &num);
    EGLint pbAttribs[] = { EGL_WIDTH, W, EGL_HEIGHT, H, EGL_NONE };
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pbAttribs);
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, nullptr);
    eglMakeCurrent(dpy, surf, surf, ctx);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);

    buildProceduralTextures();
    buildTerrainAndSeaDisplayList();
    buildQuayAndPromenadeDisplayList();
    buildIzbanAndRoadsDisplayList();
    buildLandmarksAndAirportsDisplayList();

    const std::string outDir = resolveRenderOutDir();
    auto saveFramePng = [&](const char* pngName) {
        const char* ppmPath = "/tmp/izmir_verif_frame.ppm";
        savePPM(ppmPath, W, H);
        char pngPath[512];
        std::snprintf(pngPath, sizeof(pngPath), "%s/%s", outDir.c_str(), pngName);
        char cmd[1200];
        std::snprintf(cmd, sizeof(cmd), "ffmpeg -y -loglevel error -i \"%s\" \"%s\" && rm -f \"%s\"", ppmPath, pngPath, ppmPath);
        int rc = system(cmd);
        (void)rc;
    };

    if (!nightAndBenchOnly) {
        std::printf("\n====================================================================\n");
        std::printf(" OTONOM IZMIR SEHIR DENETIM TURU : 24 ILCE VE BOLGE GOZLEMI\n");
        std::printf("====================================================================\n");

        g_timeOfDay = 0;
        for (int i = 0; i < NUM_TOUR_STOPS; i++) {
            const auto& ts = TOUR_STOPS[i];
            Vec2 p = geoPt(ts.lat, ts.lon);
            float terr = sampleTerrainElev(p.x, p.z);
            g_ac.type = AC_CESSNA;
            g_ac.pos = Vec3(p.x, terr + ts.alt, p.z);
            setOrientationFromYawPitchRoll(ts.hdg, ts.pitch, 0.0f);
            g_ac.vel = g_ac.fwd * 45.0f;
            g_ac.airspeedMs = 45.0f;
            g_ac.aglM = ts.alt;
            syncCameraToAircraft();

            renderScene3D(W, H, 10.0f + i * 2.5f);
            renderAvionicsHUD(W, H, 60.0f);
            glFinish();

            char pngName[64];
            std::snprintf(pngName, sizeof(pngName), "verif_tour_%02d.png", i + 1);
            saveFramePng(pngName);

            std::printf(" [DENETIM %02d/%d] %-35s : Pos=(%+.0f, %+.0f, %+.0f) AGL=%.0fm ... TAMAM\n",
                        i + 1, NUM_TOUR_STOPS, ts.district, g_ac.pos.x, g_ac.pos.y, g_ac.pos.z, g_ac.aglM);
        }
    }

    // Gece, Gunbatimi, Jet Model & Korfez Performans Benchmark Cekimleri
    struct SpecialShot {
        const char* pngName;
        const char* label;
        int timeOfDay;
        int acType;
        double lat, lon;
        float alt, hdg, pitch;
    };
    static const SpecialShot SHOTS[] = {
        {"verif_night_01_konak.png",    "Gece Konak & Ay-Yildiz Korfez (F-16)",      2, AC_JET,    38.4165, 27.1245, 130.0f,  26.0f, -2.5f},
        {"verif_night_02_bayrakli.png", "Gece Bayrakli Folkart/Mistral & Karsiyaka", 2, AC_CESSNA, 38.4420, 27.1490, 220.0f,  28.0f, -4.5f},
        {"verif_night_03_alsancak.png", "Gece Alsancak Kordon & Kruvaziyer/Vapur",   2, AC_JET,    38.4310, 27.1330, 145.0f,  22.0f, -5.0f},
        {"verif_sunset_04_bostanli.png","Gunbatimi Korfez Iltisi & F-16C",           1, AC_JET,    38.4460, 27.1100, 150.0f, 265.0f, -4.0f},
        {"verif_day_05_jet_kordon.png", "Gunduz F-16C Kordon & Gulf Yakini",         0, AC_JET,    38.4250, 27.1310, 135.0f,  28.0f, -5.5f}
    };

    std::printf("\n====================================================================\n");
    std::printf(" GECE / GUNBATIMI / F-16 MODEL & KORFEZ PERFORMANS DENETIMI\n");
    std::printf("====================================================================\n");
    for (size_t s = 0; s < sizeof(SHOTS) / sizeof(SHOTS[0]); s++) {
        const auto& sh = SHOTS[s];
        g_timeOfDay = sh.timeOfDay;
        g_ac.type = sh.acType;
        g_ac.gearDown = false;
        g_ac.gearAnim = 0.0f;
        Vec2 p = geoPt(sh.lat, sh.lon);
        float terr = std::max(0.0f, sampleTerrainElev(p.x, p.z));
        g_ac.pos = Vec3(p.x, terr + sh.alt, p.z);
        setOrientationFromYawPitchRoll(sh.hdg, sh.pitch, 0.0f);
        g_ac.vel = g_ac.fwd * ((sh.acType == AC_JET) ? 140.0f : 55.0f);
        g_ac.airspeedMs = len(g_ac.vel);
        g_ac.aglM = sh.alt;
        syncCameraToAircraft();

        // Warmup + GPU benchmark
        renderScene3D(W, H, 22.0f + s * 3.1f);
        glFinish();

        double t0 = nowSec();
        const int BENCH_FRAMES = 10;
        for (int f = 0; f < BENCH_FRAMES; f++) {
            renderScene3D(W, H, 22.0f + s * 3.1f + f * 0.016f);
            glFinish();
        }
        double avgMs = ((nowSec() - t0) * 1000.0) / BENCH_FRAMES;
        renderScene3D(W, H, 22.0f + s * 3.1f);
        renderAvionicsHUD(W, H, (float)(1000.0 / std::max(0.1, avgMs)));
        glFinish();

        saveFramePng(sh.pngName);
        std::printf(" [CEKIM %zu/5] %-42s | AvgRender=%.2f ms (%.1f FPS) -> %s/%s\n",
                    s + 1, sh.label, avgMs, 1000.0 / std::max(0.1, avgMs), outDir.c_str(), sh.pngName);
    }
    std::printf("====================================================================\n\n");

    if (combatPreview) {
        teleportAircraft(3); // Alsancak Kordon / Bina Bolgesi
        g_timeOfDay = 0;
        g_ac.type = AC_JET;
        g_ac.gearDown = false;
        g_ac.gearAnim = 0.0f;
        Vec2 als = geoPt(38.4310, 27.1365);
        float terr = std::max(0.0f, sampleTerrainElev(als.x, als.z));
        g_ac.pos = Vec3(als.x - 160.0f, terr + 145.0f, als.z + 180.0f);
        setOrientationFromYawPitchRoll(38.0f, -18.0f, 0.0f);
        g_ac.vel = g_ac.fwd * 125.0f;
        g_ac.airspeedMs = 125.0f;
        syncCameraToAircraft();

        // 1. Mermi serisi + Bomba birakma + 3D/HUD nisangah cekimi
        for (int b = 0; b < 4; b++) {
            fireBulletStream();
            updateWeaponsAndFires(0.04f);
            updateParticlesAndSmoke(0.04f);
        }
        dropBomb();
        updateWeaponsAndFires(0.25f);
        updateParticlesAndSmoke(0.25f);
        precompileNearChunks(g_ac.pos, 3200.0f);
        renderScene3D(W, H, 10.0f);
        renderAvionicsHUD(W, H, 60.0f);
        glFinish();
        saveFramePng("verif_combat_01_aim_fire.png");

        // 2. Bombanin binaya carpmasi, patlama topu + sok dalgasi + bina seklinde 30sn yangin gorunumu
        bool bombImpacted = false;
        int postFrames = 0;
        for (int f = 0; f < 160; f++) {
            stepFlightPhysicsSubstep(PHYS_DT);
            updateWeaponsAndFires(0.03f);
            updateParticlesAndSmoke(0.03f);
            if (!bombImpacted && !g_activeFires.empty()) {
                bombImpacted = true;
            }
            if (bombImpacted) {
                postFrames++;
                if (postFrames == 18) { // patlamadan ~0.55s sonra hem patlama topu hem bina yangini gorunumu
                    Vec3 fireC = g_activeFires[0].center;
                    g_ac.crashed = false;
                    g_ac.pos = fireC + Vec3(-45.0f, 42.0f, 75.0f);
                    Vec3 lookDir = norm(fireC - g_ac.pos);
                    g_ac.fwd = lookDir;
                    g_ac.right = norm(cross(g_ac.fwd, Vec3(0, 1, 0)));
                    g_ac.up = norm(cross(g_ac.right, g_ac.fwd));
                    syncCameraToAircraft();
                    renderScene3D(W, H, 12.2f);
                    renderAvionicsHUD(W, H, 60.0f);
                    glFinish();
                    saveFramePng("verif_combat_02_bomb_fire.png");
                }
                if (postFrames >= 55) break; // ~1.65s sonra saf 30s bina seklindeki yangin gorunumu
            }
        }
        if (!g_activeFires.empty()) {
            Vec3 fireC = g_activeFires[0].center;
            g_ac.crashed = false;
            g_ac.pos = fireC + Vec3(-52.0f, 48.0f, 82.0f);
            g_ac.fwd = norm(fireC - g_ac.pos);
            g_ac.right = norm(cross(g_ac.fwd, Vec3(0, 1, 0)));
            g_ac.up = norm(cross(g_ac.right, g_ac.fwd));
            syncCameraToAircraft();
            renderScene3D(W, H, 13.8f);
            renderAvionicsHUD(W, H, 60.0f);
            glFinish();
            saveFramePng("verif_combat_03_bldg_blaze.png");
        }
        std::printf(" [COMBAT PREVIEW] ActiveFires=%zu Explosions=%zu Particles=%zu -> %s\n",
                    g_activeFires.size(), g_explosions.size(), g_particles.size(), outDir.c_str());
    }

    g_headlessTour = false;
    eglDestroyContext(dpy, ctx);
    eglDestroySurface(dpy, surf);
    eglTerminate(dpy);
    return 0;
}

// =====================================================================================
//  MAIN : DUNYA SENTEZI, DOGRULAMA VE X11 + GLX (4x MSAA) SIMULATOR DONGUSU
// =====================================================================================
int main(int argc, char** argv) {
    bool testGenOnly = (argc > 1 && std::strcmp(argv[1], "--test-gen") == 0);
    bool inspectCity = (argc > 1 && (std::strcmp(argv[1], "--inspect-city") == 0 || std::strcmp(argv[1], "--test-tour") == 0));
    bool inspectNight = (argc > 1 && std::strcmp(argv[1], "--inspect-night") == 0);
    bool inspectCombat = (argc > 1 && std::strcmp(argv[1], "--inspect-combat") == 0);

    double tStart = nowSec();
    initNoise(352026u);
    std::string worldBinPath = resolveOsmWorldBinPath();
    loadOsmWorldBin(worldBinPath.c_str());
    buildGeographyAndWorldMask();
    computePierSeaPositions();
    generateCityBuildingsAndTrees();
    initDynamicTraffic();
    teleportAircraft(1); // [1] LTBJ 34R Pist Basi (Yerde, Kalkisa Hazir)
    double tElapsed = nowSec() - tStart;

    std::printf("====================================================================\n");
    std::printf(" IZMIR 3D UCUS VE SEHIR SIMULATORU - MATEMATIKSEL DUNYA HAZIR\n");
    std::printf("--------------------------------------------------------------------\n");
    std::printf(" Referans Merkez : Konak Saat Kulesi (%.6f N, %.6f E)\n", REF_LAT, REF_LON);
    std::printf(" Korfez Kiyi     : %zu ornek (~15m adimli Catmull-Rom + relaxCurvature)\n", g_coastSpline.size());
    std::printf(" WorldMaskGrid   : %dx%d O(1) matris (signedSeaDist & corridorDist)\n", MASK_N, MASK_N);
    std::printf(" IZBAN Hatti     : %d istasyon, %.1f km cift hat, %d canli 3-vagonlu tren\n",
                NUM_STATIONS, g_izbanTotalLen * 0.001f, NUM_IZBAN_TRAINS);
    std::printf(" Otoyol Aglari   : %zu ana arter, %zu canli arac, %d Korfez vapuru\n",
                activeTrafficRoutes().size(), g_vehicles.size(), NUM_FERRIES);
    std::printf(" Sehir Binalari  : %d cakismasiz bina (%dx%d CityChunk), %d agac\n",
                g_totalBuildings, CHUNK_NX, CHUNK_NZ, g_totalTrees);
    std::printf(" Uretim Suresi   : %.2f saniye\n", tElapsed);
    std::printf("====================================================================\n");

    if (inspectCity || inspectNight || inspectCombat) {
        return runHeadlessCityInspectionTour(inspectNight || inspectCombat, inspectCombat);
    }

    if (testGenOnly) {
        // -----------------------------------------------------------------
        // BOLUM A: COGRAFI DOGRULUK, YER SEKILLERI VE CAKISMASIZLIK DENETIMI
        // -----------------------------------------------------------------
        int bldgInSea = 0, bldgInCorridor = 0, roadInSea = 0, railInSea = 0, ferryOnLand = 0;
        int countTypes[10] = {0};
        for (size_t i = 0; i < (size_t)g_totalBuildings && i < g_allBuildings.size(); i++) {
            const auto& b = g_allBuildings[i];
            if (b.type < 10) countTypes[b.type]++;
            const WorldMaskCell& mc = queryWorldMask(b.x, b.z);
            if (mc.signedSeaDist <= 0.0f) bldgInSea++;
            if (mc.corridorDist <= 0.0f) bldgInCorridor++;
        }
        if (g_hasV2Roads) {
            for (const auto& rp : g_v2MajorRoads) {
                if (rp.isBridge) continue;
                for (const auto& p : rp.pts) {
                    if (sampleSignedSeaDist(p.x, p.z) < 0.0f) roadInSea++;
                }
            }
        } else {
            for (const auto& rs : g_roads) {
                for (const auto& p : rs.pts) {
                    if (sampleSignedSeaDist(p.x, p.z) < 0.0f) roadInSea++;
                }
            }
        }
        for (const auto& p : g_izbanSpline) {
            if (sampleSignedSeaDist(p.x, p.z) < 0.0f) railInSea++;
        }
        for (int f = 0; f < NUM_FERRIES; f++) {
            if (sampleSignedSeaDist(g_ferries[f].pos.x, g_ferries[f].pos.z) >= 0.0f) ferryOnLand++;
        }
        Vec2 yamanP = geoPt(38.5450, 27.1850);
        Vec2 dumanP = geoPt(38.6150, 27.2100);
        Vec2 catalP = geoPt(38.3550, 26.9850);
        Vec2 kadifP = geoPt(KADIFEKALE_LAT, KADIFEKALE_LON);
        Vec2 ltbjP  = geoPt(LTBJ_LAT, LTBJ_LON);
        Vec2 ltblP  = geoPt(LTBL_LAT, LTBL_LON);
        std::printf(" [DENETIM A] Yer Sekilleri     : Yamanlar=%.0fm, Dumanlidag=%.0fm, Catalkaya=%.0fm, Kadifekale=%.0fm, LTBJ=%.1fm, LTBL=%.1fm\n",
                    sampleTerrainElev(yamanP.x, yamanP.z), sampleTerrainElev(dumanP.x, dumanP.z),
                    sampleTerrainElev(catalP.x, catalP.z), sampleTerrainElev(kadifP.x, kadifP.z),
                    sampleTerrainElev(ltbjP.x, ltbjP.z),   sampleTerrainElev(ltblP.x, ltblP.z));
        std::printf(" [DENETIM B] Sehir & Orman     : SahilYali=%d, Mavisehir=%d, Fabrika=%d, MahalleApt=%d, Kule=%d, YamacEvi=%d, Kemeralti=%d, Cami=%d, Kurum=%d, Sera=%d | Agac=%d\n",
                    countTypes[0], countTypes[1], countTypes[2], countTypes[3], countTypes[4], countTypes[5], countTypes[6],
                    countTypes[7], countTypes[8], countTypes[9], g_totalTrees);
        std::printf(" [DENETIM C] Cakismasizlik     : DenizdeBina=%d, YoldaBina=%d, DenizdeYol=%d, DenizdeRay=%d, KaradaVapur=%d\n",
                    bldgInSea, bldgInCorridor, roadInSea, railInSea, ferryOnLand);

        // -----------------------------------------------------------------
        // BOLUM B: UCAK MEKANIZMASI (MSFS CESSNA 172) VE SOL/SAG DONUS TESTLERI
        // -----------------------------------------------------------------
        // 1. Pist uzerinde park dengesi testi (3 saniye, 0 gaz)
        g_ac.type = AC_CESSNA;
        teleportAircraft(1);
        for (int step = 0; step < 240 * 3; step++) stepFlightPhysicsSubstep(PHYS_DT);
        float parkPitch = std::asin(clampf(g_ac.fwd.y, -1.0f, 1.0f)) * RAD2DEG;
        float parkRoll  = std::atan2(-g_ac.right.y, g_ac.up.y) * RAD2DEG;
        std::printf(" [TEST 1] Pist Park Dengesi   : AGL=%.2fm, Pitch=%+.2f deg, Roll=%+.2f deg, Crashed=%d\n",
                    g_ac.aglM, parkPitch, parkRoll, (int)g_ac.crashed);

        // 2. LTBJ 34R Tam gaz kalkis kosusu ve rotasyon testi (9 saniye)
        g_ac.throttle = 1.0f;
        for (int step = 0; step < 240 * 9; step++) {
            g_ac.pitchInput = (g_ac.airspeedMs > 30.0f) ? 0.45f : 0.0f;
            stepFlightPhysicsSubstep(PHYS_DT);
        }
        float toPitch = std::asin(clampf(g_ac.fwd.y, -1.0f, 1.0f)) * RAD2DEG;
        std::printf(" [TEST 2] LTBJ 34R Kalkis     : Hiz=%.1f km/h, AGL=%.1fm, Pitch=%+.1f deg, OnGround=%d, Crashed=%d\n",
                    g_ac.airspeedMs * 3.6f, g_ac.aglM, toPitch, (int)g_ac.onGround, (int)g_ac.crashed);

        // 3. Havada el birakilmis (hands-off SAS) duz ucus kararliligi (6 saniye)
        teleportAircraft(2);
        for (int step = 0; step < 240 * 6; step++) stepFlightPhysicsSubstep(PHYS_DT);
        float crPitch = std::asin(clampf(g_ac.fwd.y, -1.0f, 1.0f)) * RAD2DEG;
        float crRoll  = std::atan2(-g_ac.right.y, g_ac.up.y) * RAD2DEG;
        std::printf(" [TEST 3] SAS Duz Ucus        : Hiz=%.1f km/h, Irtifa=%.1fm, Pitch=%+.2f deg, Roll=%+.2f deg, G=%.2f\n",
                    g_ac.airspeedMs * 3.6f, g_ac.pos.y, crPitch, crRoll, g_ac.gLoad);

        // 4. Saga (D / Sag Ok / Sol Analog Sag) ve Sola (A / Sol Ok / Sol Analog Sol) Donus + Kanat Duzeltme Testi
        teleportAircraft(2);
        float hdg0 = std::atan2(g_ac.fwd.x, -g_ac.fwd.z) * RAD2DEG;
        g_ac.rollInput = +0.85f; // Saga donus komutu (3 saniye)
        for (int step = 0; step < 240 * 3; step++) stepFlightPhysicsSubstep(PHYS_DT);
        float bankRightDeg = std::atan2(-g_ac.right.y, g_ac.up.y) * RAD2DEG;
        float hdgRight = std::atan2(g_ac.fwd.x, -g_ac.fwd.z) * RAD2DEG;
        float dHdgRight = wrapAngle((hdgRight - hdg0) * DEG2RAD) * RAD2DEG;

        g_ac.rollInput = -0.85f; // Sola donus komutu (4 saniye)
        for (int step = 0; step < 240 * 4; step++) stepFlightPhysicsSubstep(PHYS_DT);
        float bankLeftDeg = std::atan2(-g_ac.right.y, g_ac.up.y) * RAD2DEG;
        float hdgLeft = std::atan2(g_ac.fwd.x, -g_ac.fwd.z) * RAD2DEG;
        float dHdgLeft = wrapAngle((hdgLeft - hdgRight) * DEG2RAD) * RAD2DEG;

        g_ac.rollInput = 0.0f; // El birakma (3 saniye)
        for (int step = 0; step < 240 * 3; step++) stepFlightPhysicsSubstep(PHYS_DT);
        float bankRecoveredDeg = std::atan2(-g_ac.right.y, g_ac.up.y) * RAD2DEG;
        std::printf(" [TEST 4] Sag/Sol Donus Testi : Saga3sn(Bank=%+.1f°, dHdg=%+.1f°) | Sola4sn(Bank=%+.1f°, dHdg=%+.1f°) | Duzelme=%+.1f°\n",
                    bankRightDeg, dHdgRight, bankLeftDeg, dHdgLeft, bankRecoveredDeg);

        // 5. DG-1000 Planor Yamanlar Dagi Yamac / Termal Kaldirma (Ridge Lift) Testi
        g_ac.type = AC_GLIDER;
        Vec2 yamSlope = geoPt(38.5150, 27.1550);
        float yamTerr = sampleTerrainElev(yamSlope.x, yamSlope.z);
        g_ac.pos = Vec3(yamSlope.x, yamTerr + 160.0f, yamSlope.z);
        setOrientationFromYawPitchRoll(315.0f, 1.0f, 0.0f);
        g_ac.vel = g_ac.fwd * 38.0f;
        g_ac.omega = Vec3(0, 0, 0);
        g_ac.onGround = false;
        g_ac.crashed = false;
        for (int step = 0; step < 240 * 4; step++) stepFlightPhysicsSubstep(PHYS_DT);
        std::printf(" [TEST 5] DG-1000 Planor Lift : Yamanlar AGL=%.1fm, Hiz=%.1f km/h, RidgeLift=%+.2f m/s, VSI=%+.2f m/s\n",
                    g_ac.aglM, g_ac.airspeedMs * 3.6f, g_ac.ridgeLiftMs, g_ac.vsiMs);

        // -----------------------------------------------------------------
        // BOLUM C: TAM HARITA SIMULASYON UCUSU (MSFS CESSNA 172 ILE LTBJ -> LTBL)
        // -----------------------------------------------------------------
        g_ac.type = AC_CESSNA;
        teleportAircraft(1); // LTBJ 34R Pist Basi
        g_ac.throttle = 1.0f;
        struct TourWaypoint { const char* label; double lat, lon; float targetAlt; };
        static const TourWaypoint TOUR[] = {
            {"Gaziemir - ESBAS",          38.3486, 27.1389, 320.0f},
            {"Sirinyer - Buca",           38.3908, 27.1506, 340.0f},
            {"Kadifekale Surlari",        38.4142, 27.1453, 360.0f},
            {"Konak Saat Kulesi",         38.4188, 27.1287, 240.0f},
            {"Karatas Tarihi Asansor",    38.4086, 27.1175, 240.0f},
            {"Alsancak Liman & Kordon",   38.4375, 27.1398, 240.0f},
            {"Bayrakli Folkart & Mistral",38.4542, 27.1755, 320.0f},
            {"Karsiyaka & Bostanli Sahil",38.4556, 27.0972, 240.0f},
            {"Mavisehir Rezidanslar",     38.4789, 27.0764, 240.0f},
            {"Cigli 2. Ana Jet Ussu LTBL",38.5130, 27.0100, 200.0f}
        };
        const int numWps = (int)(sizeof(TOUR) / sizeof(TOUR[0]));
        int curWp = 0;
        std::printf("--------------------------------------------------------------------\n");
        std::printf(" [TEST 6] MSFS CESSNA 172 TAM HARITA UCUS SIMULASYONU (LTBJ -> LTBL)\n");

        // Ilk 9 saniye LTBJ 34R pistinden kalkis
        for (int step = 0; step < 240 * 9; step++) {
            g_ac.pitchInput = (g_ac.airspeedMs > 30.0f) ? 0.45f : 0.0f;
            stepFlightPhysicsSubstep(PHYS_DT);
        }

        // Tam harita sehir turu ucus simulasyonu
        for (int sec = 0; sec < 420 && curWp < numWps; sec++) {
            for (int sub = 0; sub < 240; sub++) {
                Vec2 wp = geoPt(TOUR[curWp].lat, TOUR[curWp].lon);
                Vec2 toWp(wp.x - g_ac.pos.x, wp.z - g_ac.pos.z);
                float distWp = len2(toWp);
                if (distWp < 680.0f) {
                    std::printf("   -> [OK] %-26s gecildi | Ilce: %-11s | Irtifa: %5.1fm | AGL: %5.1fm | Hiz: %5.1f km/h | Crashed=%d\n",
                                TOUR[curWp].label, queryDistrictName(g_ac.pos.x, g_ac.pos.z),
                                g_ac.pos.y, g_ac.aglM, g_ac.airspeedMs * 3.6f, (int)g_ac.crashed);
                    curWp++;
                    if (curWp >= numWps) break;
                    wp = geoPt(TOUR[curWp].lat, TOUR[curWp].lon);
                    toWp = Vec2(wp.x - g_ac.pos.x, wp.z - g_ac.pos.z);
                }
                Vec2 dirWp = norm2(toWp);
                Vec2 fwd2D = norm2(Vec2(g_ac.fwd.x, g_ac.fwd.z));
                float crossY = fwd2D.x * dirWp.z - fwd2D.z * dirWp.x;
                float dotY   = clampf(dot2(fwd2D, dirWp), -1.0f, 1.0f);
                float hdgErr = std::atan2(crossY, dotY);
                g_ac.rollInput = clampf(hdgErr * 1.45f, -1.0f, 1.0f);

                float safeTargetAlt = std::max(TOUR[curWp].targetAlt, sampleTerrainElev(g_ac.pos.x, g_ac.pos.z) + 140.0f);
                float altErr = safeTargetAlt - g_ac.pos.y;
                float desiredPitch = clampf(altErr * 0.0045f, -0.18f, 0.25f);
                float curPitch = std::asin(clampf(g_ac.fwd.y, -1.0f, 1.0f));
                g_ac.pitchInput = clampf((desiredPitch - curPitch) * 3.2f, -0.8f, 0.8f);
                g_ac.yawInput = clampf(hdgErr * 0.35f, -0.45f, 0.45f);
                g_ac.throttle = clampf(0.75f + (95.0f - g_ac.airspeedMs) * 0.025f, 0.25f, 1.15f);
                g_ac.brakeInput = 0.0f;

                stepFlightPhysicsSubstep(PHYS_DT);
            }
            updateDynamicTraffic(1.0f);
        }

        // [TEST 7] 100m Dinamik Bina Carpisma Esigi Dogrulamasi (>100m gorsel-only, <=100m carpilabilir)
        if (!g_allBuildings.empty()) {
            const BuildingInst* probeB = nullptr;
            for (size_t i = 0; i < g_allBuildings.size(); i++) {
                const auto& cand = g_allBuildings[i];
                if (cand.h < 22.0f || cand.y > 25.0f) continue;
                bool isLocalPeak = true;
                int x0 = clampi((int)((cand.x - 140.0f - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
                int x1 = clampi((int)((cand.x + 140.0f - WORLD_X0) / SH_CELL), 0, SH_NX - 1);
                int z0 = clampi((int)((cand.z - 140.0f - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
                int z1 = clampi((int)((cand.z + 140.0f - WORLD_Z0) / SH_CELL), 0, SH_NZ - 1);
                for (int gz = z0; gz <= z1 && isLocalPeak; gz++) {
                    for (int gx = x0; gx <= x1; gx++) {
                        for (int nIdx : g_spatialHash[gz * SH_NX + gx]) {
                            if (g_allBuildings[nIdx].y + g_allBuildings[nIdx].h > cand.y + cand.h + 1.0f) {
                                isLocalPeak = false;
                                break;
                            }
                        }
                    }
                }
                if (isLocalPeak) { probeB = &cand; break; }
            }
            if (probeB) {
                Vec3 insidePt(probeB->x, probeB->y + probeB->h * 0.5f, probeB->z);
                Vec3 acFar(probeB->x, probeB->y + probeB->h + 125.0f, probeB->z);
                g_collidersDirty = true;
                refreshActiveColliders100m(acFar);
                size_t farCacheSz = g_activeColliders100m.size();
                bool staticHit = checkBuildingCollision(insidePt, 1.8f);
                bool farHit = checkAircraftBuildingCollision100m(insidePt, acFar, 1.8f);

                Vec3 acNear(probeB->x, probeB->y + probeB->h + 75.0f, probeB->z);
                g_collidersDirty = true;
                refreshActiveColliders100m(acNear);
                size_t nearCacheSz = g_activeColliders100m.size();
                bool nearHit = checkAircraftBuildingCollision100m(insidePt, acNear, 1.8f);

                std::printf(" [TEST 7] 100m Bina Carpismasi : StaticHit=%d | >100m(125m): Cache=%zu, Collide=%d | <=100m(75m): Cache=%zu, Collide=%d\n",
                            (int)staticHit, farCacheSz, (int)farHit, nearCacheSz, (int)nearHit);

                // [TEST 8] Drone Modu 100m Carpma, Ekran Titremesi, Mermi & Bina-Olcekli 30s Bomba Yangini
                teleportAircraft(2);
                g_ac.droneMode = true;
                g_ac.pos = Vec3(probeB->x, probeB->y + probeB->h * 0.5f, probeB->z + probeB->d * 0.5f + 8.0f);
                g_ac.fwd = Vec3(0, 0, -1);
                g_ac.right = Vec3(1, 0, 0);
                g_ac.up = Vec3(0, 1, 0);
                g_collidersDirty = true;
                for (int s = 0; s < 120 && !g_ac.crashed; s++) stepFlightPhysicsSubstep(PHYS_DT);
                bool droneCrashed = g_ac.crashed;
                bool shakeActive = (g_camShakeTimer > 0.0f && !g_explosions.empty());

                teleportAircraft(2);
                g_ac.droneMode = false;
                // Yuksek binanin ~120m disindan nisangah ve atis/bomba testi
                g_ac.pos = Vec3(probeB->x, probeB->y + probeB->h + 55.0f, probeB->z + 90.0f);
                g_ac.fwd = norm(Vec3(0.0f, -0.5f, -1.0f));
                g_ac.up = Vec3(0, 1, 0);
                g_ac.right = norm(cross(g_ac.fwd, Vec3(0, 1, 0)));
                g_ac.up = norm(cross(g_ac.right, g_ac.fwd));
                g_ac.vel = g_ac.fwd * 65.0f;
                fireBulletStream();
                dropBomb();
                for (int s = 0; s < 240; s++) {
                    updateWeaponsAndFires(0.05f);
                    updateParticlesAndSmoke(0.05f);
                }
                bool fireStarted = (!g_activeFires.empty());
                float initFireDur = fireStarted ? g_activeFires[0].maxDuration : 0.0f;
                size_t fireNodes = fireStarted ? g_activeFires[0].nodes.size() : 0;
                // Toplam 12s + 17s = 29s aninda hala yanmali, 31s sonra sonmus olmali
                for (int s = 0; s < 340; s++) {
                    updateWeaponsAndFires(0.05f);
                }
                bool burningAt29s = (!g_activeFires.empty());
                for (int s = 0; s < 50; s++) {
                    updateWeaponsAndFires(0.05f);
                }
                bool outAt31s = g_activeFires.empty();
                std::printf(" [TEST 8] Silah/Yangin/Drone   : DroneCrash=%d, Shake&Exp=%d | BombaYangin=%d (Dur=%.0fs, Nodes=%zu) | 29s=%d -> 31sSond=%d\n",
                            (int)droneCrashed, (int)shakeActive, (int)fireStarted, initFireDur, fireNodes,
                            (int)burningAt29s, (int)outAt31s);
                if (!staticHit || farHit || !nearHit || !droneCrashed || !shakeActive || !fireStarted || !burningAt29s || !outAt31s) {
                    return 1;
                }
            }
        }
        std::printf("====================================================================\n");
        return 0;
    }

    Display* dpy = XOpenDisplay(nullptr);
    if (!dpy) {
        std::fprintf(stderr, "HATA: X11 ekrani acilamadi (DISPLAY kontrol edin).\n");
        return 1;
    }
    int screen = DefaultScreen(dpy);

    // 4x MSAA GLX gorsel secimi (desteklenmiyorsa standart 24-bit depth gorsele yedeklenir)
    int attribsMSAA[] = {
        GLX_RGBA, GLX_DOUBLEBUFFER,
        GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        GLX_SAMPLE_BUFFERS, 1, GLX_SAMPLES, 4,
        None
    };
    int attribsStd[] = {
        GLX_RGBA, GLX_DOUBLEBUFFER,
        GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        None
    };
    XVisualInfo* vi = glXChooseVisual(dpy, screen, attribsMSAA);
    if (!vi) vi = glXChooseVisual(dpy, screen, attribsStd);
    if (!vi) {
        std::fprintf(stderr, "HATA: Uygun GLX gorseli bulunamadi.\n");
        XCloseDisplay(dpy);
        return 1;
    }

    Window root = RootWindow(dpy, vi->screen);
    XSetWindowAttributes swa;
    swa.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | StructureNotifyMask;

    int winW = 1440, winH = 900;
    Window win = XCreateWindow(dpy, root, 0, 0, winW, winH, 0, vi->depth, InputOutput,
                               vi->visual, CWColormap | CWEventMask, &swa);
    XStoreName(dpy, win, "Izmir 3D Ucus ve Sehir Simulatoru - C++17 / OpenGL / 35x45 km Gercek WGS84");
    Atom wmDelete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wmDelete, 1);
    XMapWindow(dpy, win);

    GLXContext glc = glXCreateContext(dpy, vi, nullptr, GL_TRUE);
    glXMakeCurrent(dpy, win, glc);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);

    // 12 adet Doku (11x 512x512 Prosedurel HD Doku + 1x 2048x2048 GIS) ve Tile Display List'ler
    buildProceduralTextures();
    buildTerrainAndSeaDisplayList();
    buildQuayAndPromenadeDisplayList();
    buildIzbanAndRoadsDisplayList();
    buildLandmarksAndAirportsDisplayList();
    precompileNearChunks(g_ac.pos, 1800.0f);

    bool keys[65536] = {false};
    bool running = true;
    double prevTime = nowSec();
    float physAccum = 0.0f;
    float simTime = 0.0f;
    float fpsSmooth = 60.0f;

    while (running) {
        while (XPending(dpy) > 0) {
            XEvent xev;
            XNextEvent(dpy, &xev);
            if (xev.type == ClientMessage && (Atom)xev.xclient.data.l[0] == wmDelete) {
                running = false;
            } else if (xev.type == ConfigureNotify) {
                winW = std::max(320, xev.xconfigure.width);
                winH = std::max(240, xev.xconfigure.height);
            } else if (xev.type == KeyPress) {
                KeySym ks = XLookupKeysym(&xev.xkey, 0);
                if (ks < 65536) keys[ks] = true;
                if (ks == XK_Escape) running = false;
                else if (ks == XK_1) teleportAircraft(1);
                else if (ks == XK_2) teleportAircraft(2);
                else if (ks == XK_3) teleportAircraft(3);
                else if (ks == XK_4) teleportAircraft(4);
                else if (ks == XK_5) teleportAircraft(5);
                else if (ks == XK_6) teleportAircraft(6);
                else if (ks == XK_7) teleportAircraft(7);
                else if (ks == XK_8) teleportAircraft(8);
                else if (ks == XK_9) teleportAircraft(9);
                else if (ks == XK_0) teleportAircraft(0);
                else if (ks == XK_g || ks == XK_G) g_ac.gearDown = !g_ac.gearDown;
                else if (ks == XK_h || ks == XK_H) g_ac.flapStage = (g_ac.flapStage + 1) % 3;
                else if (ks == XK_b || ks == XK_B) g_ac.smokeOn = !g_ac.smokeOn;
                else if (ks == XK_c || ks == XK_C) g_camMode = (g_camMode + 1) % NUM_CAMS;
                else if (ks == XK_v || ks == XK_V) cycleAircraftType();
                else if (ks == XK_f || ks == XK_F) g_ac.droneMode = !g_ac.droneMode;
                else if (ks == XK_t || ks == XK_T) g_timeOfDay = (g_timeOfDay + 1) % 3;
                else if (ks == XK_z || ks == XK_Z) g_radarZoomIdx = (g_radarZoomIdx + 1) % 3;
                else if (ks == XK_p || ks == XK_P) g_ac.autopilot = !g_ac.autopilot;
                else if (ks == XK_i || ks == XK_I) {
                    g_tourActive = !g_tourActive;
                    if (g_tourActive) {
                        g_ac.type = AC_CESSNA;
                        g_tourIndex = 0;
                        g_tourTimer = 0.0f;
                        g_ac.crashed = false;
                        g_ac.smokeOn = false;
                        float terr = sampleTerrainElev(g_ac.pos.x, g_ac.pos.z);
                        if (g_ac.pos.y < terr + 80.0f) {
                            g_ac.pos.y = terr + 180.0f;
                            g_ac.vel = g_ac.fwd * 45.0f;
                            g_ac.airspeedMs = 45.0f;
                            g_ac.onGround = false;
                        }
                    }
                }
                else if (ks == XK_r || ks == XK_R) resetAndRecoverAircraftInAir();
                else if (ks == XK_j || ks == XK_J || ks == XK_Return) g_gunTriggerEdge = true;
                else if (ks == XK_m || ks == XK_M || ks == XK_u || ks == XK_U) g_bombTriggerEdge = true;
            } else if (xev.type == KeyRelease) {
                if (XEventsQueued(dpy, QueuedAfterReading)) {
                    XEvent nev;
                    XPeekEvent(dpy, &nev);
                    if (nev.type == KeyPress && nev.xkey.time == xev.xkey.time &&
                        nev.xkey.keycode == xev.xkey.keycode) {
                        continue; // X11 auto-repeat filtrele
                    }
                }
                KeySym ks = XLookupKeysym(&xev.xkey, 0);
                if (ks < 65536) keys[ks] = false;
            }
        }

        double curTime = nowSec();
        float frameDt = clampf((float)(curTime - prevTime), 0.001f, 0.05f);
        prevTime = curTime;
        simTime += frameDt;
        fpsSmooth = lerpf(fpsSmooth, 1.0f / frameDt, 0.08f);

        // /dev/input/js0 otomatik Linux Gamepad / Joystick okuma (Logitech F710 / Xbox)
        pollLinuxJoystick(frameDt);

        // Klavye + Gamepad Gaz (Throttle) kontrolu:
        // W/S, PageUp/PageDown, Gamepad RT (Axis 5) / LT (Axis 2), Sag Analog Y (Axis 4) ve A/B (Btn 0/1)
        float thrDelta = 0.0f;
        if (keys[XK_w] || keys[XK_W] || keys[XK_Page_Up]   || g_jsBtn[0]) thrDelta += 0.90f;
        if (keys[XK_s] || keys[XK_S] || keys[XK_Page_Down] || g_jsBtn[1]) thrDelta -= 0.95f;
        thrDelta += g_jsAxis[5] * 0.95f; // RT Tetik -> Gaz Ver
        thrDelta -= g_jsAxis[2] * 0.95f; // LT Tetik -> Gaz Kes
        thrDelta -= g_jsAxis[4] * 0.85f; // Sag Analog Yukari (-Y) -> Gaz Ver, Asagi (+Y) -> Gaz Kes
        g_ac.throttle = clampf(g_ac.throttle + thrDelta * frameDt, 0.0f, 1.25f);

        // Pitch (Yunuslama): Ok Yukari/Asagi, K, Gamepad Sol Analog Y (Axis 1) ve D-Pad Y (Axis 7)
        float pitchTarget = g_jsAxis[1] + g_jsAxis[7] * 0.85f;
        if (keys[XK_Down] || keys[XK_k] || keys[XK_K]) pitchTarget += 1.0f; // Burun yukari (Pull up)
        if (keys[XK_Up])   pitchTarget -= 1.0f; // Burun asagi (Push down)

        // Roll & Donus (Sol/Sag): A/D, Sol/Sag Ok, Gamepad Sol Analog X (Axis 0) ve D-Pad X (Axis 6)
        float rollTarget = g_jsAxis[0] + g_jsAxis[6] * 0.90f;
        if (keys[XK_Right] || keys[XK_d] || keys[XK_D]) rollTarget += 1.0f;
        if (keys[XK_Left]  || keys[XK_a] || keys[XK_A]) rollTarget -= 1.0f;

        // Rudder / Yaw (Yon Dumeni): Q/E ve Gamepad Sag Analog X (Axis 3) (LB/RB silahlara ayrildi)
        float yawTarget = g_jsAxis[3] * 0.85f;
        if (keys[XK_e] || keys[XK_E]) yawTarget += 1.0f;
        if (keys[XK_q] || keys[XK_Q]) yawTarget -= 1.0f;

        float brakeTarget = (keys[XK_space] || g_jsBtn[1]) ? 1.0f : g_jsAxis[2];

        // Manuel pilot mudahalesi varsa Otonom Tur'u devreden cikar
        bool userManual = (keys[XK_Up] || keys[XK_Down] || keys[XK_Left] || keys[XK_Right] ||
                           keys[XK_a]  || keys[XK_A]    || keys[XK_d]    || keys[XK_D] ||
                           std::abs(g_jsAxis[0]) > 0.25f || std::abs(g_jsAxis[1]) > 0.25f);
        if (userManual && g_tourActive) {
            g_tourActive = false;
        }

        // [I] Otonom Sehir Denetim Turu Otopilot Rehberligi
        if (g_tourActive) {
            g_tourTimer += frameDt;
            const auto& ts = TOUR_STOPS[g_tourIndex];
            Vec2 targetPt = geoPt(ts.lat, ts.lon);
            Vec2 toTarget(targetPt.x - g_ac.pos.x, targetPt.z - g_ac.pos.z);
            float dist = len2(toTarget);
            if (dist < 450.0f || g_tourTimer > 35.0f) {
                g_tourIndex = (g_tourIndex + 1) % NUM_TOUR_STOPS;
                g_tourTimer = 0.0f;
            }
            Vec2 dirT = norm2(toTarget);
            Vec2 fwd2D = norm2(Vec2(g_ac.fwd.x, g_ac.fwd.z));
            float crossY = fwd2D.x * dirT.z - fwd2D.z * dirT.x;
            float dotY   = clampf(dot2(fwd2D, dirT), -1.0f, 1.0f);
            float hdgErr = std::atan2(crossY, dotY);
            rollTarget = clampf(hdgErr * 1.45f, -0.75f, 0.75f);

            float terrH = sampleTerrainElev(g_ac.pos.x, g_ac.pos.z);
            float desiredAlt = std::max(terrH + ts.alt, terrH + 120.0f);
            float altErr = desiredAlt - g_ac.pos.y;
            float desiredPitch = clampf(altErr * 0.005f, -0.15f, 0.20f);
            float curPitch = std::asin(clampf(g_ac.fwd.y, -1.0f, 1.0f));
            pitchTarget = clampf((desiredPitch - curPitch) * 3.0f, -0.7f, 0.7f);
            yawTarget = clampf(hdgErr * 0.3f, -0.3f, 0.3f);
            g_ac.throttle = clampf(0.85f + (80.0f - g_ac.airspeedMs) * 0.02f, 0.4f, 1.05f);
        }

        g_ac.pitchInput = lerpf(g_ac.pitchInput, clampf(pitchTarget, -1.0f, 1.0f), std::min(1.0f, 14.0f * frameDt));
        g_ac.rollInput  = lerpf(g_ac.rollInput,  clampf(rollTarget,  -1.0f, 1.0f), std::min(1.0f, 14.0f * frameDt));
        g_ac.yawInput   = lerpf(g_ac.yawInput,   clampf(yawTarget,   -1.0f, 1.0f), std::min(1.0f, 14.0f * frameDt));
        g_ac.brakeInput = lerpf(g_ac.brakeInput, clampf(brakeTarget,  0.0f, 1.0f), std::min(1.0f, 10.0f * frameDt));

        // 240 Hz alt-adimli (PHYS_DT = 1.0f/240.0f) 6-DOF fizik motoru
        physAccum += frameDt;
        while (physAccum >= PHYS_DT) {
            stepFlightPhysicsSubstep(PHYS_DT);
            physAccum -= PHYS_DT;
        }
        updateDynamicTraffic(frameDt);
        bool wantGun = (keys[XK_j] || keys[XK_J] || keys[XK_Return] || g_jsBtn[5] || g_gunTriggerEdge);
        if (wantGun && g_gunFireCd <= 0.0f && !g_ac.crashed) {
            fireBulletStream();
            g_gunFireCd = 0.068f;
            g_gunTriggerEdge = false;
        } else {
            g_gunTriggerEdge = false;
        }
        bool wantBomb = (g_bombTriggerEdge || keys[XK_m] || keys[XK_M] || keys[XK_u] || keys[XK_U] || g_jsBtn[4]);
        if (wantBomb && g_bombDropCd <= 0.0f && !g_ac.crashed) {
            dropBomb();
            g_bombDropCd = 0.42f;
            g_bombTriggerEdge = false;
        } else {
            g_bombTriggerEdge = false;
        }
        updateWeaponsAndFires(frameDt);
        updateParticlesAndSmoke(frameDt);

        // Sahne ve Aviyonik HUD cizimi
        renderScene3D(winW, winH, simTime);
        renderAvionicsHUD(winW, winH, fpsSmooth);
        glXSwapBuffers(dpy, win);
    }

    if (g_jsFd >= 0) close(g_jsFd);
    glXMakeCurrent(dpy, None, nullptr);
    glXDestroyContext(dpy, glc);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    return 0;
}
