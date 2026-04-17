#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PLANET_RADIUS 1.8f
#define SEA_LEVEL 0.0f
#define SUBDIVISIONS 5
#define MAX_TILE_CORNERS 8
#define POLY_BLEND_TANGENT 0.004f
#define POLY_BLEND_RADIAL 0.000f
#define DRAW_WIRES 0
#define MIN_PLATES 9
#define MAX_PLATES 12
#define MIN_MAJOR_PLATES 5
#define MAX_MAJOR_PLATES 7
#define TECTONIC_TIME_SCALE 7.0f
#define TECTONIC_REBUILD_HZ 8.0f
#define WEATHER_TIME_SCALE 1.6f
#define WEATHER_UPDATE_HZ 24.0f
#define CLOUD_LAYER_HEIGHT 0.030f
#define MAX_TILE_NEIGHBORS 8
#define FLOW_ARROW_SEGMENTS 5

typedef struct OrbitCamera {
    float yaw;
    float pitch;
    float yawTarget;
    float pitchTarget;
    float distance;
    float distanceTarget;
    Vector3 target;
} OrbitCamera;

typedef struct Triangle {
    int a;
    int b;
    int c;
} Triangle;

typedef struct VertexBuffer {
    Vector3 *items;
    int count;
    int capacity;
} VertexBuffer;

typedef struct TriangleBuffer {
    Triangle *items;
    int count;
    int capacity;
} TriangleBuffer;

typedef struct EdgeEntry {
    uint64_t key;
    int midpointIndex;
    bool used;
} EdgeEntry;

typedef struct EdgeCache {
    EdgeEntry *entries;
    int capacity;
} EdgeCache;

typedef struct Tile {
    Vector3 baseCenterDir;
    Vector3 center;
    Vector3 corners[MAX_TILE_CORNERS];
    int cornerTriangleIds[MAX_TILE_CORNERS];
    int cornerCount;
    int plateId;
    float elevation;
    Color terrainFill;
    Color plateFill;
} Tile;

typedef struct TerrainSample {
    float elevation;
    float moisture;
    float ruggedness;
    float snowcap;
    float shade;
    int plateId;
    float boundary;
    float convergence;
} TerrainSample;

typedef struct Plate {
    Vector3 seedDir;
    Vector3 driftDir;
    Vector3 spinAxis;
    float angularSpeed;
    float sizeBias;
    float oceanic;
    float density;
    float crustAge;
    float continentalBias;
    float warpFreq;
    float warpAmp;
    float warpPhase;
    bool major;
    Color color;
} Plate;

typedef struct TileGraphNode {
    int neighborCount;
    int neighbors[MAX_TILE_NEIGHBORS];
} TileGraphNode;

typedef struct WeatherCell {
    Vector3 normal;
    Vector3 wind;
    Vector3 current;
    float temperature;
    float pressure;
    float humidity;
    float cloud;
    float precipitation;
    float vorticity;
    float evaporation;
    float snow;
    float storm;
    float elevation;
    float ocean;
    float surfaceTemperature;
    float soilMoisture;
    float cloudWater;
    float recentRain;
    float rainShadow;
    float orographicLift;
} WeatherCell;

typedef enum WeatherViewMode {
    WEATHER_VIEW_TEMPERATURE = 0,
    WEATHER_VIEW_PRESSURE,
    WEATHER_VIEW_WIND,
    WEATHER_VIEW_CURRENT,
    WEATHER_VIEW_HUMIDITY,
    WEATHER_VIEW_CLOUD,
    WEATHER_VIEW_RAIN,
    WEATHER_VIEW_VORTICITY,
    WEATHER_VIEW_STORM,
    WEATHER_VIEW_EVAPORATION,
    WEATHER_VIEW_SNOW,
    WEATHER_VIEW_COUNT
} WeatherViewMode;

static float SphericalNoise3(Vector3 unitDirection, float scale, Vector3 offset, int octaves, float lacunarity, float gain);
static float WarpedClimateLatitude(Vector3 unitDirection, float strength);
static Vector3 TangentEast(Vector3 normal);
static Vector3 TangentNorth(Vector3 normal);
static Vector3 ClimateEddyFlow(Vector3 normal, float scale, Vector3 offset, float strength);

static float ClampFloat(float value, float minValue, float maxValue)
{
    if (value < minValue) return minValue;
    if (value > maxValue) return maxValue;
    return value;
}

static float LerpFloat(float a, float b, float t)
{
    return a + (b - a) * t;
}

static float SmoothStep01(float x)
{
    x = ClampFloat(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

static float DisplayElevation(float elevation)
{
    return fmaxf(elevation, SEA_LEVEL);
}

static Color LerpColor(Color a, Color b, float t)
{
    t = ClampFloat(t, 0.0f, 1.0f);
    return (Color){
        (unsigned char)(a.r + (b.r - a.r) * t),
        (unsigned char)(a.g + (b.g - a.g) * t),
        (unsigned char)(a.b + (b.b - a.b) * t),
        255
    };
}

static Color ScaleColorBrightness(Color color, float amount)
{
    return (Color){
        (unsigned char)ClampFloat((float)color.r * amount, 0.0f, 255.0f),
        (unsigned char)ClampFloat((float)color.g * amount, 0.0f, 255.0f),
        (unsigned char)ClampFloat((float)color.b * amount, 0.0f, 255.0f),
        color.a
    };
}

static Color ShadeSurfaceColor(Color color, Vector3 normal, float strength)
{
    Vector3 lightDir = Vector3Normalize((Vector3){ -0.38f, 0.70f, 0.60f });
    float light = ClampFloat(Vector3DotProduct(normal, lightDir) * 0.5f + 0.5f, 0.0f, 1.0f);
    float amount = LerpFloat(1.0f - strength * 0.32f, 1.0f + strength * 0.24f, light);
    return ScaleColorBrightness(color, amount);
}

static float WeatherTemperatureC(float temperature)
{
    return LerpFloat(-35.0f, 45.0f, ClampFloat(temperature, 0.0f, 1.0f));
}

static float WeatherPressureHpa(float pressure)
{
    float t = ClampFloat((pressure - 0.18f) / 0.70f, 0.0f, 1.0f);
    return LerpFloat(870.0f, 1090.0f, t);
}

static float WeatherSaturation(float temperature)
{
    return ClampFloat(0.19f + powf(ClampFloat(temperature, 0.0f, 1.0f), 1.35f) * 0.86f, 0.16f, 1.24f);
}

static float WeatherRelativeHumidity(float humidity, float temperature)
{
    return ClampFloat(humidity / WeatherSaturation(temperature), 0.0f, 1.4f);
}

static float WeatherWindMetersPerSecond(Vector3 wind)
{
    return Vector3Length(wind) * 250.0f;
}

static float WeatherCurrentMetersPerSecond(Vector3 current)
{
    return Vector3Length(current) * 25.0f;
}

static float TerrainElevationMeters(float elevation)
{
    return elevation * 36000.0f;
}

static float RandomFloat01(void)
{
    return (float)rand() / (float)RAND_MAX;
}

static float RandomRange(float minValue, float maxValue)
{
    return minValue + (maxValue - minValue) * RandomFloat01();
}

static Vector3 RandomUnitVector(void)
{
    float z = RandomRange(-1.0f, 1.0f);
    float a = RandomRange(0.0f, 2.0f * PI);
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return (Vector3){ r * cosf(a), z, r * sinf(a) };
}

static Vector3 RandomTangentVector(Vector3 normal)
{
    Vector3 tangent = Vector3CrossProduct(normal, RandomUnitVector());
    if (Vector3LengthSqr(tangent) < 0.000001f) tangent = Vector3CrossProduct(normal, (Vector3){ 0.0f, 1.0f, 0.0f });
    if (Vector3LengthSqr(tangent) < 0.000001f) tangent = Vector3CrossProduct(normal, (Vector3){ 1.0f, 0.0f, 0.0f });
    tangent = Vector3Normalize(tangent);
    if (RandomFloat01() < 0.5f) tangent = Vector3Scale(tangent, -1.0f);
    return tangent;
}

static int GenerateRandomPlates(Plate *plates)
{
    int plateCount = MIN_PLATES + (rand() % (MAX_PLATES - MIN_PLATES + 1));
    int majorCount = MIN_MAJOR_PLATES + (rand() % (MAX_MAJOR_PLATES - MIN_MAJOR_PLATES + 1));
    if (majorCount > plateCount - 2) majorCount = plateCount - 2;
    if (majorCount < 3) majorCount = 3;

    for (int i = 0; i < plateCount; i++) {
        bool major = i < majorCount;
        Vector3 bestSeed = RandomUnitVector();
        float bestScore = -1000.0f;
        int attempts = major ? 72 : 44;

        for (int attempt = 0; attempt < attempts; attempt++) {
            Vector3 candidate = RandomUnitVector();
            if (!major && majorCount > 1 && RandomFloat01() < 0.78f) {
                int a = rand() % majorCount;
                int b = rand() % majorCount;
                if (a == b) b = (b + 1) % majorCount;
                float mix = RandomRange(0.38f, 0.62f);
                candidate = Vector3Add(
                    Vector3Scale(plates[a].seedDir, mix),
                    Vector3Scale(plates[b].seedDir, 1.0f - mix)
                );
                candidate = Vector3Add(candidate, Vector3Scale(RandomUnitVector(), RandomRange(0.10f, 0.34f)));
                if (Vector3LengthSqr(candidate) < 0.000001f) candidate = RandomUnitVector();
                candidate = Vector3Normalize(candidate);
            }

            float minSeparation = 1.0f;
            for (int p = 0; p < i; p++) {
                float d = 1.0f - Vector3DotProduct(candidate, plates[p].seedDir);
                if (d < minSeparation) minSeparation = d;
            }
            if (i == 0) minSeparation = 1.0f;

            float edgeScore = 0.0f;
            if (!major) {
                float bestMajor = -2.0f;
                float secondMajor = -2.0f;
                for (int p = 0; p < majorCount; p++) {
                    float d = Vector3DotProduct(candidate, plates[p].seedDir);
                    if (d > bestMajor) {
                        secondMajor = bestMajor;
                        bestMajor = d;
                    } else if (d > secondMajor) {
                        secondMajor = d;
                    }
                }
                edgeScore = 1.0f - SmoothStep01((bestMajor - secondMajor) / 0.42f);
            }

            float score = major
                ? minSeparation + RandomRange(-0.09f, 0.09f)
                : edgeScore * 0.62f + minSeparation * 0.38f + RandomRange(-0.16f, 0.16f);
            if (score > bestScore) {
                bestScore = score;
                bestSeed = candidate;
            }
        }

        plates[i].seedDir = bestSeed;
        plates[i].spinAxis = RandomUnitVector();
        plates[i].driftDir = Vector3CrossProduct(plates[i].spinAxis, bestSeed);
        if (Vector3LengthSqr(plates[i].driftDir) < 0.000001f) plates[i].driftDir = RandomTangentVector(bestSeed);
        plates[i].driftDir = Vector3Normalize(plates[i].driftDir);
        plates[i].spinAxis = Vector3Normalize(Vector3CrossProduct(bestSeed, plates[i].driftDir));

        float oceanic = major
            ? ((i < 2 || RandomFloat01() < 0.46f) ? RandomRange(0.08f, 0.36f) : RandomRange(0.56f, 0.92f))
            : RandomRange(0.50f, 0.98f);
        float speed = major ? RandomRange(0.006f, 0.020f) : RandomRange(0.014f, 0.044f);
        if (RandomFloat01() < 0.5f) speed = -speed;

        plates[i].angularSpeed = speed * LerpFloat(0.82f, 1.28f, oceanic);
        plates[i].sizeBias = major ? RandomRange(0.030f, 0.085f) : RandomRange(-0.070f, -0.012f);
        plates[i].oceanic = oceanic;
        plates[i].density = ClampFloat(LerpFloat(0.36f, 0.94f, oceanic) + RandomRange(-0.045f, 0.045f), 0.25f, 1.0f);
        plates[i].crustAge = ClampFloat((major ? RandomRange(0.34f, 0.96f) : RandomRange(0.05f, 0.58f)) * LerpFloat(1.10f, 0.82f, oceanic), 0.02f, 1.0f);
        plates[i].continentalBias = ClampFloat(0.78f - oceanic * 0.47f + RandomRange(-0.07f, 0.08f), 0.18f, 0.82f);
        plates[i].warpFreq = major ? RandomRange(3.1f, 5.2f) : RandomRange(5.2f, 8.9f);
        plates[i].warpAmp = major ? RandomRange(0.018f, 0.046f) : RandomRange(0.044f, 0.086f);
        plates[i].warpPhase = RandomRange(0.0f, 2.0f * PI);
        plates[i].major = major;
        plates[i].color = ColorFromHSV(RandomRange(0.0f, 360.0f), major ? RandomRange(0.58f, 0.86f) : RandomRange(0.78f, 0.98f), major ? RandomRange(0.70f, 0.94f) : RandomRange(0.78f, 1.0f));
    }

    return plateCount;
}

static Vector3 RotateAroundAxis(Vector3 v, Vector3 axis, float angle)
{
    Matrix r = MatrixRotate(axis, angle);
    return Vector3Transform(v, r);
}

static void AdvancePlateSimulation(Plate *plates, int plateCount, float simulationDt)
{
    for (int i = 0; i < plateCount; i++) {
        float step = plates[i].angularSpeed * simulationDt;
        if (step == 0.0f) continue;

        plates[i].seedDir = Vector3Normalize(RotateAroundAxis(plates[i].seedDir, plates[i].spinAxis, step));
        plates[i].driftDir = Vector3Normalize(RotateAroundAxis(plates[i].driftDir, plates[i].spinAxis, step));
        plates[i].spinAxis = Vector3Normalize(Vector3CrossProduct(plates[i].seedDir, plates[i].driftDir));
        if (Vector3LengthSqr(plates[i].spinAxis) < 0.000001f) {
            plates[i].driftDir = RandomTangentVector(plates[i].seedDir);
            plates[i].spinAxis = Vector3Normalize(Vector3CrossProduct(plates[i].seedDir, plates[i].driftDir));
        }
    }
}

static float PlateWarpScore(Vector3 direction, const Plate *plate, int plateIndex)
{
    float f = plate->warpFreq;
    float p = plate->warpPhase + (float)plateIndex * 0.61f;
    float waveA = sinf(direction.x * f + direction.y * (f * 0.73f) + direction.z * (f * 1.21f) + p);
    float waveB = cosf(direction.x * (f * 1.31f) - direction.y * (f * 0.57f) + direction.z * (f * 0.87f) + p * 1.7f);
    float wave = waveA * 0.58f + waveB * 0.42f;
    return wave * plate->warpAmp;
}

static void FindClosestPlates(
    Vector3 unitDirection, const Plate *plates, int plateCount,
    int *primaryOut, int *secondaryOut, float *primaryDotOut, float *secondaryDotOut
)
{
    int primary = 0;
    int secondary = (plateCount > 1) ? 1 : 0;
    float primaryDot = -1000.0f;
    float secondaryDot = -1000.0f;

    for (int i = 0; i < plateCount; i++) {
        float d = Vector3DotProduct(unitDirection, plates[i].seedDir) + plates[i].sizeBias + PlateWarpScore(unitDirection, &plates[i], i);
        if (d > primaryDot) {
            secondaryDot = primaryDot;
            secondary = primary;
            primaryDot = d;
            primary = i;
        } else if (d > secondaryDot) {
            secondaryDot = d;
            secondary = i;
        }
    }

    if (primaryOut) *primaryOut = primary;
    if (secondaryOut) *secondaryOut = secondary;
    if (primaryDotOut) *primaryDotOut = primaryDot;
    if (secondaryDotOut) *secondaryDotOut = secondaryDot;
}

static Vector3 GetCameraDirection(float yaw, float pitch)
{
    Vector3 dir = {
        cosf(pitch) * cosf(yaw),
        sinf(pitch),
        cosf(pitch) * sinf(yaw)
    };
    return Vector3Normalize(dir);
}

static float SmoothLerpFactor(float speed, float deltaTime)
{
    return 1.0f - expf(-speed * deltaTime);
}

static void UpdateOrbitCamera(OrbitCamera *orbit, Camera3D *camera)
{
    float dt = GetFrameTime();
    if (dt > 0.1f) dt = 0.1f;
    Vector2 mouseDelta = GetMouseDelta();

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        orbit->yawTarget += mouseDelta.x * 0.0075f;
        orbit->pitchTarget -= mouseDelta.y * 0.0075f;
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        orbit->yawTarget += mouseDelta.x * 0.0075f;
        orbit->pitchTarget -= mouseDelta.y * 0.0075f;
    }
    orbit->pitchTarget = ClampFloat(orbit->pitchTarget, -1.46f, 1.46f);

    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        orbit->distanceTarget *= expf(-wheel * 0.115f);
        orbit->distanceTarget = ClampFloat(orbit->distanceTarget, PLANET_RADIUS * 1.42f, 36.0f);
    }

    while (orbit->yawTarget - orbit->yaw > PI) orbit->yawTarget -= 2.0f * PI;
    while (orbit->yawTarget - orbit->yaw < -PI) orbit->yawTarget += 2.0f * PI;

    float rotateBlend = SmoothLerpFactor(12.0f, dt);
    float zoomBlend = SmoothLerpFactor(11.0f, dt);
    orbit->yaw += (orbit->yawTarget - orbit->yaw) * rotateBlend;
    orbit->pitch += (orbit->pitchTarget - orbit->pitch) * rotateBlend;
    orbit->distance += (orbit->distanceTarget - orbit->distance) * zoomBlend;

    orbit->target = (Vector3){ 0.0f, 0.0f, 0.0f };
    Vector3 forward = GetCameraDirection(orbit->yaw, orbit->pitch);
    camera->target = orbit->target;
    camera->position = Vector3Add(orbit->target, Vector3Scale(forward, orbit->distance));
    camera->up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera->fovy = 45.0f;
    camera->projection = CAMERA_PERSPECTIVE;
}

static void EnsureVertexCapacity(VertexBuffer *vertices, int minCapacity)
{
    if (vertices->capacity >= minCapacity) return;
    int nextCapacity = (vertices->capacity == 0) ? 64 : vertices->capacity;
    while (nextCapacity < minCapacity) nextCapacity *= 2;
    vertices->items = (Vector3 *)realloc(vertices->items, sizeof(Vector3) * (size_t)nextCapacity);
    vertices->capacity = nextCapacity;
}

static int PushVertex(VertexBuffer *vertices, Vector3 value)
{
    EnsureVertexCapacity(vertices, vertices->count + 1);
    vertices->items[vertices->count] = value;
    vertices->count += 1;
    return vertices->count - 1;
}

static void EnsureTriangleCapacity(TriangleBuffer *triangles, int minCapacity)
{
    if (triangles->capacity >= minCapacity) return;
    int nextCapacity = (triangles->capacity == 0) ? 64 : triangles->capacity;
    while (nextCapacity < minCapacity) nextCapacity *= 2;
    triangles->items = (Triangle *)realloc(triangles->items, sizeof(Triangle) * (size_t)nextCapacity);
    triangles->capacity = nextCapacity;
}

static void PushTriangle(TriangleBuffer *triangles, int a, int b, int c)
{
    EnsureTriangleCapacity(triangles, triangles->count + 1);
    triangles->items[triangles->count] = (Triangle){ a, b, c };
    triangles->count += 1;
}

static int NextPow2(int value)
{
    int result = 1;
    while (result < value) result <<= 1;
    return result;
}

static uint32_t Hash64(uint64_t key)
{
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27;
    key *= 0x94d049bb133111ebULL;
    key ^= key >> 31;
    return (uint32_t)(key & 0xffffffffU);
}

static uint64_t MakeEdgeKey(int a, int b)
{
    uint32_t minV = (a < b) ? (uint32_t)a : (uint32_t)b;
    uint32_t maxV = (a < b) ? (uint32_t)b : (uint32_t)a;
    return ((uint64_t)minV << 32) | (uint64_t)maxV;
}

static EdgeCache CreateEdgeCache(int expectedPairs)
{
    EdgeCache cache = { 0 };
    cache.capacity = NextPow2(expectedPairs * 2);
    cache.entries = (EdgeEntry *)calloc((size_t)cache.capacity, sizeof(EdgeEntry));
    return cache;
}

static void DestroyEdgeCache(EdgeCache *cache)
{
    free(cache->entries);
    cache->entries = NULL;
    cache->capacity = 0;
}

static int FindEdgeMidpoint(const EdgeCache *cache, uint64_t key)
{
    int mask = cache->capacity - 1;
    int slot = (int)(Hash64(key) & (uint32_t)mask);
    while (cache->entries[slot].used) {
        if (cache->entries[slot].key == key) return cache->entries[slot].midpointIndex;
        slot = (slot + 1) & mask;
    }
    return -1;
}

static void InsertEdgeMidpoint(EdgeCache *cache, uint64_t key, int midpointIndex)
{
    int mask = cache->capacity - 1;
    int slot = (int)(Hash64(key) & (uint32_t)mask);
    while (cache->entries[slot].used) {
        if (cache->entries[slot].key == key) {
            cache->entries[slot].midpointIndex = midpointIndex;
            return;
        }
        slot = (slot + 1) & mask;
    }
    cache->entries[slot].used = true;
    cache->entries[slot].key = key;
    cache->entries[slot].midpointIndex = midpointIndex;
}

static int GetMidpointVertex(VertexBuffer *vertices, EdgeCache *cache, int a, int b)
{
    uint64_t key = MakeEdgeKey(a, b);
    int cached = FindEdgeMidpoint(cache, key);
    if (cached >= 0) return cached;

    Vector3 midpoint = Vector3Scale(Vector3Add(vertices->items[a], vertices->items[b]), 0.5f);
    midpoint = Vector3Normalize(midpoint);
    int newIndex = PushVertex(vertices, midpoint);
    InsertEdgeMidpoint(cache, key, newIndex);
    return newIndex;
}

static void BuildIcosphere(VertexBuffer *vertices, TriangleBuffer *triangles, int subdivisions, float radius)
{
    const float t = (1.0f + sqrtf(5.0f)) * 0.5f;
    const Vector3 baseVerts[12] = {
        { -1.0f,  t,  0.0f }, {  1.0f,  t,  0.0f }, { -1.0f, -t,  0.0f }, {  1.0f, -t,  0.0f },
        {  0.0f, -1.0f,  t }, {  0.0f,  1.0f,  t }, {  0.0f, -1.0f, -t }, {  0.0f,  1.0f, -t },
        {  t,  0.0f, -1.0f }, {  t,  0.0f,  1.0f }, { -t,  0.0f, -1.0f }, { -t,  0.0f,  1.0f }
    };
    const Triangle baseFaces[20] = {
        { 0, 11, 5 }, { 0, 5, 1 }, { 0, 1, 7 }, { 0, 7, 10 }, { 0, 10, 11 },
        { 1, 5, 9 }, { 5, 11, 4 }, { 11, 10, 2 }, { 10, 7, 6 }, { 7, 1, 8 },
        { 3, 9, 4 }, { 3, 4, 2 }, { 3, 2, 6 }, { 3, 6, 8 }, { 3, 8, 9 },
        { 4, 9, 5 }, { 2, 4, 11 }, { 6, 2, 10 }, { 8, 6, 7 }, { 9, 8, 1 }
    };

    for (int i = 0; i < 12; i++) PushVertex(vertices, Vector3Normalize(baseVerts[i]));
    for (int i = 0; i < 20; i++) PushTriangle(triangles, baseFaces[i].a, baseFaces[i].b, baseFaces[i].c);

    for (int step = 0; step < subdivisions; step++) {
        EdgeCache cache = CreateEdgeCache(triangles->count * 3);
        TriangleBuffer next = { 0 };
        EnsureTriangleCapacity(&next, triangles->count * 4);

        for (int i = 0; i < triangles->count; i++) {
            Triangle tri = triangles->items[i];
            int ab = GetMidpointVertex(vertices, &cache, tri.a, tri.b);
            int bc = GetMidpointVertex(vertices, &cache, tri.b, tri.c);
            int ca = GetMidpointVertex(vertices, &cache, tri.c, tri.a);

            PushTriangle(&next, tri.a, ab, ca);
            PushTriangle(&next, tri.b, bc, ab);
            PushTriangle(&next, tri.c, ca, bc);
            PushTriangle(&next, ab, bc, ca);
        }

        free(triangles->items);
        *triangles = next;
        DestroyEdgeCache(&cache);
    }

    for (int i = 0; i < vertices->count; i++) {
        vertices->items[i] = Vector3Scale(Vector3Normalize(vertices->items[i]), radius);
    }
}

static void AddNeighborUnique(TileGraphNode *graph, int from, int to)
{
    if (from == to) return;
    TileGraphNode *node = &graph[from];
    for (int i = 0; i < node->neighborCount; i++) {
        if (node->neighbors[i] == to) return;
    }
    if (node->neighborCount < MAX_TILE_NEIGHBORS) {
        node->neighbors[node->neighborCount++] = to;
    }
}

static TileGraphNode *BuildTileGraph(const VertexBuffer *vertices, const TriangleBuffer *triangles, int *outCount)
{
    int count = vertices->count;
    TileGraphNode *graph = (TileGraphNode *)calloc((size_t)count, sizeof(TileGraphNode));
    for (int i = 0; i < triangles->count; i++) {
        Triangle t = triangles->items[i];
        AddNeighborUnique(graph, t.a, t.b);
        AddNeighborUnique(graph, t.b, t.a);
        AddNeighborUnique(graph, t.b, t.c);
        AddNeighborUnique(graph, t.c, t.b);
        AddNeighborUnique(graph, t.c, t.a);
        AddNeighborUnique(graph, t.a, t.c);
    }
    if (outCount) *outCount = count;
    return graph;
}

static void RefreshWeatherSurface(WeatherCell *cells, const Tile *tiles, int count)
{
    for (int i = 0; i < count; i++) {
        Vector3 normal = tiles[i].baseCenterDir;
        float elevationRaw = tiles[i].elevation * PLANET_RADIUS;
        cells[i].normal = normal;
        cells[i].elevation = ClampFloat(elevationRaw / 0.12f, -1.0f, 1.2f);
        cells[i].ocean = SmoothStep01((-elevationRaw + 0.010f) / 0.055f);
    }
}

static float GaussianBand(float x, float center, float width)
{
    float d = (x - center) / width;
    return expf(-d * d);
}

static void InitializeWeather(WeatherCell *cells, const Tile *tiles, int count)
{
    RefreshWeatherSurface(cells, tiles, count);
    for (int i = 0; i < count; i++) {
        float latAbs = fabsf(cells[i].normal.y);
        float climateLat = WarpedClimateLatitude(cells[i].normal, 0.145f);
        float planetaryWave = SphericalNoise3(cells[i].normal, 1.65f, (Vector3){ 23.7f, -18.2f, 9.4f }, 3, 1.92f, 0.54f);
        float moistureWave = SphericalNoise3(cells[i].normal, 4.35f, (Vector3){ -42.1f, 7.8f, 31.6f }, 3, 2.07f, 0.52f);
        float eddyWave = SphericalNoise3(cells[i].normal, 7.8f, (Vector3){ 15.2f, 51.4f, -22.7f }, 2, 2.16f, 0.48f);
        float pressureSeed = SphericalNoise3(cells[i].normal, 2.85f, (Vector3){ 88.1f, -34.6f, 12.5f }, 3, 2.04f, 0.52f);
        float equatorHeat = ClampFloat(1.0f - climateLat * climateLat + (planetaryWave - 0.5f) * 0.045f, 0.0f, 1.0f);
        float westerly = GaussianBand(climateLat, 0.45f, 0.17f);
        float trade = GaussianBand(climateLat, 0.20f, 0.14f);
        float polar = GaussianBand(climateLat, 0.78f, 0.11f);
        float itcz = GaussianBand(climateLat, 0.04f, 0.22f) * LerpFloat(0.62f, 1.18f, moistureWave);
        float subtropicalHigh = GaussianBand(climateLat, 0.31f, 0.15f);
        float stormTrack = GaussianBand(climateLat, 0.55f, 0.22f) * LerpFloat(0.70f, 1.16f, planetaryWave);
        float polarHigh = GaussianBand(climateLat, 0.91f, 0.16f);

        float land = 1.0f - cells[i].ocean;
        cells[i].temperature = ClampFloat(0.08f + 0.88f * powf(equatorHeat, 0.86f) + land * (equatorHeat - 0.45f) * 0.070f + cells[i].ocean * 0.020f - fmaxf(cells[i].elevation, 0.0f) * 0.34f - fmaxf(-cells[i].elevation, 0.0f) * 0.025f, 0.0f, 1.0f);
        cells[i].pressure = ClampFloat(0.56f - itcz * 0.018f + subtropicalHigh * 0.024f - stormTrack * 0.018f + polarHigh * 0.015f + (pressureSeed - 0.5f) * 0.035f - land * (cells[i].temperature - 0.50f) * 0.075f - fmaxf(cells[i].elevation, 0.0f) * 0.060f, 0.18f, 0.88f);
        cells[i].humidity = ClampFloat(0.19f + cells[i].ocean * 0.54f + itcz * 0.060f + stormTrack * 0.040f + (moistureWave - 0.5f) * 0.060f - subtropicalHigh * 0.050f + cells[i].temperature * 0.08f - fmaxf(cells[i].elevation, 0.0f) * 0.11f, 0.0f, 1.2f);
        cells[i].cloud = ClampFloat(0.05f + itcz * 0.065f + stormTrack * 0.060f + cells[i].humidity * 0.18f + (moistureWave - 0.5f) * 0.055f - subtropicalHigh * 0.040f + fmaxf(cells[i].elevation, 0.0f) * 0.05f, 0.0f, 1.0f);
        cells[i].precipitation = 0.0f;
        cells[i].vorticity = 0.0f;
        cells[i].evaporation = 0.0f;
        float mountainSnow = SmoothStep01((cells[i].elevation - 0.58f) / 0.44f) * SmoothStep01((0.52f - cells[i].temperature) / 0.24f);
        cells[i].snow = ClampFloat((GaussianBand(latAbs, 0.92f, 0.22f) + mountainSnow * 0.75f) * (1.0f - cells[i].ocean * 0.55f) * SmoothStep01((0.42f - cells[i].temperature) / 0.30f), 0.0f, 1.0f);
        cells[i].storm = 0.0f;
        cells[i].surfaceTemperature = cells[i].temperature;
        cells[i].soilMoisture = ClampFloat((1.0f - cells[i].ocean) * (0.10f + cells[i].humidity * 0.32f + stormTrack * 0.10f - subtropicalHigh * 0.08f - fmaxf(cells[i].elevation, 0.0f) * 0.08f), 0.0f, 1.0f);
        cells[i].cloudWater = cells[i].cloud * 0.35f;
        cells[i].recentRain = 0.0f;
        cells[i].rainShadow = 0.0f;
        cells[i].orographicLift = 0.0f;

        Vector3 east = TangentEast(cells[i].normal);
        Vector3 north = TangentNorth(cells[i].normal);

        float zonal = (0.088f * westerly - 0.066f * trade - 0.035f * polar) * LerpFloat(0.64f, 1.10f, planetaryWave) * 0.62f;
        float meridional = ((planetaryWave - 0.5f) * 0.052f + (eddyWave - 0.5f) * 0.034f) * (1.0f - latAbs * 0.35f);
        Vector3 windEddy = ClimateEddyFlow(cells[i].normal, 3.4f, (Vector3){ -8.7f, 61.4f, 25.1f }, 0.010f);
        cells[i].wind = Vector3Add(
            Vector3Add(Vector3Add(Vector3Scale(east, zonal), Vector3Scale(north, meridional)), windEddy),
            Vector3Scale(RandomTangentVector(cells[i].normal), RandomRange(-0.006f, 0.006f))
        );

        // Initial ocean gyres use latitude bands, with planetary waves to avoid perfectly parallel rings.
        float oceanJet = (0.052f * westerly - 0.045f * trade - 0.020f * polar) * cells[i].ocean * LerpFloat(0.82f, 1.06f, planetaryWave) * 0.32f;
        Vector3 currentEddy = ClimateEddyFlow(cells[i].normal, 2.6f, (Vector3){ 37.8f, -14.2f, 73.6f }, 0.015f * cells[i].ocean);
        cells[i].current = Vector3Add(
            Vector3Add(Vector3Add(Vector3Scale(east, oceanJet), Vector3Scale(north, meridional * cells[i].ocean * 0.30f)), currentEddy),
            Vector3Scale(RandomTangentVector(cells[i].normal), RandomRange(-0.003f, 0.003f) * cells[i].ocean)
        );
    }
}

static void StepWeatherSimulation(
    const TileGraphNode *graph,
    const WeatherCell *src,
    WeatherCell *dst,
    int count,
    float dt
)
{
    for (int i = 0; i < count; i++) {
        const WeatherCell *c = &src[i];
        const TileGraphNode *node = &graph[i];

        float tempAvg = c->temperature;
        float humidAvg = c->humidity;
        float cloudAvg = c->cloud;
        float cloudWaterAvg = c->cloudWater;
        float soilMoistureAvg = c->soilMoisture;
        float pressAvg = c->pressure;
        Vector3 windAvg = c->wind;
        Vector3 currentAvg = c->current;
        float scalarWeight = 1.0f;
        Vector3 pressureGradient = { 0 };
        Vector3 terrainGradient = { 0 };
        float divergence = 0.0f;
        float vorticity = 0.0f;
        float orographic = 0.0f;
        float oceanAvg = c->ocean;
        float elevationAvg = c->elevation;
        float coastalContrast = 0.0f;
        float upwindOcean = c->ocean;
        float upwindOceanWeight = 1.0f;
        Vector3 transportLocal = Vector3Add(c->wind, Vector3Scale(c->current, 0.85f));

        for (int n = 0; n < node->neighborCount; n++) {
            int nb = node->neighbors[n];
            const WeatherCell *k = &src[nb];
            Vector3 toNb = Vector3Subtract(k->normal, Vector3Scale(c->normal, Vector3DotProduct(k->normal, c->normal)));
            float toNbLen = Vector3Length(toNb);
            if (toNbLen < 0.000001f) continue;
            toNb = Vector3Scale(toNb, 1.0f / toNbLen);

            float flow = Vector3DotProduct(transportLocal, toNb);
            float upwindWeight = 1.0f + fmaxf(0.0f, -flow * 2.6f);

            tempAvg += k->temperature * upwindWeight;
            humidAvg += k->humidity * upwindWeight;
            cloudAvg += k->cloud * upwindWeight;
            cloudWaterAvg += k->cloudWater * upwindWeight;
            windAvg = Vector3Add(windAvg, Vector3Scale(k->wind, upwindWeight));
            currentAvg = Vector3Add(currentAvg, Vector3Scale(k->current, upwindWeight));
            scalarWeight += upwindWeight;
            soilMoistureAvg += k->soilMoisture;

            pressAvg += k->pressure;
            pressureGradient = Vector3Add(pressureGradient, Vector3Scale(toNb, (k->pressure - c->pressure)));
            terrainGradient = Vector3Add(terrainGradient, Vector3Scale(toNb, (k->elevation - c->elevation)));
            divergence += Vector3DotProduct(Vector3Subtract(k->wind, c->wind), toNb);
            vorticity += Vector3DotProduct(Vector3CrossProduct(toNb, Vector3Subtract(k->wind, c->wind)), c->normal);
            oceanAvg += k->ocean;
            elevationAvg += k->elevation;
            coastalContrast += fabsf(k->ocean - c->ocean);

            float rise = k->elevation - c->elevation;
            if (rise > 0.0f && flow > 0.0f) orographic += rise * flow;
            float inflow = fmaxf(0.0f, -flow);
            upwindOcean += k->ocean * inflow;
            upwindOceanWeight += inflow;
        }

        float invWeight = 1.0f / scalarWeight;
        tempAvg *= invWeight;
        humidAvg *= invWeight;
        cloudAvg *= invWeight;
        cloudWaterAvg *= invWeight;
        windAvg = Vector3Scale(windAvg, invWeight);
        currentAvg = Vector3Scale(currentAvg, invWeight);
        if (node->neighborCount > 0) {
            soilMoistureAvg /= (float)(node->neighborCount + 1);
            pressAvg /= (float)(node->neighborCount + 1);
            terrainGradient = Vector3Scale(terrainGradient, 1.0f / (float)node->neighborCount);
            divergence /= (float)node->neighborCount;
            vorticity /= (float)node->neighborCount;
            orographic = ClampFloat(orographic / (float)node->neighborCount, 0.0f, 1.0f);
            oceanAvg /= (float)(node->neighborCount + 1);
            elevationAvg /= (float)(node->neighborCount + 1);
            coastalContrast = ClampFloat(coastalContrast / (float)node->neighborCount, 0.0f, 1.0f);
        }
        upwindOcean = ClampFloat(upwindOcean / upwindOceanWeight, 0.0f, 1.0f);
        float terrainSlope = ClampFloat(Vector3Length(terrainGradient), 0.0f, 1.0f);

        float latAbs = fabsf(c->normal.y);
        float climateLat = WarpedClimateLatitude(c->normal, 0.145f);
        float planetaryWave = SphericalNoise3(c->normal, 1.65f, (Vector3){ 23.7f, -18.2f, 9.4f }, 3, 1.92f, 0.54f);
        float moistureWave = SphericalNoise3(c->normal, 4.35f, (Vector3){ -42.1f, 7.8f, 31.6f }, 3, 2.07f, 0.52f);
        float eddyWave = SphericalNoise3(c->normal, 7.8f, (Vector3){ 15.2f, 51.4f, -22.7f }, 2, 2.16f, 0.48f);
        float equatorHeat = ClampFloat(1.0f - climateLat * climateLat + (planetaryWave - 0.5f) * 0.045f, 0.0f, 1.0f);
        float itcz = GaussianBand(climateLat, 0.04f, 0.17f) * LerpFloat(0.84f, 1.12f, moistureWave);
        float subtropicalHigh = GaussianBand(climateLat, 0.31f, 0.13f);
        float stormTrack = GaussianBand(climateLat, 0.55f, 0.22f) * LerpFloat(0.86f, 1.08f, planetaryWave);
        float polarHigh = GaussianBand(climateLat, 0.91f, 0.16f);
        float convergence = ClampFloat(-divergence * 1.8f, 0.0f, 1.0f);
        Vector3 transport = Vector3Add(c->wind, Vector3Scale(c->current, 0.85f));
        float advection = ClampFloat(Vector3Length(transport) * 4.2f, 0.0f, 1.0f);
        float land = 1.0f - c->ocean;
        float albedo = ClampFloat(0.07f * c->ocean + 0.23f * land + 0.34f * c->snow + 0.12f * c->cloud, 0.0f, 0.75f);
        float lapseCooling = fmaxf(c->elevation, 0.0f) * 0.38f + fmaxf(c->elevation - 0.55f, 0.0f) * 0.22f + fmaxf(-c->elevation, 0.0f) * 0.025f;
        float landHeating = land * (equatorHeat - 0.45f) * 0.090f;
        float marineModeration = c->ocean * (0.50f - c->temperature) * 0.055f;
        float terrainCooling = fmaxf(elevationAvg, 0.0f) * 0.070f;
        float temperatureTarget = ClampFloat(0.08f + 0.88f * powf(equatorHeat, 0.86f) + landHeating + marineModeration - lapseCooling - terrainCooling - albedo * 0.16f + c->humidity * 0.015f, 0.0f, 1.0f);
        float basePressure = 0.56f - itcz * 0.025f + subtropicalHigh * 0.032f - stormTrack * 0.018f + polarHigh * 0.015f;
        float thermalLow = land * (temperatureTarget - 0.50f) * 0.110f;
        float terrainLow = fmaxf(c->elevation, 0.0f) * 0.060f + terrainSlope * land * 0.020f;
        float coastalBreezePressure = coastalContrast * (c->ocean - oceanAvg) * 0.050f;
        float pressureTarget = basePressure - thermalLow - c->humidity * 0.015f - terrainLow + coastalBreezePressure;

        float pressure = c->pressure;
        pressure += (pressureTarget - pressure) * 0.42f * dt;
        pressure += (pressAvg - pressure) * 0.72f * dt;
        pressure -= convergence * 0.045f * dt;
        pressure = ClampFloat(pressure, 0.18f, 0.88f);

        Vector3 east = TangentEast(c->normal);
        Vector3 north = TangentNorth(c->normal);
        float northLen = Vector3Length(north);

        float westerly = GaussianBand(climateLat, 0.45f, 0.17f);
        float trade = GaussianBand(climateLat, 0.20f, 0.14f);
        float polar = GaussianBand(climateLat, 0.78f, 0.11f);
        float zonalStrength = (0.088f * westerly - 0.066f * trade - 0.035f * polar) * LerpFloat(0.78f, 1.08f, planetaryWave) * 0.52f;
        float meridionalStrength = ((planetaryWave - 0.5f) * 0.020f + (eddyWave - 0.5f) * 0.015f) * (1.0f - latAbs * 0.35f);
        Vector3 terrainDownslope = Vector3Scale(terrainGradient, -0.055f * land);
        Vector3 terrainContour = Vector3CrossProduct(c->normal, terrainGradient);
        if (Vector3LengthSqr(terrainContour) > 0.000001f) terrainContour = Vector3Normalize(terrainContour);
        Vector3 terrainSteering = Vector3Add(
            terrainDownslope,
            Vector3Scale(terrainContour, Vector3DotProduct(c->wind, terrainGradient) * -0.090f * land)
        );
        float valleyChannel = land * SmoothStep01((-c->elevation) / 0.12f) * terrainSlope;
        Vector3 valleyBoost = Vector3Scale(terrainContour, Vector3DotProduct(c->wind, terrainContour) * valleyChannel * 0.07f);
        Vector3 jet = Vector3Add(Vector3Add(Vector3Add(Vector3Scale(east, zonalStrength), Vector3Scale(north, meridionalStrength)), terrainSteering), valleyBoost);

        Vector3 pressureForce = Vector3Scale(pressureGradient, -1.45f);
        Vector3 coriolis = Vector3Scale(Vector3CrossProduct(c->normal, c->wind), c->normal.y * 0.92f);
        Vector3 wind = c->wind;
        wind = Vector3Add(wind, Vector3Scale(Vector3Add(Vector3Add(pressureForce, coriolis), jet), dt));
        wind = Vector3Add(wind, Vector3Scale(Vector3Subtract(windAvg, wind), (0.54f + 0.34f * advection) * dt));

        float surfaceRoughness = land * (0.30f + terrainSlope * 0.72f + fmaxf(c->elevation, 0.0f) * 0.14f + c->snow * 0.08f);
        float mountainBlocking = land * terrainSlope * SmoothStep01((fabsf(Vector3DotProduct(wind, terrainGradient)) - 0.004f) / 0.030f);
        float drag = 0.22f + surfaceRoughness + mountainBlocking * 0.52f + c->cloud * 0.08f;
        wind = Vector3Scale(wind, ClampFloat(1.0f - drag * dt, 0.0f, 1.0f));
        float windLen = Vector3Length(wind);
        if (windLen > 0.18f) wind = Vector3Scale(wind, 0.18f / windLen);

        // Ocean currents use gyre bands, wind stress, pressure gradient, and Coriolis.
        Vector3 current = c->current;
        Vector3 gyre = Vector3Add(
            Vector3Add(
                Vector3Scale(east, (0.052f * westerly - 0.045f * trade - 0.020f * polar) * c->ocean * LerpFloat(0.82f, 1.06f, planetaryWave) * 0.32f),
                Vector3Scale(north, meridionalStrength * c->ocean * 0.24f)
            ),
            Vector3Scale(terrainGradient, -0.018f * c->ocean)
        );
        Vector3 currentPressureForce = Vector3Scale(pressureGradient, -0.18f * c->ocean);
        Vector3 windStress = Vector3Scale(wind, 0.42f * c->ocean);
        Vector3 currentCoriolis = Vector3Scale(Vector3CrossProduct(c->normal, current), c->normal.y * 0.45f);
        current = Vector3Add(current, Vector3Scale(Vector3Add(Vector3Add(currentPressureForce, windStress), currentCoriolis), dt));
        current = Vector3Add(current, Vector3Scale(Vector3Subtract(gyre, current), 0.42f * dt));
        current = Vector3Add(current, Vector3Scale(Vector3Subtract(currentAvg, current), 0.50f * dt));

        float currentDrag = 0.14f + land * 1.05f;
        current = Vector3Scale(current, ClampFloat(1.0f - currentDrag * dt, 0.0f, 1.0f));
        float currentLen = Vector3Length(current);
        if (currentLen > 0.12f) current = Vector3Scale(current, 0.12f / currentLen);

        float polewardCurrent = 0.0f;
        if (northLen > 0.000001f) {
            polewardCurrent = Vector3DotProduct(current, north) * ((c->normal.y >= 0.0f) ? 1.0f : -1.0f);
        }

        float upslopeFlow = ClampFloat(Vector3DotProduct(wind, terrainGradient) * 18.0f, 0.0f, 1.0f);
        float downslopeFlow = ClampFloat(-Vector3DotProduct(wind, terrainGradient) * 20.0f, 0.0f, 1.0f);
        float orographicLift = ClampFloat(orographic * 1.85f + upslopeFlow * 1.25f + terrainSlope * land * 0.20f, 0.0f, 1.0f);
        float rainShadow = ClampFloat(c->rainShadow * 0.70f + downslopeFlow * 0.78f + terrainSlope * land * 0.14f - upwindOcean * 0.12f - orographicLift * 0.22f, 0.0f, 0.90f);

        float surfaceTemperature = c->surfaceTemperature;
        float coldPool = land * fmaxf(0.0f, -c->elevation) * ClampFloat(1.0f - Vector3Length(c->wind) * 8.0f, 0.0f, 1.0f) * 0.055f;
        float surfaceTarget = ClampFloat(
            temperatureTarget + land * (equatorHeat - 0.50f) * 0.060f - c->snow * 0.12f - c->soilMoisture * land * 0.035f + rainShadow * land * 0.050f - coldPool,
            0.0f,
            1.0f
        );
        float surfaceResponse = LerpFloat(0.20f, 0.80f, land) * (1.0f - c->soilMoisture * land * 0.35f);
        surfaceTemperature += (surfaceTarget - surfaceTemperature) * surfaceResponse * dt;
        surfaceTemperature = ClampFloat(surfaceTemperature, 0.0f, 1.0f);

        float temperature = c->temperature;
        float heatResponse = LerpFloat(0.42f, 0.76f, land);
        temperature += (surfaceTemperature - temperature) * heatResponse * dt;
        temperature += (tempAvg - temperature) * (0.15f + 0.76f * advection) * dt;
        temperature += polewardCurrent * c->ocean * 0.10f * dt;
        temperature += rainShadow * land * 0.070f * dt;
        temperature -= c->cloud * 0.018f * dt;
        temperature = ClampFloat(temperature, 0.0f, 1.0f);

        float humidity = c->humidity;
        float saturation = WeatherSaturation(temperature);
        float coastalMoisture = ClampFloat(oceanAvg * 0.42f + upwindOcean * 0.72f + coastalContrast * 0.30f, 0.0f, 1.0f);
        float landWetness = ClampFloat(
            c->soilMoisture * 0.62f + c->recentRain * 0.64f + coastalMoisture * 0.20f + c->cloudWater * 0.18f + c->snow * 0.22f,
            0.0f,
            1.0f
        );
        float surfaceWetness = ClampFloat(c->ocean + land * landWetness, 0.0f, 1.0f);
        float humidityDeficit = ClampFloat((saturation - humidity + 0.16f) / 0.84f, 0.0f, 1.0f);
        float climateHumidity = ClampFloat(0.16f + c->ocean * 0.43f + coastalMoisture * 0.23f + land * c->soilMoisture * 0.28f + itcz * 0.090f + stormTrack * 0.040f - subtropicalHigh * 0.070f + temperature * 0.08f - fmaxf(c->elevation, 0.0f) * 0.085f - rainShadow * 0.30f, 0.0f, 1.18f);
        float evaporationRate = surfaceWetness * (0.018f + surfaceTemperature * 0.128f + windLen * 0.24f + land * 0.018f) * humidityDeficit;
        humidity += evaporationRate * dt;
        humidity += (climateHumidity - humidity) * 0.11f * dt;
        humidity += (humidAvg - humidity) * (0.12f + 0.72f * advection) * dt;
        humidity -= rainShadow * land * 0.090f * dt;

        float lift = convergence * 0.40f + orographicLift * 1.05f + fabsf(vorticity) * 0.06f;
        float relativeHumidity = ClampFloat(humidity / saturation, 0.0f, 2.0f);
        float condensation = fmaxf(0.0f, humidity - saturation * 0.96f) * (0.80f + 1.40f * lift) * dt;
        condensation += humidity * lift * SmoothStep01((relativeHumidity - 0.58f) / 0.46f) * 0.055f * dt;
        humidity -= condensation;

        float cloudWater = c->cloudWater;
        cloudWater += condensation * 1.20f;
        cloudWater += (cloudWaterAvg - cloudWater) * (0.18f + advection * 0.36f) * dt;
        cloudWater -= rainShadow * land * 0.040f * dt;
        cloudWater = ClampFloat(cloudWater, 0.0f, 1.20f);

        float cloud = c->cloud;
        float terrainCloud = land * c->soilMoisture * SmoothStep01((relativeHumidity - 0.40f) / 0.45f);
        float cloudTarget = ClampFloat(SmoothStep01(cloudWater / 0.34f) * 0.72f + (relativeHumidity - 0.58f) * 0.42f + lift * 0.28f + terrainCloud * 0.20f + coastalMoisture * 0.06f - rainShadow * 0.16f, 0.0f, 1.0f);
        cloud += (cloudTarget - cloud) * (0.42f + 0.18f * advection) * dt;
        cloud += (cloudAvg - cloud) * 0.18f * dt;

        float storm = ClampFloat(lift * 0.95f + orographicLift * 0.44f + fabsf(vorticity) * 0.50f + convergence * 0.22f + fmaxf(0.0f, relativeHumidity - 0.82f) * 0.88f + cloudWater * 0.28f, 0.0f, 1.0f);
        float rainRate = cloudWater * (0.030f + 0.46f * storm + 0.14f * convergence + 0.58f * orographicLift + terrainCloud * 0.065f);
        float rain = rainRate * dt;
        cloudWater = ClampFloat(cloudWater - rain * 1.55f, 0.0f, 1.20f);
        float coldPrecip = SmoothStep01((0.35f - temperature) / 0.18f);
        float snow = c->snow;
        snow += rain * coldPrecip * (land + c->ocean * 0.35f) * 0.85f;
        snow -= fmaxf(0.0f, temperature - 0.33f) * (0.18f + rain * 0.8f) * dt;
        float mountainSnow = SmoothStep01((c->elevation - 0.58f) / 0.44f) * SmoothStep01((0.52f - temperature) / 0.24f);
        snow += (GaussianBand(latAbs, 0.94f, 0.16f) + mountainSnow * 0.85f) * coldPrecip * land * 0.010f * dt;
        snow = ClampFloat(snow, 0.0f, 1.0f);

        float liquidRain = rain * (1.0f - coldPrecip * 0.78f);
        float runoff = ClampFloat((terrainSlope * 0.95f + c->soilMoisture * 0.35f - 0.12f) * land, 0.0f, 0.65f);
        float soilMoisture = c->soilMoisture;
        soilMoisture += liquidRain * land * (4.60f * (1.0f - runoff));
        soilMoisture += fmaxf(0.0f, temperature - 0.36f) * c->snow * land * 0.018f * dt;
        soilMoisture += (soilMoistureAvg - soilMoisture) * 0.055f * dt;
        soilMoisture -= evaporationRate * land * 0.78f * dt;
        soilMoisture -= runoff * 0.060f * dt;
        soilMoisture = ClampFloat(soilMoisture * land, 0.0f, 1.0f);

        cloud -= rain * 0.42f;
        humidity -= rain * 0.30f;
        cloud -= (0.030f + (1.0f - temperature) * 0.012f + c->snow * 0.020f + rainShadow * land * 0.035f) * dt;

        humidity = ClampFloat(humidity, 0.0f, 1.4f);
        cloud = ClampFloat(cloud, 0.0f, 1.0f);
        float precipitationTarget = ClampFloat(rainRate * 3.80f, 0.0f, 1.0f);
        float precipitation = ClampFloat(c->precipitation + (precipitationTarget - c->precipitation) * (0.75f + storm * 1.40f) * dt, 0.0f, 1.0f);
        float evaporation = ClampFloat(c->evaporation + (evaporationRate * 9.0f - c->evaporation) * 0.72f * dt, 0.0f, 1.0f);
        float recentRain = ClampFloat(c->recentRain + (precipitationTarget - c->recentRain) * (0.45f + storm * 1.20f) * dt, 0.0f, 1.0f);

        dst[i] = *c;
        dst[i].wind = wind;
        dst[i].current = current;
        dst[i].temperature = temperature;
        dst[i].pressure = pressure;
        dst[i].humidity = humidity;
        dst[i].cloud = cloud;
        dst[i].precipitation = precipitation;
        dst[i].vorticity = ClampFloat(vorticity * 13.0f, -1.0f, 1.0f);
        dst[i].evaporation = evaporation;
        dst[i].snow = snow;
        dst[i].storm = storm;
        dst[i].surfaceTemperature = surfaceTemperature;
        dst[i].soilMoisture = soilMoisture;
        dst[i].cloudWater = cloudWater;
        dst[i].recentRain = recentRain;
        dst[i].rainShadow = rainShadow;
        dst[i].orographicLift = orographicLift;
    }
}

static const unsigned char kPerlinPerm[256] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

static int PerlinPerm(int x)
{
    return (int)kPerlinPerm[x & 255];
}

static float PerlinFade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float PerlinGrad(int hash, float x, float y, float z)
{
    int h = hash & 15;
    float u = (h < 8) ? x : y;
    float v = (h < 4) ? y : ((h == 12 || h == 14) ? x : z);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

static float PerlinNoise3(float x, float y, float z)
{
    int ix0 = (int)floorf(x);
    int iy0 = (int)floorf(y);
    int iz0 = (int)floorf(z);
    float fx0 = x - (float)ix0;
    float fy0 = y - (float)iy0;
    float fz0 = z - (float)iz0;
    float fx1 = fx0 - 1.0f;
    float fy1 = fy0 - 1.0f;
    float fz1 = fz0 - 1.0f;

    int X = ix0 & 255;
    int Y = iy0 & 255;
    int Z = iz0 & 255;

    int A = (PerlinPerm(X) + Y) & 255;
    int B = (PerlinPerm((X + 1) & 255) + Y) & 255;
    int AA = (PerlinPerm(A) + Z) & 255;
    int AB = (PerlinPerm((A + 1) & 255) + Z) & 255;
    int BA = (PerlinPerm(B) + Z) & 255;
    int BB = (PerlinPerm((B + 1) & 255) + Z) & 255;

    float u = PerlinFade(fx0);
    float v = PerlinFade(fy0);
    float w = PerlinFade(fz0);

    float x00 = LerpFloat(PerlinGrad(PerlinPerm(AA), fx0, fy0, fz0), PerlinGrad(PerlinPerm(BA), fx1, fy0, fz0), u);
    float x10 = LerpFloat(PerlinGrad(PerlinPerm(AB), fx0, fy1, fz0), PerlinGrad(PerlinPerm(BB), fx1, fy1, fz0), u);
    float x01 = LerpFloat(PerlinGrad(PerlinPerm((AA + 1) & 255), fx0, fy0, fz1), PerlinGrad(PerlinPerm((BA + 1) & 255), fx1, fy0, fz1), u);
    float x11 = LerpFloat(PerlinGrad(PerlinPerm((AB + 1) & 255), fx0, fy1, fz1), PerlinGrad(PerlinPerm((BB + 1) & 255), fx1, fy1, fz1), u);

    float y0 = LerpFloat(x00, x10, v);
    float y1 = LerpFloat(x01, x11, v);
    float value = LerpFloat(y0, y1, w);
    return value * 0.5f + 0.5f;
}

static float FbmNoise3(Vector3 p, int octaves, float lacunarity, float gain)
{
    float sum = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float norm = 0.0f;

    for (int i = 0; i < octaves; i++) {
        sum += PerlinNoise3(p.x * frequency, p.y * frequency, p.z * frequency) * amplitude;
        norm += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    if (norm <= 0.0f) return 0.0f;
    return sum / norm;
}

static float RidgedFbmNoise3(Vector3 p, int octaves, float lacunarity, float gain)
{
    float sum = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float norm = 0.0f;

    for (int i = 0; i < octaves; i++) {
        float n = PerlinNoise3(p.x * frequency, p.y * frequency, p.z * frequency);
        float ridge = 1.0f - fabsf(n * 2.0f - 1.0f);
        sum += ridge * ridge * amplitude;
        norm += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    if (norm <= 0.0f) return 0.0f;
    return sum / norm;
}

static float SphericalNoise3(Vector3 unitDirection, float scale, Vector3 offset, int octaves, float lacunarity, float gain)
{
    Vector3 p = Vector3Add(Vector3Scale(unitDirection, scale), offset);
    return FbmNoise3(p, octaves, lacunarity, gain);
}

static float WarpedClimateLatitude(Vector3 unitDirection, float strength)
{
    float latAbs = fabsf(unitDirection.y);
    float planetaryWave = SphericalNoise3(unitDirection, 1.65f, (Vector3){ 23.7f, -18.2f, 9.4f }, 3, 1.92f, 0.54f);
    float eddyWave = SphericalNoise3(unitDirection, 7.8f, (Vector3){ 15.2f, 51.4f, -22.7f }, 2, 2.16f, 0.48f);
    float cellWave = SphericalNoise3(unitDirection, 3.20f, (Vector3){ -40.4f, 21.3f, 66.8f }, 3, 2.08f, 0.52f);
    float warp = (planetaryWave - 0.5f) * strength + (eddyWave - 0.5f) * strength * 0.58f + (cellWave - 0.5f) * strength * 0.36f;
    return ClampFloat(fabsf(unitDirection.y + warp * (1.0f - latAbs * 0.22f)), 0.0f, 1.0f);
}

static Vector3 TangentEast(Vector3 normal)
{
    Vector3 east = Vector3CrossProduct((Vector3){ 0.0f, 1.0f, 0.0f }, normal);
    if (Vector3LengthSqr(east) < 0.000001f) east = Vector3CrossProduct((Vector3){ 1.0f, 0.0f, 0.0f }, normal);
    return Vector3Normalize(east);
}

static Vector3 TangentNorth(Vector3 normal)
{
    Vector3 north = Vector3Subtract((Vector3){ 0.0f, 1.0f, 0.0f }, Vector3Scale(normal, normal.y));
    if (Vector3LengthSqr(north) < 0.000001f) north = Vector3CrossProduct(normal, TangentEast(normal));
    return Vector3Normalize(north);
}

static Vector3 ClimateEddyFlow(Vector3 normal, float scale, Vector3 offset, float strength)
{
    const float eps = 0.040f;
    Vector3 east = TangentEast(normal);
    Vector3 north = TangentNorth(normal);
    Vector3 e0 = Vector3Normalize(Vector3Add(normal, Vector3Scale(east, eps)));
    Vector3 e1 = Vector3Normalize(Vector3Subtract(normal, Vector3Scale(east, eps)));
    Vector3 n0 = Vector3Normalize(Vector3Add(normal, Vector3Scale(north, eps)));
    Vector3 n1 = Vector3Normalize(Vector3Subtract(normal, Vector3Scale(north, eps)));

    float dEast = SphericalNoise3(e0, scale, offset, 4, 2.04f, 0.52f) - SphericalNoise3(e1, scale, offset, 4, 2.04f, 0.52f);
    float dNorth = SphericalNoise3(n0, scale, offset, 4, 2.04f, 0.52f) - SphericalNoise3(n1, scale, offset, 4, 2.04f, 0.52f);

    return Vector3Add(Vector3Scale(east, dNorth * strength / eps), Vector3Scale(north, -dEast * strength / eps));
}

static TerrainSample EvaluateTerrain(Vector3 unitDirection, const Plate *plates, int plateCount)
{
    int p0 = 0;
    int p1 = 0;
    float dot0 = 0.0f;
    float dot1 = 0.0f;
    FindClosestPlates(unitDirection, plates, plateCount, &p0, &p1, &dot0, &dot1);

    float gap = dot0 - dot1;
    float boundary = 1.0f - SmoothStep01(gap / 0.18f);

    Vector3 boundaryDir = Vector3Subtract(plates[p1].seedDir, plates[p0].seedDir);
    boundaryDir = Vector3Subtract(boundaryDir, Vector3Scale(unitDirection, Vector3DotProduct(boundaryDir, unitDirection)));
    if (Vector3LengthSqr(boundaryDir) < 0.000001f) boundaryDir = Vector3CrossProduct(unitDirection, (Vector3){ 0.0f, 1.0f, 0.0f });
    if (Vector3LengthSqr(boundaryDir) < 0.000001f) boundaryDir = Vector3CrossProduct(unitDirection, (Vector3){ 1.0f, 0.0f, 0.0f });
    boundaryDir = Vector3Normalize(boundaryDir);
    Vector3 shearDir = Vector3Normalize(Vector3CrossProduct(unitDirection, boundaryDir));

    Vector3 velocity0 = Vector3Scale(plates[p0].driftDir, plates[p0].angularSpeed * 42.0f);
    Vector3 velocity1 = Vector3Scale(plates[p1].driftDir, plates[p1].angularSpeed * 42.0f);
    Vector3 relativeDrift = Vector3Subtract(velocity0, velocity1);
    float convergenceRaw = Vector3DotProduct(relativeDrift, boundaryDir);
    float shearRaw = fabsf(Vector3DotProduct(relativeDrift, shearDir));
    float convergent = ClampFloat(convergenceRaw * 0.54f, 0.0f, 1.0f) * boundary;
    float divergent = ClampFloat(-convergenceRaw * 0.54f, 0.0f, 1.0f) * boundary;
    float transform = ClampFloat(shearRaw * 0.58f, 0.0f, 1.0f) * boundary;
    float ocean0 = plates[p0].oceanic;
    float ocean1 = plates[p1].oceanic;
    float plateMix = boundary * 0.72f;
    float oceanMix = LerpFloat(ocean0, (ocean0 + ocean1) * 0.5f, plateMix);
    float densityMix = LerpFloat(plates[p0].density, (plates[p0].density + plates[p1].density) * 0.5f, plateMix);
    float ageMix = LerpFloat(plates[p0].crustAge, (plates[p0].crustAge + plates[p1].crustAge) * 0.5f, plateMix);
    float biasMix = LerpFloat(plates[p0].continentalBias, (plates[p0].continentalBias + plates[p1].continentalBias) * 0.5f, plateMix);
    float continental0 = 1.0f - oceanMix;
    float continental1 = 1.0f - ocean1;
    float densityDelta = densityMix - plates[p1].density;
    float oceanContrast = ClampFloat(fabsf(ocean0 - ocean1) * 1.35f, 0.0f, 1.0f);
    float continentCollision = convergent * continental0 * continental1;
    float oceanCollision = convergent * ocean0 * ocean1;
    float subduction = convergent * oceanContrast;
    float thisPlateSubducts = subduction * SmoothStep01((densityDelta + 0.10f) / 0.34f);
    float otherPlateSubducts = subduction * SmoothStep01((-densityDelta + 0.10f) / 0.34f);

    Vector3 superPosA = Vector3Add(Vector3Scale(unitDirection, 0.52f), (Vector3){ 18.7f, -6.3f, 11.9f });
    Vector3 superPosB = Vector3Add(Vector3Scale(unitDirection, 0.60f), (Vector3){ -21.1f, 12.4f, -16.8f });
    Vector3 macroPos = Vector3Add(Vector3Scale(unitDirection, 1.12f), (Vector3){ 31.4f, -9.8f, 17.7f });
    Vector3 detailPos = Vector3Add(Vector3Scale(unitDirection, 1.9f), (Vector3){ -47.2f, 18.4f, -13.1f });
    Vector3 ridgePos = Vector3Add(Vector3Scale(unitDirection, 6.7f), (Vector3){ 59.1f, -25.4f, 6.2f });
    Vector3 rangePos = Vector3Add(Vector3Scale(unitDirection, 4.4f), (Vector3){ 9.8f, 63.4f, -37.2f });
    Vector3 peakPos = Vector3Add(Vector3Scale(unitDirection, 12.8f), (Vector3){ -73.6f, 31.7f, 19.5f });
    Vector3 cragPos = Vector3Add(Vector3Scale(unitDirection, 27.0f), (Vector3){ 101.3f, -58.2f, 44.9f });
    Vector3 moistPos = Vector3Add(Vector3Scale(unitDirection, 3.9f), (Vector3){ -11.3f, 5.2f, 41.8f });

    float superA = FbmNoise3(superPosA, 3, 1.82f, 0.55f);
    float superB = FbmNoise3(superPosB, 3, 1.88f, 0.55f);
    float super = fmaxf(superA, superB * 0.98f);
    float macro = FbmNoise3(macroPos, 3, 1.92f, 0.52f);
    float detail = FbmNoise3(detailPos, 2, 2.00f, 0.50f);
    float ridgeNoise = FbmNoise3(ridgePos, 3, 2.06f, 0.50f);
    float rangeNoise = RidgedFbmNoise3(rangePos, 3, 2.00f, 0.52f);
    float peakNoise = RidgedFbmNoise3(peakPos, 4, 2.16f, 0.50f);
    float cragNoise = RidgedFbmNoise3(cragPos, 2, 2.32f, 0.46f);
    float moisture = FbmNoise3(moistPos, 4, 2.12f, 0.56f);

    float continentShape = SmoothStep01((super - 0.420f) / 0.355f);
    float continental = continentShape * 0.52f + macro * 0.22f + detail * 0.030f + biasMix * 0.35f - oceanMix * 0.12f;
    float landFactor = SmoothStep01((continental - 0.445f) / 0.300f);
    landFactor = ClampFloat(landFactor + continentCollision * 0.055f + otherPlateSubducts * 0.035f - divergent * 0.040f, 0.0f, 1.0f);
    float ridge = 1.0f - fabsf(ridgeNoise * 2.0f - 1.0f);

    float shelf = SmoothStep01((landFactor - 0.10f) / 0.28f);
    float upland = SmoothStep01((landFactor - 0.46f) / 0.38f);
    float landRelief = SmoothStep01((landFactor - 0.22f) / 0.30f);
    float ridgeSharp = powf(ClampFloat(ridge, 0.0f, 1.0f), 2.20f);
    float rangeCore = SmoothStep01((rangeNoise - 0.22f) / 0.48f);
    float peakMask = ClampFloat(rangeCore * (0.45f + peakNoise * 0.95f) * (0.55f + cragNoise * 0.70f), 0.0f, 1.0f);
    float boundaryBelt = SmoothStep01((boundary - 0.20f) / 0.55f);
    float orogeny = ClampFloat(continentCollision * 1.45f + otherPlateSubducts * 1.18f + transform * 0.30f, 0.0f, 1.0f);
    float oceanFloor = -0.066f + (macro - 0.5f) * 0.009f - oceanMix * 0.007f - ageMix * oceanMix * 0.008f;
    float continentalRise = -0.002f + landFactor * 0.024f + biasMix * 0.010f + continentShape * 0.014f;
    float baseElevation = LerpFloat(oceanFloor, continentalRise, shelf);
    float broadRelief = shelf * ((macro - 0.5f) * 0.022f + (super - 0.5f) * 0.014f);
    float hillVariation = landRelief * ((detail - 0.5f) * 0.020f + (ridge - 0.48f) * 0.012f);
    float mountainMask = ClampFloat(peakMask * (0.48f + upland * 0.70f) * (0.65f + boundaryBelt * 0.70f), 0.0f, 1.0f);
    float cragLift = cragNoise * mountainMask * (orogeny * boundaryBelt + upland * 0.30f) * 0.016f;
    float continentMountains = orogeny * boundaryBelt * (0.028f + mountainMask * 0.130f);
    float ancientHighlands = upland * rangeCore * peakMask * SmoothStep01((landFactor - 0.56f) / 0.30f) * 0.034f;
    float volcanicArc = otherPlateSubducts * boundaryBelt * (0.016f + mountainMask * 0.080f);
    float islandArc = oceanCollision * boundaryBelt * ridgeSharp * 0.026f;
    float midOceanRidge = divergent * oceanMix * (0.006f + ridgeSharp * 0.010f);
    float continentalRift = divergent * continental0 * 0.016f;
    float transformLift = transform * boundaryBelt * mountainMask * 0.015f;
    float trenchDrop = thisPlateSubducts * (0.014f + oceanMix * 0.018f);
    float coastalShelf = (1.0f - shelf) * (macro - 0.5f) * 0.008f;
    float elevation = ClampFloat(
        baseElevation + broadRelief + hillVariation + continentMountains + ancientHighlands + volcanicArc + islandArc + midOceanRidge + transformLift + cragLift + coastalShelf
        - continentalRift - trenchDrop,
        -0.094f,
        0.248f
    );

    float ruggedness = ClampFloat((ridgeSharp * 0.45f + rangeCore * 0.38f + cragNoise * 0.34f) * landFactor + orogeny * 0.46f + subduction * 0.18f + transform * 0.10f + divergent * 0.05f, 0.0f, 1.0f);
    float latAbs = fabsf(unitDirection.y);
    float landSnow = SmoothStep01((elevation - 0.055f) / 0.065f);
    float peakSnow = SmoothStep01((elevation - 0.105f) / 0.075f) * SmoothStep01((ruggedness - 0.35f) / 0.40f);
    float polarSnow = SmoothStep01((latAbs - 0.68f) / 0.24f) * SmoothStep01((elevation - 0.010f) / 0.085f);
    float snowcap = ClampFloat(landSnow * (peakSnow + polarSnow * 0.70f), 0.0f, 1.0f);
    float shade = ClampFloat((peakNoise - 0.42f) * 0.48f + (cragNoise - 0.36f) * 0.70f + (detail - 0.5f) * 0.22f, -1.0f, 1.0f);
    return (TerrainSample){ elevation, moisture, ruggedness, snowcap, shade, p0, boundary, convergenceRaw * boundary };
}

static Color TerrainColor(const TerrainSample *sample, int cornerCount)
{
    float e = sample->elevation;
    float m = sample->moisture;
    float r = sample->ruggedness;
    float s = sample->snowcap;
    float shade = sample->shade;

    Color deepOcean = (Color){ 8, 35, 94, 255 };
    Color shallowOcean = (Color){ 21, 93, 154, 255 };
    Color beach = (Color){ 198, 186, 140, 255 };
    Color dryLowland = (Color){ 125, 133, 84, 255 };
    Color wetLowland = (Color){ 74, 144, 86, 255 };
    Color dryHighland = (Color){ 146, 128, 92, 255 };
    Color wetHighland = (Color){ 110, 132, 92, 255 };
    Color mountainBrown = (Color){ 132, 104, 76, 255 };
    Color mountainGray = (Color){ 120, 120, 122, 255 };
    Color snowShadow = (Color){ 198, 207, 214, 255 };
    Color snowLight = (Color){ 248, 250, 252, 255 };

    Color lowland = LerpColor(dryLowland, wetLowland, SmoothStep01((m - 0.22f) / 0.66f));
    Color highland = LerpColor(dryHighland, wetHighland, SmoothStep01((m - 0.28f) / 0.54f));

    // Continuous blend chain: smooth gradients with no hard biome edges.
    Color color = deepOcean;
    color = LerpColor(color, shallowOcean, SmoothStep01((e + 0.14f) / 0.11f));
    color = LerpColor(color, beach, SmoothStep01((e + 0.03f) / 0.10f));
    color = LerpColor(color, lowland, SmoothStep01((e - 0.00f) / 0.09f));
    color = LerpColor(color, highland, SmoothStep01((e - 0.07f) / 0.10f));
    float mountainAmount = SmoothStep01((e - 0.070f) / 0.095f) * SmoothStep01((r - 0.22f) / 0.50f);
    float grayAmount = SmoothStep01((e - 0.130f) / 0.085f) * SmoothStep01((r - 0.42f) / 0.38f);
    color = LerpColor(color, mountainBrown, mountainAmount);
    color = LerpColor(color, mountainGray, grayAmount);
    float reliefShade = ClampFloat(SmoothStep01((e - 0.025f) / 0.145f) * (r * 0.72f + mountainAmount * 0.48f), 0.0f, 1.0f);
    color = LerpColor(color, (Color){ 54, 49, 45, 255 }, ClampFloat(-shade * reliefShade * 0.34f, 0.0f, 0.34f));
    color = LerpColor(color, (Color){ 222, 216, 198, 255 }, ClampFloat(shade * reliefShade * 0.24f, 0.0f, 0.24f));
    color = LerpColor(color, LerpColor(snowShadow, snowLight, ClampFloat(0.55f + shade * 0.35f, 0.0f, 1.0f)), s);

    if (cornerCount == 5) color = LerpColor(color, (Color){ 170, 160, 128, 255 }, 0.08f);
    return color;
}

static float FractFloat(float value)
{
    return value - floorf(value);
}

static float HashFromDirection(Vector3 direction, float seed)
{
    float n = direction.x * 127.1f + direction.y * 311.7f + direction.z * 74.7f + seed;
    return FractFloat(sinf(n) * 43758.5453f);
}

static Vector3 WarpCornerPoint(Vector3 unitDirection, float radius)
{
    Vector3 tangentA = Vector3CrossProduct(unitDirection, (Vector3){ 0.0f, 1.0f, 0.0f });
    if (Vector3LengthSqr(tangentA) < 0.000001f) tangentA = Vector3CrossProduct(unitDirection, (Vector3){ 1.0f, 0.0f, 0.0f });
    tangentA = Vector3Normalize(tangentA);
    Vector3 tangentB = Vector3Normalize(Vector3CrossProduct(tangentA, unitDirection));

    float jitterA = (HashFromDirection(unitDirection, 19.0f) * 2.0f - 1.0f) * POLY_BLEND_TANGENT;
    float jitterB = (HashFromDirection(unitDirection, 57.0f) * 2.0f - 1.0f) * POLY_BLEND_TANGENT;
    float radialJitter = (HashFromDirection(unitDirection, 93.0f) * 2.0f - 1.0f) * POLY_BLEND_RADIAL;

    Vector3 tangentOffset = Vector3Add(Vector3Scale(tangentA, jitterA), Vector3Scale(tangentB, jitterB));
    Vector3 warpedDirection = Vector3Normalize(Vector3Add(unitDirection, tangentOffset));
    return Vector3Scale(warpedDirection, radius * (1.0f + radialJitter));
}

static void SortCornersByAngle(Vector3 *corners, int *cornerIds, float *angles, int count)
{
    for (int i = 1; i < count; i++) {
        Vector3 corner = corners[i];
        int cornerId = cornerIds[i];
        float angle = angles[i];
        int j = i - 1;
        while (j >= 0 && angles[j] > angle) {
            angles[j + 1] = angles[j];
            corners[j + 1] = corners[j];
            cornerIds[j + 1] = cornerIds[j];
            j--;
        }
        angles[j + 1] = angle;
        corners[j + 1] = corner;
        cornerIds[j + 1] = cornerId;
    }
}

static Vector3 *BuildTriangleDirections(const VertexBuffer *vertices, const TriangleBuffer *triangles)
{
    Vector3 *triangleDirections = (Vector3 *)malloc(sizeof(Vector3) * (size_t)triangles->count);
    for (int i = 0; i < triangles->count; i++) {
        Triangle tri = triangles->items[i];
        Vector3 sum = Vector3Add(vertices->items[tri.a], vertices->items[tri.b]);
        sum = Vector3Add(sum, vertices->items[tri.c]);
        triangleDirections[i] = Vector3Normalize(sum);
    }
    return triangleDirections;
}

static void UpdatePlanetTiles(
    Tile *tiles,
    int tileCount,
    const Vector3 *triangleDirections,
    Vector3 *triangleSurfacePoints,
    int triangleCount,
    const Plate *plates,
    int plateCount,
    float radius
)
{
    for (int i = 0; i < triangleCount; i++) {
        TerrainSample cornerTerrain = EvaluateTerrain(triangleDirections[i], plates, plateCount);
        float cornerRadius = radius * (1.0f + DisplayElevation(cornerTerrain.elevation));
        triangleSurfacePoints[i] = WarpCornerPoint(triangleDirections[i], cornerRadius);
    }

    for (int v = 0; v < tileCount; v++) {
        Tile *tile = &tiles[v];
        TerrainSample centerTerrain = EvaluateTerrain(tile->baseCenterDir, plates, plateCount);
        float centerRadius = radius * (1.0f + DisplayElevation(centerTerrain.elevation));

        tile->center = Vector3Scale(tile->baseCenterDir, centerRadius);
        tile->plateId = centerTerrain.plateId;
        tile->elevation = centerTerrain.elevation;
        tile->terrainFill = TerrainColor(&centerTerrain, tile->cornerCount);
        tile->plateFill = plates[centerTerrain.plateId].color;
        if (centerTerrain.boundary > 0.48f) {
            tile->plateFill = LerpColor(tile->plateFill, (Color){ 232, 232, 232, 255 }, 0.18f);
        }
        for (int i = 0; i < tile->cornerCount; i++) {
            tile->corners[i] = triangleSurfacePoints[tile->cornerTriangleIds[i]];
        }
    }
}

static Tile *BuildPlanetTiles(
    const VertexBuffer *vertices,
    const TriangleBuffer *triangles,
    const Vector3 *triangleDirections,
    Vector3 *triangleSurfacePoints,
    const Plate *plates,
    int plateCount,
    float radius,
    int *outCount
)
{
    int vertexCount = vertices->count;
    int triangleCount = triangles->count;

    int *counts = (int *)calloc((size_t)vertexCount, sizeof(int));
    for (int i = 0; i < triangleCount; i++) {
        Triangle tri = triangles->items[i];
        counts[tri.a] += 1;
        counts[tri.b] += 1;
        counts[tri.c] += 1;
    }

    int *offsets = (int *)malloc(sizeof(int) * (size_t)(vertexCount + 1));
    offsets[0] = 0;
    for (int i = 0; i < vertexCount; i++) offsets[i + 1] = offsets[i] + counts[i];

    int adjacencyCount = offsets[vertexCount];
    int *adjacentTriangles = (int *)malloc(sizeof(int) * (size_t)adjacencyCount);
    int *cursor = (int *)malloc(sizeof(int) * (size_t)vertexCount);
    memcpy(cursor, offsets, sizeof(int) * (size_t)vertexCount);

    for (int i = 0; i < triangleCount; i++) {
        Triangle tri = triangles->items[i];
        adjacentTriangles[cursor[tri.a]++] = i;
        adjacentTriangles[cursor[tri.b]++] = i;
        adjacentTriangles[cursor[tri.c]++] = i;
    }

    Tile *tiles = (Tile *)malloc(sizeof(Tile) * (size_t)vertexCount);
    for (int v = 0; v < vertexCount; v++) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int degree = end - start;
        if (degree > MAX_TILE_CORNERS) degree = MAX_TILE_CORNERS;

        Vector3 normal = Vector3Normalize(vertices->items[v]);
        Vector3 reference = Vector3CrossProduct(normal, (Vector3){ 0.0f, 1.0f, 0.0f });
        if (Vector3LengthSqr(reference) < 0.000001f) {
            reference = Vector3CrossProduct(normal, (Vector3){ 1.0f, 0.0f, 0.0f });
        }
        reference = Vector3Normalize(reference);
        Vector3 reference2 = Vector3Normalize(Vector3CrossProduct(normal, reference));

        Vector3 corners[MAX_TILE_CORNERS] = { 0 };
        int cornerIds[MAX_TILE_CORNERS] = { 0 };
        float angles[MAX_TILE_CORNERS] = { 0 };
        for (int i = 0; i < degree; i++) {
            int triangleId = adjacentTriangles[start + i];
            Vector3 corner = triangleDirections[triangleId];
            Vector3 tangent = Vector3Subtract(corner, Vector3Scale(normal, Vector3DotProduct(corner, normal)));
            corners[i] = corner;
            cornerIds[i] = triangleId;
            angles[i] = atan2f(Vector3DotProduct(tangent, reference2), Vector3DotProduct(tangent, reference));
        }

        SortCornersByAngle(corners, cornerIds, angles, degree);

        tiles[v].baseCenterDir = normal;
        tiles[v].cornerCount = degree;
        for (int i = 0; i < degree; i++) tiles[v].cornerTriangleIds[i] = cornerIds[i];
    }

    free(cursor);
    free(adjacentTriangles);
    free(offsets);
    free(counts);

    *outCount = vertexCount;
    UpdatePlanetTiles(tiles, vertexCount, triangleDirections, triangleSurfacePoints, triangleCount, plates, plateCount, radius);
    return tiles;
}

static Color Gradient3(float t, Color a, Color b, Color c)
{
    t = ClampFloat(t, 0.0f, 1.0f);
    if (t < 0.5f) return LerpColor(a, b, t * 2.0f);
    return LerpColor(b, c, (t - 0.5f) * 2.0f);
}

static const char *WeatherViewName(WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE: return "Temperature";
        case WEATHER_VIEW_PRESSURE: return "Pressure";
        case WEATHER_VIEW_WIND: return "Wind Speed";
        case WEATHER_VIEW_CURRENT: return "Ocean Currents";
        case WEATHER_VIEW_HUMIDITY: return "Humidity";
        case WEATHER_VIEW_CLOUD: return "Cloud Cover";
        case WEATHER_VIEW_RAIN: return "Precipitation";
        case WEATHER_VIEW_VORTICITY: return "Vorticity";
        case WEATHER_VIEW_STORM: return "Storm Lift";
        case WEATHER_VIEW_EVAPORATION: return "Evaporation";
        case WEATHER_VIEW_SNOW: return "Snow/Ice";
        default: return "Unknown";
    }
}

static Color GetWeatherViewColor(const WeatherCell *w, WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE: {
            return Gradient3(
                w->temperature,
                (Color){ 60, 108, 172, 255 },
                (Color){ 89, 158, 104, 255 },
                (Color){ 209, 156, 86, 255 }
            );
        }
        case WEATHER_VIEW_PRESSURE: {
            float anomaly = ClampFloat((w->pressure - 0.46f) / 0.22f, 0.0f, 1.0f);
            return Gradient3(
                anomaly,
                (Color){ 60, 84, 176, 255 },
                (Color){ 57, 151, 143, 255 },
                (Color){ 225, 157, 76, 255 }
            );
        }
        case WEATHER_VIEW_WIND: {
            float speed = ClampFloat(Vector3Length(w->wind) / 0.22f, 0.0f, 1.0f);
            return Gradient3(
                speed,
                (Color){ 31, 50, 108, 255 },
                (Color){ 78, 177, 211, 255 },
                (Color){ 232, 246, 252, 255 }
            );
        }
        case WEATHER_VIEW_CURRENT: {
            float speed = ClampFloat(SmoothStep01(Vector3Length(w->current) / 0.15f), 0.0f, 1.0f);
            Color oceanCalm = (Color){ 22, 42, 86, 255 };
            Color oceanFlow = (Color){ 44, 154, 214, 255 };
            Color oceanFast = (Color){ 245, 236, 149, 255 };
            return Gradient3(speed, oceanCalm, oceanFlow, oceanFast);
        }
        case WEATHER_VIEW_HUMIDITY: {
            float rh = ClampFloat((WeatherRelativeHumidity(w->humidity, w->temperature) - 0.24f) / 0.86f, 0.0f, 1.0f);
            return Gradient3(
                SmoothStep01(rh),
                (Color){ 166, 125, 65, 255 },
                (Color){ 73, 153, 98, 255 },
                (Color){ 54, 122, 190, 255 }
            );
        }
        case WEATHER_VIEW_CLOUD: {
            float cover = SmoothStep01((fmaxf(w->cloud, w->cloudWater * 1.55f) - 0.035f) / 0.48f);
            return Gradient3(
                cover,
                (Color){ 36, 53, 82, 255 },
                (Color){ 103, 127, 151, 255 },
                (Color){ 224, 231, 238, 255 }
            );
        }
        case WEATHER_VIEW_RAIN: {
            float rain = SmoothStep01(fmaxf(w->precipitation, w->recentRain * 0.55f) / 0.22f);
            return Gradient3(
                rain,
                (Color){ 34, 47, 63, 255 },
                (Color){ 42, 133, 204, 255 },
                (Color){ 226, 112, 104, 255 }
            );
        }
        case WEATHER_VIEW_VORTICITY: {
            float t = ClampFloat((w->vorticity + 1.0f) * 0.5f, 0.0f, 1.0f);
            return Gradient3(
                t,
                (Color){ 58, 128, 218, 255 },
                (Color){ 42, 48, 60, 255 },
                (Color){ 222, 96, 76, 255 }
            );
        }
        case WEATHER_VIEW_STORM: {
            float storm = SmoothStep01((fmaxf(w->storm, w->orographicLift * 0.65f) - 0.04f) / 0.55f);
            return Gradient3(
                storm,
                (Color){ 36, 42, 56, 255 },
                (Color){ 102, 90, 168, 255 },
                (Color){ 244, 185, 72, 255 }
            );
        }
        case WEATHER_VIEW_EVAPORATION: {
            float evaporation = SmoothStep01((w->evaporation + w->soilMoisture * 0.12f) / 0.42f);
            return Gradient3(
                evaporation,
                (Color){ 38, 65, 78, 255 },
                (Color){ 66, 157, 139, 255 },
                (Color){ 240, 210, 118, 255 }
            );
        }
        case WEATHER_VIEW_SNOW: {
            return Gradient3(
                w->snow,
                (Color){ 43, 66, 92, 255 },
                (Color){ 146, 181, 204, 255 },
                (Color){ 248, 250, 252, 255 }
            );
        }
        default:
            return (Color){ 255, 0, 255, 255 };
    }
}

static float WeatherViewOverlayStrength(WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_CLOUD:
        case WEATHER_VIEW_RAIN:
        case WEATHER_VIEW_SNOW:
            return 0.52f;
        case WEATHER_VIEW_TEMPERATURE:
        case WEATHER_VIEW_HUMIDITY:
            return 0.48f;
        default:
            return 0.62f;
    }
}

static void DrawPlanetTiles(
    const Tile *tiles,
    const WeatherCell *weather,
    int tileCount,
    bool showPlateView,
    bool weatherEnabled,
    WeatherViewMode weatherView,
    int selectedTile
)
{
    Color wire = (Color){ 16, 24, 20, 255 };
    rlDisableBackfaceCulling();
    for (int i = 0; i < tileCount; i++) {
        const Tile *tile = &tiles[i];
        Color fill = tile->terrainFill;
        if (showPlateView) fill = tile->plateFill;
        else if (weatherEnabled && weather != NULL) {
            Color weatherFill = GetWeatherViewColor(&weather[i], weatherView);
            fill = LerpColor(fill, weatherFill, WeatherViewOverlayStrength(weatherView));
        }
        bool isSelected = (i == selectedTile);
        if (isSelected) fill = (Color){ 246, 248, 252, 255 };
        for (int j = 0; j < tile->cornerCount; j++) {
            Vector3 a = tile->corners[j];
            Vector3 b = tile->corners[(j + 1) % tile->cornerCount];

            Vector3 edgeA = Vector3Subtract(a, tile->center);
            Vector3 edgeB = Vector3Subtract(b, tile->center);
            float winding = Vector3DotProduct(Vector3CrossProduct(edgeA, edgeB), tile->center);
            Vector3 faceNormal = Vector3CrossProduct(edgeA, edgeB);
            if (winding < 0.0f) faceNormal = Vector3Scale(faceNormal, -1.0f);
            if (Vector3LengthSqr(faceNormal) > 0.000001f) faceNormal = Vector3Normalize(faceNormal);
            else faceNormal = Vector3Normalize(tile->center);
            Color triFill = isSelected ? fill : ShadeSurfaceColor(fill, faceNormal, showPlateView ? 0.16f : 0.44f);
            if (winding >= 0.0f) DrawTriangle3D(tile->center, a, b, triFill);
            else DrawTriangle3D(tile->center, b, a, triFill);

            if (DRAW_WIRES || isSelected) {
                Color lineColor = isSelected ? (Color){ 255, 255, 255, 255 } : wire;
                DrawLine3D(a, b, lineColor);
            }
        }
    }
    rlEnableBackfaceCulling();
}

static int PickTileFromMouse(const Tile *tiles, int tileCount, Camera3D camera)
{
    Ray ray = GetMouseRay(GetMousePosition(), camera);
    int bestTile = -1;
    float bestDistance = FLT_MAX;

    for (int i = 0; i < tileCount; i++) {
        const Tile *tile = &tiles[i];
        for (int j = 0; j < tile->cornerCount; j++) {
            Vector3 a = tile->corners[j];
            Vector3 b = tile->corners[(j + 1) % tile->cornerCount];
            RayCollision hit = GetRayCollisionTriangle(ray, tile->center, a, b);
            if (!hit.hit) hit = GetRayCollisionTriangle(ray, tile->center, b, a);
            if (hit.hit && hit.distance < bestDistance) {
                bestDistance = hit.distance;
                bestTile = i;
            }
        }
    }

    return bestTile;
}

static void DrawSelectedTileInfo(
    const Tile *tiles,
    const Plate *plates,
    const WeatherCell *weather,
    int tileCount,
    int selectedTile,
    bool tectonicsPaused,
    bool weatherEnabled,
    WeatherViewMode weatherView
)
{
    if (selectedTile < 0 || selectedTile >= tileCount) return;

    const Tile *tile = &tiles[selectedTile];
    const Plate *plate = &plates[tile->plateId];
    const WeatherCell *w = &weather[selectedTile];
    Vector3 normal = tile->baseCenterDir;
    float elevationM = TerrainElevationMeters(tile->elevation);
    float latitude = asinf(normal.y) * RAD2DEG;
    float longitude = atan2f(normal.z, normal.x) * RAD2DEG;
    float windSpeed = WeatherWindMetersPerSecond(w->wind);
    float currentSpeed = WeatherCurrentMetersPerSecond(w->current);
    float temperatureC = WeatherTemperatureC(w->temperature);
    float pressureHpa = WeatherPressureHpa(w->pressure);
    float humidityPct = WeatherRelativeHumidity(w->humidity, w->temperature) * 100.0f;
    float cloudPct = ClampFloat(fmaxf(w->cloud, w->cloudWater), 0.0f, 1.0f) * 100.0f;
    float precipitationMmH = w->precipitation * 25.0f;
    float evaporationMmDay = w->evaporation * 12.0f;
    float snowPct = ClampFloat(w->snow, 0.0f, 1.0f) * 100.0f;
    float soilPct = ClampFloat(w->soilMoisture, 0.0f, 1.0f) * 100.0f;
    float rainShadowPct = ClampFloat(w->rainShadow, 0.0f, 1.0f) * 100.0f;
    float liftPct = ClampFloat(w->orographicLift, 0.0f, 1.0f) * 100.0f;
    float vorticityE5 = w->vorticity * 8.0f;

    DrawRectangle(14, 14, 460, 292, (Color){ 6, 10, 16, 198 });
    DrawRectangleLines(14, 14, 460, 292, (Color){ 175, 189, 209, 180 });

    int x = 26;
    int y = 24;
    int lh = 20;
    char line[160];

    snprintf(line, sizeof(line), "Tile %d  Plate %d  %s", selectedTile, tile->plateId + 1, plate->major ? "Major" : "Minor");
    DrawText(line, x, y, 18, (Color){ 244, 248, 255, 255 }); y += lh;
    snprintf(line, sizeof(line), "Lat %.1f  Lon %.1f", latitude, longitude);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Elevation %.0f m", elevationM);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Oceanic %.2f  Density %.2f  Age %.2f", plate->oceanic, plate->density, plate->crustAge);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Temperature %.1f C", temperatureC);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Pressure %.0f hPa", pressureHpa);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Humidity %.0f%%  Cloud %.0f%%", humidityPct, cloudPct);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Rain %.1fmm/h  Wind %.1fm/s  Cur %.2fm/s", precipitationMmH, windSpeed, currentSpeed);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Vort %.1f 1e-5/s", vorticityE5);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Storm %.0f%%  Evap %.1fmm/day  Snow %.0f%%", w->storm * 100.0f, evaporationMmDay, snowPct);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Soil %.0f%%  Lift %.0f%%  Shadow %.0f%%", soilPct, liftPct, rainShadowPct);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Tectonics %s  Weather %s", tectonicsPaused ? "Paused" : "Running", weatherEnabled ? "On" : "Off");
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Weather View: %s", WeatherViewName(weatherView));
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 });
}

static void DrawViewThemeInfo(
    bool showPlateView,
    bool weatherEnabled,
    WeatherViewMode weatherView,
    bool tectonicsPaused
)
{
    const char *viewLine = showPlateView ? "View: Plates" : (weatherEnabled ? "View: Weather" : "View: Terrain");
    char modeLine[96];
    snprintf(modeLine, sizeof(modeLine), "Weather Mode: %s", WeatherViewName(weatherView));
    char stateLine[96];
    snprintf(stateLine, sizeof(stateLine), "Weather: %s  Tectonics: %s", weatherEnabled ? "On" : "Off", tectonicsPaused ? "Paused" : "Running");

    int font = 18;
    int pad = 12;
    int gap = 6;
    int w1 = MeasureText(viewLine, font);
    int w2 = MeasureText(modeLine, font);
    int w3 = MeasureText(stateLine, font);
    int width = w1;
    if (w2 > width) width = w2;
    if (w3 > width) width = w3;
    width += pad * 2;
    int height = pad * 2 + font * 3 + gap * 2;

    int x = GetScreenWidth() - width - 14;
    int y = 14;
    DrawRectangle(x, y, width, height, (Color){ 6, 10, 16, 198 });
    DrawRectangleLines(x, y, width, height, (Color){ 175, 189, 209, 180 });
    DrawText(viewLine, x + pad, y + pad, font, (Color){ 244, 248, 255, 255 });
    DrawText(modeLine, x + pad, y + pad + font + gap, font, (Color){ 210, 220, 236, 255 });
    DrawText(stateLine, x + pad, y + pad + (font + gap) * 2, font, (Color){ 210, 220, 236, 255 });
}

static void DrawWeatherClouds(
    const Tile *tiles,
    const WeatherCell *weather,
    int tileCount,
    WeatherViewMode mode
)
{
    if (mode != WEATHER_VIEW_CLOUD && mode != WEATHER_VIEW_RAIN) return;
    rlDisableBackfaceCulling();
    for (int i = 0; i < tileCount; i++) {
        const Tile *tile = &tiles[i];
        float cloud = weather[i].cloud;
        float storm = weather[i].precipitation;
        float alphaBlend = cloud * 0.78f + storm * 0.50f - 0.18f;
        if (alphaBlend <= 0.0f) continue;

        unsigned char alpha = (unsigned char)(ClampFloat(alphaBlend, 0.0f, 1.0f) * 170.0f);
        Color cloudColor = LerpColor((Color){ 246, 248, 252, 255 }, (Color){ 192, 201, 212, 255 }, ClampFloat(storm * 0.9f, 0.0f, 1.0f));
        cloudColor.a = alpha;

        float lift = CLOUD_LAYER_HEIGHT + cloud * 0.008f;
        Vector3 cDir = Vector3Normalize(tile->center);
        Vector3 c = Vector3Scale(cDir, Vector3Length(tile->center) + lift);

        for (int j = 0; j < tile->cornerCount; j++) {
            Vector3 a = tile->corners[j];
            Vector3 b = tile->corners[(j + 1) % tile->cornerCount];
            Vector3 aCloud = Vector3Scale(Vector3Normalize(a), Vector3Length(a) + lift);
            Vector3 bCloud = Vector3Scale(Vector3Normalize(b), Vector3Length(b) + lift);

            Vector3 edgeA = Vector3Subtract(aCloud, c);
            Vector3 edgeB = Vector3Subtract(bCloud, c);
            float winding = Vector3DotProduct(Vector3CrossProduct(edgeA, edgeB), c);
            if (winding >= 0.0f) DrawTriangle3D(c, aCloud, bCloud, cloudColor);
            else DrawTriangle3D(c, bCloud, aCloud, cloudColor);
        }
    }
    rlEnableBackfaceCulling();
}

static void DrawRibbonSegment3D(Vector3 a, Vector3 b, float width, Color color)
{
    Vector3 tangent = Vector3Subtract(b, a);
    if (Vector3LengthSqr(tangent) < 0.000001f) return;
    tangent = Vector3Normalize(tangent);

    Vector3 normal = Vector3Normalize(Vector3Add(a, b));
    Vector3 side = Vector3CrossProduct(tangent, normal);
    if (Vector3LengthSqr(side) < 0.000001f) side = TangentEast(normal);
    side = Vector3Scale(Vector3Normalize(side), width * 0.5f);

    Vector3 a0 = Vector3Subtract(a, side);
    Vector3 a1 = Vector3Add(a, side);
    Vector3 b0 = Vector3Subtract(b, side);
    Vector3 b1 = Vector3Add(b, side);

    DrawTriangle3D(a0, b0, b1, color);
    DrawTriangle3D(a0, b1, a1, color);
    DrawTriangle3D(a0, b1, b0, color);
    DrawTriangle3D(a0, a1, b1, color);
}

static Vector3 TangentComponent(Vector3 vector, Vector3 normal)
{
    return Vector3Subtract(vector, Vector3Scale(normal, Vector3DotProduct(vector, normal)));
}

static Vector3 FlowArrowSurfacePoint(Vector3 normal, Vector3 tangentOffset, float radius)
{
    Vector3 direction = Vector3Normalize(Vector3Add(normal, Vector3Scale(tangentOffset, 1.0f / radius)));
    return Vector3Scale(direction, radius);
}

static void DrawFlowArrowHead(
    Vector3 normal,
    Vector3 dir,
    Vector3 side,
    float radius,
    float tipOffset,
    float headLength,
    float headWidth,
    Color color
)
{
    Vector3 tip = FlowArrowSurfacePoint(normal, Vector3Scale(dir, tipOffset), radius);
    Vector3 base = Vector3Scale(dir, tipOffset - headLength);
    Vector3 left = FlowArrowSurfacePoint(normal, Vector3Add(base, Vector3Scale(side, headWidth * 0.5f)), radius);
    Vector3 right = FlowArrowSurfacePoint(normal, Vector3Subtract(base, Vector3Scale(side, headWidth * 0.5f)), radius);

    DrawTriangle3D(tip, left, right, color);
    DrawTriangle3D(tip, right, left, color);
}

static void DrawFlowArrow(
    const Tile *tile,
    const WeatherCell *cell,
    Vector3 vector,
    float speedScale,
    float minSpeed,
    float lift,
    float clock,
    Color slowColor,
    Color fastColor,
    float lengthBase,
    float widthBase,
    float seed
)
{
    (void)clock; (void)seed;
    Vector3 normal = cell->normal;
    Vector3 tangent = TangentComponent(vector, normal);
    float speed = Vector3Length(tangent);
    if (speed < minSpeed) return;

    Vector3 dir = Vector3Scale(tangent, 1.0f / speed);
    Vector3 side = Vector3CrossProduct(dir, normal);
    if (Vector3LengthSqr(side) < 0.000001f) side = TangentEast(normal);
    side = Vector3Normalize(side);

    float speed01 = ClampFloat(speed / speedScale, 0.0f, 1.0f);
    float radius = Vector3Length(tile->center) + lift;
    float length = lengthBase * (0.65f + speed01 * 0.70f);
    float shaftWidth = widthBase * (0.70f + speed01 * 0.35f);
    float headLength = length * 0.32f;
    float headWidth = shaftWidth * 3.0f;
    float tail = -length * 0.46f;
    float tip = length * 0.54f;
    float shaftEnd = tip - headLength * 0.55f;

    Color arrowColor = LerpColor(slowColor, fastColor, speed01);
    arrowColor.a = (unsigned char)(140.0f + speed01 * 80.0f);

    Vector3 a = FlowArrowSurfacePoint(normal, Vector3Scale(dir, tail), radius);
    Vector3 b = FlowArrowSurfacePoint(normal, Vector3Scale(dir, shaftEnd), radius);
    DrawRibbonSegment3D(a, b, shaftWidth, arrowColor);

    DrawFlowArrowHead(normal, dir, side, radius, tip, headLength, headWidth, arrowColor);
}

static void DrawWindVectors(const Tile *tiles, const WeatherCell *weather, int tileCount)
{
    float clock = (float)GetTime();
    for (int i = 0; i < tileCount; i += 13) {
        Vector3 normal = weather[i].normal;
        if (HashFromDirection(normal, 143.0f) < 0.17f) continue;
        DrawFlowArrow(
            &tiles[i],
            &weather[i],
            weather[i].wind,
            0.22f,
            0.010f,
            CLOUD_LAYER_HEIGHT + 0.020f,
            clock,
            (Color){ 198, 88, 72, 190 },
            (Color){ 255, 205, 122, 245 },
            0.185f,
            0.018f,
            211.0f
        );
    }
}

static void DrawCurrentVectors(const Tile *tiles, const WeatherCell *weather, int tileCount)
{
    float clock = (float)GetTime() * 0.72f;
    for (int i = 0; i < tileCount; i += 11) {
        Vector3 normal = weather[i].normal;
        float ocean = weather[i].ocean;
        if (ocean < 0.74f || HashFromDirection(normal, 319.0f) < 0.17f) continue;

        Color slow = (Color){ 54, 190, 224, (unsigned char)(148.0f + ocean * 58.0f) };
        Color fast = (Color){ 188, 247, 255, (unsigned char)(198.0f + ocean * 45.0f) };
        DrawFlowArrow(
            &tiles[i],
            &weather[i],
            weather[i].current,
            0.15f,
            0.006f,
            CLOUD_LAYER_HEIGHT + 0.014f,
            clock,
            slow,
            fast,
            0.155f + ocean * 0.040f,
            0.020f,
            421.0f
        );
    }
}

int main(void)
{
    srand((unsigned int)time(NULL));

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(1280, 800, "Planet");
    SetTargetFPS(60);

    VertexBuffer vertices = { 0 };
    TriangleBuffer triangles = { 0 };
    BuildIcosphere(&vertices, &triangles, SUBDIVISIONS, PLANET_RADIUS);
    Vector3 *triangleDirections = BuildTriangleDirections(&vertices, &triangles);
    Vector3 *triangleSurfacePoints = (Vector3 *)malloc(sizeof(Vector3) * (size_t)triangles.count);

    int graphCount = 0;
    TileGraphNode *tileGraph = BuildTileGraph(&vertices, &triangles, &graphCount);

    Plate plates[MAX_PLATES] = { 0 };
    int plateCount = GenerateRandomPlates(plates);

    int tileCount = 0;
    Tile *tiles = BuildPlanetTiles(&vertices, &triangles, triangleDirections, triangleSurfacePoints, plates, plateCount, PLANET_RADIUS, &tileCount);
    WeatherCell *weatherA = (WeatherCell *)calloc((size_t)tileCount, sizeof(WeatherCell));
    WeatherCell *weatherB = (WeatherCell *)calloc((size_t)tileCount, sizeof(WeatherCell));
    InitializeWeather(weatherA, tiles, tileCount);
    memcpy(weatherB, weatherA, sizeof(WeatherCell) * (size_t)tileCount);

    OrbitCamera orbit = {
        .yaw = 0.8f,
        .pitch = 0.3f,
        .yawTarget = 0.8f,
        .pitchTarget = 0.3f,
        .distance = 6.5f,
        .distanceTarget = 6.5f,
        .target = { 0.0f, 0.0f, 0.0f }
    };
    Camera3D camera = { 0 };
    bool showPlateView = false;
    bool tectonicsPaused = false;
    bool weatherEnabled = true;
    WeatherViewMode weatherView = WEATHER_VIEW_TEMPERATURE;
    int selectedTile = -1;
    Vector2 clickStart = { 0 };
    float tectonicRebuildTimer = 0.0f;
    const float tectonicRebuildStep = 1.0f / TECTONIC_REBUILD_HZ;
    float weatherTimer = 0.0f;
    const float weatherStep = 1.0f / WEATHER_UPDATE_HZ;

    while (!WindowShouldClose()) {
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) clickStart = GetMousePosition();
        if (IsKeyPressed(KEY_C)) showPlateView = !showPlateView;
        if (IsKeyPressed(KEY_SPACE)) tectonicsPaused = !tectonicsPaused;
        if (IsKeyPressed(KEY_W)) weatherEnabled = !weatherEnabled;
        if (IsKeyPressed(KEY_TAB)) weatherView = (WeatherViewMode)((weatherView + 1) % WEATHER_VIEW_COUNT);
        if (IsKeyPressed(KEY_ONE)) weatherView = WEATHER_VIEW_TEMPERATURE;
        if (IsKeyPressed(KEY_TWO)) weatherView = WEATHER_VIEW_PRESSURE;
        if (IsKeyPressed(KEY_THREE)) weatherView = WEATHER_VIEW_WIND;
        if (IsKeyPressed(KEY_FOUR)) weatherView = WEATHER_VIEW_CURRENT;
        if (IsKeyPressed(KEY_FIVE)) weatherView = WEATHER_VIEW_HUMIDITY;
        if (IsKeyPressed(KEY_SIX)) weatherView = WEATHER_VIEW_CLOUD;
        if (IsKeyPressed(KEY_SEVEN)) weatherView = WEATHER_VIEW_RAIN;
        if (IsKeyPressed(KEY_EIGHT)) weatherView = WEATHER_VIEW_VORTICITY;
        if (IsKeyPressed(KEY_NINE)) weatherView = WEATHER_VIEW_STORM;
        if (IsKeyPressed(KEY_E)) weatherView = WEATHER_VIEW_EVAPORATION;
        if (IsKeyPressed(KEY_ZERO)) weatherView = WEATHER_VIEW_SNOW;
        float dt = GetFrameTime();

        if (!tectonicsPaused) {
            AdvancePlateSimulation(plates, plateCount, dt * TECTONIC_TIME_SCALE);
            tectonicRebuildTimer += dt;
            if (tectonicRebuildTimer >= tectonicRebuildStep) {
                tectonicRebuildTimer -= tectonicRebuildStep;
                UpdatePlanetTiles(tiles, tileCount, triangleDirections, triangleSurfacePoints, triangles.count, plates, plateCount, PLANET_RADIUS);
            }
        }

        if (weatherEnabled) {
            weatherTimer += dt * WEATHER_TIME_SCALE;
            while (weatherTimer >= weatherStep) {
                RefreshWeatherSurface(weatherA, tiles, tileCount);
                StepWeatherSimulation(tileGraph, weatherA, weatherB, tileCount, weatherStep);
                WeatherCell *swap = weatherA;
                weatherA = weatherB;
                weatherB = swap;
                weatherTimer -= weatherStep;
            }
        }

        UpdateOrbitCamera(&orbit, &camera);
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            Vector2 end = GetMousePosition();
            float dx = end.x - clickStart.x;
            float dy = end.y - clickStart.y;
            if ((dx * dx + dy * dy) <= 25.0f) {
                selectedTile = PickTileFromMouse(tiles, tileCount, camera);
            }
        }

        BeginDrawing();
        ClearBackground((Color){ 8, 10, 16, 255 });

        BeginMode3D(camera);
        DrawPlanetTiles(tiles, weatherA, tileCount, showPlateView, weatherEnabled, weatherView, selectedTile);
        if (!showPlateView && weatherEnabled) DrawWeatherClouds(tiles, weatherA, tileCount, weatherView);
        if (!showPlateView && weatherEnabled && weatherView == WEATHER_VIEW_WIND) DrawWindVectors(tiles, weatherA, tileCount);
        if (!showPlateView && weatherEnabled && weatherView == WEATHER_VIEW_CURRENT) DrawCurrentVectors(tiles, weatherA, tileCount);
        EndMode3D();

        DrawSelectedTileInfo(tiles, plates, weatherA, tileCount, selectedTile, tectonicsPaused, weatherEnabled, weatherView);
        DrawViewThemeInfo(showPlateView, weatherEnabled, weatherView, tectonicsPaused);

        EndDrawing();
    }

    free(tiles);
    free(weatherB);
    free(weatherA);
    free(tileGraph);
    free(triangleSurfacePoints);
    free(triangleDirections);
    free(triangles.items);
    free(vertices.items);

    CloseWindow();
    return 0;
}
