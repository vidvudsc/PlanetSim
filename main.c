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
#define ATMOSPHERE_SURFACE_MARGIN 0.20f
#define ATMOSPHERE_DENSITY_FALLOFF 2.9f
#define ATMOSPHERE_SCATTERING_STRENGTH 24.0f
#define MAX_TILE_NEIGHBORS 8
#define FLOW_ARROW_SEGMENTS 5
#define WEATHER_MIN_TEMP_C -120.0f
#define WEATHER_MAX_TEMP_C 95.0f
#define WEATHER_PRESSURE_MIN_HPA 540.0f
#define WEATHER_PRESSURE_MAX_HPA 1065.0f
#define WEATHER_SEA_LEVEL_PRESSURE_HPA 1013.25f
#define WEATHER_TERRAIN_ALTITUDE_RANGE_M 7200.0f
#define WEATHER_TROPO_LAPSE_C_PER_KM 6.5f
#define WEATHER_ATMOSPHERE_SCALE_HEIGHT_M 8400.0f
#define CLIMATE_CHART_BINS 96
#define CLIMATE_CHART_SAMPLE_CAP 48.0f

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
    float oceanTemperature;
    float soilMoisture;
    float cloudWater;
    float recentRain;
    float rainShadow;
    float orographicLift;
    float pressureAnomaly;
    float frontStrength;
    float airMassAge;
    float climateTemperature;
    float climatePrecipitation;
    float climateSoilMoisture;
    float seasonalTemperatureMin;
    float seasonalTemperatureMax;
} WeatherCell;

typedef struct ClimateSettings {
    bool panelOpen;
    bool autoAdvanceTime;
    bool dayNightEnabled;
    bool seasonsEnabled;
    bool showSunOrbit;
    bool showTiltAxis;
    float dayPhase;
    float yearPhase;
    float daySpeed;
    float yearSpeed;
    float axialTiltDegrees;
    float solarIntensity;
    float orbitDistanceAu;
    float orbitEccentricity;
    float stellarLuminosity;
    float stellarTemperatureK;
    float greenhouseC;
    float orbitDistanceRangeMin;
    float orbitDistanceRangeMax;
    float orbitEccentricityRangeMin;
    float orbitEccentricityRangeMax;
    float stellarLuminosityRangeMin;
    float stellarLuminosityRangeMax;
    float stellarTemperatureRangeMin;
    float stellarTemperatureRangeMax;
    float greenhouseRangeMin;
    float greenhouseRangeMax;
    float temperatureContrastRangeMin;
    float temperatureContrastRangeMax;
    float weatherTimeScale;
    float temperatureContrast;
    float atmosphereDensityFalloff;
    float atmosphereScatteringScale;
    float panelScroll;
    float panelContentHeight;
} ClimateSettings;

typedef struct SolarState {
    Vector3 northPole;
    Vector3 orbitRight;
    Vector3 orbitForward;
    Vector3 lightDir;
    float declination;
    float orbitDistanceAu;
    float stellarFlux;
    float equilibriumTemperatureC;
    float habitableZoneInnerAu;
    float habitableZoneOuterAu;
    float stellarTemperatureK;
    Color starColor;
} SolarState;

typedef struct WeatherForcing {
    float latAbs;
    float climateLat;
    float planetaryWave;
    float moistureWave;
    float eddyWave;
    float synopticWave;
    float equatorHeat;
    float itcz;
    float subtropicalHigh;
    float stormTrack;
    float polarHigh;
    float westerly;
    float trade;
    float polar;
    Vector3 east;
    Vector3 north;
} WeatherForcing;

typedef struct WeatherTerrain {
    float elevationMeters;
    float altitudeKm;
    float oceanDepthKm;
    float land;
    float ocean;
    float coast;
    float slope;
    float roughness;
    float highland;
    float mountain;
    float valley;
    float exposure;
} WeatherTerrain;

typedef struct WeatherFlux {
    float temperature;
    float pressure;
    float humidity;
    float cloudWater;
    float soilMoisture;
    float oceanTemperature;
    Vector3 wind;
    Vector3 current;
} WeatherFlux;

typedef struct PanelLayout {
    Rectangle bounds;
    Rectangle clipRect;
    float cursorY;
    float contentX;
    float contentWidth;
    float scrollY;
} PanelLayout;

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
    WEATHER_VIEW_OCEAN_TEMP,
    WEATHER_VIEW_BIOME,
    WEATHER_VIEW_COUNT
} WeatherViewMode;

typedef struct ClimateChartHistory {
    float values[WEATHER_VIEW_COUNT][CLIMATE_CHART_BINS];
    float latest[WEATHER_VIEW_COUNT];
    float sampleCounts[CLIMATE_CHART_BINS];
    int lastBin;
    bool initialized;
} ClimateChartHistory;

static float SphericalNoise3(Vector3 unitDirection, float scale, Vector3 offset, int octaves, float lacunarity, float gain);
static float WarpedClimateLatitudeFromAxis(Vector3 unitDirection, Vector3 climateNorth, float equatorShift, float strength);
static Vector3 TangentEast(Vector3 normal);
static Vector3 TangentEastFromAxis(Vector3 normal, Vector3 climateNorth);
static Vector3 TangentNorthFromAxis(Vector3 normal, Vector3 climateNorth);
static Vector3 ClimateEddyFlowFromAxis(Vector3 normal, Vector3 climateNorth, float scale, Vector3 offset, float strength);
static float MaxSurfaceRadius(const Tile *tiles, int tileCount);

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

static float Wrap01(float value)
{
    value = fmodf(value, 1.0f);
    if (value < 0.0f) value += 1.0f;
    return value;
}

static Vector3 BuildClimateNorthPole(float axialTiltDegrees)
{
    float tilt = axialTiltDegrees * DEG2RAD;
    return Vector3Normalize((Vector3){ sinf(tilt), cosf(tilt), 0.0f });
}

static Color StarColorFromTemperature(float kelvin)
{
    kelvin = ClampFloat(kelvin, 1800.0f, 12000.0f) / 100.0f;
    float red = 255.0f;
    float green = 255.0f;
    float blue = 255.0f;

    if (kelvin <= 66.0f) {
        red = 255.0f;
        green = 99.4708025861f * logf(kelvin) - 161.1195681661f;
        blue = (kelvin <= 19.0f) ? 0.0f : 138.5177312231f * logf(kelvin - 10.0f) - 305.0447927307f;
    } else {
        red = 329.698727446f * powf(kelvin - 60.0f, -0.1332047592f);
        green = 288.1221695283f * powf(kelvin - 60.0f, -0.0755148492f);
        blue = 255.0f;
    }

    return (Color){
        (unsigned char)ClampFloat(red, 0.0f, 255.0f),
        (unsigned char)ClampFloat(green, 0.0f, 255.0f),
        (unsigned char)ClampFloat(blue, 0.0f, 255.0f),
        255
    };
}

static SolarState BuildSolarState(const ClimateSettings *climate)
{
    SolarState solar = { 0 };
    solar.northPole = BuildClimateNorthPole(climate->axialTiltDegrees);

    Vector3 orbitRight = Vector3CrossProduct((Vector3){ 0.0f, 0.0f, 1.0f }, solar.northPole);
    if (Vector3LengthSqr(orbitRight) < 0.000001f) orbitRight = Vector3CrossProduct((Vector3){ 1.0f, 0.0f, 0.0f }, solar.northPole);
    orbitRight = Vector3Normalize(orbitRight);
    Vector3 orbitForward = Vector3Normalize(Vector3CrossProduct(solar.northPole, orbitRight));
    solar.orbitRight = orbitRight;
    solar.orbitForward = orbitForward;

    float dayAngle = climate->dayPhase * 2.0f * PI;
    float yearAngle = climate->yearPhase * 2.0f * PI;
    solar.declination = climate->seasonsEnabled ? sinf(yearAngle) * climate->axialTiltDegrees * DEG2RAD : 0.0f;
    float eccentricity = ClampFloat(climate->orbitEccentricity, 0.0f, 0.85f);
    float semiMajorAxis = fmaxf(0.05f, climate->orbitDistanceAu);
    solar.orbitDistanceAu = semiMajorAxis * (1.0f - eccentricity * eccentricity) / fmaxf(0.08f, 1.0f + eccentricity * cosf(yearAngle));
    solar.orbitDistanceAu = fmaxf(0.03f, solar.orbitDistanceAu);
    float luminosity = fmaxf(0.01f, climate->stellarLuminosity * climate->solarIntensity);
    solar.stellarFlux = luminosity / (solar.orbitDistanceAu * solar.orbitDistanceAu);
    solar.equilibriumTemperatureC = 255.0f * powf(fmaxf(solar.stellarFlux, 0.0001f), 0.25f) - 273.15f + climate->greenhouseC;
    solar.habitableZoneInnerAu = sqrtf(luminosity / 1.10f);
    solar.habitableZoneOuterAu = sqrtf(luminosity / 0.53f);
    solar.stellarTemperatureK = climate->stellarTemperatureK;
    solar.starColor = StarColorFromTemperature(climate->stellarTemperatureK);

    Vector3 equatorialSun = Vector3Add(
        Vector3Scale(solar.orbitRight, cosf(dayAngle)),
        Vector3Scale(solar.orbitForward, sinf(dayAngle))
    );
    solar.lightDir = Vector3Normalize(Vector3Add(
        Vector3Scale(equatorialSun, cosf(solar.declination)),
        Vector3Scale(solar.northPole, sinf(solar.declination))
    ));
    return solar;
}

static float SolarFacingAmount(Vector3 normal, Vector3 lightDir)
{
    return ClampFloat(Vector3DotProduct(normal, lightDir), -1.0f, 1.0f);
}

static float ClimateEquatorShift(const SolarState *solar, const ClimateSettings *climate)
{
    if (!climate->seasonsEnabled || climate->axialTiltDegrees <= 0.0f) return 0.0f;
    return ClampFloat(sinf(solar->declination) * 0.42f, -0.34f, 0.34f);
}

static float ClimateInsolation(Vector3 normal, const SolarState *solar, const ClimateSettings *climate)
{
    float equatorShift = ClimateEquatorShift(solar, climate);
    float climateLat = WarpedClimateLatitudeFromAxis(normal, solar->northPole, equatorShift, 0.145f);
    float baseHeat = 1.0f - climateLat * climateLat;
    float signedLatitude = Vector3DotProduct(normal, solar->northPole);
    float seasonBias = climate->seasonsEnabled ? signedLatitude * sinf(solar->declination) * 0.34f : 0.0f;
    float sunDot = SolarFacingAmount(normal, solar->lightDir);
    float daylightBoost = climate->dayNightEnabled ? fmaxf(0.0f, sunDot) * 0.26f : 0.12f;
    float nightsideCooling = climate->dayNightEnabled ? fmaxf(0.0f, -sunDot) * 0.08f : 0.0f;
    float insolation = ClampFloat(baseHeat + seasonBias + daylightBoost - nightsideCooling, 0.0f, 1.0f);
    float orbitalHeatScale = powf(fmaxf(solar->stellarFlux, 0.0001f), 0.25f);
    return ClampFloat(insolation * orbitalHeatScale, 0.0f, 1.0f);
}

static float ClimateMeanSeaLevelTemperatureC(const SolarState *solar)
{
    return solar->equilibriumTemperatureC + 32.0f;
}

static float ClimateLiquidWaterFactor(const SolarState *solar)
{
    float meanC = ClimateMeanSeaLevelTemperatureC(solar);
    float thaw = SmoothStep01((meanC + 18.0f) / 30.0f);
    float boilLoss = SmoothStep01((meanC - 62.0f) / 36.0f);
    return ClampFloat(thaw * (1.0f - boilLoss), 0.0f, 1.0f);
}

static const char *ClimateOrbitStatus(const SolarState *solar)
{
    if (solar->orbitDistanceAu < solar->habitableZoneInnerAu) return "Interior/hot";
    if (solar->orbitDistanceAu > solar->habitableZoneOuterAu) return "Exterior/frozen";
    return "Habitable zone";
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

static Color ShadeSurfaceColor(Color color, Vector3 normal, float strength, const SolarState *solar)
{
    Vector3 lightDir = solar ? solar->lightDir : Vector3Normalize((Vector3){ -0.38f, 0.70f, 0.60f });
    float light = ClampFloat(Vector3DotProduct(normal, lightDir) * 0.5f + 0.5f, 0.0f, 1.0f);
    float amount = LerpFloat(1.0f - strength * 0.32f, 1.0f + strength * 0.24f, light);
    Color shaded = ScaleColorBrightness(color, amount);
    if (!solar) return shaded;
    float rTint = LerpFloat(0.82f, 1.18f, (float)solar->starColor.r / 255.0f);
    float gTint = LerpFloat(0.82f, 1.18f, (float)solar->starColor.g / 255.0f);
    float bTint = LerpFloat(0.82f, 1.18f, (float)solar->starColor.b / 255.0f);
    return (Color){
        (unsigned char)ClampFloat((float)shaded.r * rTint, 0.0f, 255.0f),
        (unsigned char)ClampFloat((float)shaded.g * gTint, 0.0f, 255.0f),
        (unsigned char)ClampFloat((float)shaded.b * bTint, 0.0f, 255.0f),
        shaded.a
    };
}

static float WeatherTemperatureC(float temperature)
{
    return LerpFloat(WEATHER_MIN_TEMP_C, WEATHER_MAX_TEMP_C, ClampFloat(temperature, 0.0f, 1.0f));
}

static float WeatherTemperature01FromC(float temperatureC)
{
    return ClampFloat((temperatureC - WEATHER_MIN_TEMP_C) / (WEATHER_MAX_TEMP_C - WEATHER_MIN_TEMP_C), 0.0f, 1.0f);
}

static float WeatherPressureHpa(float pressure)
{
    return LerpFloat(WEATHER_PRESSURE_MIN_HPA, WEATHER_PRESSURE_MAX_HPA, ClampFloat(pressure, 0.0f, 1.0f));
}

static float WeatherPressure01FromHpa(float pressureHpa)
{
    return ClampFloat((pressureHpa - WEATHER_PRESSURE_MIN_HPA) / (WEATHER_PRESSURE_MAX_HPA - WEATHER_PRESSURE_MIN_HPA), 0.0f, 1.0f);
}

static float WeatherSaturation(float temperature)
{
    float temperatureC = WeatherTemperatureC(temperature);
    float vaporPressure = 6.112f * expf((17.67f * temperatureC) / (temperatureC + 243.5f));
    return ClampFloat(vaporPressure / 42.0f, 0.08f, 1.35f);
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

static float WeatherElevationMeters(float normalizedElevation)
{
    return normalizedElevation * WEATHER_TERRAIN_ALTITUDE_RANGE_M;
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

static void UpdateOrbitCamera(OrbitCamera *orbit, Camera3D *camera, bool allowInput)
{
    float dt = GetFrameTime();
    if (dt > 0.1f) dt = 0.1f;
    Vector2 mouseDelta = GetMouseDelta();

    if (allowInput && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        orbit->yawTarget += mouseDelta.x * 0.0075f;
        orbit->pitchTarget -= mouseDelta.y * 0.0075f;
    }

    if (allowInput && (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT))) {
        orbit->yawTarget += mouseDelta.x * 0.0075f;
        orbit->pitchTarget -= mouseDelta.y * 0.0075f;
    }
    orbit->pitchTarget = ClampFloat(orbit->pitchTarget, -1.46f, 1.46f);

    float wheel = GetMouseWheelMove();
    if (allowInput && wheel != 0.0f) {
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
        float elevationMeters = TerrainElevationMeters(tiles[i].elevation);
        cells[i].normal = normal;
        cells[i].elevation = ClampFloat(elevationMeters / WEATHER_TERRAIN_ALTITUDE_RANGE_M, -0.75f, 1.35f);
        cells[i].ocean = SmoothStep01((-elevationMeters + 140.0f) / 720.0f);
    }
}

static float GaussianBand(float x, float center, float width)
{
    float d = (x - center) / width;
    return expf(-d * d);
}

static WeatherForcing BuildWeatherForcing(Vector3 normal, const ClimateSettings *climate, const SolarState *solar)
{
    WeatherForcing f = { 0 };
    f.latAbs = fabsf(Vector3DotProduct(normal, solar->northPole));
    float equatorShift = ClimateEquatorShift(solar, climate);
    f.climateLat = WarpedClimateLatitudeFromAxis(normal, solar->northPole, equatorShift, 0.145f);
    f.planetaryWave = SphericalNoise3(normal, 1.65f, (Vector3){ 23.7f, -18.2f, 9.4f }, 3, 1.92f, 0.54f);
    f.moistureWave = SphericalNoise3(normal, 4.35f, (Vector3){ -42.1f, 7.8f, 31.6f }, 3, 2.07f, 0.52f);
    f.eddyWave = SphericalNoise3(normal, 7.8f, (Vector3){ 15.2f, 51.4f, -22.7f }, 2, 2.16f, 0.48f);

    Vector3 driftingNormal = RotateAroundAxis(normal, solar->northPole, climate->dayPhase * -2.4f + climate->yearPhase * 0.9f);
    f.synopticWave = SphericalNoise3(driftingNormal, 2.35f, (Vector3){ 81.2f, -12.6f, 43.5f }, 4, 2.0f, 0.53f);
    f.equatorHeat = ClampFloat(ClimateInsolation(normal, solar, climate) + (f.planetaryWave - 0.5f) * 0.045f, 0.0f, 1.0f);

    f.westerly = GaussianBand(f.climateLat, 0.45f, 0.17f);
    f.trade = GaussianBand(f.climateLat, 0.20f, 0.14f);
    f.polar = GaussianBand(f.climateLat, 0.78f, 0.11f);
    f.itcz = GaussianBand(f.climateLat, 0.04f, 0.17f) * LerpFloat(0.84f, 1.12f, f.moistureWave);
    f.subtropicalHigh = GaussianBand(f.climateLat, 0.31f, 0.13f);
    f.stormTrack = GaussianBand(f.climateLat, 0.55f, 0.22f) * LerpFloat(0.86f, 1.08f, f.planetaryWave);
    f.polarHigh = GaussianBand(f.climateLat, 0.91f, 0.16f);
    f.east = TangentEastFromAxis(normal, solar->northPole);
    f.north = TangentNorthFromAxis(normal, solar->northPole);
    return f;
}

static Vector3 LimitTangentVector(Vector3 value, Vector3 normal, float maxLength)
{
    value = Vector3Subtract(value, Vector3Scale(normal, Vector3DotProduct(value, normal)));
    float len = Vector3Length(value);
    if (len > maxLength && len > 0.000001f) value = Vector3Scale(value, maxLength / len);
    return value;
}

static WeatherTerrain BuildWeatherTerrain(const WeatherCell *cell, float terrainSlope, float coastalContrast, float elevationAvg)
{
    WeatherTerrain t = { 0 };
    t.elevationMeters = WeatherElevationMeters(cell->elevation);
    t.altitudeKm = fmaxf(t.elevationMeters, 0.0f) * 0.001f;
    t.oceanDepthKm = fmaxf(-t.elevationMeters, 0.0f) * 0.001f;
    t.ocean = ClampFloat(cell->ocean, 0.0f, 1.0f);
    t.land = 1.0f - t.ocean;
    t.coast = ClampFloat(coastalContrast * (0.40f + 0.60f * (1.0f - fabsf(t.ocean - 0.5f) * 2.0f)), 0.0f, 1.0f);
    t.slope = ClampFloat(terrainSlope, 0.0f, 1.0f);
    t.highland = t.land * SmoothStep01((t.altitudeKm - 0.75f) / 2.25f);
    t.mountain = t.land * SmoothStep01((t.altitudeKm - 1.60f) / 3.60f);
    t.valley = t.land * SmoothStep01((elevationAvg - cell->elevation + 0.035f) / 0.20f) * SmoothStep01((0.62f - cell->elevation) / 0.80f);
    t.roughness = t.land * ClampFloat(0.20f + t.slope * 1.08f + t.highland * 0.20f + t.mountain * 0.30f + cell->snow * 0.08f, 0.0f, 1.0f);
    t.exposure = ClampFloat(t.slope * (0.58f + t.mountain * 0.46f) + t.coast * 0.22f + t.highland * 0.14f, 0.0f, 1.0f);
    return t;
}

static float TerrainPressureFactor(float altitudeKm)
{
    return expf(-(altitudeKm * 1000.0f) / WEATHER_ATMOSPHERE_SCALE_HEIGHT_M);
}

static void AddSymmetricScalarFlux(float *a, float *b, float transfer)
{
    *a -= transfer;
    *b += transfer;
}

static void AddSymmetricVectorFlux(Vector3 *a, Vector3 *b, Vector3 transfer)
{
    *a = Vector3Subtract(*a, transfer);
    *b = Vector3Add(*b, transfer);
}

static void AccumulateWeatherFluxes(const TileGraphNode *graph, const WeatherCell *src, WeatherFlux *fluxes, int count, float dt)
{
    for (int i = 0; i < count; i++) {
        const WeatherCell *a = &src[i];
        const TileGraphNode *node = &graph[i];

        for (int n = 0; n < node->neighborCount; n++) {
            int nb = node->neighbors[n];
            if (nb <= i) continue;

            const WeatherCell *b = &src[nb];
            Vector3 edge = Vector3Subtract(b->normal, Vector3Scale(a->normal, Vector3DotProduct(b->normal, a->normal)));
            float edgeLen = Vector3Length(edge);
            if (edgeLen < 0.000001f) continue;
            edge = Vector3Scale(edge, 1.0f / edgeLen);

            Vector3 airTransport = Vector3Scale(Vector3Add(a->wind, b->wind), 0.5f);
            Vector3 seaTransport = Vector3Scale(Vector3Add(a->current, b->current), 0.5f);
            float airFlow = Vector3DotProduct(airTransport, edge);
            float seaFlow = Vector3DotProduct(seaTransport, edge);
            float landA = 1.0f - a->ocean;
            float landB = 1.0f - b->ocean;
            float oceanGate = fminf(a->ocean, b->ocean);
            float landGate = fminf(landA, landB);
            float elevationJump = fabsf(b->elevation - a->elevation);
            float mountainBarrier = SmoothStep01((elevationJump - 0.10f) / 0.34f) * landGate;
            float coastMixing = fabsf(a->ocean - b->ocean);
            float airConductance = (1.0f - mountainBarrier * 0.68f) * (1.0f + coastMixing * 0.16f);
            float oceanConductance = oceanGate * (1.0f - coastMixing * 0.45f);

            float airAdvection = ClampFloat(fabsf(airFlow) * 10.5f * dt * airConductance, 0.0f, 0.24f);
            float airDiffusion = (0.018f + 0.026f * (1.0f - mountainBarrier)) * dt;
            float oceanAdvection = ClampFloat(fabsf(seaFlow) * 9.0f * dt * oceanConductance, 0.0f, 0.18f);
            float oceanDiffusion = (0.010f + 0.022f * oceanConductance) * dt;

            float temperatureTransfer = (airFlow >= 0.0f)
                ? (a->temperature - b->temperature) * airAdvection
                : (a->temperature - b->temperature) * airAdvection;
            temperatureTransfer += (a->temperature - b->temperature) * airDiffusion;
            AddSymmetricScalarFlux(&fluxes[i].temperature, &fluxes[nb].temperature, temperatureTransfer);

            float pressureTransfer = (a->pressure - b->pressure) * (airDiffusion * 0.60f + airAdvection * 0.20f);
            AddSymmetricScalarFlux(&fluxes[i].pressure, &fluxes[nb].pressure, pressureTransfer);

            float humidityTransfer = (a->humidity - b->humidity) * (airDiffusion * 0.82f + airAdvection * 0.72f);
            float cloudTransfer = (a->cloudWater - b->cloudWater) * (airDiffusion * 0.68f + airAdvection * 0.52f);
            AddSymmetricScalarFlux(&fluxes[i].humidity, &fluxes[nb].humidity, humidityTransfer);
            AddSymmetricScalarFlux(&fluxes[i].cloudWater, &fluxes[nb].cloudWater, cloudTransfer);

            Vector3 windTransfer = Vector3Scale(Vector3Subtract(a->wind, b->wind), airAdvection * 0.34f + airDiffusion * 0.42f);
            AddSymmetricVectorFlux(&fluxes[i].wind, &fluxes[nb].wind, windTransfer);

            float oceanHeatTransfer = (a->oceanTemperature - b->oceanTemperature) * (oceanDiffusion + oceanAdvection);
            AddSymmetricScalarFlux(&fluxes[i].oceanTemperature, &fluxes[nb].oceanTemperature, oceanHeatTransfer);
            Vector3 currentTransfer = Vector3Scale(Vector3Subtract(a->current, b->current), oceanAdvection * 0.42f + oceanDiffusion * 0.36f);
            AddSymmetricVectorFlux(&fluxes[i].current, &fluxes[nb].current, currentTransfer);

            float downhillAB = fmaxf(0.0f, a->elevation - b->elevation);
            float downhillBA = fmaxf(0.0f, b->elevation - a->elevation);
            float soilExchange = (a->soilMoisture - b->soilMoisture) * (0.010f * dt * landGate);
            soilExchange += a->soilMoisture * downhillAB * 0.060f * dt * landGate;
            soilExchange -= b->soilMoisture * downhillBA * 0.060f * dt * landGate;
            AddSymmetricScalarFlux(&fluxes[i].soilMoisture, &fluxes[nb].soilMoisture, soilExchange);
        }
    }
}

static void InitializeWeather(WeatherCell *cells, const Tile *tiles, int count, const ClimateSettings *climate, const SolarState *solar)
{
    RefreshWeatherSurface(cells, tiles, count);
    for (int i = 0; i < count; i++) {
        WeatherForcing forcing = BuildWeatherForcing(cells[i].normal, climate, solar);
        float pressureSeed = SphericalNoise3(cells[i].normal, 2.85f, (Vector3){ 88.1f, -34.6f, 12.5f }, 3, 2.04f, 0.52f);
        float initializationItcz = GaussianBand(forcing.climateLat, 0.04f, 0.22f) * LerpFloat(0.62f, 1.18f, forcing.moistureWave);
        float initializationHigh = GaussianBand(forcing.climateLat, 0.31f, 0.15f);
        float initializationStormTrack = GaussianBand(forcing.climateLat, 0.55f, 0.22f) * LerpFloat(0.70f, 1.16f, forcing.planetaryWave);

        WeatherTerrain terrain = BuildWeatherTerrain(&cells[i], 0.0f, 0.0f, cells[i].elevation);
        float meanTemperatureC = ClimateMeanSeaLevelTemperatureC(solar);
        float seaLevelTemperatureC = meanTemperatureC + (powf(forcing.equatorHeat, 0.82f) - 0.55f) * (74.0f * climate->temperatureContrast);
        seaLevelTemperatureC += terrain.land * (forcing.equatorHeat - 0.48f) * 8.0f;
        seaLevelTemperatureC += terrain.ocean * 1.8f - terrain.oceanDepthKm * 0.25f;
        float temperatureC = seaLevelTemperatureC - terrain.altitudeKm * WEATHER_TROPO_LAPSE_C_PER_KM - terrain.highland * 2.5f;
        cells[i].temperature = WeatherTemperature01FromC(temperatureC);
        cells[i].pressureAnomaly = (forcing.synopticWave - 0.5f) * 0.060f + (pressureSeed - 0.5f) * 0.030f;
        float seaLevelPressureHpa = WEATHER_SEA_LEVEL_PRESSURE_HPA - initializationItcz * 11.0f + initializationHigh * 15.0f - initializationStormTrack * 9.0f + forcing.polarHigh * 7.0f + cells[i].pressureAnomaly * 170.0f;
        seaLevelPressureHpa -= terrain.land * (temperatureC - 13.0f) * 0.85f;
        float pressureHpa = seaLevelPressureHpa * TerrainPressureFactor(terrain.altitudeKm);
        cells[i].pressure = WeatherPressure01FromHpa(pressureHpa);
        float saturation = WeatherSaturation(cells[i].temperature);
        float liquidWaterFactor = ClimateLiquidWaterFactor(solar);
        float relativeHumidity = ClampFloat(0.20f + terrain.ocean * 0.48f * liquidWaterFactor + initializationItcz * 0.16f * liquidWaterFactor + initializationStormTrack * 0.12f + terrain.coast * 0.10f * liquidWaterFactor + (forcing.moistureWave - 0.5f) * 0.14f - initializationHigh * 0.16f - terrain.altitudeKm * 0.035f, 0.04f, 1.05f);
        cells[i].humidity = ClampFloat(relativeHumidity * saturation, 0.0f, 1.35f);
        cells[i].cloud = ClampFloat(0.03f + initializationItcz * 0.10f + initializationStormTrack * 0.09f + WeatherRelativeHumidity(cells[i].humidity, cells[i].temperature) * 0.22f + (forcing.moistureWave - 0.5f) * 0.06f - initializationHigh * 0.06f + terrain.mountain * 0.08f, 0.0f, 1.0f);
        cells[i].precipitation = 0.0f;
        cells[i].vorticity = 0.0f;
        cells[i].evaporation = 0.0f;
        float mountainSnow = SmoothStep01((cells[i].elevation - 0.58f) / 0.44f) * SmoothStep01((0.52f - cells[i].temperature) / 0.24f);
        cells[i].snow = ClampFloat((GaussianBand(forcing.latAbs, 0.92f, 0.22f) + mountainSnow * 0.75f) * (1.0f - cells[i].ocean * 0.55f) * SmoothStep01((0.42f - cells[i].temperature) / 0.30f), 0.0f, 1.0f);
        cells[i].storm = 0.0f;
        cells[i].surfaceTemperature = cells[i].temperature;
        cells[i].oceanTemperature = cells[i].temperature;
        cells[i].soilMoisture = ClampFloat(terrain.land * liquidWaterFactor * (0.08f + relativeHumidity * 0.34f + initializationStormTrack * 0.14f + terrain.coast * 0.10f - initializationHigh * 0.11f - terrain.altitudeKm * 0.025f), 0.0f, 1.0f);
        cells[i].cloudWater = cells[i].cloud * 0.35f;
        cells[i].recentRain = 0.0f;
        cells[i].rainShadow = 0.0f;
        cells[i].orographicLift = 0.0f;
        cells[i].frontStrength = 0.0f;
        cells[i].airMassAge = RandomRange(0.0f, 1.0f);
        cells[i].climateTemperature = cells[i].temperature;
        cells[i].climatePrecipitation = 0.0f;
        cells[i].climateSoilMoisture = cells[i].soilMoisture;
        cells[i].seasonalTemperatureMin = cells[i].temperature;
        cells[i].seasonalTemperatureMax = cells[i].temperature;

        float zonal = (0.088f * forcing.westerly - 0.066f * forcing.trade - 0.035f * forcing.polar) * LerpFloat(0.64f, 1.10f, forcing.planetaryWave) * 0.62f;
        float meridional = ((forcing.planetaryWave - 0.5f) * 0.052f + (forcing.eddyWave - 0.5f) * 0.034f) * (1.0f - forcing.latAbs * 0.35f);
        Vector3 windEddy = ClimateEddyFlowFromAxis(cells[i].normal, solar->northPole, 3.4f, (Vector3){ -8.7f, 61.4f, 25.1f }, 0.010f);
        cells[i].wind = Vector3Add(
            Vector3Add(Vector3Add(Vector3Scale(forcing.east, zonal), Vector3Scale(forcing.north, meridional)), windEddy),
            Vector3Scale(RandomTangentVector(cells[i].normal), RandomRange(-0.006f, 0.006f))
        );

        // Initial ocean gyres use latitude bands, with planetary waves to avoid perfectly parallel rings.
        float oceanJet = (0.052f * forcing.westerly - 0.045f * forcing.trade - 0.020f * forcing.polar) * cells[i].ocean * LerpFloat(0.82f, 1.06f, forcing.planetaryWave) * 0.32f;
        Vector3 currentEddy = ClimateEddyFlowFromAxis(cells[i].normal, solar->northPole, 2.6f, (Vector3){ 37.8f, -14.2f, 73.6f }, 0.015f * cells[i].ocean);
        cells[i].current = Vector3Add(
            Vector3Add(Vector3Add(Vector3Scale(forcing.east, oceanJet), Vector3Scale(forcing.north, meridional * cells[i].ocean * 0.30f)), currentEddy),
            Vector3Scale(RandomTangentVector(cells[i].normal), RandomRange(-0.003f, 0.003f) * cells[i].ocean)
        );
    }
}

static void StepWeatherSimulation(
    const TileGraphNode *graph,
    const WeatherCell *src,
    WeatherCell *dst,
    int count,
    float dt,
    const ClimateSettings *climate,
    const SolarState *solar
)
{
    WeatherFlux *fluxes = (WeatherFlux *)calloc((size_t)count, sizeof(WeatherFlux));
    if (fluxes) AccumulateWeatherFluxes(graph, src, fluxes, count, dt);
    const WeatherFlux zeroFlux = { 0 };

    for (int i = 0; i < count; i++) {
        const WeatherCell *c = &src[i];
        const TileGraphNode *node = &graph[i];
        const WeatherFlux *flux = fluxes ? &fluxes[i] : &zeroFlux;

        float tempAvg = c->temperature;
        float humidAvg = c->humidity;
        float cloudAvg = c->cloud;
        float cloudWaterAvg = c->cloudWater;
        float soilMoistureAvg = c->soilMoisture;
        float pressAvg = c->pressure;
        Vector3 windAvg = c->wind;
        Vector3 currentAvg = c->current;
        float oceanTempAvg = c->oceanTemperature;
        float oceanTempWeight = 1.0f;
        float scalarWeight = 1.0f;
        Vector3 pressureGradient = { 0 };
        Vector3 terrainGradient = { 0 };
        float divergence = 0.0f;
        float vorticity = 0.0f;
        float orographic = 0.0f;
        float frontContrast = 0.0f;
        float frontWeight = 0.0f;
        float oceanAvg = c->ocean;
        float elevationAvg = c->elevation;
        float coastalContrast = 0.0f;
        float upwindOcean = c->ocean;
        float upwindOceanWeight = 1.0f;
        Vector3 transportLocal = Vector3Add(c->wind, Vector3Scale(c->current, 0.85f));

        // Transport pass: sample mostly from upwind cells, with a small diffusion floor.
        for (int n = 0; n < node->neighborCount; n++) {
            int nb = node->neighbors[n];
            const WeatherCell *k = &src[nb];
            Vector3 toNb = Vector3Subtract(k->normal, Vector3Scale(c->normal, Vector3DotProduct(k->normal, c->normal)));
            float toNbLen = Vector3Length(toNb);
            if (toNbLen < 0.000001f) continue;
            toNb = Vector3Scale(toNb, 1.0f / toNbLen);

            float flow = Vector3DotProduct(transportLocal, toNb);
            float inflow = fmaxf(0.0f, -flow);
            float outflow = fmaxf(0.0f, flow);
            float upwindWeight = 0.18f + inflow * 24.0f;

            tempAvg += k->temperature * upwindWeight;
            humidAvg += k->humidity * upwindWeight;
            cloudAvg += k->cloud * upwindWeight;
            cloudWaterAvg += k->cloudWater * upwindWeight;
            windAvg = Vector3Add(windAvg, Vector3Scale(k->wind, upwindWeight));
            currentAvg = Vector3Add(currentAvg, Vector3Scale(k->current, upwindWeight));
            scalarWeight += upwindWeight;
            soilMoistureAvg += k->soilMoisture;

            float currentFlow = Vector3DotProduct(c->current, toNb);
            float oceanWeight = 1.0f + fmaxf(0.0f, -currentFlow * 4.0f);
            oceanTempAvg += k->oceanTemperature * oceanWeight;
            oceanTempWeight += oceanWeight;

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
            upwindOcean += k->ocean * inflow;
            upwindOceanWeight += inflow;

            float thermalContrast = fabsf(k->temperature - c->temperature);
            float moistureContrast = fabsf(WeatherRelativeHumidity(k->humidity, k->temperature) - WeatherRelativeHumidity(c->humidity, c->temperature));
            float boundaryFlow = inflow + outflow * 0.35f;
            frontContrast += (thermalContrast * 1.35f + moistureContrast * 0.38f) * boundaryFlow;
            frontWeight += boundaryFlow;
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
        oceanTempAvg /= oceanTempWeight;
        float terrainSlope = ClampFloat(Vector3Length(terrainGradient), 0.0f, 1.0f);
        float incomingFront = (frontWeight > 0.000001f) ? ClampFloat(frontContrast / frontWeight * 6.5f, 0.0f, 1.0f) : 0.0f;

        WeatherForcing forcing = BuildWeatherForcing(c->normal, climate, solar);
        float convergence = ClampFloat(-divergence * 1.8f, 0.0f, 1.0f);
        Vector3 transport = Vector3Add(c->wind, Vector3Scale(c->current, 0.85f));
        float advection = ClampFloat(Vector3Length(transport) * 4.2f, 0.0f, 1.0f);
        WeatherTerrain terrain = BuildWeatherTerrain(c, terrainSlope, coastalContrast, elevationAvg);
        float land = terrain.land;
        float signedClimateLatitude = Vector3DotProduct(c->normal, solar->northPole);
        float albedo = ClampFloat(0.06f * terrain.ocean + 0.21f * land + 0.48f * c->snow + 0.18f * c->cloud, 0.0f, 0.82f);
        float meanTemperatureC = ClimateMeanSeaLevelTemperatureC(solar);
        float seaLevelTemperatureC = meanTemperatureC + (powf(forcing.equatorHeat, 0.80f) - 0.55f) * (76.0f * climate->temperatureContrast);
        seaLevelTemperatureC += land * (forcing.equatorHeat - 0.46f) * (10.0f * climate->temperatureContrast);
        seaLevelTemperatureC += terrain.ocean * (WeatherTemperatureC(c->oceanTemperature) - seaLevelTemperatureC) * 0.18f;
        seaLevelTemperatureC += terrain.coast * (WeatherTemperatureC(c->oceanTemperature) - seaLevelTemperatureC) * 0.08f;
        seaLevelTemperatureC -= albedo * 10.5f;
        seaLevelTemperatureC += WeatherRelativeHumidity(c->humidity, c->temperature) * 1.6f;
        seaLevelTemperatureC -= terrain.mountain * 2.3f;
        float temperatureTargetC = seaLevelTemperatureC - terrain.altitudeKm * WEATHER_TROPO_LAPSE_C_PER_KM;
        float synopticTarget = (forcing.synopticWave - 0.5f) * (0.055f + forcing.stormTrack * 0.045f) - convergence * 0.018f + forcing.subtropicalHigh * 0.010f;
        float pressureAnomaly = c->pressureAnomaly + (synopticTarget - c->pressureAnomaly) * (0.10f + forcing.stormTrack * 0.10f) * dt;
        pressureAnomaly = ClampFloat(pressureAnomaly, -0.105f, 0.105f);
        float seaLevelPressureHpa = WEATHER_SEA_LEVEL_PRESSURE_HPA - forcing.itcz * 14.0f + forcing.subtropicalHigh * 19.0f - forcing.stormTrack * 10.0f + forcing.polarHigh * 8.0f + pressureAnomaly * 185.0f;
        float thermalLowHpa = land * fmaxf(temperatureTargetC - 14.0f, -10.0f) * 0.82f;
        float humidityLowHpa = WeatherRelativeHumidity(c->humidity, c->temperature) * 4.5f;
        float leeTroughHpa = terrain.exposure * land * 5.0f;
        float coastalBreezeHpa = terrain.coast * (terrain.ocean - oceanAvg) * 10.0f;
        float pressureTargetHpa = seaLevelPressureHpa * TerrainPressureFactor(terrain.altitudeKm) - thermalLowHpa - humidityLowHpa - leeTroughHpa + coastalBreezeHpa;
        float pressureTarget = WeatherPressure01FromHpa(pressureTargetHpa);

        // Pressure and wind pass: persistent synoptic anomalies ride on the climate bands.
        float pressure = c->pressure;
        pressure += (pressureTarget - pressure) * 0.42f * dt;
        pressure += (pressAvg - pressure) * 0.72f * dt;
        pressure += flux->pressure;
        pressure -= convergence * 0.045f * dt;
        pressure = ClampFloat(pressure, 0.04f, 0.96f);

        Vector3 east = forcing.east;
        Vector3 north = forcing.north;
        float northLen = Vector3Length(north);

        float zonalStrength = (0.088f * forcing.westerly - 0.066f * forcing.trade - 0.035f * forcing.polar) * LerpFloat(0.78f, 1.08f, forcing.planetaryWave) * 0.52f;
        float meridionalStrength = ((forcing.planetaryWave - 0.5f) * 0.020f + (forcing.eddyWave - 0.5f) * 0.015f) * (1.0f - forcing.latAbs * 0.35f);
        Vector3 terrainDownslope = Vector3Scale(terrainGradient, -0.045f * land * (0.45f + terrain.exposure));
        Vector3 terrainContour = Vector3CrossProduct(c->normal, terrainGradient);
        if (Vector3LengthSqr(terrainContour) > 0.000001f) terrainContour = Vector3Normalize(terrainContour);
        Vector3 terrainSteering = Vector3Add(
            terrainDownslope,
            Vector3Scale(terrainContour, Vector3DotProduct(c->wind, terrainGradient) * -0.085f * land * (0.45f + terrain.roughness))
        );
        float valleyChannel = terrain.valley * terrainSlope;
        Vector3 valleyBoost = Vector3Scale(terrainContour, Vector3DotProduct(c->wind, terrainContour) * valleyChannel * 0.07f);
        Vector3 jet = Vector3Add(Vector3Add(Vector3Add(Vector3Scale(east, zonalStrength), Vector3Scale(north, meridionalStrength)), terrainSteering), valleyBoost);

        Vector3 pressureForce = Vector3Scale(pressureGradient, -1.45f);
        Vector3 coriolis = Vector3Scale(Vector3CrossProduct(c->normal, c->wind), signedClimateLatitude * 0.92f);
        Vector3 wind = c->wind;
        wind = Vector3Add(wind, flux->wind);
        wind = Vector3Add(wind, Vector3Scale(Vector3Add(Vector3Add(pressureForce, coriolis), jet), dt));
        wind = Vector3Add(wind, Vector3Scale(Vector3Subtract(windAvg, wind), (0.54f + 0.34f * advection) * dt));

        float surfaceRoughness = terrain.roughness;
        float mountainBlocking = terrain.mountain * SmoothStep01((fabsf(Vector3DotProduct(wind, terrainGradient)) - 0.004f) / 0.030f);
        float drag = 0.22f + surfaceRoughness + mountainBlocking * 0.52f + c->cloud * 0.08f;
        wind = Vector3Scale(wind, ClampFloat(1.0f - drag * dt, 0.0f, 1.0f));
        float windLen = Vector3Length(wind);
        if (windLen > 0.18f) wind = Vector3Scale(wind, 0.18f / windLen);
        wind = LimitTangentVector(wind, c->normal, 0.18f);

        // Ocean currents use gyre bands, wind stress, pressure gradient, and Coriolis.
        Vector3 current = c->current;
        current = Vector3Add(current, flux->current);
        Vector3 gyre = Vector3Add(
            Vector3Add(
                Vector3Scale(east, (0.052f * forcing.westerly - 0.045f * forcing.trade - 0.020f * forcing.polar) * c->ocean * LerpFloat(0.82f, 1.06f, forcing.planetaryWave) * 0.32f),
                Vector3Scale(north, meridionalStrength * c->ocean * 0.24f)
            ),
            Vector3Scale(terrainGradient, -0.018f * c->ocean)
        );
        Vector3 currentPressureForce = Vector3Scale(pressureGradient, -0.18f * c->ocean);
        Vector3 windStress = Vector3Scale(wind, 0.42f * c->ocean);
        Vector3 currentCoriolis = Vector3Scale(Vector3CrossProduct(c->normal, current), signedClimateLatitude * 0.45f);
        current = Vector3Add(current, Vector3Scale(Vector3Add(Vector3Add(currentPressureForce, windStress), currentCoriolis), dt));
        current = Vector3Add(current, Vector3Scale(Vector3Subtract(gyre, current), 0.42f * dt));
        current = Vector3Add(current, Vector3Scale(Vector3Subtract(currentAvg, current), 0.50f * dt));

        float currentDrag = 0.14f + land * 1.05f + terrain.coast * terrain.ocean * 0.18f;
        current = Vector3Scale(current, ClampFloat(1.0f - currentDrag * dt, 0.0f, 1.0f));
        float currentLen = Vector3Length(current);
        if (currentLen > 0.12f) current = Vector3Scale(current, 0.12f / currentLen);
        current = LimitTangentVector(current, c->normal, 0.12f);

        float polewardCurrent = 0.0f;
        if (northLen > 0.000001f) {
            polewardCurrent = Vector3DotProduct(current, north) * ((Vector3DotProduct(c->normal, solar->northPole) >= 0.0f) ? 1.0f : -1.0f);
        }

        float sunDot = SolarFacingAmount(c->normal, solar->lightDir);
        float daylightHeating = climate->dayNightEnabled ? fmaxf(0.0f, sunDot) * (0.16f * powf(fmaxf(solar->stellarFlux, 0.0001f), 0.25f)) : 0.08f;
        float radiativeCooling = climate->dayNightEnabled ? fmaxf(0.0f, -sunDot) * (0.10f + land * 0.05f) : 0.0f;

        float upslopeFlow = ClampFloat(Vector3DotProduct(wind, terrainGradient) * 18.0f, 0.0f, 1.0f);
        float downslopeFlow = ClampFloat(-Vector3DotProduct(wind, terrainGradient) * 20.0f, 0.0f, 1.0f);
        float orographicLift = ClampFloat(orographic * 1.75f + upslopeFlow * (0.90f + terrain.mountain * 0.86f) + terrain.exposure * 0.18f, 0.0f, 1.0f);
        float rainShadow = ClampFloat(c->rainShadow * 0.64f + downslopeFlow * (0.58f + terrain.mountain * 0.50f) + terrain.exposure * land * 0.10f - upwindOcean * 0.14f - orographicLift * 0.24f, 0.0f, 0.92f);

        float surfaceTemperature = c->surfaceTemperature;
        float coldPoolC = terrain.valley * land * ClampFloat(1.0f - Vector3Length(c->wind) * 8.0f, 0.0f, 1.0f) * 5.0f;
        float surfaceTargetC = temperatureTargetC;
        surfaceTargetC += land * (forcing.equatorHeat - 0.50f) * (7.5f * climate->temperatureContrast);
        surfaceTargetC += daylightHeating * (16.0f + land * 6.0f);
        surfaceTargetC -= radiativeCooling * 14.0f;
        surfaceTargetC -= c->snow * 11.0f + c->soilMoisture * land * 3.2f;
        surfaceTargetC += rainShadow * land * 4.6f;
        surfaceTargetC -= coldPoolC;
        float surfaceTarget = WeatherTemperature01FromC(surfaceTargetC);
        float surfaceResponse = LerpFloat(0.18f, 0.86f, land) * (1.0f - c->soilMoisture * land * 0.35f) * (1.0f + terrain.exposure * 0.18f);
        surfaceTemperature += (surfaceTarget - surfaceTemperature) * surfaceResponse * dt;
        surfaceTemperature = ClampFloat(surfaceTemperature, 0.0f, 1.0f);

        // Surface pass: land reacts quickly, ocean carries heat with much higher inertia.
        float oceanTemperature = c->oceanTemperature;
        oceanTemperature += flux->oceanTemperature;
        float liquidWaterFactor = ClimateLiquidWaterFactor(solar);
        float oceanBaseC = meanTemperatureC + (powf(forcing.equatorHeat, 0.88f) - 0.54f) * 42.0f + daylightHeating * 2.8f - radiativeCooling * 1.2f;
        float upwellingC = terrain.coast * (1.0f - upwindOcean) * (5.5f + terrain.oceanDepthKm * 0.45f);
        float oceanTempTarget = WeatherTemperature01FromC(oceanBaseC - upwellingC - terrain.oceanDepthKm * 0.35f);
        float oceanInertia = 0.035f * c->ocean;
        float currentAdvection = ClampFloat(Vector3Length(c->current) * 6.0f, 0.0f, 1.0f);
        oceanTemperature += (oceanTempTarget - oceanTemperature) * oceanInertia * dt;
        oceanTemperature += (oceanTempAvg - oceanTemperature) * (0.12f + 0.68f * currentAdvection) * dt;
        oceanTemperature += polewardCurrent * c->ocean * 0.075f * dt;
        oceanTemperature = ClampFloat(oceanTemperature, 0.0f, 1.0f);

        float temperature = c->temperature;
        float heatResponse = LerpFloat(0.42f, 0.76f, land);
        temperature += flux->temperature;
        temperature += (surfaceTemperature - temperature) * heatResponse * dt;
        temperature += (oceanTemperature - temperature) * c->ocean * 0.16f * dt;
        temperature += (tempAvg - temperature) * (0.15f + 0.76f * advection) * dt;
        temperature += polewardCurrent * c->ocean * 0.10f * dt;
        temperature += rainShadow * land * 0.070f * dt;
        temperature -= c->cloud * 0.018f * dt;
        temperature = ClampFloat(temperature, 0.0f, 1.0f);

        // Water cycle pass: vapor condenses into cloud water, then rains or snows out.
        float humidity = c->humidity;
        humidity += flux->humidity;
        float saturation = WeatherSaturation(temperature);
        float coastalMoisture = ClampFloat((oceanAvg * 0.38f + upwindOcean * 0.78f + terrain.coast * 0.34f) * liquidWaterFactor, 0.0f, 1.0f);
        float landWetness = ClampFloat(
            c->soilMoisture * 0.62f + c->recentRain * 0.64f + coastalMoisture * 0.20f + c->cloudWater * 0.18f + c->snow * 0.22f,
            0.0f,
            1.0f
        );
        float surfaceWetness = ClampFloat(c->ocean * liquidWaterFactor + land * landWetness, 0.0f, 1.0f);
        float evaporationSurfaceTemp = LerpFloat(surfaceTemperature, oceanTemperature, c->ocean);
        float humidityDeficit = ClampFloat((saturation - humidity + 0.16f) / 0.84f, 0.0f, 1.0f);
        float climateRelativeHumidity = ClampFloat(0.32f + c->ocean * 0.40f + coastalMoisture * 0.30f + land * c->soilMoisture * 0.30f + forcing.itcz * 0.15f + forcing.stormTrack * 0.08f - forcing.subtropicalHigh * 0.18f - terrain.altitudeKm * 0.030f - rainShadow * 0.33f - c->airMassAge * land * 0.08f, 0.08f, 1.10f);
        float climateHumidity = ClampFloat(climateRelativeHumidity * saturation, 0.0f, 1.30f);
        float aerodynamicDrying = 0.55f + windLen * 7.2f + terrain.exposure * 0.34f;
        float evaporationRate = surfaceWetness * (0.010f + evaporationSurfaceTemp * 0.116f + land * 0.014f) * humidityDeficit * aerodynamicDrying;
        humidity += evaporationRate * dt;
        humidity += (climateHumidity - humidity) * 0.11f * dt;
        humidity += (humidAvg - humidity) * (0.12f + 0.72f * advection) * dt;
        humidity -= rainShadow * land * (0.080f + terrain.mountain * 0.045f) * dt;

        float frontStrength = ClampFloat(c->frontStrength + (incomingFront - c->frontStrength) * (0.48f + advection * 0.70f) * dt, 0.0f, 1.0f);
        float lift = convergence * 0.40f + orographicLift * 1.05f + fabsf(vorticity) * 0.06f + frontStrength * (0.20f + forcing.stormTrack * 0.34f);
        float relativeHumidity = ClampFloat(humidity / saturation, 0.0f, 2.0f);
        float condensation = fmaxf(0.0f, humidity - saturation * 0.96f) * (0.80f + 1.40f * lift) * dt;
        condensation += humidity * lift * SmoothStep01((relativeHumidity - 0.58f) / 0.46f) * 0.055f * dt;
        humidity -= condensation;

        float cloudWater = c->cloudWater;
        cloudWater += flux->cloudWater;
        cloudWater += condensation * 1.20f;
        cloudWater += (cloudWaterAvg - cloudWater) * (0.18f + advection * 0.36f) * dt;
        cloudWater -= rainShadow * land * 0.040f * dt;
        cloudWater = ClampFloat(cloudWater, 0.0f, 1.20f);

        float cloud = c->cloud;
        float terrainCloud = land * c->soilMoisture * SmoothStep01((relativeHumidity - 0.40f) / 0.45f);
        float capCloud = terrain.mountain * SmoothStep01((relativeHumidity - 0.52f) / 0.34f) * (0.35f + upslopeFlow * 0.65f);
        float cloudTarget = ClampFloat(SmoothStep01(cloudWater / 0.34f) * 0.72f + (relativeHumidity - 0.58f) * 0.42f + lift * 0.28f + terrainCloud * 0.20f + capCloud * 0.18f + coastalMoisture * 0.06f - rainShadow * 0.18f, 0.0f, 1.0f);
        cloud += (cloudTarget - cloud) * (0.42f + 0.18f * advection) * dt;
        cloud += (cloudAvg - cloud) * 0.18f * dt;

        float storm = ClampFloat(lift * 0.88f + orographicLift * 0.38f + fabsf(vorticity) * 0.42f + convergence * 0.20f + frontStrength * 0.58f + fmaxf(0.0f, relativeHumidity - 0.82f) * 0.80f + cloudWater * 0.25f, 0.0f, 1.0f);
        float rainRate = cloudWater * (0.020f + 0.38f * storm + 0.13f * convergence + 0.58f * orographicLift + frontStrength * 0.25f + terrainCloud * 0.055f + capCloud * 0.070f);
        float rain = rainRate * dt;
        cloudWater = ClampFloat(cloudWater - rain * 1.55f, 0.0f, 1.20f);
        float coldPrecip = SmoothStep01((0.35f - temperature) / 0.18f);
        float snow = c->snow;
        snow += rain * coldPrecip * (land + c->ocean * 0.35f) * 0.85f;
        snow -= fmaxf(0.0f, temperature - 0.33f) * (0.18f + rain * 0.8f) * dt;
        float mountainSnow = SmoothStep01((c->elevation - 0.58f) / 0.44f) * SmoothStep01((0.52f - temperature) / 0.24f);
        snow += (GaussianBand(forcing.latAbs, 0.94f, 0.16f) + mountainSnow * 0.85f) * coldPrecip * land * 0.010f * dt;
        snow = ClampFloat(snow, 0.0f, 1.0f);

        float liquidRain = rain * (1.0f - coldPrecip * 0.78f);
        float runoff = ClampFloat((terrain.slope * 0.90f + terrain.mountain * 0.22f + c->soilMoisture * 0.35f - 0.12f) * land, 0.0f, 0.68f);
        float soilMoisture = c->soilMoisture;
        soilMoisture += flux->soilMoisture;
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
        // Climate-memory pass: biomes use long-term conditions instead of frame weather.
        float climateBlend = ClampFloat(0.018f * dt, 0.0001f, 0.018f);
        float climateTemperature = LerpFloat(c->climateTemperature, temperature, climateBlend);
        float climatePrecipitation = LerpFloat(c->climatePrecipitation, precipitationTarget, climateBlend);
        float climateSoilMoisture = LerpFloat(c->climateSoilMoisture, soilMoisture, climateBlend);
        float seasonalRecovery = ClampFloat(0.010f * dt, 0.0f, 0.010f);
        float seasonalTemperatureMin = LerpFloat(fminf(c->seasonalTemperatureMin, temperature), climateTemperature, seasonalRecovery);
        float seasonalTemperatureMax = LerpFloat(fmaxf(c->seasonalTemperatureMax, temperature), climateTemperature, seasonalRecovery);
        float airMassAge = ClampFloat((c->airMassAge + dt * (0.055f + land * 0.030f)) * (1.0f - (convergence + precipitationTarget) * 0.070f * dt), 0.0f, 1.0f);

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
        dst[i].oceanTemperature = oceanTemperature;
        dst[i].soilMoisture = soilMoisture;
        dst[i].cloudWater = cloudWater;
        dst[i].recentRain = recentRain;
        dst[i].rainShadow = rainShadow;
        dst[i].orographicLift = orographicLift;
        dst[i].pressureAnomaly = pressureAnomaly;
        dst[i].frontStrength = frontStrength;
        dst[i].airMassAge = airMassAge;
        dst[i].climateTemperature = climateTemperature;
        dst[i].climatePrecipitation = climatePrecipitation;
        dst[i].climateSoilMoisture = climateSoilMoisture;
        dst[i].seasonalTemperatureMin = seasonalTemperatureMin;
        dst[i].seasonalTemperatureMax = seasonalTemperatureMax;
    }

    free(fluxes);
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

static float WarpedClimateLatitudeFromAxis(Vector3 unitDirection, Vector3 climateNorth, float equatorShift, float strength)
{
    float axisLatitude = Vector3DotProduct(unitDirection, climateNorth) - equatorShift;
    axisLatitude = ClampFloat(axisLatitude, -1.0f, 1.0f);
    float latAbs = fabsf(axisLatitude);
    float planetaryWave = SphericalNoise3(unitDirection, 1.65f, (Vector3){ 23.7f, -18.2f, 9.4f }, 3, 1.92f, 0.54f);
    float eddyWave = SphericalNoise3(unitDirection, 7.8f, (Vector3){ 15.2f, 51.4f, -22.7f }, 2, 2.16f, 0.48f);
    float cellWave = SphericalNoise3(unitDirection, 3.20f, (Vector3){ -40.4f, 21.3f, 66.8f }, 3, 2.08f, 0.52f);
    float warp = (planetaryWave - 0.5f) * strength + (eddyWave - 0.5f) * strength * 0.58f + (cellWave - 0.5f) * strength * 0.36f;
    return ClampFloat(fabsf(axisLatitude + warp * (1.0f - latAbs * 0.22f)), 0.0f, 1.0f);
}

static Vector3 TangentEast(Vector3 normal)
{
    Vector3 east = Vector3CrossProduct((Vector3){ 0.0f, 1.0f, 0.0f }, normal);
    if (Vector3LengthSqr(east) < 0.000001f) east = Vector3CrossProduct((Vector3){ 1.0f, 0.0f, 0.0f }, normal);
    return Vector3Normalize(east);
}

static Vector3 TangentEastFromAxis(Vector3 normal, Vector3 climateNorth)
{
    Vector3 east = Vector3CrossProduct(climateNorth, normal);
    if (Vector3LengthSqr(east) < 0.000001f) east = Vector3CrossProduct((Vector3){ 0.0f, 0.0f, 1.0f }, normal);
    if (Vector3LengthSqr(east) < 0.000001f) east = Vector3CrossProduct((Vector3){ 1.0f, 0.0f, 0.0f }, normal);
    return Vector3Normalize(east);
}

static Vector3 TangentNorthFromAxis(Vector3 normal, Vector3 climateNorth)
{
    Vector3 north = Vector3Subtract(climateNorth, Vector3Scale(normal, Vector3DotProduct(climateNorth, normal)));
    if (Vector3LengthSqr(north) < 0.000001f) north = Vector3CrossProduct(normal, TangentEastFromAxis(normal, climateNorth));
    return Vector3Normalize(north);
}

static Vector3 ClimateEddyFlowFromAxis(Vector3 normal, Vector3 climateNorth, float scale, Vector3 offset, float strength)
{
    const float eps = 0.040f;
    Vector3 east = TangentEastFromAxis(normal, climateNorth);
    Vector3 north = TangentNorthFromAxis(normal, climateNorth);
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
        case WEATHER_VIEW_OCEAN_TEMP: return "Ocean Temperature";
        case WEATHER_VIEW_BIOME: return "Biomes";
        default: return "Unknown";
    }
}

static const char *WeatherViewShortName(WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE: return "Temperature";
        case WEATHER_VIEW_PRESSURE: return "Pressure";
        case WEATHER_VIEW_WIND: return "Wind";
        case WEATHER_VIEW_CURRENT: return "Currents";
        case WEATHER_VIEW_HUMIDITY: return "Humidity";
        case WEATHER_VIEW_CLOUD: return "Clouds";
        case WEATHER_VIEW_RAIN: return "Rain";
        case WEATHER_VIEW_VORTICITY: return "Vorticity";
        case WEATHER_VIEW_STORM: return "Storm";
        case WEATHER_VIEW_EVAPORATION: return "Evaporation";
        case WEATHER_VIEW_SNOW: return "Snow/Ice";
        case WEATHER_VIEW_OCEAN_TEMP: return "Ocean Temp";
        case WEATHER_VIEW_BIOME: return "Biomes";
        default: return "Unknown";
    }
}

static Color GetBiomeColor(const WeatherCell *w)
{
    if (w->ocean > 0.65f) {
        return (Color){ 18, 62, 118, 255 };
    }
    if (w->snow > 0.35f || w->surfaceTemperature < 0.08f) {
        return (Color){ 228, 234, 240, 255 };
    }

    float temp = w->climateTemperature;
    float seasonalRange = ClampFloat(w->seasonalTemperatureMax - w->seasonalTemperatureMin, 0.0f, 1.0f);
    float moisture = ClampFloat(w->climateSoilMoisture * 0.58f + w->climatePrecipitation * 0.34f + WeatherRelativeHumidity(w->humidity, w->temperature) * 0.08f, 0.0f, 1.0f);
    moisture = ClampFloat(moisture - seasonalRange * 0.06f + w->rainShadow * -0.08f, 0.0f, 1.0f);

    if (temp < 0.18f) {
        if (moisture < 0.22f) return (Color){ 182, 178, 166, 255 };
        return (Color){ 120, 150, 124, 255 };
    }
    if (temp < 0.38f) {
        if (moisture < 0.16f) return (Color){ 178, 162, 120, 255 };
        if (moisture < 0.32f) return (Color){ 96, 138, 72, 255 };
        return (Color){ 48, 108, 58, 255 };
    }
    if (temp < 0.58f) {
        if (moisture < 0.12f) return (Color){ 186, 168, 118, 255 };
        if (moisture < 0.28f) return (Color){ 108, 148, 62, 255 };
        return (Color){ 42, 102, 52, 255 };
    }
    if (temp < 0.78f) {
        if (moisture < 0.10f) return (Color){ 194, 174, 122, 255 };
        if (moisture < 0.24f) return (Color){ 118, 154, 58, 255 };
        return (Color){ 38, 96, 48, 255 };
    }
    if (moisture < 0.08f) return (Color){ 202, 180, 128, 255 };
    if (moisture < 0.20f) return (Color){ 128, 148, 52, 255 };
    return (Color){ 34, 88, 44, 255 };
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
            float rain = SmoothStep01(fmaxf(fmaxf(w->precipitation, w->recentRain * 0.55f), w->frontStrength * 0.16f) / 0.22f);
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
            float storm = SmoothStep01((fmaxf(fmaxf(w->storm, w->orographicLift * 0.65f), w->frontStrength * 0.72f) - 0.04f) / 0.55f);
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
        case WEATHER_VIEW_OCEAN_TEMP: {
            return Gradient3(
                w->oceanTemperature,
                (Color){ 22, 42, 86, 255 },
                (Color){ 44, 154, 214, 255 },
                (Color){ 232, 80, 60, 255 }
            );
        }
        case WEATHER_VIEW_BIOME: {
            return GetBiomeColor(w);
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
        case WEATHER_VIEW_OCEAN_TEMP:
            return 0.48f;
        case WEATHER_VIEW_BIOME:
            return 0.72f;
        default:
            return 0.62f;
    }
}

static float WeatherIceAmount(const WeatherCell *w)
{
    float ocean = ClampFloat(w->ocean, 0.0f, 1.0f);
    float land = 1.0f - ocean;
    float oceanTempC = WeatherTemperatureC(w->oceanTemperature);
    float surfaceTempC = WeatherTemperatureC(w->surfaceTemperature);
    float seaIce = ocean * ClampFloat(
        SmoothStep01((-1.8f - oceanTempC) / 7.0f) * 0.72f +
        SmoothStep01((-8.0f - surfaceTempC) / 16.0f) * 0.42f +
        w->snow * 0.28f,
        0.0f,
        1.0f
    );
    float landIce = land * ClampFloat(
        w->snow * 0.76f +
        SmoothStep01((-6.0f - surfaceTempC) / 18.0f) * 0.32f +
        SmoothStep01((w->elevation - 0.45f) / 0.50f) * 0.22f,
        0.0f,
        1.0f
    );
    return ClampFloat(fmaxf(seaIce, landIce), 0.0f, 1.0f);
}

static Color WeatherIceColor(const WeatherCell *w)
{
    Color glacier = (Color){ 234, 244, 250, 255 };
    Color seaIce = (Color){ 198, 226, 240, 255 };
    return LerpColor(glacier, seaIce, ClampFloat(w->ocean, 0.0f, 1.0f));
}

static float WeatherChartMetricValue(const WeatherCell *w, WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE:
            return ClampFloat(w->temperature, 0.0f, 1.0f);
        case WEATHER_VIEW_PRESSURE:
            return ClampFloat(w->pressure, 0.0f, 1.0f);
        case WEATHER_VIEW_WIND:
            return SmoothStep01(Vector3Length(w->wind) / 0.18f);
        case WEATHER_VIEW_CURRENT:
            return SmoothStep01(Vector3Length(w->current) / 0.12f);
        case WEATHER_VIEW_HUMIDITY:
            return ClampFloat((WeatherRelativeHumidity(w->humidity, w->temperature) - 0.10f) / 1.05f, 0.0f, 1.0f);
        case WEATHER_VIEW_CLOUD:
            return ClampFloat(fmaxf(w->cloud, w->cloudWater * 1.15f), 0.0f, 1.0f);
        case WEATHER_VIEW_RAIN:
            return SmoothStep01(fmaxf(fmaxf(w->precipitation, w->recentRain * 0.55f), w->frontStrength * 0.16f) / 0.28f);
        case WEATHER_VIEW_VORTICITY:
            return ClampFloat(fabsf(w->vorticity), 0.0f, 1.0f);
        case WEATHER_VIEW_STORM:
            return ClampFloat(fmaxf(fmaxf(w->storm, w->orographicLift * 0.65f), w->frontStrength * 0.72f), 0.0f, 1.0f);
        case WEATHER_VIEW_EVAPORATION:
            return SmoothStep01((w->evaporation + w->soilMoisture * 0.12f) / 0.42f);
        case WEATHER_VIEW_SNOW:
            return ClampFloat(w->snow, 0.0f, 1.0f);
        case WEATHER_VIEW_OCEAN_TEMP:
            return ClampFloat(w->oceanTemperature, 0.0f, 1.0f);
        case WEATHER_VIEW_BIOME: {
            float land = 1.0f - ClampFloat(w->ocean, 0.0f, 1.0f);
            float moisture = ClampFloat(w->climateSoilMoisture * 0.58f + w->climatePrecipitation * 0.34f, 0.0f, 1.0f);
            float warmth = SmoothStep01((w->climateTemperature - 0.18f) / 0.62f);
            return ClampFloat(land * (moisture * 0.62f + warmth * 0.38f), 0.0f, 1.0f);
        }
        default:
            return 0.0f;
    }
}

static Color WeatherChartLineColor(WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE: return (Color){ 232, 142, 78, 255 };
        case WEATHER_VIEW_PRESSURE: return (Color){ 108, 158, 238, 255 };
        case WEATHER_VIEW_WIND: return (Color){ 236, 232, 170, 255 };
        case WEATHER_VIEW_CURRENT: return (Color){ 78, 198, 236, 255 };
        case WEATHER_VIEW_HUMIDITY: return (Color){ 94, 184, 124, 255 };
        case WEATHER_VIEW_CLOUD: return (Color){ 204, 216, 232, 255 };
        case WEATHER_VIEW_RAIN: return (Color){ 72, 148, 238, 255 };
        case WEATHER_VIEW_VORTICITY: return (Color){ 216, 108, 116, 255 };
        case WEATHER_VIEW_STORM: return (Color){ 190, 148, 246, 255 };
        case WEATHER_VIEW_EVAPORATION: return (Color){ 224, 190, 92, 255 };
        case WEATHER_VIEW_SNOW: return (Color){ 232, 244, 252, 255 };
        case WEATHER_VIEW_OCEAN_TEMP: return (Color){ 238, 96, 82, 255 };
        case WEATHER_VIEW_BIOME: return (Color){ 88, 184, 84, 255 };
        default: return (Color){ 255, 255, 255, 255 };
    }
}

static float WeatherChartDisplayValue(WeatherViewMode mode, float normalized)
{
    normalized = ClampFloat(normalized, 0.0f, 1.0f);
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE:
        case WEATHER_VIEW_OCEAN_TEMP:
            return WeatherTemperatureC(normalized);
        case WEATHER_VIEW_PRESSURE:
            return WeatherPressureHpa(normalized);
        case WEATHER_VIEW_WIND:
            return normalized * 45.0f;
        case WEATHER_VIEW_CURRENT:
            return normalized * 3.0f;
        case WEATHER_VIEW_RAIN:
            return normalized * 7.0f;
        case WEATHER_VIEW_VORTICITY:
            return normalized * 8.0f;
        case WEATHER_VIEW_EVAPORATION:
            return normalized * 12.0f;
        default:
            return normalized * 100.0f;
    }
}

static const char *WeatherChartUnit(WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE:
        case WEATHER_VIEW_OCEAN_TEMP:
            return "C";
        case WEATHER_VIEW_PRESSURE:
            return "hPa";
        case WEATHER_VIEW_WIND:
        case WEATHER_VIEW_CURRENT:
            return "m/s";
        case WEATHER_VIEW_RAIN:
            return "mm/h";
        case WEATHER_VIEW_VORTICITY:
            return "1e-5/s";
        case WEATHER_VIEW_EVAPORATION:
            return "mm/day";
        default:
            return "%";
    }
}

static void FormatWeatherChartValue(WeatherViewMode mode, float normalized, char *buffer, int bufferSize)
{
    float value = WeatherChartDisplayValue(mode, normalized);
    const char *unit = WeatherChartUnit(mode);
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE:
        case WEATHER_VIEW_OCEAN_TEMP:
        case WEATHER_VIEW_PRESSURE:
        case WEATHER_VIEW_HUMIDITY:
        case WEATHER_VIEW_CLOUD:
        case WEATHER_VIEW_STORM:
        case WEATHER_VIEW_SNOW:
        case WEATHER_VIEW_BIOME:
            snprintf(buffer, (size_t)bufferSize, "%.0f %s", value, unit);
            break;
        case WEATHER_VIEW_WIND:
        case WEATHER_VIEW_CURRENT:
        case WEATHER_VIEW_RAIN:
        case WEATHER_VIEW_VORTICITY:
        case WEATHER_VIEW_EVAPORATION:
            snprintf(buffer, (size_t)bufferSize, "%.1f %s", value, unit);
            break;
        default:
            snprintf(buffer, (size_t)bufferSize, "%.0f %s", value, unit);
            break;
    }
}

static float WeatherChartMinimumSpan(WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE:
        case WEATHER_VIEW_OCEAN_TEMP:
            return 8.0f / 90.0f;
        case WEATHER_VIEW_PRESSURE:
            return 24.0f / 525.0f;
        case WEATHER_VIEW_WIND:
            return 6.0f / 45.0f;
        case WEATHER_VIEW_CURRENT:
            return 0.55f / 3.0f;
        case WEATHER_VIEW_RAIN:
            return 1.0f / 7.0f;
        case WEATHER_VIEW_VORTICITY:
            return 1.2f / 8.0f;
        case WEATHER_VIEW_EVAPORATION:
            return 1.6f / 12.0f;
        default:
            return 0.14f;
    }
}

static void ResetClimateChartHistory(ClimateChartHistory *history)
{
    memset(history, 0, sizeof(*history));
    history->lastBin = -1;
}

static void UpdateClimateChartHistory(ClimateChartHistory *history, const WeatherCell *weather, int tileCount, float yearPhase)
{
    if (tileCount <= 0) return;

    float sums[WEATHER_VIEW_COUNT] = { 0 };
    float weights[WEATHER_VIEW_COUNT] = { 0 };
    for (int i = 0; i < tileCount; i++) {
        for (int mode = 0; mode < WEATHER_VIEW_COUNT; mode++) {
            float weight = 1.0f;
            if (mode == WEATHER_VIEW_CURRENT || mode == WEATHER_VIEW_OCEAN_TEMP) weight = ClampFloat(weather[i].ocean, 0.05f, 1.0f);
            sums[mode] += WeatherChartMetricValue(&weather[i], (WeatherViewMode)mode) * weight;
            weights[mode] += weight;
        }
    }

    int bin = (int)(Wrap01(yearPhase) * (float)CLIMATE_CHART_BINS);
    if (bin < 0) bin = 0;
    if (bin >= CLIMATE_CHART_BINS) bin = CLIMATE_CHART_BINS - 1;

    if (!history->initialized) {
        for (int mode = 0; mode < WEATHER_VIEW_COUNT; mode++) {
            float value = (weights[mode] > 0.0f) ? sums[mode] / weights[mode] : 0.0f;
            history->latest[mode] = value;
            for (int b = 0; b < CLIMATE_CHART_BINS; b++) history->values[mode][b] = value;
        }
        for (int b = 0; b < CLIMATE_CHART_BINS; b++) history->sampleCounts[b] = 1.0f;
        history->initialized = true;
    }

    float count = fminf(history->sampleCounts[bin], CLIMATE_CHART_SAMPLE_CAP);
    float blend = 1.0f / (count + 1.0f);
    for (int mode = 0; mode < WEATHER_VIEW_COUNT; mode++) {
        float value = (weights[mode] > 0.0f) ? sums[mode] / weights[mode] : 0.0f;
        history->latest[mode] = value;
        history->values[mode][bin] = LerpFloat(history->values[mode][bin], value, blend);
    }
    history->sampleCounts[bin] = fminf(count + 1.0f, CLIMATE_CHART_SAMPLE_CAP);
    history->lastBin = bin;
}

static uint64_t HashInts3(int a, int b, int c)
{
    return ((uint64_t)(uint32_t)a << 32) ^ ((uint64_t)(uint32_t)b << 1) ^ (uint64_t)(uint32_t)c;
}

static float Hash2D01(int x, int y, int seed)
{
    uint32_t h = Hash64(HashInts3(x * 73856093, y * 19349663, seed * 83492791));
    return (float)h / (float)UINT32_MAX;
}

static Vector3 SunLightDirection(const SolarState *solar)
{
    return solar->lightDir;
}

static float MaxSurfaceRadius(const Tile *tiles, int tileCount)
{
    float maxRadius = PLANET_RADIUS;
    for (int i = 0; i < tileCount; i++) {
        float centerRadius = Vector3Length(tiles[i].center);
        if (centerRadius > maxRadius) maxRadius = centerRadius;
        for (int j = 0; j < tiles[i].cornerCount; j++) {
            float cornerRadius = Vector3Length(tiles[i].corners[j]);
            if (cornerRadius > maxRadius) maxRadius = cornerRadius;
        }
    }
    return maxRadius;
}

static Vector3 AtmosphereScatterCoefficients(float strength)
{
    float redWavelength = 700.0f;
    float greenWavelength = 530.0f;
    float blueWavelength = 440.0f;

    return (Vector3){
        powf(400.0f / redWavelength, 4.0f) * strength,
        powf(400.0f / greenWavelength, 4.0f) * strength,
        powf(400.0f / blueWavelength, 4.0f) * strength
    };
}

static void DrawBillboardDisc(Vector3 center, Vector3 right, Vector3 up, float radius, Color color, int segments)
{
    if (radius <= 0.0f || color.a == 0 || segments < 8) return;
    for (int i = 0; i < segments; i++) {
        float a0 = ((float)i / (float)segments) * 2.0f * PI;
        float a1 = ((float)(i + 1) / (float)segments) * 2.0f * PI;
        Vector3 p0 = Vector3Add(center, Vector3Add(Vector3Scale(right, cosf(a0) * radius), Vector3Scale(up, sinf(a0) * radius)));
        Vector3 p1 = Vector3Add(center, Vector3Add(Vector3Scale(right, cosf(a1) * radius), Vector3Scale(up, sinf(a1) * radius)));
        DrawTriangle3D(center, p1, p0, color);
    }
}

static void DrawSunBillboard(Camera3D camera, const SolarState *solar)
{
    Vector3 lightDir = SunLightDirection(solar);
    Vector3 sunWorld = Vector3Scale(lightDir, 120.0f);
    Vector3 toSun = Vector3Normalize(Vector3Subtract(sunWorld, camera.position));
    Vector3 cameraForward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    if (Vector3DotProduct(cameraForward, toSun) <= 0.0f) return;

    Vector3 right = Vector3Normalize(Vector3CrossProduct(cameraForward, camera.up));
    if (Vector3LengthSqr(right) < 0.000001f) right = (Vector3){ 1.0f, 0.0f, 0.0f };
    Vector3 up = Vector3Normalize(Vector3CrossProduct(right, cameraForward));

    float luminosity = fmaxf(0.01f, solar->stellarFlux * solar->orbitDistanceAu * solar->orbitDistanceAu);
    float temperatureRatio = fmaxf(0.20f, solar->stellarTemperatureK / 5778.0f);
    float starRadiusSolar = sqrtf(luminosity) / fmaxf(0.20f, temperatureRatio * temperatureRatio);
    float apparentScale = ClampFloat(starRadiusSolar / fmaxf(0.08f, solar->orbitDistanceAu), 0.26f, 4.2f);
    float coreRadius = 0.42f + apparentScale * 0.48f;
    float haloRadius = coreRadius * (4.6f + apparentScale * 0.45f);
    Color star = solar->starColor;
    Color warmCorona = LerpColor(star, (Color){ 255, 194, 112, 255 }, 0.28f);
    Color hotCore = LerpColor(star, (Color){ 255, 252, 232, 255 }, 0.72f);

    rlDisableBackfaceCulling();
    DrawBillboardDisc(sunWorld, right, up, haloRadius, (Color){ warmCorona.r, warmCorona.g, warmCorona.b, 18 }, 64);
    DrawBillboardDisc(sunWorld, right, up, coreRadius * 3.0f, (Color){ star.r, star.g, star.b, 28 }, 64);

    for (int i = 0; i < 18; i++) {
        float angle = ((float)i / 18.0f) * 2.0f * PI;
        float length = coreRadius * LerpFloat(2.8f, 5.8f, Hash2D01(i, 0, 131));
        float width = coreRadius * LerpFloat(0.045f, 0.095f, Hash2D01(i, 0, 149));
        Vector3 rayDir = Vector3Add(Vector3Scale(right, cosf(angle)), Vector3Scale(up, sinf(angle)));
        Vector3 raySide = Vector3Add(Vector3Scale(right, -sinf(angle)), Vector3Scale(up, cosf(angle)));
        Vector3 base = Vector3Add(sunWorld, Vector3Scale(rayDir, coreRadius * 1.22f));
        Vector3 tip = Vector3Add(sunWorld, Vector3Scale(rayDir, length));
        Vector3 p0 = Vector3Add(base, Vector3Scale(raySide, width));
        Vector3 p1 = Vector3Subtract(base, Vector3Scale(raySide, width));
        DrawTriangle3D(tip, p0, p1, (Color){ warmCorona.r, warmCorona.g, warmCorona.b, (unsigned char)LerpFloat(18.0f, 42.0f, Hash2D01(i, 0, 167)) });
    }

    DrawBillboardDisc(sunWorld, right, up, coreRadius * 1.36f, (Color){ star.r, star.g, star.b, 116 }, 64);
    DrawBillboardDisc(sunWorld, right, up, coreRadius * 0.82f, (Color){ hotCore.r, hotCore.g, hotCore.b, 238 }, 64);
    DrawBillboardDisc(Vector3Add(sunWorld, Vector3Add(Vector3Scale(right, -coreRadius * 0.18f), Vector3Scale(up, coreRadius * 0.18f))), right, up, coreRadius * 0.30f, (Color){ 255, 255, 250, 168 }, 32);
    rlEnableBackfaceCulling();
}

static void DrawSunOrbitGuide(Camera3D camera, const SolarState *solar)
{
    const float sunDistance = 120.0f;
    const int segments = 96;
    Vector3 cameraForward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    bool havePrev = false;
    Vector2 prevScreen = { 0 };

    for (int i = 0; i <= segments; i++) {
        float t = (float)i / (float)segments;
        float angle = t * 2.0f * PI;
        Vector3 equatorialSun = Vector3Add(
            Vector3Scale(solar->orbitRight, cosf(angle)),
            Vector3Scale(solar->orbitForward, sinf(angle))
        );
        Vector3 orbitDir = Vector3Normalize(Vector3Add(
            Vector3Scale(equatorialSun, cosf(solar->declination)),
            Vector3Scale(solar->northPole, sinf(solar->declination))
        ));
        Vector3 orbitPoint = Vector3Scale(orbitDir, sunDistance);
        Vector3 toPoint = Vector3Normalize(Vector3Subtract(orbitPoint, camera.position));
        bool visible = Vector3DotProduct(cameraForward, toPoint) > 0.0f;

        if (!visible) {
            havePrev = false;
            continue;
        }

        Vector2 screen = GetWorldToScreen(orbitPoint, camera);
        if (havePrev) {
            DrawLineEx(prevScreen, screen, 1.8f, (Color){ 108, 154, 216, 110 });
        }
        prevScreen = screen;
        havePrev = true;
    }
}

static void DrawTiltAxisGuide(const Tile *tiles, int tileCount, const SolarState *solar)
{
    float surfaceRadius = MaxSurfaceRadius(tiles, tileCount);
    float innerRadius = surfaceRadius + 0.050f;
    float outerRadius = surfaceRadius + 0.62f;
    Vector3 northInner = Vector3Scale(solar->northPole, innerRadius);
    Vector3 northOuter = Vector3Scale(solar->northPole, outerRadius);
    Vector3 southInner = Vector3Scale(solar->northPole, -innerRadius);
    Vector3 southOuter = Vector3Scale(solar->northPole, -outerRadius);
    Vector3 northTip = Vector3Scale(solar->northPole, outerRadius + 0.14f);
    Vector3 southTip = Vector3Scale(solar->northPole, -(outerRadius + 0.14f));

    DrawLine3D(southOuter, northOuter, (Color){ 226, 238, 255, 150 });
    DrawLine3D(northInner, northOuter, (Color){ 122, 190, 255, 255 });
    DrawLine3D(southInner, southOuter, (Color){ 255, 160, 120, 255 });
    DrawSphere(northTip, 0.030f, (Color){ 142, 206, 255, 235 });
    DrawSphere(southTip, 0.030f, (Color){ 255, 178, 138, 235 });

    Vector3 planetUp = (Vector3){ 0.0f, 1.0f, 0.0f };
    DrawCylinderEx(Vector3Scale(planetUp, innerRadius), Vector3Scale(planetUp, outerRadius * 0.92f), 0.010f, 0.010f, 10, (Color){ 248, 250, 255, 210 });
    DrawCylinderEx(Vector3Scale(planetUp, -innerRadius), Vector3Scale(planetUp, -outerRadius * 0.92f), 0.010f, 0.010f, 10, (Color){ 248, 250, 255, 185 });
}

static void DrawSpaceBackground(int screenWidth, int screenHeight, Camera3D camera, float clock)
{
    DrawRectangleGradientV(0, 0, screenWidth, screenHeight, (Color){ 5, 8, 18, 255 }, (Color){ 1, 2, 7, 255 });

    Vector3 cameraForward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    const int starCount = 720;
    const float starDistance = 240.0f;
    for (int i = 0; i < starCount; i++) {
        float u = Hash2D01(i, 0, 17);
        float v = Hash2D01(i, 0, 29);
        float z = 1.0f - 2.0f * u;
        float angle = 2.0f * PI * v;
        float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        Vector3 dir = { r * cosf(angle), z, r * sinf(angle) };
        if (Vector3DotProduct(cameraForward, dir) <= 0.02f) continue;

        Vector3 starWorld = Vector3Scale(dir, starDistance);
        Vector2 screen = GetWorldToScreen(starWorld, camera);
        if (screen.x < -8.0f || screen.x > (float)screenWidth + 8.0f || screen.y < -8.0f || screen.y > (float)screenHeight + 8.0f) continue;

        float brightness = Hash2D01(i, 0, 43);
        float phase = Hash2D01(i, 0, 61) * 2.0f * PI;
        float twinkle = 0.82f + 0.18f * sinf(clock * (0.45f + Hash2D01(i, 0, 73) * 1.1f) + phase);
        float hueMix = Hash2D01(i, 0, 97);
        Color cool = (Color){ 156, 194, 255, 255 };
        Color warm = (Color){ 255, 232, 198, 255 };
        Color star = LerpColor(cool, warm, hueMix * 0.42f);
        star.a = (unsigned char)ClampFloat((120.0f + brightness * 135.0f) * twinkle, 0.0f, 255.0f);

        int size = brightness > 0.985f ? 3 : (brightness > 0.940f ? 2 : 1);
        DrawRectangle((int)screen.x, (int)screen.y, size, size, star);
        if (size > 1) {
            Color glow = star;
            glow.a = (unsigned char)(star.a * 0.40f);
            DrawRectangle((int)screen.x - 2, (int)screen.y, size + 4, 1, glow);
            DrawRectangle((int)screen.x, (int)screen.y - 2, 1, size + 4, glow);
        }
    }
}

static void DrawPlanetTiles(
    const Tile *tiles,
    const WeatherCell *weather,
    int tileCount,
    bool showPlateView,
    bool weatherEnabled,
    WeatherViewMode weatherView,
    int selectedTile,
    const SolarState *solar
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
        if (!showPlateView && weather != NULL) {
            float ice = WeatherIceAmount(&weather[i]);
            if (ice > 0.01f) {
                float strength = (weatherEnabled && weatherView == WEATHER_VIEW_SNOW) ? 0.72f : 0.42f;
                fill = LerpColor(fill, WeatherIceColor(&weather[i]), ClampFloat(ice * strength, 0.0f, 0.78f));
            }
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
            Color triFill = isSelected ? fill : ShadeSurfaceColor(fill, faceNormal, showPlateView ? 0.16f : 0.44f, solar);
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
    float surfaceTemperatureC = WeatherTemperatureC(w->surfaceTemperature);
    float oceanTemperatureC = WeatherTemperatureC(w->oceanTemperature);
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

    DrawRectangle(14, 14, 460, 312, (Color){ 6, 10, 16, 198 });
    DrawRectangleLines(14, 14, 460, 312, (Color){ 175, 189, 209, 180 });

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
    snprintf(line, sizeof(line), "Surface %.1f C  Ocean %.1f C", surfaceTemperatureC, oceanTemperatureC);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Pressure %.0f hPa", pressureHpa);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Humidity %.0f%%  Cloud %.0f%%", humidityPct, cloudPct);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Rain %.1fmm/h  Wind %.1fm/s  Cur %.2fm/s", precipitationMmH, windSpeed, currentSpeed);
    DrawText(line, x, y, 18, (Color){ 210, 220, 236, 255 }); y += lh;
    snprintf(line, sizeof(line), "Vort %.1f 1e-5/s  Front %.0f%%", vorticityE5, w->frontStrength * 100.0f);
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

static void WeatherLegendPalette(WeatherViewMode mode, Color *low, Color *mid, Color *high)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE:
            *low = (Color){ 60, 108, 172, 255 };
            *mid = (Color){ 89, 158, 104, 255 };
            *high = (Color){ 209, 156, 86, 255 };
            break;
        case WEATHER_VIEW_PRESSURE:
            *low = (Color){ 60, 84, 176, 255 };
            *mid = (Color){ 57, 151, 143, 255 };
            *high = (Color){ 225, 157, 76, 255 };
            break;
        case WEATHER_VIEW_WIND:
            *low = (Color){ 31, 50, 108, 255 };
            *mid = (Color){ 78, 177, 211, 255 };
            *high = (Color){ 232, 246, 252, 255 };
            break;
        case WEATHER_VIEW_CURRENT:
            *low = (Color){ 22, 42, 86, 255 };
            *mid = (Color){ 44, 154, 214, 255 };
            *high = (Color){ 245, 236, 149, 255 };
            break;
        case WEATHER_VIEW_HUMIDITY:
            *low = (Color){ 166, 125, 65, 255 };
            *mid = (Color){ 73, 153, 98, 255 };
            *high = (Color){ 54, 122, 190, 255 };
            break;
        case WEATHER_VIEW_CLOUD:
            *low = (Color){ 36, 53, 82, 255 };
            *mid = (Color){ 103, 127, 151, 255 };
            *high = (Color){ 224, 231, 238, 255 };
            break;
        case WEATHER_VIEW_RAIN:
            *low = (Color){ 34, 47, 63, 255 };
            *mid = (Color){ 42, 133, 204, 255 };
            *high = (Color){ 226, 112, 104, 255 };
            break;
        case WEATHER_VIEW_VORTICITY:
            *low = (Color){ 58, 128, 218, 255 };
            *mid = (Color){ 42, 48, 60, 255 };
            *high = (Color){ 222, 96, 76, 255 };
            break;
        case WEATHER_VIEW_STORM:
            *low = (Color){ 36, 42, 56, 255 };
            *mid = (Color){ 102, 90, 168, 255 };
            *high = (Color){ 244, 185, 72, 255 };
            break;
        case WEATHER_VIEW_EVAPORATION:
            *low = (Color){ 38, 65, 78, 255 };
            *mid = (Color){ 66, 157, 139, 255 };
            *high = (Color){ 240, 210, 118, 255 };
            break;
        case WEATHER_VIEW_SNOW:
            *low = (Color){ 43, 66, 92, 255 };
            *mid = (Color){ 146, 181, 204, 255 };
            *high = (Color){ 248, 250, 252, 255 };
            break;
        case WEATHER_VIEW_OCEAN_TEMP:
            *low = (Color){ 22, 42, 86, 255 };
            *mid = (Color){ 44, 154, 214, 255 };
            *high = (Color){ 232, 80, 60, 255 };
            break;
        case WEATHER_VIEW_BIOME:
            *low = (Color){ 18, 62, 118, 255 };
            *mid = (Color){ 194, 174, 122, 255 };
            *high = (Color){ 34, 88, 44, 255 };
            break;
        default:
            *low = (Color){ 255, 0, 255, 255 };
            *mid = (Color){ 255, 255, 255, 255 };
            *high = (Color){ 255, 0, 255, 255 };
            break;
    }
}

static void WeatherLegendLabels(WeatherViewMode mode, char *low, int lowSize, char *mid, int midSize, char *high, int highSize)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE:
        case WEATHER_VIEW_OCEAN_TEMP:
            snprintf(low, (size_t)lowSize, "-120 C");
            snprintf(mid, (size_t)midSize, "-12 C");
            snprintf(high, (size_t)highSize, "95 C");
            break;
        case WEATHER_VIEW_PRESSURE:
            snprintf(low, (size_t)lowSize, "low 780 hPa");
            snprintf(mid, (size_t)midSize, "normal 1013");
            snprintf(high, (size_t)highSize, "high 1305");
            break;
        case WEATHER_VIEW_WIND:
            snprintf(low, (size_t)lowSize, "calm");
            snprintf(mid, (size_t)midSize, "28 m/s");
            snprintf(high, (size_t)highSize, "55+ m/s");
            break;
        case WEATHER_VIEW_CURRENT:
            snprintf(low, (size_t)lowSize, "still");
            snprintf(mid, (size_t)midSize, "1.9 m/s");
            snprintf(high, (size_t)highSize, "3.8+ m/s");
            break;
        case WEATHER_VIEW_HUMIDITY:
            snprintf(low, (size_t)lowSize, "dry");
            snprintf(mid, (size_t)midSize, "humid");
            snprintf(high, (size_t)highSize, "saturated");
            break;
        case WEATHER_VIEW_CLOUD:
            snprintf(low, (size_t)lowSize, "clear");
            snprintf(mid, (size_t)midSize, "broken");
            snprintf(high, (size_t)highSize, "overcast");
            break;
        case WEATHER_VIEW_RAIN:
            snprintf(low, (size_t)lowSize, "dry");
            snprintf(mid, (size_t)midSize, "rain");
            snprintf(high, (size_t)highSize, "downpour/front");
            break;
        case WEATHER_VIEW_VORTICITY:
            snprintf(low, (size_t)lowSize, "- spin");
            snprintf(mid, (size_t)midSize, "calm");
            snprintf(high, (size_t)highSize, "+ spin");
            break;
        case WEATHER_VIEW_STORM:
            snprintf(low, (size_t)lowSize, "stable");
            snprintf(mid, (size_t)midSize, "lift");
            snprintf(high, (size_t)highSize, "severe");
            break;
        case WEATHER_VIEW_EVAPORATION:
            snprintf(low, (size_t)lowSize, "low evap");
            snprintf(mid, (size_t)midSize, "moist flux");
            snprintf(high, (size_t)highSize, "hot/dry");
            break;
        case WEATHER_VIEW_SNOW:
            snprintf(low, (size_t)lowSize, "bare");
            snprintf(mid, (size_t)midSize, "snow");
            snprintf(high, (size_t)highSize, "ice/glacier");
            break;
        case WEATHER_VIEW_BIOME:
            snprintf(low, (size_t)lowSize, "ocean");
            snprintf(mid, (size_t)midSize, "dry/open");
            snprintf(high, (size_t)highSize, "wet forest");
            break;
        default:
            snprintf(low, (size_t)lowSize, "low");
            snprintf(mid, (size_t)midSize, "mid");
            snprintf(high, (size_t)highSize, "high");
            break;
    }
}

static void DrawWeatherColorLegend(WeatherViewMode mode, bool weatherEnabled, bool showPlateView)
{
    if (!weatherEnabled || showPlateView) return;

    float screenWidth = (float)GetScreenWidth();
    float screenHeight = (float)GetScreenHeight();
    float width = fminf(520.0f, screenWidth - 36.0f);
    float height = 66.0f;
    float bottomMargin = (screenHeight < 760.0f) ? 92.0f : 78.0f;
    Rectangle bounds = { (screenWidth - width) * 0.5f, screenHeight - height - bottomMargin, width, height };
    Rectangle bar = { bounds.x + 18.0f, bounds.y + 30.0f, bounds.width - 36.0f, 14.0f };

    Color lowColor;
    Color midColor;
    Color highColor;
    WeatherLegendPalette(mode, &lowColor, &midColor, &highColor);
    DrawRectangleRounded(bounds, 0.12f, 8, (Color){ 5, 10, 17, 214 });
    DrawRectangleRoundedLinesEx(bounds, 0.12f, 8, 1.0f, (Color){ 69, 101, 150, 190 });
    DrawText(WeatherViewName(mode), (int)bounds.x + 18, (int)bounds.y + 9, 15, (Color){ 232, 240, 252, 255 });

    int segments = 36;
    for (int i = 0; i < segments; i++) {
        float t0 = (float)i / (float)segments;
        float t1 = (float)(i + 1) / (float)segments;
        float x0 = bar.x + bar.width * t0;
        float x1 = bar.x + bar.width * t1;
        Color color = (t0 < 0.5f)
            ? LerpColor(lowColor, midColor, t0 * 2.0f)
            : LerpColor(midColor, highColor, (t0 - 0.5f) * 2.0f);
        DrawRectangle((int)x0, (int)bar.y, (int)ceilf(x1 - x0), (int)bar.height, color);
    }
    DrawRectangleLinesEx(bar, 1.0f, (Color){ 190, 212, 238, 190 });

    char low[32];
    char mid[32];
    char high[32];
    WeatherLegendLabels(mode, low, (int)sizeof(low), mid, (int)sizeof(mid), high, (int)sizeof(high));
    int midWidth = MeasureText(mid, 12);
    int highWidth = MeasureText(high, 12);
    DrawText(low, (int)bar.x, (int)(bar.y + bar.height + 6.0f), 12, (Color){ 184, 204, 230, 245 });
    DrawText(mid, (int)(bar.x + bar.width * 0.5f - midWidth * 0.5f), (int)(bar.y + bar.height + 6.0f), 12, (Color){ 184, 204, 230, 245 });
    DrawText(high, (int)(bar.x + bar.width - highWidth), (int)(bar.y + bar.height + 6.0f), 12, (Color){ 184, 204, 230, 245 });
}

static int gPanelActiveWidgetId = 0;
static int gPanelNextWidgetId = 1;
static Vector2 gClimateChartsPosition = { 28.0f, 36.0f };
static Vector2 gClimateChartsSize = { 740.0f, 560.0f };
static Vector2 gClimateChartsDragOffset = { 0.0f, 0.0f };
static bool gClimateChartsPositionInitialized = false;

static Rectangle ControlPanelBounds(void)
{
    float width = 418.0f;
    float screenHeight = (float)GetScreenHeight();
    float height = fmaxf(320.0f, screenHeight - 36.0f);
    height = fminf(height, fmaxf(120.0f, screenHeight - 36.0f));
    float x = fmaxf(18.0f, (float)GetScreenWidth() - width - 18.0f);
    return (Rectangle){ x, 18.0f, width, height };
}

static Rectangle SidebarToggleBounds(bool panelOpen)
{
    if (panelOpen) {
        Rectangle bounds = ControlPanelBounds();
        return (Rectangle){ bounds.x + bounds.width - 42.0f, bounds.y + 10.0f, 28.0f, 28.0f };
    }

    float width = 40.0f;
    float height = 118.0f;
    float x = (float)GetScreenWidth() - width - 18.0f;
    return (Rectangle){ x, 18.0f, width, height };
}

static void PanelBeginFrame(void)
{
    gPanelNextWidgetId = 1;
}

static int PanelNextWidgetId(void)
{
    return gPanelNextWidgetId++;
}

static Rectangle PanelConsumeRect(PanelLayout *layout, float height)
{
    Rectangle rect = { layout->contentX, layout->clipRect.y + layout->cursorY - layout->scrollY, layout->contentWidth, height };
    layout->cursorY += height + 6.0f;
    return rect;
}

static bool PanelIsInteractive(PanelLayout *layout, Rectangle rect)
{
    Vector2 mouse = GetMousePosition();
    return CheckCollisionPointRec(mouse, rect) && CheckCollisionPointRec(mouse, layout->clipRect);
}

static void PanelDrawSectionTitle(PanelLayout *layout, const char *title)
{
    Rectangle rect = PanelConsumeRect(layout, 24.0f);
    DrawRectangleRounded(rect, 0.06f, 6, (Color){ 19, 38, 68, 240 });
    DrawRectangleRoundedLinesEx(rect, 0.06f, 6, 1.0f, (Color){ 52, 92, 150, 210 });
    DrawText(title, (int)rect.x + 12, (int)rect.y + 3, 18, (Color){ 238, 244, 252, 255 });
}

static void PanelDrawTextRow(PanelLayout *layout, const char *left, const char *right)
{
    Rectangle rect = PanelConsumeRect(layout, 18.0f);
    DrawText(left, (int)rect.x + 2, (int)rect.y, 17, (Color){ 214, 224, 239, 255 });
    int valueWidth = MeasureText(right, 17);
    DrawText(right, (int)(rect.x + rect.width - valueWidth), (int)rect.y, 17, (Color){ 176, 201, 233, 255 });
}

static bool PanelButton(PanelLayout *layout, const char *label)
{
    Rectangle rect = PanelConsumeRect(layout, 28.0f);
    bool hovered = PanelIsInteractive(layout, rect);
    int id = PanelNextWidgetId();
    if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = id;
    bool triggered = (gPanelActiveWidgetId == id && hovered && IsMouseButtonReleased(MOUSE_BUTTON_LEFT));
    if (gPanelActiveWidgetId == id && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;

    Color fill = (gPanelActiveWidgetId == id && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) ? (Color){ 30, 74, 132, 255 }
        : hovered ? (Color){ 26, 61, 110, 255 }
        : (Color){ 18, 42, 78, 255 };
    DrawRectangleRounded(rect, 0.08f, 8, fill);
    DrawRectangleRoundedLinesEx(rect, 0.08f, 8, 1.0f, (Color){ 63, 119, 192, 220 });
    int textWidth = MeasureText(label, 17);
    DrawText(label, (int)(rect.x + rect.width * 0.5f - textWidth * 0.5f), (int)rect.y + 5, 17, (Color){ 238, 244, 252, 255 });
    return triggered;
}

static bool PanelCheckbox(PanelLayout *layout, const char *label, bool *value)
{
    Rectangle rect = PanelConsumeRect(layout, 24.0f);
    bool hovered = PanelIsInteractive(layout, rect);
    int id = PanelNextWidgetId();
    if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = id;
    bool changed = false;
    if (gPanelActiveWidgetId == id && hovered && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        *value = !*value;
        changed = true;
    }
    if (gPanelActiveWidgetId == id && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;

    Rectangle box = { rect.x + 2.0f, rect.y + 2.0f, 18.0f, 18.0f };
    DrawRectangleRounded(box, 0.12f, 6, *value ? (Color){ 44, 102, 178, 255 } : (Color){ 12, 21, 36, 255 });
    DrawRectangleRoundedLinesEx(box, 0.12f, 6, 1.0f, hovered ? (Color){ 115, 176, 244, 255 } : (Color){ 71, 111, 168, 230 });
    if (*value) {
        DrawLineEx((Vector2){ box.x + 4.0f, box.y + 10.0f }, (Vector2){ box.x + 9.0f, box.y + 15.0f }, 2.5f, (Color){ 242, 247, 255, 255 });
        DrawLineEx((Vector2){ box.x + 9.0f, box.y + 15.0f }, (Vector2){ box.x + 16.0f, box.y + 5.0f }, 2.5f, (Color){ 242, 247, 255, 255 });
    }
    DrawText(label, (int)rect.x + 32, (int)rect.y + 1, 17, hovered ? (Color){ 236, 242, 251, 255 } : (Color){ 214, 224, 239, 255 });
    return changed;
}

static bool PanelSliderFloat(PanelLayout *layout, const char *label, float *value, float minValue, float maxValue, const char *format)
{
    Rectangle rect = PanelConsumeRect(layout, 34.0f);
    Rectangle track = { rect.x, rect.y + 21.0f, rect.width, 6.0f };
    Vector2 mouse = GetMousePosition();
    int id = PanelNextWidgetId();
    bool hovered = CheckCollisionPointRec(mouse, (Rectangle){ track.x - 4.0f, rect.y, track.width + 8.0f, rect.height }) && CheckCollisionPointRec(mouse, layout->clipRect);

    if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = id;
    bool changed = false;
    if (gPanelActiveWidgetId == id) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            float t = ClampFloat((mouse.x - track.x) / track.width, 0.0f, 1.0f);
            *value = LerpFloat(minValue, maxValue, t);
            changed = true;
        } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            gPanelActiveWidgetId = 0;
        }
    }

    float t = (*value - minValue) / (maxValue - minValue);
    t = ClampFloat(t, 0.0f, 1.0f);
    float knobX = track.x + t * track.width;
    char valueText[64];
    snprintf(valueText, sizeof(valueText), format, *value);
    DrawText(label, (int)rect.x + 2, (int)rect.y, 17, (Color){ 214, 224, 239, 255 });
    int textWidth = MeasureText(valueText, 17);
    DrawText(valueText, (int)(rect.x + rect.width - textWidth), (int)rect.y, 17, (Color){ 176, 201, 233, 255 });
    DrawRectangleRounded(track, 0.5f, 8, (Color){ 12, 21, 36, 255 });
    Rectangle fill = { track.x, track.y, track.width * t, track.height };
    DrawRectangleRounded(fill, 0.5f, 8, (Color){ 47, 112, 194, 255 });
    DrawCircleV((Vector2){ knobX, track.y + track.height * 0.5f }, 6.5f,
        (gPanelActiveWidgetId == id) ? (Color){ 214, 236, 255, 255 } : hovered ? (Color){ 178, 220, 255, 255 } : (Color){ 132, 190, 244, 255 });
    return changed;
}

static void ExpandPanelSliderRange(float value, float *rangeMin, float *rangeMax, float floorValue, float ceilingValue, float minSpan)
{
    if (*rangeMax - *rangeMin < minSpan) {
        *rangeMin = value - minSpan * 0.5f;
        *rangeMax = value + minSpan * 0.5f;
    }
    if (value < *rangeMin) *rangeMin = value;
    if (value > *rangeMax) *rangeMax = value;

    float span = fmaxf(minSpan, *rangeMax - *rangeMin);
    if (value < *rangeMin + span * 0.08f) *rangeMin -= span * 0.45f;
    if (value > *rangeMax - span * 0.08f) *rangeMax += span * 0.45f;

    if (floorValue > -FLT_MAX * 0.5f && *rangeMin < floorValue) *rangeMin = floorValue;
    if (ceilingValue < FLT_MAX * 0.5f && *rangeMax > ceilingValue) *rangeMax = ceilingValue;
    if (*rangeMax - *rangeMin < minSpan) {
        float center = ClampFloat(value, *rangeMin, *rangeMax);
        *rangeMin = center - minSpan * 0.5f;
        *rangeMax = center + minSpan * 0.5f;
        if (floorValue > -FLT_MAX * 0.5f && *rangeMin < floorValue) {
            *rangeMax += floorValue - *rangeMin;
            *rangeMin = floorValue;
        }
        if (ceilingValue < FLT_MAX * 0.5f && *rangeMax > ceilingValue) {
            *rangeMin -= *rangeMax - ceilingValue;
            *rangeMax = ceilingValue;
        }
    }
}

static bool PanelSliderFloatAutoRange(
    PanelLayout *layout,
    const char *label,
    float *value,
    float *rangeMin,
    float *rangeMax,
    float floorValue,
    float ceilingValue,
    float minSpan,
    const char *format
)
{
    ExpandPanelSliderRange(*value, rangeMin, rangeMax, floorValue, ceilingValue, minSpan);
    bool changed = PanelSliderFloat(layout, label, value, *rangeMin, *rangeMax, format);
    ExpandPanelSliderRange(*value, rangeMin, rangeMax, floorValue, ceilingValue, minSpan);
    return changed;
}

static void PanelDrawWeatherViewSelector(PanelLayout *layout, WeatherViewMode *weatherView)
{
    Rectangle row = PanelConsumeRect(layout, 28.0f);
    Rectangle left = { row.x, row.y, 36.0f, row.height };
    Rectangle right = { row.x + row.width - 36.0f, row.y, 36.0f, row.height };
    Rectangle center = { left.x + left.width + 8.0f, row.y, row.width - left.width - right.width - 16.0f, row.height };
    int leftId = PanelNextWidgetId();
    bool hoverLeft = PanelIsInteractive(layout, left);
    if (hoverLeft && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = leftId;
    if (gPanelActiveWidgetId == leftId && hoverLeft && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        *weatherView = (WeatherViewMode)((*weatherView + WEATHER_VIEW_COUNT - 1) % WEATHER_VIEW_COUNT);
    }
    if (gPanelActiveWidgetId == leftId && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;

    int rightId = PanelNextWidgetId();
    bool hoverRight = PanelIsInteractive(layout, right);
    if (hoverRight && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = rightId;
    if (gPanelActiveWidgetId == rightId && hoverRight && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        *weatherView = (WeatherViewMode)((*weatherView + 1) % WEATHER_VIEW_COUNT);
    }
    if (gPanelActiveWidgetId == rightId && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;

    DrawRectangleRounded(left, 0.10f, 8, hoverLeft ? (Color){ 26, 61, 110, 255 } : (Color){ 18, 42, 78, 255 });
    DrawRectangleRounded(right, 0.10f, 8, hoverRight ? (Color){ 26, 61, 110, 255 } : (Color){ 18, 42, 78, 255 });
    DrawRectangleRounded(center, 0.08f, 8, (Color){ 12, 21, 36, 255 });
    DrawRectangleRoundedLinesEx(left, 0.10f, 8, 1.0f, (Color){ 63, 119, 192, 220 });
    DrawRectangleRoundedLinesEx(right, 0.10f, 8, 1.0f, (Color){ 63, 119, 192, 220 });
    DrawRectangleRoundedLinesEx(center, 0.08f, 8, 1.0f, (Color){ 52, 92, 150, 210 });
    DrawText("<", (int)left.x + 11, (int)left.y + 4, 17, (Color){ 238, 244, 252, 255 });
    DrawText(">", (int)right.x + 11, (int)right.y + 4, 17, (Color){ 238, 244, 252, 255 });
    const char *label = WeatherViewShortName(*weatherView);
    int fontSize = 16;
    int labelWidth = MeasureText(label, fontSize);
    if (labelWidth > (int)center.width - 18) {
        fontSize = 15;
        labelWidth = MeasureText(label, fontSize);
    }
    DrawText(label, (int)(center.x + center.width * 0.5f - labelWidth * 0.5f), (int)center.y + 5, fontSize, (Color){ 238, 244, 252, 255 });
}

static bool DrawControlPanel(
    ClimateSettings *climate,
    bool *showPlateView,
    bool *atmosphereEnabled,
    bool *weatherEnabled,
    bool *tectonicsPaused,
    WeatherViewMode *weatherView,
    bool *resetWeatherRequested,
    bool *showClimateCharts,
    const SolarState *solar
)
{
    Rectangle bounds = ControlPanelBounds();
    Vector2 mouse = GetMousePosition();
    bool hovered = climate->panelOpen && CheckCollisionPointRec(mouse, bounds);
    if (!climate->panelOpen) return false;

    Rectangle clipRect = {
        bounds.x + 16.0f,
        bounds.y + 66.0f,
        bounds.width - 44.0f,
        bounds.height - 82.0f
    };
    float maxScroll = fmaxf(0.0f, climate->panelContentHeight - clipRect.height);
    if (hovered) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) climate->panelScroll = ClampFloat(climate->panelScroll - wheel * 34.0f, 0.0f, maxScroll);
    }

    PanelBeginFrame();
    DrawRectangleRounded(bounds, 0.03f, 8, (Color){ 10, 14, 22, 240 });
    DrawRectangleRoundedLinesEx(bounds, 0.03f, 8, 1.0f, (Color){ 41, 70, 119, 220 });
    DrawRectangle((int)bounds.x, (int)bounds.y, (int)bounds.width, 52, (Color){ 16, 29, 52, 250 });
    DrawText("Controls", (int)bounds.x + 16, (int)bounds.y + 14, 26, (Color){ 240, 245, 252, 255 });

    Rectangle toggleRect = SidebarToggleBounds(true);
    bool toggleHover = CheckCollisionPointRec(mouse, toggleRect);
    if (toggleHover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = -100;
    if (gPanelActiveWidgetId == -100 && toggleHover && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) climate->panelOpen = false;
    if (gPanelActiveWidgetId == -100 && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;
    DrawRectangleRounded(toggleRect, 0.18f, 6, toggleHover ? (Color){ 30, 66, 118, 255 } : (Color){ 20, 44, 80, 255 });
    DrawRectangleRoundedLinesEx(toggleRect, 0.18f, 6, 1.0f, (Color){ 89, 144, 216, 220 });
    DrawText("X", (int)toggleRect.x + 9, (int)toggleRect.y + 5, 18, (Color){ 238, 244, 252, 255 });

    PanelLayout layout = {
        .bounds = bounds,
        .clipRect = clipRect,
        .cursorY = 0.0f,
        .contentX = clipRect.x + 6.0f,
        .contentWidth = clipRect.width - 12.0f,
        .scrollY = climate->panelScroll,
    };

    BeginScissorMode((int)clipRect.x, (int)clipRect.y, (int)clipRect.width, (int)clipRect.height);

    char summary[96];
    snprintf(summary, sizeof(summary), "Day %.2f   Year %.2f", climate->dayPhase, climate->yearPhase);
    PanelDrawTextRow(&layout, "Solar Clock", summary);
    snprintf(summary, sizeof(summary), "Sun %.0f deg", asinf(SolarFacingAmount(solar->lightDir, solar->northPole)) * RAD2DEG);
    PanelDrawTextRow(&layout, "Declination", summary);
    snprintf(summary, sizeof(summary), "%.2f AU  %.2fx flux", solar->orbitDistanceAu, solar->stellarFlux);
    PanelDrawTextRow(&layout, "Orbit", summary);
    snprintf(summary, sizeof(summary), "%s  %.1f C", ClimateOrbitStatus(solar), ClimateMeanSeaLevelTemperatureC(solar));
    PanelDrawTextRow(&layout, "Climate", summary);
    snprintf(summary, sizeof(summary), "FPS %.1f", GetFPS() * 1.0f);
    PanelDrawTextRow(&layout, "Performance", summary);

    PanelDrawSectionTitle(&layout, "Simulation");
    PanelCheckbox(&layout, "Atmosphere", atmosphereEnabled);
    PanelCheckbox(&layout, "Weather", weatherEnabled);
    PanelCheckbox(&layout, "Plate View", showPlateView);
    PanelCheckbox(&layout, "Pause Tectonics", tectonicsPaused);

    PanelDrawSectionTitle(&layout, "Time");
    PanelCheckbox(&layout, "Auto Advance Time", &climate->autoAdvanceTime);
    PanelCheckbox(&layout, "Day/Night Heating", &climate->dayNightEnabled);
    PanelCheckbox(&layout, "Seasonal Shift", &climate->seasonsEnabled);
    PanelSliderFloat(&layout, "Day Phase", &climate->dayPhase, 0.0f, 1.0f, "%.2f");
    climate->dayPhase = Wrap01(climate->dayPhase);
    PanelSliderFloat(&layout, "Year Phase", &climate->yearPhase, 0.0f, 1.0f, "%.2f");
    climate->yearPhase = Wrap01(climate->yearPhase);
    PanelSliderFloat(&layout, "Day Speed", &climate->daySpeed, 0.0f, 4.0f, "%.2fx");
    PanelSliderFloat(&layout, "Year Speed", &climate->yearSpeed, 0.0f, 4.0f, "%.2fx");

    PanelDrawSectionTitle(&layout, "Orbit");
    PanelSliderFloatAutoRange(&layout, "Orbit Distance", &climate->orbitDistanceAu, &climate->orbitDistanceRangeMin, &climate->orbitDistanceRangeMax, 0.001f, FLT_MAX, 0.20f, "%.2f AU");
    PanelSliderFloatAutoRange(&layout, "Orbit Eccentricity", &climate->orbitEccentricity, &climate->orbitEccentricityRangeMin, &climate->orbitEccentricityRangeMax, 0.0f, 0.98f, 0.08f, "%.2f");
    PanelSliderFloat(&layout, "Axial Tilt", &climate->axialTiltDegrees, 0.0f, 89.0f, "%.1f deg");

    PanelDrawSectionTitle(&layout, "Star");
    PanelSliderFloatAutoRange(&layout, "Star Luminosity", &climate->stellarLuminosity, &climate->stellarLuminosityRangeMin, &climate->stellarLuminosityRangeMax, 0.001f, FLT_MAX, 0.25f, "%.2f L");
    PanelSliderFloatAutoRange(&layout, "Star Color Temp", &climate->stellarTemperatureK, &climate->stellarTemperatureRangeMin, &climate->stellarTemperatureRangeMax, 300.0f, FLT_MAX, 400.0f, "%.0f K");
    PanelSliderFloat(&layout, "Sun Brightness", &climate->solarIntensity, 0.35f, 1.65f, "%.2fx");

    PanelDrawSectionTitle(&layout, "Climate");
    PanelSliderFloatAutoRange(&layout, "Greenhouse", &climate->greenhouseC, &climate->greenhouseRangeMin, &climate->greenhouseRangeMax, -FLT_MAX, FLT_MAX, 10.0f, "%.0f C");
    PanelSliderFloatAutoRange(&layout, "Temp Contrast", &climate->temperatureContrast, &climate->temperatureContrastRangeMin, &climate->temperatureContrastRangeMax, 0.0f, FLT_MAX, 0.20f, "%.2fx");

    PanelDrawSectionTitle(&layout, "Weather And Atmosphere");
    PanelSliderFloat(&layout, "Weather Speed", &climate->weatherTimeScale, 0.25f, 6.0f, "%.2fx");
    PanelSliderFloat(&layout, "Atmo Density", &climate->atmosphereDensityFalloff, 1.20f, 4.60f, "%.2f");
    PanelSliderFloat(&layout, "Atmo Scatter", &climate->atmosphereScatteringScale, 0.25f, 2.40f, "%.2fx");
    if (PanelButton(&layout, "Reset Weather To Current Climate")) *resetWeatherRequested = true;

    PanelDrawSectionTitle(&layout, "Views");
    PanelCheckbox(&layout, "Show Sun Orbit", &climate->showSunOrbit);
    PanelCheckbox(&layout, "Show Tilt Axis", &climate->showTiltAxis);
    PanelDrawWeatherViewSelector(&layout, weatherView);
    if (PanelButton(&layout, *showClimateCharts ? "Hide Climate Charts" : "Pop Out Climate Charts")) {
        *showClimateCharts = !*showClimateCharts;
    }

    EndScissorMode();

    climate->panelContentHeight = layout.cursorY;
    maxScroll = fmaxf(0.0f, climate->panelContentHeight - clipRect.height);
    climate->panelScroll = ClampFloat(climate->panelScroll, 0.0f, maxScroll);

    DrawRectangleRounded(clipRect, 0.03f, 6, (Color){ 0, 0, 0, 0 });
    DrawRectangleLinesEx(clipRect, 1.0f, (Color){ 28, 45, 74, 180 });
    if (maxScroll > 0.0f) {
        Rectangle bar = { bounds.x + bounds.width - 14.0f, clipRect.y, 6.0f, clipRect.height };
        float thumbHeight = fmaxf(28.0f, clipRect.height * (clipRect.height / climate->panelContentHeight));
        float thumbTravel = bar.height - thumbHeight;
        float thumbOffset = (maxScroll > 0.0f) ? (climate->panelScroll / maxScroll) * thumbTravel : 0.0f;
        Rectangle thumb = { bar.x, bar.y + thumbOffset, bar.width, thumbHeight };
        DrawRectangleRounded(bar, 0.5f, 8, (Color){ 12, 21, 36, 255 });
        DrawRectangleRounded(thumb, 0.5f, 8, hovered ? (Color){ 84, 145, 222, 255 } : (Color){ 58, 110, 182, 255 });
    }

    return hovered;
}

static bool DrawCollapsedSidebarToggle(bool *panelOpen)
{
    Rectangle rect = SidebarToggleBounds(false);
    Vector2 mouse = GetMousePosition();
    bool hovered = CheckCollisionPointRec(mouse, rect);
    if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = -101;
    bool opened = false;
    if (gPanelActiveWidgetId == -101 && hovered && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        *panelOpen = true;
        opened = true;
    }
    if (gPanelActiveWidgetId == -101 && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;

    DrawRectangleRounded(rect, 0.18f, 8, hovered ? (Color){ 18, 42, 78, 240 } : (Color){ 10, 20, 36, 240 });
    DrawRectangleRoundedLinesEx(rect, 0.18f, 8, 1.0f, (Color){ 89, 144, 216, 220 });
    DrawText(">", (int)rect.x + 12, (int)rect.y + 10, 20, (Color){ 238, 244, 252, 255 });
    DrawText("UI", (int)rect.x + 9, (int)rect.y + 42, 18, (Color){ 214, 224, 239, 255 });
    DrawText("Open", (int)rect.x + 4, (int)rect.y + 70, 16, (Color){ 176, 201, 233, 255 });
    return hovered || opened;
}

static Rectangle ClimateChartsBounds(bool collapsed)
{
    float screenWidth = (float)GetScreenWidth();
    float screenHeight = (float)GetScreenHeight();
    float minWidth = fminf(560.0f, fmaxf(360.0f, screenWidth - 36.0f));
    float minHeight = fminf(460.0f, fmaxf(390.0f, screenHeight - 36.0f));
    float maxWidth = fmaxf(minWidth, screenWidth - 36.0f);
    float maxHeight = fmaxf(minHeight, screenHeight - 36.0f);
    if (!gClimateChartsPositionInitialized) {
        gClimateChartsPosition = (Vector2){ 28.0f, 36.0f };
        gClimateChartsSize = (Vector2){ fminf(740.0f, maxWidth), fminf(560.0f, maxHeight) };
        gClimateChartsPositionInitialized = true;
    }
    gClimateChartsSize.x = ClampFloat(gClimateChartsSize.x, minWidth, maxWidth);
    gClimateChartsSize.y = ClampFloat(gClimateChartsSize.y, minHeight, maxHeight);
    float width = gClimateChartsSize.x;
    float height = collapsed ? 48.0f : gClimateChartsSize.y;
    gClimateChartsPosition.x = ClampFloat(gClimateChartsPosition.x, 8.0f, fmaxf(8.0f, screenWidth - width - 8.0f));
    gClimateChartsPosition.y = ClampFloat(gClimateChartsPosition.y, 8.0f, fmaxf(8.0f, screenHeight - height - 8.0f));
    return (Rectangle){ gClimateChartsPosition.x, gClimateChartsPosition.y, width, height };
}

static void DrawChartLineRange(Rectangle chart, const float *values, int bins, int startBin, float minValue, float maxValue, Color color, float thickness)
{
    if (bins <= 1) return;
    float span = fmaxf(0.001f, maxValue - minValue);
    Vector2 prev = { 0 };
    bool havePrev = false;
    for (int i = 0; i < bins; i++) {
        int bin = (startBin + i) % bins;
        float x = chart.x + ((float)i / (float)(bins - 1)) * chart.width;
        float normalized = ClampFloat((values[bin] - minValue) / span, 0.0f, 1.0f);
        float y = chart.y + (1.0f - normalized) * chart.height;
        Vector2 p = { x, y };
        if (havePrev) DrawLineEx(prev, p, thickness, color);
        prev = p;
        havePrev = true;
    }
}

static void DrawChartLine(Rectangle chart, const float *values, int bins, int startBin, Color color, float thickness)
{
    DrawChartLineRange(chart, values, bins, startBin, 0.0f, 1.0f, color, thickness);
}

static void DrawChartFrame(Rectangle chart, int currentBin)
{
    DrawRectangleRounded(chart, 0.035f, 6, (Color){ 8, 15, 27, 235 });
    DrawRectangleRoundedLinesEx(chart, 0.035f, 6, 1.0f, (Color){ 55, 83, 128, 210 });
    for (int i = 1; i < 4; i++) {
        float y = chart.y + chart.height * ((float)i / 4.0f);
        DrawLineEx((Vector2){ chart.x, y }, (Vector2){ chart.x + chart.width, y }, 1.0f, (Color){ 54, 70, 96, 95 });
    }
    for (int i = 1; i < 4; i++) {
        float x = chart.x + chart.width * ((float)i / 4.0f);
        DrawLineEx((Vector2){ x, chart.y }, (Vector2){ x, chart.y + chart.height }, 1.0f, (Color){ 54, 70, 96, 75 });
    }
    if (currentBin >= 0) {
        float t = (float)currentBin / (float)(CLIMATE_CHART_BINS - 1);
        float x = chart.x + t * chart.width;
        DrawLineEx((Vector2){ x, chart.y }, (Vector2){ x, chart.y + chart.height }, 2.0f, (Color){ 250, 252, 255, 120 });
    }
}

static void DrawChartRelativeLabels(Rectangle chart, const char *xLabel)
{
    DrawText("high", (int)chart.x - 38, (int)chart.y - 2, 12, (Color){ 158, 180, 208, 235 });
    DrawText("mid", (int)chart.x - 32, (int)(chart.y + chart.height * 0.5f - 6), 12, (Color){ 128, 150, 178, 220 });
    DrawText("low", (int)chart.x - 32, (int)(chart.y + chart.height - 12), 12, (Color){ 158, 180, 208, 235 });
    DrawText("relative trend", (int)chart.x + 8, (int)chart.y + 6, 13, (Color){ 178, 204, 234, 245 });
    int xWidth = MeasureText(xLabel, 13);
    DrawText(xLabel, (int)(chart.x + chart.width - xWidth), (int)(chart.y + chart.height + 8), 13, (Color){ 178, 204, 234, 245 });
}

static void ClimateChartRange(const ClimateChartHistory *history, WeatherViewMode mode, float *minOut, float *maxOut)
{
    int selected = (int)mode;
    float minValue = history->initialized ? history->values[selected][0] : history->latest[selected];
    float maxValue = minValue;
    for (int i = 0; i < CLIMATE_CHART_BINS; i++) {
        float value = history->values[selected][i];
        minValue = fminf(minValue, value);
        maxValue = fmaxf(maxValue, value);
    }

    float minimumSpan = WeatherChartMinimumSpan(mode);
    float span = maxValue - minValue;
    float center = (minValue + maxValue) * 0.5f;
    if (span < minimumSpan) {
        minValue = center - minimumSpan * 0.5f;
        maxValue = center + minimumSpan * 0.5f;
    } else {
        float padding = span * 0.16f;
        minValue -= padding;
        maxValue += padding;
    }

    if (minValue < 0.0f) {
        maxValue = fminf(1.0f, maxValue - minValue);
        minValue = 0.0f;
    }
    if (maxValue > 1.0f) {
        minValue = fmaxf(0.0f, minValue - (maxValue - 1.0f));
        maxValue = 1.0f;
    }
    if (maxValue - minValue < 0.001f) maxValue = fminf(1.0f, minValue + 0.001f);

    *minOut = minValue;
    *maxOut = maxValue;
}

static void DrawChartMetricLabelsRange(Rectangle chart, WeatherViewMode mode, float minValue, float maxValue, const char *xLabel)
{
    char top[32];
    char mid[32];
    char low[32];
    float midValue = (minValue + maxValue) * 0.5f;
    FormatWeatherChartValue(mode, maxValue, top, (int)sizeof(top));
    FormatWeatherChartValue(mode, midValue, mid, (int)sizeof(mid));
    FormatWeatherChartValue(mode, minValue, low, (int)sizeof(low));
    int topWidth = MeasureText(top, 12);
    int midWidth = MeasureText(mid, 12);
    int lowWidth = MeasureText(low, 12);
    DrawText(top, (int)(chart.x - topWidth - 8), (int)chart.y - 2, 12, (Color){ 158, 180, 208, 235 });
    DrawText(mid, (int)(chart.x - midWidth - 8), (int)(chart.y + chart.height * 0.5f - 6), 12, (Color){ 128, 150, 178, 220 });
    DrawText(low, (int)(chart.x - lowWidth - 8), (int)(chart.y + chart.height - 12), 12, (Color){ 158, 180, 208, 235 });
    char label[80];
    snprintf(label, sizeof(label), "%s (%s, adaptive)", WeatherViewShortName(mode), WeatherChartUnit(mode));
    DrawText(label, (int)chart.x + 8, (int)chart.y + 6, 13, (Color){ 178, 204, 234, 245 });
    int xWidth = MeasureText(xLabel, 13);
    DrawText(xLabel, (int)(chart.x + chart.width - xWidth), (int)(chart.y + chart.height + 8), 13, (Color){ 178, 204, 234, 245 });
}

static const char *WeatherChartDescription(WeatherViewMode mode)
{
    switch (mode) {
        case WEATHER_VIEW_TEMPERATURE: return "global air temperature index";
        case WEATHER_VIEW_PRESSURE: return "surface pressure index";
        case WEATHER_VIEW_WIND: return "wind speed index";
        case WEATHER_VIEW_CURRENT: return "ocean current speed index";
        case WEATHER_VIEW_HUMIDITY: return "relative humidity index";
        case WEATHER_VIEW_CLOUD: return "cloud and condensate coverage";
        case WEATHER_VIEW_RAIN: return "rainfall and frontal precipitation";
        case WEATHER_VIEW_VORTICITY: return "cyclonic/rotational activity";
        case WEATHER_VIEW_STORM: return "storm lift and instability";
        case WEATHER_VIEW_EVAPORATION: return "evaporation and drying";
        case WEATHER_VIEW_SNOW: return "snow and ice coverage";
        case WEATHER_VIEW_OCEAN_TEMP: return "ocean mixed-layer temperature";
        case WEATHER_VIEW_BIOME: return "land greenness/moisture index";
        default: return "weather index";
    }
}

static bool ClimateMetricRow(Rectangle rect, WeatherViewMode mode, WeatherViewMode *selectedMode, const ClimateChartHistory *history)
{
    Vector2 mouse = GetMousePosition();
    bool hovered = CheckCollisionPointRec(mouse, rect);
    int id = -300 - (int)mode;
    if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = id;
    bool selected = (*selectedMode == mode);
    bool triggered = false;
    if (gPanelActiveWidgetId == id && hovered && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        *selectedMode = mode;
        selected = true;
        triggered = true;
    }
    if (gPanelActiveWidgetId == id && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;

    Color fill = selected ? (Color){ 25, 58, 104, 242 }
        : hovered ? (Color){ 18, 38, 68, 238 }
        : (Color){ 10, 20, 34, 228 };
    DrawRectangleRounded(rect, 0.08f, 6, fill);
    DrawRectangleRoundedLinesEx(rect, 0.08f, 6, 1.0f, selected ? (Color){ 112, 172, 238, 220 } : (Color){ 42, 66, 104, 180 });
    Color color = WeatherChartLineColor(mode);
    DrawRectangle((int)rect.x + 9, (int)rect.y + 9, 10, 10, color);
    DrawText(WeatherViewShortName(mode), (int)rect.x + 26, (int)rect.y + 5, 14, (Color){ 226, 236, 250, 255 });
    char latest[24];
    float latestValue = history->initialized ? history->latest[(int)mode] : 0.0f;
    FormatWeatherChartValue(mode, latestValue, latest, (int)sizeof(latest));
    int latestWidth = MeasureText(latest, 13);
    DrawText(latest, (int)(rect.x + rect.width - latestWidth - 10), (int)rect.y + 7, 13, (Color){ 166, 194, 226, 255 });
    return triggered;
}

static bool DrawClimateChartsPopup(
    bool *open,
    bool *collapsed,
    const ClimateChartHistory *history,
    WeatherViewMode *selectedMode,
    float yearPhase
)
{
    if (!*open) return false;

    Rectangle bounds = ClimateChartsBounds(*collapsed);
    Vector2 mouse = GetMousePosition();
    bool hovered = CheckCollisionPointRec(mouse, bounds);
    int dragId = -221;
    int resizeId = -222;
    int collapseId = -223;
    Rectangle dragRect = { bounds.x, bounds.y, bounds.width - 82.0f, 48.0f };
    Rectangle resizeRect = { bounds.x + bounds.width - 24.0f, bounds.y + bounds.height - 24.0f, 18.0f, 18.0f };
    bool dragHover = CheckCollisionPointRec(mouse, dragRect);
    bool resizeHover = !*collapsed && CheckCollisionPointRec(mouse, resizeRect);
    if (resizeHover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        gPanelActiveWidgetId = resizeId;
        gClimateChartsDragOffset = (Vector2){ bounds.x + bounds.width - mouse.x, bounds.y + bounds.height - mouse.y };
    } else if (dragHover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        gPanelActiveWidgetId = dragId;
        gClimateChartsDragOffset = (Vector2){ mouse.x - bounds.x, mouse.y - bounds.y };
    }
    if (gPanelActiveWidgetId == dragId) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            gClimateChartsPosition = (Vector2){ mouse.x - gClimateChartsDragOffset.x, mouse.y - gClimateChartsDragOffset.y };
            bounds = ClimateChartsBounds(*collapsed);
            dragRect = (Rectangle){ bounds.x, bounds.y, bounds.width - 82.0f, 48.0f };
            resizeRect = (Rectangle){ bounds.x + bounds.width - 24.0f, bounds.y + bounds.height - 24.0f, 18.0f, 18.0f };
        } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            gPanelActiveWidgetId = 0;
        }
    }
    if (gPanelActiveWidgetId == resizeId) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            gClimateChartsSize = (Vector2){
                mouse.x - bounds.x + gClimateChartsDragOffset.x,
                mouse.y - bounds.y + gClimateChartsDragOffset.y
            };
            bounds = ClimateChartsBounds(*collapsed);
            dragRect = (Rectangle){ bounds.x, bounds.y, bounds.width - 82.0f, 48.0f };
            resizeRect = (Rectangle){ bounds.x + bounds.width - 24.0f, bounds.y + bounds.height - 24.0f, 18.0f, 18.0f };
        } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            gPanelActiveWidgetId = 0;
        }
    }
    int closeId = -220;
    Rectangle closeRect = { bounds.x + bounds.width - 38.0f, bounds.y + 10.0f, 26.0f, 26.0f };
    Rectangle collapseRect = { closeRect.x - 34.0f, bounds.y + 10.0f, 26.0f, 26.0f };
    bool closeHover = CheckCollisionPointRec(mouse, closeRect);
    bool collapseHover = CheckCollisionPointRec(mouse, collapseRect);
    if (collapseHover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = collapseId;
    if (gPanelActiveWidgetId == collapseId && collapseHover && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) *collapsed = !*collapsed;
    if (gPanelActiveWidgetId == collapseId && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;
    if (closeHover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = closeId;
    if (gPanelActiveWidgetId == closeId && closeHover && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) *open = false;
    if (gPanelActiveWidgetId == closeId && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) gPanelActiveWidgetId = 0;

    DrawRectangleRounded(bounds, 0.025f, 8, (Color){ 7, 10, 17, 246 });
    DrawRectangleRoundedLinesEx(bounds, 0.025f, 8, 1.0f, (Color){ 66, 99, 152, 230 });
    DrawRectangle((int)bounds.x, (int)bounds.y, (int)bounds.width, 48, (Color){ 15, 29, 52, 250 });
    DrawText("Planet Weather Cycles", (int)bounds.x + 18, (int)bounds.y + 13, 24, (Color){ 242, 247, 255, 255 });
    char phaseText[64];
    int dayOfYear = 1 + (int)(Wrap01(yearPhase) * 365.0f);
    if (dayOfYear > 365) dayOfYear = 365;
    snprintf(phaseText, sizeof(phaseText), "Day %d", dayOfYear);
    int phaseWidth = MeasureText(phaseText, 16);
    DrawText(phaseText, (int)(collapseRect.x - phaseWidth - 18), (int)bounds.y + 18, 16, (Color){ 172, 198, 232, 255 });
    DrawRectangleRounded(collapseRect, 0.18f, 6, collapseHover ? (Color){ 38, 78, 132, 255 } : (Color){ 20, 44, 80, 255 });
    DrawText(*collapsed ? "+" : "-", (int)collapseRect.x + 8, (int)collapseRect.y + 3, 19, (Color){ 238, 244, 252, 255 });
    DrawRectangleRounded(closeRect, 0.18f, 6, closeHover ? (Color){ 38, 78, 132, 255 } : (Color){ 20, 44, 80, 255 });
    DrawText("X", (int)closeRect.x + 8, (int)closeRect.y + 4, 17, (Color){ 238, 244, 252, 255 });
    if (*collapsed) return hovered;

    DrawLineEx((Vector2){ resizeRect.x + 4.0f, resizeRect.y + 16.0f }, (Vector2){ resizeRect.x + 16.0f, resizeRect.y + 4.0f }, 2.0f, resizeHover ? (Color){ 210, 232, 255, 255 } : (Color){ 112, 150, 196, 220 });
    DrawLineEx((Vector2){ resizeRect.x + 9.0f, resizeRect.y + 16.0f }, (Vector2){ resizeRect.x + 16.0f, resizeRect.y + 9.0f }, 2.0f, resizeHover ? (Color){ 210, 232, 255, 255 } : (Color){ 112, 150, 196, 220 });

    float margin = 18.0f;
    int selected = (int)(*selectedMode);
    int currentBin = (int)(Wrap01(yearPhase) * (float)CLIMATE_CHART_BINS);
    if (currentBin >= CLIMATE_CHART_BINS) currentBin = CLIMATE_CHART_BINS - 1;
    int currentMarker = CLIMATE_CHART_BINS - 1;

    Rectangle summary = { bounds.x + margin + 42.0f, bounds.y + 86.0f, bounds.width - margin * 2.0f - 42.0f, 106.0f };
    DrawText("Summary", (int)summary.x, (int)summary.y - 24, 17, (Color){ 220, 230, 244, 255 });
    DrawChartFrame(summary, currentMarker);
    char dayLabel[64];
    snprintf(dayLabel, sizeof(dayLabel), "year cycle -> day %d", dayOfYear);
    DrawChartRelativeLabels(summary, dayLabel);

    int startBin = (currentBin + 1) % CLIMATE_CHART_BINS;
    const WeatherViewMode summaryModes[] = {
        WEATHER_VIEW_TEMPERATURE,
        WEATHER_VIEW_HUMIDITY,
        WEATHER_VIEW_CLOUD,
        WEATHER_VIEW_RAIN,
        WEATHER_VIEW_WIND,
        WEATHER_VIEW_SNOW
    };
    if (history->initialized) {
        for (int i = 0; i < (int)(sizeof(summaryModes) / sizeof(summaryModes[0])); i++) {
            int mode = (int)summaryModes[i];
            Color color = WeatherChartLineColor(summaryModes[i]);
            color.a = 205;
            DrawChartLine(summary, history->values[mode], CLIMATE_CHART_BINS, startBin, color, 1.8f);
        }
    }

    float legendX = summary.x;
    float legendY = summary.y + summary.height + 18.0f;
    for (int i = 0; i < (int)(sizeof(summaryModes) / sizeof(summaryModes[0])); i++) {
        WeatherViewMode mode = summaryModes[i];
        Color color = WeatherChartLineColor(mode);
        Rectangle swatch = { legendX, legendY, 9.0f, 9.0f };
        DrawRectangleRec(swatch, color);
        DrawText(WeatherViewShortName(mode), (int)swatch.x + 13, (int)swatch.y - 3, 12, (Color){ 178, 198, 224, 235 });
        legendX += 96.0f;
    }

    float contentTop = legendY + 34.0f;
    float listWidth = 230.0f;
    Rectangle listBounds = { bounds.x + margin, contentTop, listWidth, bounds.y + bounds.height - contentTop - margin };
    Rectangle detail = { listBounds.x + listWidth + 72.0f, contentTop + 64.0f, bounds.x + bounds.width - (listBounds.x + listWidth + 72.0f) - margin, listBounds.height - 104.0f };

    DrawText("Metric", (int)listBounds.x, (int)listBounds.y - 24, 17, (Color){ 220, 230, 244, 255 });
    DrawText("latest", (int)(listBounds.x + listBounds.width - 46), (int)listBounds.y - 21, 12, (Color){ 148, 174, 204, 235 });
    float rowStep = fminf(25.0f, listBounds.height / (float)WEATHER_VIEW_COUNT);
    float rowHeight = fmaxf(16.0f, rowStep - 3.0f);
    for (int mode = 0; mode < WEATHER_VIEW_COUNT; mode++) {
        Rectangle row = { listBounds.x, listBounds.y + (float)mode * rowStep, listBounds.width, rowHeight };
        ClimateMetricRow(row, (WeatherViewMode)mode, selectedMode, history);
    }

    selected = (int)(*selectedMode);
    Color selectedColor = WeatherChartLineColor(*selectedMode);
    DrawRectangle((int)detail.x, (int)contentTop, 12, 12, selectedColor);
    DrawText(WeatherViewName(*selectedMode), (int)detail.x + 18, (int)contentTop - 4, 20, (Color){ 238, 244, 252, 255 });
    DrawText(WeatherChartDescription(*selectedMode), (int)detail.x, (int)contentTop + 20, 14, (Color){ 164, 190, 220, 245 });
    char latestText[64];
    float latestValue = history->initialized ? history->latest[selected] : 0.0f;
    char valueText[32];
    FormatWeatherChartValue(*selectedMode, latestValue, valueText, (int)sizeof(valueText));
    snprintf(latestText, sizeof(latestText), "current %s", valueText);
    int latestWidth = MeasureText(latestText, 15);
    DrawText(latestText, (int)(detail.x + detail.width - latestWidth), (int)contentTop + 2, 15, (Color){ 190, 214, 242, 255 });

    float rangeMin = 0.0f;
    float rangeMax = 1.0f;
    ClimateChartRange(history, *selectedMode, &rangeMin, &rangeMax);
    DrawChartFrame(detail, currentMarker);
    DrawChartMetricLabelsRange(detail, *selectedMode, rangeMin, rangeMax, dayLabel);
    if (history->initialized) DrawChartLineRange(detail, history->values[selected], CLIMATE_CHART_BINS, startBin, rangeMin, rangeMax, selectedColor, 3.0f);

    return hovered;
}

static void DrawWeatherClouds(
    const Tile *tiles,
    const WeatherCell *weather,
    int tileCount,
    WeatherViewMode mode
)
{
    bool focusMode = (mode == WEATHER_VIEW_CLOUD || mode == WEATHER_VIEW_RAIN || mode == WEATHER_VIEW_STORM);
    rlDisableBackfaceCulling();
    for (int i = 0; i < tileCount; i++) {
        const Tile *tile = &tiles[i];
        float cloud = fmaxf(weather[i].cloud, weather[i].cloudWater * 1.34f);
        float storm = fmaxf(weather[i].precipitation, weather[i].storm);
        float breakup = SphericalNoise3(weather[i].normal, 18.0f, (Vector3){ (float)GetTime() * 0.020f, 37.0f, -19.0f }, 2, 2.1f, 0.50f);
        float cellular = HashFromDirection(weather[i].normal, 701.0f);
        float cloudShape = ClampFloat(cloud + (breakup - 0.5f) * 0.30f + (cellular - 0.5f) * 0.12f, 0.0f, 1.0f);
        float alphaBlend = cloudShape * (focusMode ? 0.86f : 0.44f) + storm * (focusMode ? 0.42f : 0.20f) - (focusMode ? 0.16f : 0.24f);
        if (alphaBlend <= 0.0f) continue;

        unsigned char alpha = (unsigned char)(ClampFloat(alphaBlend, 0.0f, 1.0f) * (focusMode ? 178.0f : 112.0f));
        Color fairCloud = LerpColor((Color){ 250, 252, 255, 255 }, (Color){ 224, 231, 238, 255 }, ClampFloat(cloudShape, 0.0f, 1.0f));
        Color stormCloud = LerpColor((Color){ 168, 178, 190, 255 }, (Color){ 86, 92, 108, 255 }, ClampFloat(storm * 1.2f, 0.0f, 1.0f));
        Color cloudColor = LerpColor(fairCloud, stormCloud, ClampFloat(storm * 0.85f + weather[i].frontStrength * 0.20f, 0.0f, 1.0f));
        cloudColor.a = alpha;

        float lift = CLOUD_LAYER_HEIGHT + cloudShape * 0.010f + storm * 0.010f + weather[i].frontStrength * 0.004f;
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

    Shader atmosphereShader = LoadShader("shaders/atmosphere.vs", "shaders/atmosphere.fs");
    int atmosphereScreenSizeLoc = GetShaderLocation(atmosphereShader, "screenSize");
    int atmosphereCameraPosLoc = GetShaderLocation(atmosphereShader, "cameraPos");
    int atmosphereCameraForwardLoc = GetShaderLocation(atmosphereShader, "cameraForward");
    int atmosphereCameraRightLoc = GetShaderLocation(atmosphereShader, "cameraRight");
    int atmosphereCameraUpLoc = GetShaderLocation(atmosphereShader, "cameraUp");
    int atmosphereCameraFovYLoc = GetShaderLocation(atmosphereShader, "cameraFovY");
    int atmosphereAspectRatioLoc = GetShaderLocation(atmosphereShader, "aspectRatio");
    int atmospherePlanetRadiusLoc = GetShaderLocation(atmosphereShader, "planetRadius");
    int atmosphereAtmosphereRadiusLoc = GetShaderLocation(atmosphereShader, "atmosphereRadius");
    int atmosphereLightDirLoc = GetShaderLocation(atmosphereShader, "lightDir");
    int atmosphereStarColorLoc = GetShaderLocation(atmosphereShader, "starColor");
    int atmosphereScatteringCoefficientsLoc = GetShaderLocation(atmosphereShader, "scatteringCoefficients");
    int atmosphereDensityFalloffLoc = GetShaderLocation(atmosphereShader, "densityFalloff");
    int atmosphereScatteringStrengthLoc = GetShaderLocation(atmosphereShader, "scatteringStrength");

    ClimateSettings climate = {
        .panelOpen = true,
        .autoAdvanceTime = true,
        .dayNightEnabled = true,
        .seasonsEnabled = true,
        .showSunOrbit = false,
        .showTiltAxis = false,
        .dayPhase = 0.18f,
        .yearPhase = 0.08f,
        .daySpeed = 1.0f,
        .yearSpeed = 1.0f,
        .axialTiltDegrees = 23.5f,
        .solarIntensity = 1.0f,
        .orbitDistanceAu = 1.0f,
        .orbitEccentricity = 0.0167f,
        .stellarLuminosity = 1.0f,
        .stellarTemperatureK = 5778.0f,
        .greenhouseC = 0.0f,
        .orbitDistanceRangeMin = 0.20f,
        .orbitDistanceRangeMax = 5.00f,
        .orbitEccentricityRangeMin = 0.0f,
        .orbitEccentricityRangeMax = 0.65f,
        .stellarLuminosityRangeMin = 0.05f,
        .stellarLuminosityRangeMax = 5.00f,
        .stellarTemperatureRangeMin = 1800.0f,
        .stellarTemperatureRangeMax = 12000.0f,
        .greenhouseRangeMin = -30.0f,
        .greenhouseRangeMax = 65.0f,
        .temperatureContrastRangeMin = 0.50f,
        .temperatureContrastRangeMax = 1.80f,
        .weatherTimeScale = WEATHER_TIME_SCALE,
        .temperatureContrast = 1.0f,
        .atmosphereDensityFalloff = ATMOSPHERE_DENSITY_FALLOFF,
        .atmosphereScatteringScale = 1.0f,
        .panelScroll = 0.0f,
        .panelContentHeight = 0.0f,
    };
    SolarState solar = BuildSolarState(&climate);

    Vector3 atmosphereScatterCoefficients = AtmosphereScatterCoefficients(ATMOSPHERE_SCATTERING_STRENGTH);
    Vector3 sunLightDirection = SunLightDirection(&solar);
    Vector3 starColor = { (float)solar.starColor.r / 255.0f, (float)solar.starColor.g / 255.0f, (float)solar.starColor.b / 255.0f };
    float atmospherePlanetRadius = PLANET_RADIUS;
    float atmosphereOuterRadius = PLANET_RADIUS + ATMOSPHERE_SURFACE_MARGIN;
    float atmosphereDensityFalloff = climate.atmosphereDensityFalloff;
    float atmosphereScatteringStrength = climate.atmosphereScatteringScale;

    SetShaderValue(atmosphereShader, atmospherePlanetRadiusLoc, &atmospherePlanetRadius, SHADER_UNIFORM_FLOAT);
    SetShaderValue(atmosphereShader, atmosphereAtmosphereRadiusLoc, &atmosphereOuterRadius, SHADER_UNIFORM_FLOAT);
    SetShaderValue(atmosphereShader, atmosphereLightDirLoc, &sunLightDirection.x, SHADER_UNIFORM_VEC3);
    SetShaderValue(atmosphereShader, atmosphereStarColorLoc, &starColor.x, SHADER_UNIFORM_VEC3);
    SetShaderValue(atmosphereShader, atmosphereScatteringCoefficientsLoc, &atmosphereScatterCoefficients.x, SHADER_UNIFORM_VEC3);
    SetShaderValue(atmosphereShader, atmosphereDensityFalloffLoc, &atmosphereDensityFalloff, SHADER_UNIFORM_FLOAT);
    SetShaderValue(atmosphereShader, atmosphereScatteringStrengthLoc, &atmosphereScatteringStrength, SHADER_UNIFORM_FLOAT);

    RenderTexture2D sceneTexture = LoadRenderTexture(GetScreenWidth(), GetScreenHeight());

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
    InitializeWeather(weatherA, tiles, tileCount, &climate, &solar);
    memcpy(weatherB, weatherA, sizeof(WeatherCell) * (size_t)tileCount);
    ClimateChartHistory climateCharts = { 0 };
    ResetClimateChartHistory(&climateCharts);
    UpdateClimateChartHistory(&climateCharts, weatherA, tileCount, climate.yearPhase);
    atmosphereOuterRadius = MaxSurfaceRadius(tiles, tileCount) + ATMOSPHERE_SURFACE_MARGIN;
    SetShaderValue(atmosphereShader, atmosphereAtmosphereRadiusLoc, &atmosphereOuterRadius, SHADER_UNIFORM_FLOAT);

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
    bool atmosphereEnabled = true;
    bool showPlateView = false;
    bool tectonicsPaused = false;
    bool weatherEnabled = true;
    bool showClimateCharts = false;
    bool showClimateChartsCollapsed = false;
    WeatherViewMode weatherView = WEATHER_VIEW_TEMPERATURE;
    WeatherViewMode climateChartView = WEATHER_VIEW_TEMPERATURE;
    int selectedTile = -1;
    Vector2 clickStart = { 0 };
    float tectonicRebuildTimer = 0.0f;
    const float tectonicRebuildStep = 1.0f / TECTONIC_REBUILD_HZ;
    float weatherTimer = 0.0f;
    const float weatherStep = 1.0f / WEATHER_UPDATE_HZ;
    bool resetWeatherRequested = false;

    while (!WindowShouldClose()) {
        Rectangle controlPanelRect = ControlPanelBounds();
        Rectangle sidebarToggleRect = SidebarToggleBounds(climate.panelOpen);
        bool controlPanelHovered = climate.panelOpen && CheckCollisionPointRec(GetMousePosition(), controlPanelRect);
        bool sidebarToggleHovered = CheckCollisionPointRec(GetMousePosition(), sidebarToggleRect);
        bool climateChartsHovered = showClimateCharts && CheckCollisionPointRec(GetMousePosition(), ClimateChartsBounds(showClimateChartsCollapsed));
        bool climateChartsCapturing = showClimateCharts && gPanelActiveWidgetId <= -220;
        bool uiHovered = controlPanelHovered || sidebarToggleHovered || climateChartsHovered || climateChartsCapturing;
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !uiHovered) clickStart = GetMousePosition();
        if (IsKeyPressed(KEY_A)) atmosphereEnabled = !atmosphereEnabled;
        if (IsKeyPressed(KEY_C)) showPlateView = !showPlateView;
        if (IsKeyPressed(KEY_SPACE)) tectonicsPaused = !tectonicsPaused;
        if (IsKeyPressed(KEY_P)) climate.autoAdvanceTime = !climate.autoAdvanceTime;
        if (IsKeyPressed(KEY_W)) weatherEnabled = !weatherEnabled;
        if (IsKeyPressed(KEY_F1)) climate.panelOpen = !climate.panelOpen;
        if (IsKeyPressed(KEY_T)) climate.showTiltAxis = !climate.showTiltAxis;
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
        if (IsKeyPressed(KEY_R)) weatherView = WEATHER_VIEW_OCEAN_TEMP;
        if (IsKeyPressed(KEY_B)) weatherView = WEATHER_VIEW_BIOME;
        float dt = GetFrameTime();

        if (climate.autoAdvanceTime) {
            climate.dayPhase = Wrap01(climate.dayPhase + dt * 0.020f * climate.daySpeed);
            climate.yearPhase = Wrap01(climate.yearPhase + dt * 0.0035f * climate.yearSpeed);
        }
        solar = BuildSolarState(&climate);
        sunLightDirection = SunLightDirection(&solar);
        starColor = (Vector3){ (float)solar.starColor.r / 255.0f, (float)solar.starColor.g / 255.0f, (float)solar.starColor.b / 255.0f };
        atmosphereDensityFalloff = climate.atmosphereDensityFalloff;
        atmosphereScatteringStrength = climate.atmosphereScatteringScale;
        SetShaderValue(atmosphereShader, atmosphereLightDirLoc, &sunLightDirection.x, SHADER_UNIFORM_VEC3);
        SetShaderValue(atmosphereShader, atmosphereStarColorLoc, &starColor.x, SHADER_UNIFORM_VEC3);
        SetShaderValue(atmosphereShader, atmosphereDensityFalloffLoc, &atmosphereDensityFalloff, SHADER_UNIFORM_FLOAT);
        SetShaderValue(atmosphereShader, atmosphereScatteringStrengthLoc, &atmosphereScatteringStrength, SHADER_UNIFORM_FLOAT);

        if (resetWeatherRequested) {
            InitializeWeather(weatherA, tiles, tileCount, &climate, &solar);
            memcpy(weatherB, weatherA, sizeof(WeatherCell) * (size_t)tileCount);
            ResetClimateChartHistory(&climateCharts);
            UpdateClimateChartHistory(&climateCharts, weatherA, tileCount, climate.yearPhase);
            weatherTimer = 0.0f;
            resetWeatherRequested = false;
        }

        if (!tectonicsPaused) {
            AdvancePlateSimulation(plates, plateCount, dt * TECTONIC_TIME_SCALE);
            tectonicRebuildTimer += dt;
            if (tectonicRebuildTimer >= tectonicRebuildStep) {
                tectonicRebuildTimer -= tectonicRebuildStep;
                UpdatePlanetTiles(tiles, tileCount, triangleDirections, triangleSurfacePoints, triangles.count, plates, plateCount, PLANET_RADIUS);
                atmosphereOuterRadius = MaxSurfaceRadius(tiles, tileCount) + ATMOSPHERE_SURFACE_MARGIN;
                SetShaderValue(atmosphereShader, atmosphereAtmosphereRadiusLoc, &atmosphereOuterRadius, SHADER_UNIFORM_FLOAT);
            }
        }

        if (weatherEnabled) {
            weatherTimer += dt * climate.weatherTimeScale;
            while (weatherTimer >= weatherStep) {
                RefreshWeatherSurface(weatherA, tiles, tileCount);
                StepWeatherSimulation(tileGraph, weatherA, weatherB, tileCount, weatherStep, &climate, &solar);
                WeatherCell *swap = weatherA;
                weatherA = weatherB;
                weatherB = swap;
                UpdateClimateChartHistory(&climateCharts, weatherA, tileCount, climate.yearPhase);
                weatherTimer -= weatherStep;
            }
        }

        UpdateOrbitCamera(&orbit, &camera, !uiHovered);
        if (IsWindowResized()) {
            int width = GetScreenWidth();
            int height = GetScreenHeight();
            if (width > 0 && height > 0) {
                UnloadRenderTexture(sceneTexture);
                sceneTexture = LoadRenderTexture(width, height);
            }
        }
        if (!uiHovered && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            Vector2 end = GetMousePosition();
            float dx = end.x - clickStart.x;
            float dy = end.y - clickStart.y;
            if ((dx * dx + dy * dy) <= 25.0f) {
                selectedTile = PickTileFromMouse(tiles, tileCount, camera);
            }
        }

        BeginTextureMode(sceneTexture);
        ClearBackground((Color){ 0, 0, 0, 0 });
        BeginMode3D(camera);
        DrawSunBillboard(camera, &solar);
        DrawPlanetTiles(tiles, weatherA, tileCount, showPlateView, weatherEnabled, weatherView, selectedTile, &solar);
        if (climate.showTiltAxis) DrawTiltAxisGuide(tiles, tileCount, &solar);
        if (!showPlateView && weatherEnabled) DrawWeatherClouds(tiles, weatherA, tileCount, weatherView);
        if (!showPlateView && weatherEnabled && weatherView == WEATHER_VIEW_WIND) DrawWindVectors(tiles, weatherA, tileCount);
        if (!showPlateView && weatherEnabled && weatherView == WEATHER_VIEW_CURRENT) DrawCurrentVectors(tiles, weatherA, tileCount);
        EndMode3D();
        EndTextureMode();

        int screenWidth = GetScreenWidth();
        int screenHeight = GetScreenHeight();
        Vector2 screenSize = { (float)screenWidth, (float)screenHeight };
        Vector3 cameraForward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
        Vector3 cameraRight = Vector3Normalize(Vector3CrossProduct(cameraForward, camera.up));
        Vector3 cameraTrueUp = Vector3Normalize(Vector3CrossProduct(cameraRight, cameraForward));
        float cameraFovY = camera.fovy;
        float aspectRatio = (float)screenWidth / (float)screenHeight;

        SetShaderValue(atmosphereShader, atmosphereScreenSizeLoc, &screenSize.x, SHADER_UNIFORM_VEC2);
        SetShaderValue(atmosphereShader, atmosphereCameraPosLoc, &camera.position.x, SHADER_UNIFORM_VEC3);
        SetShaderValue(atmosphereShader, atmosphereCameraForwardLoc, &cameraForward.x, SHADER_UNIFORM_VEC3);
        SetShaderValue(atmosphereShader, atmosphereCameraRightLoc, &cameraRight.x, SHADER_UNIFORM_VEC3);
        SetShaderValue(atmosphereShader, atmosphereCameraUpLoc, &cameraTrueUp.x, SHADER_UNIFORM_VEC3);
        SetShaderValue(atmosphereShader, atmosphereCameraFovYLoc, &cameraFovY, SHADER_UNIFORM_FLOAT);
        SetShaderValue(atmosphereShader, atmosphereAspectRatioLoc, &aspectRatio, SHADER_UNIFORM_FLOAT);

        Rectangle source = { 0.0f, 0.0f, (float)sceneTexture.texture.width, -(float)sceneTexture.texture.height };
        Rectangle destination = { 0.0f, 0.0f, (float)screenWidth, (float)screenHeight };

        BeginDrawing();
        DrawSpaceBackground(screenWidth, screenHeight, camera, (float)GetTime());
        if (climate.showSunOrbit) DrawSunOrbitGuide(camera, &solar);
        if (atmosphereEnabled) {
            BeginShaderMode(atmosphereShader);
            DrawTexturePro(sceneTexture.texture, source, destination, (Vector2){ 0.0f, 0.0f }, 0.0f, WHITE);
            EndShaderMode();
        } else {
            DrawTexturePro(sceneTexture.texture, source, destination, (Vector2){ 0.0f, 0.0f }, 0.0f, WHITE);
        }

        DrawSelectedTileInfo(tiles, plates, weatherA, tileCount, selectedTile, tectonicsPaused, weatherEnabled, weatherView);
        DrawWeatherColorLegend(weatherView, weatherEnabled, showPlateView);
        if (climate.panelOpen) {
            DrawControlPanel(&climate, &showPlateView, &atmosphereEnabled, &weatherEnabled, &tectonicsPaused, &weatherView, &resetWeatherRequested, &showClimateCharts, &solar);
        } else {
            DrawCollapsedSidebarToggle(&climate.panelOpen);
        }
        DrawClimateChartsPopup(&showClimateCharts, &showClimateChartsCollapsed, &climateCharts, &climateChartView, climate.yearPhase);

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
    UnloadRenderTexture(sceneTexture);
    UnloadShader(atmosphereShader);

    CloseWindow();
    return 0;
}
