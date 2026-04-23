#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

uniform sampler2D texture0;
uniform vec2 screenSize;
uniform vec3 cameraPos;
uniform vec3 cameraForward;
uniform vec3 cameraRight;
uniform vec3 cameraUp;
uniform float cameraFovY;
uniform float aspectRatio;
uniform float planetRadius;
uniform float atmosphereRadius;
uniform vec3 lightDir;
uniform vec3 scatteringCoefficients;
uniform float densityFalloff;
uniform float scatteringStrength;

const int INSCATTER_POINTS = 10;
const int OPTICAL_DEPTH_POINTS = 8;

float rayleighPhase(float cosTheta)
{
    return 0.75 * (1.0 + cosTheta * cosTheta);
}

vec2 raySphere(vec3 rayOrigin, vec3 rayDir, float sphereRadius)
{
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - sphereRadius * sphereRadius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) return vec2(1e9, 0.0);

    float s = sqrt(discriminant);
    float nearHit = -b - s;
    float farHit = -b + s;
    if (farHit < 0.0) return vec2(1e9, 0.0);

    float distanceToSphere = max(nearHit, 0.0);
    float distanceThroughSphere = max(farHit - distanceToSphere, 0.0);
    return vec2(distanceToSphere, distanceThroughSphere);
}

float densityAtPoint(vec3 samplePoint)
{
    float height = length(samplePoint) - planetRadius;
    float atmosphereHeight = max(atmosphereRadius - planetRadius, 0.0001);
    float height01 = clamp(height / atmosphereHeight, 0.0, 1.0);
    float density = exp(-height01 * densityFalloff) * (1.0 - height01 * 0.55);
    return max(density, 0.0);
}

float opticalDepth(vec3 rayOrigin, vec3 rayDir, float rayLength)
{
    if (rayLength <= 0.0) return 0.0;

    float stepSize = rayLength / float(OPTICAL_DEPTH_POINTS);
    float depth = 0.0;
    vec3 samplePoint = rayOrigin + rayDir * (stepSize * 0.5);
    for (int i = 0; i < OPTICAL_DEPTH_POINTS; i++) {
        depth += densityAtPoint(samplePoint) * stepSize;
        samplePoint += rayDir * stepSize;
    }
    return depth;
}

vec3 calculateAtmosphere(vec3 rayOrigin, vec3 rayDir, float rayLength, out float totalViewDepth)
{
    float stepSize = rayLength / float(INSCATTER_POINTS);
    vec3 inScattered = vec3(0.0);
    totalViewDepth = 0.0;
    vec3 samplePoint = rayOrigin + rayDir * (stepSize * 0.5);
    float cosTheta = dot(lightDir, -rayDir);
    float phase = rayleighPhase(cosTheta);

    for (int i = 0; i < INSCATTER_POINTS; i++) {
        float localDensity = densityAtPoint(samplePoint);
        totalViewDepth += localDensity * stepSize;

        vec2 sunPlanetHit = raySphere(samplePoint, lightDir, planetRadius);
        bool sunBlocked = sunPlanetHit.y > 0.0;
        if (sunBlocked || localDensity <= 0.00001) {
            samplePoint += rayDir * stepSize;
            continue;
        }

        vec2 sunHit = raySphere(samplePoint, lightDir, atmosphereRadius);
        float sunRayLength = sunHit.y;
        float sunDepth = opticalDepth(samplePoint, lightDir, sunRayLength);
        vec3 transmittance = exp(-(sunDepth + totalViewDepth) * scatteringCoefficients);
        float horizonGlow = pow(1.0 - abs(dot(normalize(samplePoint), lightDir)), 2.2);
        vec3 horizonTint = mix(vec3(1.0), vec3(1.18, 0.94, 0.78), horizonGlow * 0.55);
        inScattered += localDensity * transmittance * scatteringCoefficients * stepSize * phase * horizonTint;
        samplePoint += rayDir * stepSize;
    }

    return inScattered * scatteringStrength;
}

void main()
{
    vec4 sceneColor = texture(texture0, fragTexCoord) * fragColor;

    vec2 screenUV = vec2(fragTexCoord.x, 1.0 - fragTexCoord.y);
    vec2 ndc = screenUV * 2.0 - 1.0;
    float tanHalfFov = tan(radians(cameraFovY) * 0.5);
    vec3 rayDir = normalize(
        cameraForward
        + cameraRight * (ndc.x * aspectRatio * tanHalfFov)
        + cameraUp * (ndc.y * tanHalfFov)
    );

    vec2 atmosphereHit = raySphere(cameraPos, rayDir, atmosphereRadius);
    if (atmosphereHit.y <= 0.0) {
        finalColor = sceneColor;
        return;
    }

    vec2 planetHit = raySphere(cameraPos, rayDir, planetRadius);
    float distanceToAtmosphere = atmosphereHit.x;
    float distanceThroughAtmosphere = atmosphereHit.y;
    if (planetHit.y > 0.0) {
        distanceThroughAtmosphere = min(distanceThroughAtmosphere, max(planetHit.x - distanceToAtmosphere, 0.0));
    }
    if (distanceThroughAtmosphere <= 0.0) {
        finalColor = sceneColor;
        return;
    }

    vec3 atmosphereStart = cameraPos + rayDir * distanceToAtmosphere;
    float totalViewDepth = 0.0;
    vec3 atmosphereLight = calculateAtmosphere(atmosphereStart, rayDir, distanceThroughAtmosphere, totalViewDepth);
    vec3 viewTransmittance = exp(-totalViewDepth * scatteringCoefficients * 0.35);
    vec3 color = sceneColor.rgb * viewTransmittance + atmosphereLight;
    float atmosphereAlpha = clamp(max(max(atmosphereLight.r, atmosphereLight.g), atmosphereLight.b), 0.0, 1.0);
    finalColor = vec4(color, max(sceneColor.a, atmosphereAlpha));
}
