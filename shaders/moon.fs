#version 330

in vec3 fragNormal;
in vec3 fragLocal;

uniform vec3 lightDir;          // direction TO the sun (already normalized)
uniform vec3 moonColor;         // lit-side base
uniform vec3 moonDarkColor;     // shadow-side base

out vec4 finalColor;

float hash3(vec3 p)
{
    p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float vnoise(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash3(i + vec3(0,0,0));
    float n100 = hash3(i + vec3(1,0,0));
    float n010 = hash3(i + vec3(0,1,0));
    float n110 = hash3(i + vec3(1,1,0));
    float n001 = hash3(i + vec3(0,0,1));
    float n101 = hash3(i + vec3(1,0,1));
    float n011 = hash3(i + vec3(0,1,1));
    float n111 = hash3(i + vec3(1,1,1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
               f.z);
}

float fbm(vec3 p)
{
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * vnoise(p);
        p *= 2.07;
        a *= 0.5;
    }
    return v;
}

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(lightDir);
    float ndotl = dot(N, L);

    // Lambert with a soft terminator (slight forward fall-off so the dark side is not pitch-black)
    float lit = smoothstep(-0.08, 0.45, ndotl);

    // Albedo variation: large maria + small craters
    float maria   = fbm(fragLocal * 2.6);
    float craters = vnoise(fragLocal * 16.0);
    float albedo  = 0.85 + (maria - 0.5) * 0.35 + (craters - 0.5) * 0.18;

    vec3 col = mix(moonDarkColor, moonColor, lit) * albedo;

    // Subtle rim self-shadow so the limb darkens
    float rim = smoothstep(0.0, 0.25, abs(ndotl));
    col *= 0.85 + 0.15 * rim;

    finalColor = vec4(col, 1.0);
}
