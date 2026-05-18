# Physics Rework Plan — `physics-rework`

The throughline: **everything in SI units, nothing clamped, all state emergent from energy/momentum balance.** Today the sim runs on normalized `[0, 1]` proxies for temperature, pressure, humidity, etc., with hard caps that prevent extremes. After this branch, temperatures fall out of the energy budget; orbits come from real Keplerian elements; moons cause real tides and eclipses; and the simulator runs fast enough at higher subdivisions thanks to threading + decoupled sim/render.

## 0. Branch + scaffolding

- Branch from `ui-imgui` → `physics-rework`.
- Add `physics/` subdirectory grouping new units/orbit/radiation/moisture modules. Keep everything still callable from `main.cpp`; no engine rewrite.
- Add a constants header `physics/constants.h`:
  - Stefan–Boltzmann `σ = 5.670374e-8 W/m²/K⁴`
  - Gas constants, R_d / R_v, c_p_dry, L_v, `g`, `Ω_planet`, `AU`, `L_sun`
- New compile flag `-O3 -march=native`.

## 1. Units pass — promote to SI, remove clamps

The single most impactful change. Existing fields:

| Field | Today | After |
|---|---|---|
| `WeatherCell.temperature` | `[0,1]` (proxy) | `T` in **Kelvin**, no clamp |
| `WeatherCell.pressure`    | `[0,1]` (proxy) | `p` in **Pascals**, no clamp |
| `WeatherCell.humidity`    | `[0,1]` (specific) | `q` (kg/kg specific humidity) |
| `WeatherCell.cloudWater`  | `[0,1]` | `q_c` (kg/kg cloud water mixing ratio) |
| `WeatherCell.precipitation` | `[0,1]` | `P` in **mm/hr** |
| `WeatherCell.wind`        | tangent vector (m/s already) | unchanged |
| `WeatherCell.oceanTemperature` | `[0,1]` | `T_sea` in **Kelvin** |
| `WeatherCell.soilMoisture` | `[0,1]` | volumetric `θ` in **m³/m³** |

**Touchpoints:**
- `WEATHER_MIN_TEMP_C` / `WEATHER_MAX_TEMP_C` / `WEATHER_PRESSURE_*_HPA` → delete.
- `WeatherTemperatureC`, `WeatherPressureHpa`, etc. → become trivial unit converters from K/Pa to display units (UI keeps showing °C and hPa).
- `WeatherChartDisplayValue`, `WeatherChartMetricValue` → simplify (no more proxy math).
- `InitializeWeather` → seed with realistic K/Pa from a standard-atmosphere lapse + meridional T profile.
- All existing `LerpFloat` / `Clamp` calls on these fields → audit; clamps gated to physical floors only (e.g. `q ≥ 0`).

Ship this first — the UI plots will then show real Kelvins and the rest of the physics work has correct units to plug into.

## 2. Radiation balance — temperatures become emergent

Replace the current temperature update (a smooth nudge toward equilibrium) with an explicit per-cell energy budget:

```
dE/dt = SW_in − LW_out + H_sens + H_lat + H_advect
T_new = T + Δt · (dE/dt) / (ρ · c_p · h_eff)
```

with:

- **Shortwave in:** `SW_in = S · max(0, n̂·ŝ) · (1 − α_surface) · τ_atm · (1 − f_cloud · α_cloud)`. `S` is instantaneous solar flux from §3, `n̂·ŝ` is the cosine of the solar zenith, `τ_atm` is a 1-band Beer–Lambert transmittance through the airmass.
- **Longwave out:** `LW_out = ε_eff · σ · T⁴` with `ε_eff = ε_clear · (1 − f_cloud) + ε_cloud · f_cloud`. Greenhouse forcing (`climate.greenhouseC`) modifies `ε_clear` so the existing slider keeps meaning.
- **Sensible / latent fluxes** between surface and atmosphere using bulk aerodynamic formulas (existing wind feeds drag coefficient).
- **Effective heat capacity** `ρ · c_p · h_eff`: large for ocean cells (mixed-layer depth), small for land, intermediate for ice. This single change explains why polar winters get to −40 °C and deserts to +50 °C without any clamping.

Hooks: surface albedo derived from existing biome/ice fields. Cloud cover from `q_c`. Same data, real physics.

## 3. Keplerian orbits

`climate.orbitDistanceAu`, `eccentricity`, `inclinationDeg`, `axialTiltDegrees` already exist (UI). Make them mean something:

- Replace `yearPhase += dt · const` with **mean-anomaly integration**: `M += n · dt` where `n = 2π / T_orbit`. `T_orbit = 2π · √(a³ / GM_star)` in real years.
- Solve Kepler `M = E − e·sin(E)` for eccentric anomaly `E` (Newton iteration, ~3 steps). Compute true anomaly `ν` and radius `r = a(1 − e·cos E)`.
- Plug `r` into `SW`: `S = L_star / (4π r²)`. Stellar luminosity is now in watts, `r` in meters. Periapsis flux > apoapsis flux falls out automatically — and combined with axial tilt produces the asymmetric Earth-like seasons.
- Orbit map already renders the ellipse with focus offset; it just needs to read the real `ν` instead of `yearPhase * 2π`.
- Day/night uses the planet's own rotation rate `Ω_planet`, decoupled from orbital motion.

Backed out: the proxy `BuildSolarState` computation of `solar.orbitDistanceAu` from `yearPhase` and `eccentricity`. That becomes the result of Kepler integration.

## 4. Atmosphere — shallow-water layer

The current weather step is a per-tile relaxation + edge-flux scheme that doesn't conserve momentum coherently. Replace the dynamical core with **2D shallow-water primitive equations on the sphere**, keeping the existing tile graph as the spatial discretization:

Per cell prognostic variables: `h` (column thickness departure), `u`, `v` (horizontal wind).

```
∂h/∂t + ∇·(h·v) = E − P
∂v/∂t + (v·∇)v + f·k̂×v + g·∇h = −κ·v + F_drag
```

with:

- **Coriolis** `f = 2 Ω_planet sin φ` — uses planet rotation rate, real Coriolis at all latitudes.
- **Pressure gradient** `g·∇h` driving geostrophic balance.
- **Friction** `κ·v` near the surface (terrain-dependent: high over mountains).
- **Source/sink** from evaporation `E` and precipitation `P` (see §5).

Tracer transport (temperature, humidity, cloud water) is advected by `v` on the same tile-edge flux scheme already in the code — just upgraded from ad-hoc to genuine upwind/centered fluxes consistent with the dynamics.

This is the largest single piece. Mitigate risk by keeping the existing transport step as a fallback and ABing under a runtime flag.

## 5. Moisture + hydrology

- **Saturation vapor pressure** via Clausius–Clapeyron (replace the existing `WeatherSaturation` polynomial).
- **Evaporation** `E ∝ (e_sat(T_surf) − e_air) · |v| · C_E` (bulk aerodynamic).
- **Condensation** when `q > q_sat`: excess goes to `q_c`; latent heat `L_v · Δq` released into the energy budget (closes the loop with §2).
- **Precipitation**: autoconversion `q_c → P` once `q_c > q_c_crit`.
- **Snow vs rain** by surface temperature and air-column temperature.
- **River routing**: each tick, surface runoff flows downhill along the tile graph by elevation gradient. Accumulate flow; render as rivers/lakes where flux exceeds a threshold. Cheap and looks great.
- **Snow/ice mass balance**: accumulation − melt − sublimation, no longer a visual proxy.

## 6. Moons that matter

The four-moon UI from `ui-imgui` becomes physical:

- **Tides:** for each moon, equilibrium tidal potential `Φ_tide ∝ G·M_moon · (3 cos²θ − 1) / r_moon³` per tile. This enters the shallow-water `g·∇h` term as an extra forcing. Spring/neap cycles from moon alignment emerge for free.
- **Tidal heating**: a small baseline heat term in the ocean energy budget proportional to `Σ_moons (M_moon² / r_moon⁶)`.
- **Eclipses:** when a moon's projected disk intersects the planet–sun line at a tile, drop `SW_in` to a small fraction over the umbra for the eclipse duration. Visible as a dark spot moving across the globe.
- **Moonlight (optional):** tiny SW contribution at night from reflected starlight off lit moons. Negligible thermally, lovely visually.
- **Moon orbits** integrated as hierarchical Kepler around the planet (each moon's `MoonConfig` already has the right elements; period becomes physical, derived from `M_planet`).

Add per-moon `mass` (in Earth-moon masses) to `MoonConfig`. UI slider lands in the existing Moons tab.

## 7. Performance

Without these, "more detailed" means "slower." With them, the sim happily runs at subdivisions 6 (≈21k tiles) at real-time.

1. **Threaded weather step.** Per-tile update is embarrassingly parallel. `std::thread` pool partitioning tile-index ranges, double-buffered cells. Expect 4–8× on M1 Pro. **Biggest single win.** Do this early.
2. **SoA layout.** `WeatherCell` is currently AoS (~100 B). Split hot fields (`T[]`, `p[]`, `q[]`, `u[]`, `v[]`) into separate arrays. Halves cache misses, vectorises well.
3. **Decouple sim from render.** Fixed-`dt` accumulator (already half-done with `weatherTimer`). Sim ticks at e.g. 60 Hz; rendering interpolates. Cranking speed no longer destabilizes the solver.
4. **Instanced tile rendering.** Currently the tile mesh is rebuilt with per-frame colors. Keep persistent VAO, upload only the color buffer when it changes. 2–4× render-time speedup.
5. **`-O3 -march=native`** in the Makefile.
6. **(Optional, large) GPU compute step.** Upload tile state as a texture / SSBO, do radiation + advection in a compute shader. Gets you subdivisions 7 (84k tiles) real-time. Big lift, deferred until 1–5 are exhausted.
7. **Profile with macOS Instruments** before optimising further. The above is the prior, not a substitute for measurement.

## 8. Rollout order

Each step ships a runnable, coherent sim. No big-bang rewrite.

1. **Branch + units pass** (§0–1). UI immediately shows Kelvins / Pascals. Sim behaviour roughly unchanged. **Day 1–2.**
2. **Radiation balance** (§2) replacing the temperature update. Temperatures become emergent and unclamped. UI plots get interesting. **Day 3–5.**
3. **Threaded weather step** (§7.1). Buys headroom for everything else. **Day 6.**
4. **Keplerian orbit** (§3). Orbit map renders true ν, periapsis warming appears. **Day 7–8.**
5. **Shallow-water atmosphere** (§4). Jet streams, real fronts, geostrophic flow. **Day 9–13.**
6. **Hydrology + rivers + snow mass balance** (§5). **Day 14–16.**
7. **Moons: tides, eclipses, tidal heating** (§6). **Day 17–19.**
8. **SoA + render instancing + decoupled tick** (§7.2–4). **Day 20–22.**

Stop at any step and you still have a working simulator strictly better than `main`.

## 9. Out of scope (next branches)

- **Vertical levels** (2–4 layer baroclinic atmosphere) — gives proper jet structure, fronts. Branch `physics-vertical`.
- **GPU compute** weather step (§7.6) — branch `physics-gpu`.
- **True n-body** for planet + moons (currently hierarchical Kepler) — branch `physics-nbody`. Only worth it if we add Lagrange-point chaos or multi-planet systems.
- **Plate tectonics rework** with isostasy / real erosion — separate concern from atmospheric physics.
