# PlanetSim

A real-time procedural planet simulator written in C using [raylib](https://www.raylib.com/). Simulates tectonic plates, terrain generation, and a dynamic weather system — all running live on a 3D globe.

Weather is physically motivated: trade winds and westerlies drive atmospheric circulation, mountain ranges cast rain shadows, valleys channel wind, coastal regions stay humid, and the ITCZ brings equatorial rainfall. Ocean gyres and currents respond to wind stress and Coriolis forces.

![Timelapse](screenshots/timelapse.gif)

---

## Features

- **Tectonic simulation** — plates drift, collide, and build mountain ranges over time
- **Atmospheric circulation** — trade winds, westerlies, polar highs, ITCZ, and storm tracks
- **Terrain-driven weather** — orographic lift, rain shadows, valley channeling, cold pooling
- **Ocean currents** — gyre patterns, wind-driven drift, Coriolis deflection
- **Sun-driven climate** — day/night heating, axial tilt, seasons, and ocean heat transport
- **12 visualization modes** — temperature, pressure, wind, humidity, clouds, rain, snow, vorticity, ocean temperature, and more
- **Live control panel** — adjust weather speed, solar forcing, tilt, atmosphere tuning, and view modes in real time

---

## Screenshots

| Wind | Ocean Currents |
|---|---|
| ![Wind](screenshots/wind.png) | ![Currents](screenshots/currents.png) |

| Temperature | Tectonic Plates |
|---|---|
| ![Temperature](screenshots/temperature.png) | ![Plates](screenshots/plates.png) |

---

## Building

Requires [raylib](https://www.raylib.com/) installed.

```sh
make
./planet
```

---

## Controls

- **Left drag** — rotate globe
- **Scroll** — zoom
- **F1** — show or hide the control panel
- **A / C / W / Space** — atmosphere, plate view, weather, tectonics pause
- **Tab / 1-0 / E / R** — cycle or jump to weather visualization modes
