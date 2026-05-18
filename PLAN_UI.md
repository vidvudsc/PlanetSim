# UI Branch Plan — `ui-imgui`

## Goal
Replace the hand-rolled immediate-mode panel with **Dear ImGui + ImPlot** (via rlImGui), give every UI surface a floating-window UX (drag / resize / collapse / pop-out / hide), and add a **3D orbit-map mini view**. No physics changes on this branch.

## Vendoring
```
vendor/imgui/      Dear ImGui (docking branch — for snap + viewports)
vendor/implot/     ImPlot
vendor/rlImGui/    Raylib ImGui backend
```

## Build
- `main.c` → `main.cpp`; compile with `clang++ -std=c++17`.
- Makefile compiles every ImGui/ImPlot/rlImGui `.cpp` plus `main.cpp` and links one binary.
- C-to-C++ port fixes: enum increment casts, compound-literal cleanup where C++ disagrees, `malloc`/`calloc` already cast.

## Window catalogue (all floating, all toggleable)

| Window | Replaces | Notes |
|---|---|---|
| Top bar | — | Pinned upper-left, ~280×40 px. Clock + play/pause + hamburger toggle for every other window. |
| Controls | `DrawControlPanel` | Tabs: Sim · Tectonics · Atmosphere · Render · Debug. |
| Tile inspector | `DrawSelectedTileInfo` | Lat/lon, elevation, biome, plate; ImPlot sparklines for last 256 weather steps. |
| Climate charts | `DrawClimateChartsPopup` | ImPlot time-series with legend, hover values, multi-metric selection. |
| Weather legend | `DrawWeatherColorLegend` | Compact gradient bar + labels. |
| Orbit map | (new) | Embedded `RenderTexture2D` with second `Camera3D`. Drag-to-rotate inside view, scroll-zoom, maximize button to overlay fullscreen, click planet/star to refocus main camera. |

## Persistence
Dear ImGui auto-writes `imgui.ini` for window positions, sizes, collapsed/visible state. Toggle-state of each panel is mirrored to `imgui.ini` via custom save handlers — survives restart.

## Mouse routing
Replace the existing `uiHovered` plumbing with `ImGui::GetIO().WantCaptureMouse` and `WantCaptureKeyboard`. Orbit camera input only fires when ImGui isn't capturing.

## Order of operations
1. Vendor deps
2. C++ port (compile-only — UI unchanged)
3. ImGui init + hello-world window
4. Top bar
5. Controls window (full migration)
6. Tile inspector (with ImPlot)
7. Climate charts (with ImPlot)
8. Weather legend
9. Orbit map
10. Delete dead custom-panel code
11. Polish: TTF font, theme, docking flag, viewports flag, per-window alpha slider
12. Build, smoke-test, commit

## Out of scope (next branch: `physics-rework`)
- Real Keplerian orbits feeding the sim (the orbit map can render synthetically for now and switch to real elements once they exist).
- Unclamped SI temperatures, radiation balance, moons, tides, eclipses.
