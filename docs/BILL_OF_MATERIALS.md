# BraveBot — Bill of Materials

Auto-generated from `bravebot_sim/registry.py` (`python scripts/export_manifest.py`).

**Base platform:** Modified LimX Dynamics TRON 1 (WF_TRON1A) architecture  
**Frame:** URDF-local — +x forward, +y left, +z up, origin at the TRON 1 base link.

## LimX TRON 1 base (real meshes, Apache-2.0)

| Link | Mount (m) | Mesh |
|------|-----------|------|
| base | (0.000, 0.000, 0.000) | `tron1/base_Link.stl` |
| abadL | (0.056, 0.105, -0.260) | `tron1/abad_L_Link.stl` |
| abadR | (0.056, -0.105, -0.260) | `tron1/abad_R_Link.stl` |
| hipL | (-0.021, 0.126, -0.260) | `tron1/hip_L_Link.stl` |
| hipR | (-0.021, -0.126, -0.260) | `tron1/hip_R_Link.stl` |
| kneeL | (-0.171, 0.105, -0.520) | `tron1/knee_L_Link.stl` |
| kneeR | (-0.171, -0.105, -0.520) | `tron1/knee_R_Link.stl` |
| wheelL | (-0.021, 0.148, -0.780) | `tron1/wheel_L_Link.stl` |
| wheelR | (-0.021, -0.148, -0.780) | `tron1/wheel_R_Link.stl` |

## BraveBot modification components (original geometry)

| # | Component | Group | Mount (m) | Mesh | Notes |
|---|-----------|-------|-----------|------|-------|
| 1 | Payload adapter frame | structural | (0.020, 0.000, 0.045) | `bravebot/payload.stl` | Custom mounting layer bolted on top of the TRON 1 base. |
| 2 | Rugged inspection torso | structural | (0.020, 0.000, 0.215) | `bravebot/torso.stl` | Sealed IP66 enclosure housing inspection electronics and edge compute. |
| 3 | Protective rail (left) | structural | (0.030, 0.176, -0.100) | `bravebot/railL.stl` | Impact rail cage and field handling point. |
| 4 | Protective rail (right) | structural | (0.030, -0.176, -0.100) | `bravebot/railR.stl` | Impact rail cage and field handling point. |
| 5 | Sensor head housing | structural | (0.020, 0.000, 0.570) | `bravebot/head.stl` | Houses the four-sensor stack at the top of the mast. |
| 6 | Sensor mast | structural | (0.020, 0.000, 0.430) | `bravebot/mast.stl` | Raises the sensor head for an elevated vantage over racks and aisles. |
| 7 | Hot-swappable battery | power | (0.020, 0.000, 0.120) | `bravebot/battery.stl` | 2-4 h runtime, ~10 s swap for continuous 24/7 patrol. |
| 8 | Edge AI core | compute | (-0.030, 0.000, 0.220) | `bravebot/edgeai.stl` | Local multimodal reasoning: fusion, diagnosis, risk scoring. No cloud. |
| 9 | Front AI display panel | compute | (0.165, 0.000, 0.215) | `bravebot/display.stl` | Local status, patrol mode and operator feedback. |
| 10 | Rear service panel | structural | (-0.120, 0.000, 0.215) | `bravebot/rear.stl` | Tool-free maintenance access and modular service layout. |
| 11 | Acoustic imaging array | sensor | (0.140, 0.075, 0.580) | `bravebot/acoustic.stl` | Detects ultrasonic leaks, partial discharge and bearing wear. |
| 12 | Thermal camera | sensor | (0.145, 0.000, 0.600) | `bravebot/thermal.stl` | Maps hotspots across UPS, PDU, cable joints, GPUs and racks. |
| 13 | HD visual AI camera | sensor | (0.145, -0.075, 0.580) | `bravebot/hdcam.stl` | Reads gauges, assets, doors, seals and indicator lights. |
| 14 | Gas / laser telemetry sensor | sensor | (0.140, 0.000, 0.535) | `bravebot/gas.stl` | VOC / CO / H2 off-gas; optional OGI / TDLAS for industrial gas. |
| 15 | Navigation sensor cluster | sensor | (0.145, 0.000, 0.480) | `bravebot/nav.stl` | Mapping, obstacle avoidance and route following. |
| 16 | Communication antenna (left) | comms | (0.000, 0.070, 0.710) | `bravebot/antennaL.stl` | Network sync and command / communication link. |
| 17 | Communication antenna (right) | comms | (0.000, -0.070, 0.710) | `bravebot/antennaR.stl` | Network sync and command / communication link. |
| 18 | Emergency stop | safety | (-0.100, 0.110, 0.310) | `bravebot/estop.stl` | Hardware safety cut-off for operation around people and equipment. |

## Four-sensor inspection stack

| Sensor | Modality | FOV (deg) | Range (m) | Spec |
|--------|----------|-----------|-----------|------|
| Acoustic imaging array | acoustic | 35 | 8 | Phased acoustic imaging array, ultrasonic up to 100 kHz |
| Thermal camera | thermal | 24 | 12 | Radiometric infrared camera, per-zone baseline learning |
| HD visual AI camera | visual | 30 | 10 | HD camera with on-board gauge OCR, object & scene detection |
| Gas / laser telemetry sensor | gas | 45 | 4 | VOC / CO / H2 / off-gas; optional OGI / TDLAS (OTC) |
| Navigation sensor cluster | nav | 60 | 15 | LiDAR + depth navigation cluster: mapping, obstacle avoidance |

## Headline specifications

| Spec | Value |
|------|-------|
| robot type | Wheel-legged autonomous inspection robot |
| base platform | Modified LimX Dynamics TRON 1 (WF_TRON1A) architecture |
| chassis footprint | ~9 in x 11 in |
| chassis weight kg | 21.8 (48 lb body) |
| sensor payload kg | 15.0 (33 lb stack, config-dependent) |
| mobility | Wheel-legged, stair / slope capable |
| slope capability | up to 30 deg |
| step capability | up to ~10 in (estimated) |
| ruggedness | IP66 (target) |
| runtime | 2-4 h per charge (estimated) |
| battery | Hot-swappable, ~10 s swap |
| sensor stack | Acoustic / thermal / gas / visual |
| ai | Edge-resident multimodal AI / local reasoning |
| integrations | OPC UA / MQTT / REST / BMS / DCIM / CMMS |
| wheel radius m | 0.1297 |
| wheel separation m | 0.2970 |
| max speed mps | 3.0 (wheeled mode) |
