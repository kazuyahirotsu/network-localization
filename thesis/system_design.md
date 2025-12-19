# System Design & Deployment Strategy: Resilient LPWA Localization

## 1. Mission Profile & Feasibility Definition
**What are we actually building?**
We are NOT building a replacement for GPS for precise delivery (e.g., Amazon Prime Air). That requires <1m accuracy.
We ARE building a **Rapidly Deployable Navigation Infrastructure** for GNSS-denied/spoofed environments.

*   **Target Application:**
    *   **Wide-Area Search & Rescue (SAR):** Finding a lost hiker/survivor in a 10km² mountain range.
    *   **Disaster Response:** Mapping flood extent or wildfire perimeters.
    *   **Defense:** Operating in jammed environments where approximate positioning is better than "blind" flight.

### 1.1 Objective Classification & Feasibility
We categorize drone operations into three types to define "success" for our system:

1.  **Spot Arrival / Point-to-Point Transit (Guidance Phase):**
    *   *Goal:* Navigate from Base A to Target Area B (e.g., 5km away).
    *   *Accuracy Req:* **Corridor Feasibility.** The drone must stay within a "safe corridor" (e.g., 200m wide) to avoid no-fly zones or terrain.
    *   *Verdict:* **FEASIBLE.** Our simulation shows lateral deviation < 50m. The system successfully guides the drone to the *vicinity* of the target.
    *   *Note:* Precision landing is out of scope. We assume a secondary sensor (Visual Servoing / Marker Detection) takes over for the final 10 meters.

2.  **Area Sweep / Mapping (Coverage Phase):**
    *   *Goal:* Photograph or sense a defined sector (e.g., 1km x 1km) to find a missing person or map damage.
    *   *Accuracy Req:* **No-Gap Coverage.** The spacing between flight lines must ensure the camera footprint covers the entire ground despite positioning errors.
    *   *Verdict:* **HIGHLY FEASIBLE.**
        *   *Math:* With a 100m altitude and 90° FOV camera, swath width is 200m.
        *   *Deviation:* Our max lateral error is ~50-70m.
        *   *Strategy:* By setting flight line spacing to 100m (50% overlap), we absorb the localization error. The search is slightly inefficient but guaranteed to be complete.

3.  **Precision Strike / Targeted Drop (Excluded):**
    *   *Goal:* Dropping a payload on a specific 5m target.
    *   *Verdict:* **INFEASIBLE** with LoRa alone. Requires terminal guidance (Vision/IR).

## 2. System Architecture: Centralized Intelligence, Distributed Sensing
You cannot run GCN training on a $5 microcontroller. Do not pretend you can. The architecture must be **Asymmetric**.

### Components
1.  **The Edge (Beacons):** Dumb, low-power, "Listen & Chirp" devices.
2.  **The Hub (Base Station / Mother Drone):** High-compute node (GPU laptop or Jetson Orin). Runs the GCN.
3.  **The Client (Service Drone):** The user of the system. Runs multilateration.

## 3. Deployment Strategy (The "How")
Manual placement in disaster zones is dangerous and slow. We assume **Aerial Deployment**.

### Phase 1: Deployment (Air Drop)
*   **Vector:** A heavy-lift "Mother Drone" or Helicopter flies over the area.
*   **Action:** Drops beacons at rough intervals (e.g., every 500m).
*   **Anchor Initialization:**
    *   The Mother Drone records its own position (via high-grade INS or intermittent GNSS) at the moment of the drop.
    *   These coordinates become the *initial* guess for the beacon positions.
    *   *Reality Check:* The beacon will bounce/roll upon landing. The recorded drop position has an error of 10-50m immediately. This justifies the need for Chapter 4 (Self-Correction).

### Phase 2: Self-Organization (The Learning Loop)
This is where your GCN comes in.
1.  **Wake-up:** Beacons wake up.
2.  **Sensing:** Every beacon broadcasts a "Hello". All neighbors record RSSI.
3.  **Aggregation:** Beacons send their recorded RSSI lists to the **Base Station** (via multi-hop or long-range link).
4.  **Computation (Off-board):**
    *   The Base Station runs the **Physically-Informed GCN**.
    *   Input: Noisy RSSI matrix + Drop coordinates (Anchors).
    *   Output: Refined Beacon Coordinates (X, Y, Z).
5.  **Update:** Base Station broadcasts the *refined* coordinates back to each beacon.
6.  **Storage:** Beacons save their new (X, Y) to flash memory.

### Phase 3: Service Operation
*   Now the beacons act as standard references.
*   They simply broadcast: `ID: 1, Pos: [X_opt, Y_opt, Z_opt]`.
*   The Service Drone listens and runs Multilateration (Chapter 3).

## 4. Hardware Requirements

### Beacon Node (The Infrastructure)
*   **Cost Target:** < $50 USD.
*   **MCU:** ESP32 or STM32 (Low power, deep sleep capable).
*   **Comms:** LoRa Transceiver (SX1276/SX1262). 915MHz/868MHz.
*   **Power:** LiPo Battery + Solar trickle charger (optional).
*   **Sensors:** None required (maybe Barometer for Z-axis relative check, but Chapter 3 suggests decoupling Z anyway).
*   **Logic:**
    *   Mode A (Calibration): Rx neighbors, Tx RSSI report to Base.
    *   Mode B (Beaconing): Tx Coordinate Packet periodically (e.g., 1Hz).

### Service Drone (The Client)
*   **Comms:** LoRa Receiver.
*   **Sensors:** Barometer/LiDAR (Essential for Z-axis decoupling).
*   **Compute:** Raspberry Pi / Jetson (runs the Multilateration + Huber Loss + Flight Control).
*   **Flight Controller:** ArduPilot/PX4 (accepts external position injection).

## 5. Feasibility Gap Analysis
**Where does this break in reality?**

1.  **The "Hidden Node" Problem:**
    *   Beacons on the ground are easily occluded by terrain (Longley-Rice model confirms this).
    *   *Mitigation:* Deployment density must account for ~50% packet loss. Over-deploy rather than under-deploy.

2.  **Battery Life:**
    *   Continuous broadcasting kills batteries.
    *   *Strategy:* **Wake-on-Radio**. Beacons sleep until they hear a specific "Pilot Tone" from a Service Drone entering the area.

3.  **Update Rate:**
    *   LoRa is slow (Time-on-Air ~100ms-300ms).
    *   Standard Multilateration needs ~4-10 packets.
    *   *Result:* Position update rate is ~0.5Hz.
    *   *Conclusion:* This system is for **Guidance**, not **Control**. The drone's internal IMU handles the 100Hz stabilization; LoRa corrects the drift every 2-3 seconds.
