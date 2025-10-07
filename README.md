# FMCW Radar

This project demonstrates a FMCW radar processing chain with three core steps: **Range–Doppler**, **Angle of Arrival (AoA)**, and **CA‑CFAR** detection.

---

## 1) Range–Doppler (RD)

- **Range FFT (fast-time):** For each chirp, take an FFT across ADC samples to convert beat frequency into range using
- **Doppler FFT (slow-time):** Stack per‑chirp range and take an FFT across chirps to convert Doppler frequency into velocity
- **Map:** The 2‑D magnitude forms an RD heatmap where peaks correspond to targets (range vs. velocity).

---

## 2) Angle of Arrival (AoA)

- For the detected RD bin, collect complex values across all Rx elements to get a snapshot of the array response.
- Form a steering matrix
- Compute a simple beamformer / angle FFT power and take the peak

---

## 3) CA‑CFAR (Constant False Alarm Rate)

- For each Cell Under Test (CUT), use reference cells to estimate the local noise level, guard cells protect the CUT from target energy bleeding into the estimate.
- Compute a threshold
- Declare a detection if power(CUT) > T. Applied as two 1‑D passes and combined.
