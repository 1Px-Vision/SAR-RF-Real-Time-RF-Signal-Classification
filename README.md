# SAR-RF-Real-Time-RF-Signal-Classification

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/SAR_RF.jpg)

SAR-RF is an edge-ML stack for real-time RF signal classification designed for Search-and-Rescue (SAR) scenarios and future UAV/drone deployment. It ingests complex IQ samples from an SDR, classifies the modulation in real time, and exposes results to a lightweight UI and mapping layer. The project targets two hardware paths:

* DPU-FPGA (Vitis AI on Zynq/Zedboard-class devices) for low-power, on-board inference (INT8).

* GPU (CUDA-class devices / Jetson / desktop) for rapid prototyping and high-throughput lab runs.

In SAR operations, spectrum awareness helps detect, localize, and prioritize signals of interest (e.g., distress beacons, VHF/UHF comms) under tight size-weight-and-power (SWaP) constraints. SAR-RF brings robust RF classifiers to the edge so drones and field kits can act faster with limited bandwidth.

* Stream & preprocess IQ from RTL-SDR (or compatible front-ends): resampling, normalization, windowing.

* Classify common modulations (e.g., BPSK, QPSK, GMSK, FM, OOK, OQPSK, 8PSK, 16QAM, AM-DSB-WC/SC).

* Run anywhere: INT8 on DPU (Vitis AI) or FP32/FP16 on GPU with a shared model interface.

* Serve over Ethernet: FastAPI inference server on the device; Dash UI client on ground station.

* Visualize: live PSD/Waterfall, per-class confidence bars, and optional map overlays/GNSS.

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/SAR_RF_lab.jpg)

## Dataset
Over-the-air signals inherently exhibit impairments and distortions—an essential part of any realistic dataset—including multipath fading, carrier-frequency offset, timing/phase errors, and additive white Gaussian noise (AWGN). These effects reduce the signal-to-noise ratio (SNR); lower SNR makes error-free reception increasingly difficult. For a fixed SNR, higher-order constellations (larger symbol alphabets) have denser decision regions, increasing the likelihood of detection errors. We use the DeepSig dataset [Wiki](https://www.deepsig.ai/datasets/), which includes both synthetic channel simulations and over-the-air recordings spanning 27 analog and digital modulation types. 

## Hardware/Software at a glance

* Edge: Zedboard/Ultra96-class DPU (Vitis AI) or NVIDIA GPU/Jetson.

* I/O: RTL-SDR (default), pluggable front-ends.

* Runtime: Python, FastAPI (server), Dash/Plotly (UI).

* Models: CNN/ResNet-style; logits-only export for DPU; Vitis-AI quantization (INT8).

## Roadmap

* On-drone deployment (companion computer), power budgeting, and thermal envelopes.

* Bearing/DOA fusion and geotagged detections.

* Dataset expansion, domain adaptation, and semi-supervised updates.

* Optional on-device recording & replay for after-action review.

## Datasets & model shape

* Input tensor: (batch, 1024, 1, 2) (real/imag channels).

* Label space: default 10-class modulation set (easily extendable).

* Training: cross-entropy from logits (no in-graph softmax), weighted for class imbalance; robust augmentation across SNR (e.g., −20 dB to +30 dB), frequency offset, IQ imbalance.

## DPU-FPGA deployment (overview)

1. Train TF/Keras model → output logits.

2. Quantize with Vitis-AI (calibration dataset) → INT8 Keras model.

3. Compile with vai_c_tensorflow2 for your arch.json → produce .xmodel.

4. Run with VART on the target (Zynq/Kria), expose an Ethernet API (/infer) for the drone/ground UI.

5. Do softmax on host (DPU outputs logits/INT8 scores).


## Real-time constraints & target metrics

* **End-to-end latency (1×1024 window):** ≤10 ms (goal, DPU) / ≤20 ms (GPU mobile).

* **Throughput:** ≥100 FPS per stream on embedded; scalable via batching on GPU.

* **Power:** <6–8 W (DPU SoC node), 15–30 W (Jetson class).

* **Robustness:** stable classification from −10 dB SNR upward with temporal smoothing.

## Drone integration plan

* **Form factors:** SDR + DPU SoC or SDR + Jetson on companion computer.

* **Interfaces:** Ethernet to flight computer, optional MAVLink status, GNSS (NMEA) for geo-tagging.

* **Operational modes:** fixed-channel monitoring, band scanning, triggered capture, geo-fence alerts.

* **Data products:** time-stamped detections, confidence, SNR, (future) coarse bearings.
