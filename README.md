# SAR-RF-Real-Time-RF-Signal-Classification

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/SAR_RF.jpg)

SAR-RF is an edge-ML stack for real-time RF signal classification designed for Search-and-Rescue (SAR) scenarios and future UAV/drone deployment. It ingests complex IQ samples from an SDR, classifies the modulation in real time, and exposes results to a lightweight UI and mapping layer. The project targets two hardware paths:

* DPU-FPGA (Vitis AI on Zynq/Zedboard-class devices) for low-power, on-board inference (INT8).

* GPU (CUDA-class devices / Jetson / desktop) for rapid prototyping and high-throughput lab runs.

In SAR operations, spectrum awareness helps detect, localize, and prioritize signals of interest (e.g., distress beacons, VHF/UHF communications) under tight size, weight, and power (SWaP) constraints. SAR-RF brings robust RF classifiers to the edge so drones and field kits can act faster with limited bandwidth.

* Stream & preprocess IQ from RTL-SDR (or compatible front-ends): resampling, normalization, windowing.

* Classify common modulations (e.g., BPSK, QPSK, GMSK, FM, OOK, OQPSK, 8PSK, 16QAM, AM-DSB-WC/SC).

* Run anywhere: INT8 on DPU (Vitis AI) or FP32/FP16 on GPU with a shared model interface.

* Serve over Ethernet: FastAPI inference server on the device; Dash UI client on ground station.

* Visualize: live PSD/Waterfall, per-class confidence bars, and optional map overlays/GNSS.

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/SAR_RF_lab.jpg)

## Dataset
Over-the-air signals inherently exhibit impairments and distortions—an essential part of any realistic dataset—including multipath fading, carrier-frequency offset, timing/phase errors, and additive white Gaussian noise (AWGN). These effects reduce the signal-to-noise ratio (SNR); lower SNR makes error-free reception increasingly difficult. For a fixed SNR, higher-order constellations (larger symbol alphabets) have denser decision regions, increasing the likelihood of detection errors. We utilize the [DeepSig dataset](https://www.deepsig.ai/datasets/), which encompasses both synthetic channel simulations and over-the-air recordings, spanning 27 analog and digital modulation types. 

![SNR](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/ACC_SNR_RF.jpg)

## Feature-Scale SE-Net for I/Q Signal Classification

A lightweight encoder–decoder CNN that fuses multi-scale features with a Feature-Scale (FS) block and channel attention (SE) for real-time classification of I/Q sequences. Default input: (1024, 1, 2).

![dpu_M](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Model_DPU.jpg)

### Encoder (5 stages)
Repeated Conv2D → BatchNorm → ReLU blocks (ConvBlock) with stride-2 downsamples. Feature maps:

* F1: (B, 16, H/2, W/2)

* F2: (B, 32, H/4, W/4)

* F3: (B, 64, H/8, W/8)

* F4: (B,128, H/16, W/16)

* F5: (B,256, H/32, W/32)

#### Squeeze-and-Excitation (SE). 
Adaptive 2D pooling → small MLP (or 1×1 convs) → sigmoid scale to recalibrate channels of F5, producing F5_out.

#### Multi-level fusion. 
Concatenate [F1, F2, F3, F4, F5_out] to expose both shallow and deep context to the decoder.

#### Feature-Scale (FS) block. Parallel convolutions at different receptive fields, then fuse:

* 1×1 (dilation = 1)

* 3×3 (dilation = 2)

* 3×3 (dilation = 3)

* Concatenate → Conv2D to mix scales

### Decoder

Concatenate the encoder features [F1, F2, F3, F4, F5_out] after spatially aligning them (to the smallest stride). This stacked tensor is the decoder’s input.

#### Feature-Scale (FS) block (context mixer).
Apply three parallel conv paths to the input, then fuse:

* 1×1, dilation=1

* 3×3, dilation=2

* 3×3, dilation=3
* Concatenate the three outputs channel-wise → Conv2D (1×1) to mix and reduce.

#### Upsample Stage 1.

* Upsample ×2 (nearest or bilinear), ConvBlock(192→128) → ConvBlock(128→64) (ConvBlock = Conv2D → BatchNorm → ReLU.)

#### Upsample Stage 2.

* Upsample ×2, ConvBlock(64→32) → ConvBlock(32→16)

#### Classification head.

AdaptivePool2D to (B, 16, 1, 1) (global pooling; input-size agnostic), Flatten → Linear(16 → Num_Classes) → logits (apply softmax only at evaluation time)

## Vivado, Vitis, & PetaLinux 2024.2 Install on WLS-Ubuntu
FPGA design IDEs seem to sprint ahead with every release—yes, ‘accelerating’ very much intended—so installation tips never go out of style. I’m back with notes for AMD’s 2025.1 tools after running into a few new twists during setup (and, unsurprisingly, devoting yet more disk space to Vivado/Vitis).

The Vivado ([HW Developer](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools.html)) and Vitis ([SW Developer](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html)) download links point to the same installer; during setup, you choose to install Vivado, Vitis, or both

### WLS-Ubuntu Alias

```
echo 'alias vitis-gui="source /tools/Xilinx/2025.1/Vitis/settings64.sh && DISPLAY=:0.0 vitis"' >> ~/.bashrc
# (re)load your shell config
source ~/.bashrc
# Use it
vitis-gui

echo 'alias vivado-gui="source /tools/Xilinx/2025.1/Vivado/settings64.sh && DISPLAY=:0.0 vivado"' >> ~/.bashrc
# (re)load your shell config
source ~/.bashrc

# Use it
vivado-gui
```

### PetaLinux Install

To complete the Linux setup for Yocto-based PetaLinux, you must install and configure a TFTP server. Manually create the TFTP root directory and apply the correct permissions. Begin by creating the TFTP service rules file:

#### Install the required host packages
```
# Core dev + Yocto build deps
sudo apt install -y \
  gawk wget git-core diffstat unzip texinfo \
  gcc-multilib build-essential chrpath socat xterm \
  autoconf automake libtool pkg-config zlib1g-dev \
  libssl-dev libncurses5-dev libncursesw5 libtinfo5 \
  libselinux1 libsdl2-dev libglib2.0-dev \
  python3 python3-pexpect python3-git python3-jinja2 \
  cpio rsync bc iproute2 net-tools iputils-ping \
  bison flex gnupg file curl tar gzip xz-utils zstd liblz4-tool
```
#### Prep for PetaLinux Install
```
sudo apt update
sudo apt install -y xinetd tftpd-hpa

sudo nano /etc/xinetd.d/tftp

service tftp
    {
    protocol = udp
    port = 69
    socket_type = dgram
    wait = yes
    user = nobody
    server = /usr/sbin/in.tftpd
    server_args = /tftpboot
    disable = no
    }
```
Save and close the TFTP rules file, then create the TFTP root directory with the correct ownership and permissions

```
~$ sudo mkdir /tftpboot
~$ sudo chmod -R 777 /tftpboot
~$ sudo chown -R nobody /tftpboot
```

Finally, restart the xinetd network service for the changes to take effect.
```
~$ sudo /etc/init.d/xinetd stop
~$ sudo /etc/init.d/xinetd start
```

Make the desired installation directory, apply the correct permissions, and install PetaLinux in the versioned path that matches your Vivado/Vitis release.
```
~$ sudo mkdir -p /tools/Xilinx/PetaLinux/2024.2/
~$ sudo chmod -R 755 /tools/Xilinx/PetaLinux/2024.2/
~$ sudo chown -R <user>:<user> /tools/Xilinx/PetaLinux/2024.2/
```

##### Option A 
```
# 1) Create just the PetaLinux path (as root)
sudo mkdir -p /tools/Xilinx/PetaLinux/2024.2

# 2) Give YOUR user ownership of that path
sudo chown -R "$USER:$USER" /tools/Xilinx/PetaLinux/2024.2

# (Alternative: keep root owner but grant ACL to you)
# sudo setfacl -R -m u:$USER:rwx /tools/Xilinx/PetaLinux
# sudo setfacl -d -m u:$USER:rwx /tools/Xilinx/PetaLinux

# 3) Sanity check
ls -ld /tools/Xilinx/PetaLinux/2024.2
test -w /tools/Xilinx/PetaLinux/2024.2 && echo "write OK" || echo "NOT writable"

# 4) Re-run installer (note: non-root user)
./petalinux-v2024.2-11062026-installer.run --dir /tools/Xilinx/PetaLinux/2024.2

```
## Vivado Hardware Design

PS/clock/reset/IRQ backbone and the standard AXI touchpoints (GP0 + HP0)

![](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/models/DPU_Vivado_Radio.jpg)

## Build the PetaLinux and GNU Radio

### Creating the PetaLinux project

Create a PetaLinux project and configure the hardware with the XSA file created

```
# (env, once per shell)
source /tools/Xilinx/PetaLinux/202x.x/settings.sh

petalinux-create -t project --template zynq -n zedboard_202x_x-petalinux
cd zedboard_202x_x-petalinux
petalinux-config --get-hw-description=/tools/Xilinx/PetaLinux/202x.x/zedboard_202x_x-petalinux/zed_accel_x/zed_dpu_radio_wrapper.xsa

```
![PConf](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Hardware/Petalinux_Config.jpg)

### Modifying the Device Tree
We need to add some driver configurations for SD card. Modify ``` system-user.dtsi ``` file 
```
/include/ "system-conf.dtsi"
/ {
};

/* SD */
&sdhci1 {
        disable-wp;
        no-1-8-v;
};
```
### Config the Kernel

Run the following command
```
petalinux-config -c kernel
```

![conf2](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Hardware/Petalinux_Config_2.jpg)

#### Disable CPU IDLE in kernel config:

* CPU Power Management > CPU Idle > CPU idle PM support - set 'N' n the [ ] menu selection.
* CPU Power Management > CPU Frequency scaling > CPU Frequency scaling - set 'N'.
  
### GNU Radio software to RootFS file system

Yocto layer for the following SW to build in the system:

* **GNU Radio v3.8** - main program
* **gr-osmosdr v3.8** - module needed for SDR receivers
* **gr-fpga_ai v3.8** - module needed for accessing DPU on FPGA

#### Config the RootFS file system

```
grep -n 'LAYERSERIES_COMPAT' /tools/Xilinx/PetaLinux/202x.x/zedboard_202x_x-petalinux/project-spec/meta-sdr/conf/layer.conf

```

```
 nano project-spec/meta-user/conf/user-rootfsconfig file.

Note: Mention Each package in individual line
#These packages will get added into rootfs menu entry

CONFIG_gpio-demo
CONFIG_peekpoke

# Packages for base XRT support
CONFIG_xrt

# ackages for easy system management
CONFIG_dnf
CONFIG_e2fsprogs-resize2fs
CONFIG_parted
CONFIG_resize-part

# Packages for Vitis-AI dependencies support
CONFIG_packagegroup-petalinux-vitisai

# Optional Packages for natively building 
# Vitis AI applications on target board
CONFIG_packagegroup-petalinux-self-hosted
CONFIG_cmake
CONFIG_packagegroup-petalinux-vitisai-dev
CONFIG_xrt-dev
CONFIG_opencl-clhpp-dev
CONFIG_opencl-headers-dev
CONFIG_packagegroup-petalinux-opencv
CONFIG_packagegroup-petalinux-opencv-dev

# Optional Packages for running i
# Vitis-AI demo applications with GUI
CONFIG_mesa-megadriver
CONFIG_packagegroup-petalinux-x11
CONFIG_packagegroup-petalinux-v4lutils
CONFIG_packagegroup-petalinux-matchbox

# Packages for date and time settings
CONFIG_ntp
CONFIG_ntpdate
CONFIG_ntp-utils

# Some utils
CONFIG_git
CONFIG_zip
CONFIG_unzip

# Gnuradio and its modules
CONFIG_gnuradio
CONFIG_gr-osmosdr
CONFIG_gr-fpga-ai

```
Config the rootfs by running this command:

```
petalinux-config -c rootfs
```
* User Packages. Enable all, don't forget for gnuradio, gr-osmosdr and gr-fpga-ai.
* Enable OpenSSH and disable dropbear: Image Features-> Disable ssh-server-dropbear and enable ssh-server-openssh.
* Filesystem Packages -> misc -> packagegroup-core-ssh-dropbear and disable packagegroup-core-ssh-dropbear.
* Filesystem Packages -> console -> network -> openssh and enable openssh, openssh-sftp-server, openssh-sshd, openssh-scp.
*  Image Features and enable package-management and debug_tweaks.

### Image Feature Settings

## Vitis Platform with DPU for AI Inference

## Vitis Platform

## Export the Platform

## Performance Test

## Install Vitis AI

## DPU settings for ZCU104

##  Performance Test
Run the benchmark from the target board’s serial or SSH terminal.

```
Usage: python3 test_performance.py <threads> <model.xmodel> <num_frames>
```

* RF frames are read from rf_input.npy; each frame contains 1024 I/Q samples.

* The script prints the measured performance.

* The sample results shown below were obtained on a ZCU104 with two 4096-MAC DPU cores running at 300 MHz (~2.45 peak INT8 TOPS).

## Accuracy Test

On the target board (serial console or SSH), run:

```
Usage: python3 test_accuracy.py rfClassification.xmodel
```

The script reads evaluation data from:

* rf_input.npy – RF frames

* rf_snr.npy* – per-frame SNR values

* rf_classes.npy – ground-truth modulation classes

## Hardware/Software

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

# Single-board SDR platform.
Built around the Xilinx Zynq-7020 SoC and an AD936x RF transceiver (AD9363/AD9361/AD9364), this board integrates 512 MB DDR3L, 16 MB flash, and MicroSD for boot/configuration. I/O includes USB OTG, USB-JTAG/UART, and Ethernet. The Zynq architecture combines an ARM® Cortex®-A9 processing system with programmable logic, delivering a highly flexible platform where workloads can run on the ARM cores, the FPGA fabric, or a mix of both.

![HW_SDR](https://github.com/1Px-Vision/SAR-RF-Real-Time-RF-Signal-Classification/blob/main/Hardware/HW_SDR.jpg)

