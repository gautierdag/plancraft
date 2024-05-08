# Plancraft Simulation docker

Docker container to run `minerl>=1.0.0` simulation with GPU rendering in headless mode using the optimised virtualGL.

```bash
docker build . -t plancraft-sim
docker run --gpus=all --device=/dev/dri -it plancraft-sim /bin/bash
```
## Run

Inside the docker container, you may check your gpus info for egl
```bash
/opt/VirtualGL/bin/eglinfo -e
```
The output should be like
```
EGL device ID: egl0 or egl, DRI device path: /dev/dri/card0
EGL device ID: egl1, DRI device path: /dev/dri/card1
```
then you can choose a specific GPU (ex. `/dev/dri/card1` or `egl1`) for rendering, and verify if GPU rending is working use the following command
```bash
vglrun -d /dev/dri/card1 /opt/VirtualGL/bin/glxspheres64
```
or (in case of `/dev/dri/card1` is not available for whatever reason)
```bash
vglrun -d egl1 /opt/VirtualGL/bin/glxspheres64
```
If you can see something like `OpenGL Renderer: NVIDIA GeForce RTX 3090/PCIe/SSE2` in the output, GPU rendering is working.

To run your own application with GPU rendering, prepand commands with `vglrun`.
```bash
vglrun -d <eglx_or_dri_device_path> <your_command>
```

## Test

```bash
vglrun -d egl python test.py
```