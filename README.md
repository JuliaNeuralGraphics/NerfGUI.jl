# NerfGUI.jl

GUI application for [Nerf.jl](https://github.com/JuliaNeuralGraphics/Nerf.jl)

https://user-images.githubusercontent.com/17990405/213012459-5e1d2154-f921-4004-aa1f-7a55c26f042c.mp4

## Requirements

- Julia 1.8 or higher.
- AMDGPU or CUDA capable machine.

## Usage

To launch application, simply execute its main function.

```julia
using NerfGUI
NerfGUI.main()
```

## Configuration

**Note:** make sure to specify correct `backend` in `LocalPreferences.toml` before launching the application. <br/>
Available options are: `ROC` (i.e. ROCm, for AMD GPUs), `CUDA` (for NVIDIA GPUs).

See `LocalPreferences.toml` file for other available configurations.
