# The Engine Model, Trial Three

# Config File in Training
1. File_name: `configs/demo.yaml`
2. Modifications:
    a. `voxel_resolution: 800`
    b. `snerg_dtype: float16`
    c. `batch_size: 256`
3. Num_iterations: 47500
4. Did the model converge? 
    a. TODO

# The Dataset
- 100 training, 100 val, 200 testing
- depth maps were much more defined (normalized to be 0-1)
- camera was closer to the engine
- **I left only 1 light in the Blender scene**

# Results
1. Qualitative: TODO 
2. PSNR: 10.516991861087531

# Performance

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**|   TODO   |
| **# of Triangles in Mesh** | TODO |
| **# of Draw Calls** |  TODO |
| Size of Assets (MB)     | TODO       |
| GPU Memory Footprint (rounded to nearest 10 MB)         |   TODO    |

![FPS of the `Mesh` in the Browser](TODO)