# The Engine Model, Trial Two

# Config File in Training
1. File_name: `configs/demo.yaml`
2. Modifications:
    a. `voxel_resolution: 800`
    b. `snerg_dtype: float16`
    c. `batch_size: 256`
3. Num_iterations: 47500
4. Did the model converge? 
    a. pretty much - the `avg_loss` was bouncing around `0.0130`

# The Dataset
- 100 training, 100 val, 200 testing
- **depth maps were much more defined** (normalized to be 0-1)
- **camera was closer** to the engine
- I **added 4 lights around** the model in the scene, placed on both sides of the model along the X and Y axes

# Results
1. Qualitative: the overall shape was non-existent, some color showed
2. **PSNR**: 10.516991861087531

# Performance

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**|   6.08   |
| **# of Triangles in Mesh** | 2 |
| **# of Draw Calls** |  1 |
| Size of Assets (MB)     | 79.2 |
| GPU Memory Footprint         | 1.4 GB |

![No FPS graph, because visual results were too poor to expect users realistically to view it](TODO)