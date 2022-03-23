# The Engine Model, Trial One

# Config File in Training
1. File_name: `configs/demo.yaml`
2. Modifications:
    a. `voxel_resolution: 800`
    b. `snerg_dtype: float16`
    c. `batch_size: 256`
3. Num_iterations: TODO
4. Did the model converge? 
    a. TODO

# The Dataset
- 100 training, 100 val, 200 testing
- depth maps were basically invisible
- **camera was farther away** from the engine

# Results
1. Qualitative: TODO
2. **PSNR**: TODO

# Performance

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**|   TODO   |
| **# of Triangles in Mesh** | TODO |
| **# of Draw Calls** |  TODO |
| Size of Assets (MB)     | TODO       |
| GPU Memory Footprint (rounded to nearest 10 MB)         |   TODO    |

![FPS of the `Mesh` in the Browser](trial1fps.png)