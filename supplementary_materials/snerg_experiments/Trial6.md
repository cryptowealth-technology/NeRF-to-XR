# The Engine Model, Trial Six

# Config File in Training
1. File_name: `configs/trial6.yaml`
2. Modifications:
3. Num_iterations: 25000
4. Did the model converge? 
    a. TODO

# The Dataset
- 100 training, 100 val, 200 testing
- depth maps were present 
- **camera was farther away** from the engine
- **"Solid"** rendering was used, rather than "Material Preview"
- Data dir name: `engine_6_ds`

# Results
1. Qualitative: TODO
2. **PSNR**: TODO
3. Checkpoint dir: `snerg_on_engine_6` - AWS EC 2
4. Baked images: `baked_6`
## Visual Look

*Figure 1*: TODO

# Performance

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**|   TODO   |
| **# of Triangles in Mesh** | TODO |
| **# of Draw Calls** | TOSDO |
| Size of Assets (MB)     | TODO      |
| GPU Memory Footprint (rounded to nearest 10 MB) |   TODO    |

![TODO](TODO)