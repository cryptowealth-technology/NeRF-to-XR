# The Engine Model, Trial Four

# Config File in Training
1. File_name: `configs/demo.yaml`
2. Modifications:
    a. `voxel_resolution: 800`
    b. `snerg_dtype: float16`
    c. `batch_size: 256`
3. Num_iterations: 50,000
4. Did the model converge? 
    a. yes - its final `avg_loss` was 0.0047, lowest of any model I can remember seeing

# The Dataset
- 100 training, 100 val, 200 testing
- depth maps were much more defined (normalized to be 0-1)
- camera was **farther** to the engine
- I left only 1 light in the Blender scene
- Data dir name: `engine_faraway_with_normalize`

# Results
1. Qualitative: still lacks shape, like Trial 2 and 3 it has color though
2. PSNR: 14.303311971709785
    a. slight increase over 2 and 3, but below Trial 1
3. Checkpoint dir: `snerg_on_engine_4.zip` - AWS
4. Baked images: `baked_4`

## Visual Look

*Figure 1*: This is roughly the front of the cube, looking down.
![Front view, looking "down" at the model](trial4_front.png)
*Figure 2*: This is roughly the side view of the cube.
![Side view](trial4_side.png)
*Figure 3*: This is roughly the bottom view of the cube, looking up.
![Bottom view](trial4_bottom.png)

Summary: from the pictures above, we can see the rendered models has the faint outlines of a cube-like shape. The green color comes out in some areas, but not all.
# Performance

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**|   16.83   |
| **# of Triangles in Mesh** | 2 |
| **# of Draw Calls** |  1 |
| Size of Assets (MB)     |  56.5    |
| GPU Memory Footprint (rounded to nearest 10 MB)         |   770    |

![No FPS graph, because visual results were too poor to expect users realistically to view it](TODO)