# The Engine Model, Trial Eight

## Config File in Training
1. File_name: `configs/trial8.yaml`
2. Modifications: mainly, added more memory and params to the model (i.e. doubled both `snerg_dtype` and `net_width`)
3. Num_iterations: TODO
4. Did the model converge? TODO

## The Dataset - same as Trial 6
- 100 training, 100 val, 200 testing
- depth maps were present 
- **camera was farther away** from the engine
- **"Solid"** rendering was used, rather than "Material Preview"
- Data dir name: `engine_6_ds`

## Results
1. TensorBoard visualizations: [TensorBoard.dev link](TODO)
2. **PSNR**: TODO
3. Checkpoint dir: `snerg_on_engine_8` - AWS EC 2
4. Baked images: TODO

## Visual Look

*Figure 1*: TODO

## Performance 

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**| TODO  |
| **# of Triangles in Mesh** | TODO |
| **# of Draw Calls** | TODO |
| Size of Assets (MB)     | TODO   |
| GPU Memory Footprint (rounded to nearest 10 MB) |   TODO    |

![TODO](TODO.png)

## Takeaways - TODO