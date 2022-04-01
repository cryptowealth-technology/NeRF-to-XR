# The Engine Model, Trial Nine

## Config File in Training
1. File_name: `TensoRF/configs/engine.yaml`
2. Modifications: None (it's the same as `lego.txt`)
3. Num_iterations: 30K
4. Did the model converge? yes - the [TensorBoard logs](https://tensorboard.dev/experiment/bEhlsvZ3SqiRNI5yQu2kew/#scalars) do seem to show the loss and `train_PSNR` hitting a plateau

## The Dataset - same as Trial 6
- 100 training, 100 val, 200 testing
- depth maps were present 
- **camera was farther away** from the engine
- **"Solid"** rendering was used, rather than "Material Preview"
- Data dir name: `engine_6_ds`

## Results
1. TensorBoard visualizations: [TensorBoard.dev link](TODO)
2. **PSNR**: 41.59
3. Checkpoint dir: `tensorf_VM_on_engine_9`
4. Baked images: TODO

## Visual Look

*Figure 1*: 

- unfortunately, it looks like in our run that the model didn't provide visualizations of its prediction on ALL views of the engine
- also the prediction image (above) does not exactly correspond to the viewpoint in the ground truth (below)
- nonetheless, the results do seem to be just about photorealistic

![Prediction of the engine, top view](../../TensoRF/log/tensorf_VM_on_engine_9/imgs_vis/009999_003.png)

![Ground truth of the engine, top view](../../TensoRF/log/tensorf_VM_on_engine_9/imgs_test_all/010.png)

## Performance 
We will need to REVISIT this section - as there is no realtime renderer yet for TensoRF.

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**| TODO  |
| **# of Triangles in Mesh** | TODO |
| **# of Draw Calls** | TODO |
| Size of Assets (MB)     | TODO   |
| GPU Memory Footprint (rounded to nearest 10 MB) |   TODO    |

![TODO](TODO.png)

## Takeaways - TODO