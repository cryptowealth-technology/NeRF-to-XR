# The Engine Model, Trial Seven

## Config File in Training
1. File_name: `configs/trial7.yaml`
2. Modifications: mainly, increased the `voxel_resolution` to 1,000, and raised `max_steps` to 1,000,000
3. Num_iterations: 46,800
4. Did the model converge? no
    1. From the [visualization site](https://tensorboard.dev/experiment/Xvd1EzikRpeG5Ek8ydCbUA/), we can see that the `train_avg_loss` and was still steadily decreasing at the end of the 46.8K iterations. Even though the plots for the `test_PSNR` and `test_SSIM` shows that peak performance was around the point of 27.6K iterations, we can still see that they were on an upward trajectory when the training stopped.
    2. "Was the model overfitting?" is another question to consider here. I would say yes (to a small extent) - as we can see from the `train_avg_PSNR` plot for example, we can see it's final value was about 32, while that of the `test_PSNR` was only about 31.

## The Dataset - same as Trial 6
- 100 training, 100 val, 200 testing
- depth maps were present 
- **camera was farther away** from the engine
- **"Solid"** rendering was used, rather than "Material Preview"
- Data dir name: `engine_6_ds`

## Results
1. TensorBoard visualizations: [TensorBoard.dev link](https://tensorboard.dev/experiment/Xvd1EzikRpeG5Ek8ydCbUA/)
2. **PSNR**: 31.95
3. Checkpoint dir: `snerg_on_engine_7` - AWS EC 2
4. Baked images: N/A

## Visual Look

*Figure 1*: Below I have pasted one of the predictions that the model made on its last step, along with the corresponding ground truth image. Although we can see the model is inching towards accurate colors and well-defined shapes (which makes sense, because it's the highest PSNR ever), we can see that the prediction is still missing some of the finer details. E.g. the model is not yet quite able to distinctly render all holes and wires on the front of the engine.

![Trial 7 prediction image, for the front of the car](trial7_front.png)
![Trial 7 - test dataset image, for the front of the car](trial7_front_truth.png)

## Performance 
Going to SKIP for now - these come out essentially the same each trial, because there aren't that many changes happening in the algorithm/ground truth data. 

| Metric               |  `Mesh` |
|----------------------|--------|
| **Avg. FPS (over 60 s, rounded to the nearest 0.01)**| TODO  |
| **# of Triangles in Mesh** | TODO |
| **# of Draw Calls** | TODO |
| Size of Assets (MB)     | TODO   |
| GPU Memory Footprint (rounded to nearest 10 MB) |   TODO    |

![TODO](TODO.png)

## Takeaways

*Comparison to Trial 6*

1. To review, there were a few changes made in the `config` for Trial 7, to make it different from Trial 6:
    1. increase to number of training iterations, and voxel resolution (probably the most impactful)
    2. 2x increase in the batch size (mainly to see if we could get away with it, and not run out of memory)
    3. note there was a 2x decrease to the `chunk` (but that was probably irrelevant, because that's the # of inferences to use in the `eval` script. )

2. To recap - it looks like the first two changes did indeed help the model improve in PSNR, and still hasn't even really started overfitting yet. 

3. So, what can we do to continue tuning the model:
    - increase `snerg_dtype` to `float32` - perhaps it gives better resolution
    - increase `net_width` 2x to `512` - same as it is in the `blender` config
    - In the future (not on the next trial), we may also consider using a "delayed" learning rate, as the authors show in `blender.yaml`. At this time I don't think it'd be that helpful, because we haven't had a problem with diverging weights so far.