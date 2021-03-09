# RLGan_cs236g

## File Explanation

## Training

## Testing
In order to run test for GAN model:
```shell
python gan_main.py --train False --pretrained_path './models/GAN_train/10000_G.pth' --batch_size 10 --save_step 1.0 --g_conv_dim 16 --z_dim 2
```
This runs `evaulate` of `Tester` class from `gan_tester.py` with model saved at `'./models/GAN_train/10000_G.pth'`
