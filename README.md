# RLGan_cs236g

## File Explanation
### Auto-Encoder
`AE_main.py` file to run for training and testing of auto-encoder model.
`AE.py` file containing model design.
`AE_trainer.py` file containing trainer class for auto-encoder.

### GAN
`gan_main.py` file to run for training and testing of GAN model.
`gan.py` file containing model design.
`gan_trainer.py` file containing trainer class for GAN.
`gan_tester.py` file containing tester class for GAN.

### MNIST Classifier
`MNISTClassifier_main.py` file to run for training and testing of classifier model.
`MNISTClassifier.py` file containing model design.
`MNISTClassifier_trainer.py` file containing trainer class for classifier.

## Training
In order to train for GAN model:
```shell
python gan_main.py --batch_size 50 --save_step 1.0 --g_lr 0.00005 --d_lr 0.00005  --z_dim 2 --g_conv_dim 16 --d_conv_dim 16 --adv_loss hinge
```
This runs `train` of `Trainer` class from `gan_trainer.py` with models and  sample results saved at default paths of `'./models/GAN_train'` and `'./checkPoints/GAN_train'`.

important arguments

`total_step` as total number of steps to train model. `pretrained_num` as step to continue training from, `g_conv_dim` and `d_conv_dim` as number of channels in hidden layer of generator and discriminator as a measure of model capacity, `z_dim` input dimension of GAN model. `g_lr` and `d_lr` as learning rate of models.

## Testing
In order to run test for GAN model:
```shell
python gan_main.py --train False --pretrained_path './models/GAN_train/10000_G.pth' --batch_size 10 --g_conv_dim 16 --z_dim 2
```
This runs `evaulate` of `Tester` class from `gan_tester.py` with model saved at `'./models/GAN_train/10000_G.pth'`.

important arguments

`train` False to test GAN, `pretrained_path` as model path, `g_conv_dim` as number of channels in hidden layer of generator as a measure of model capacity, `z_dim` input dimension of GAN model.
