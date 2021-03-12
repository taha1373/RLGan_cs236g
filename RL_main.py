import argparse

import torch
import torch.utils.data
import torch.nn.parallel
from models.lossess import ChamferLoss, NLL, MSE, Norm


import Datasets
import models
import torchvision.transforms as transforms

import gpv_transforms
import pc_transforms
from visualizer import Visualizer
from utils import save_checkpoint,AverageMeter,get_n_params


def str2bool(v):
    """
    Parameters
    ----------
    v : str
        string of False or True
    """
    return v.lower() in ('true')


def parse_args(args):
    """
    parse args

    Parameters
    ----------
    args :
        passed arguments
    """
    parser = argparse.ArgumentParser(description='RL Agent Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # arguments for Saving Models
    parser.add_argument('--save_path', default='./RL_ckpt', help='Path to Checkpoints')
    parser.add_argument('--save', default=True, help='Save Models or not ?')  # TODO
    #  parser.add_argument('--pretrained_enc_dec',
    #                      default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/08-08-20:41/ae_pointnet,Adam,200epochs,b24,lr0.001/model_best.pth.tar',
    #                      help='Use Pretrained Encoder and Decoder for training')
    parser.add_argument('--pretrained_enc_dec',
                        default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/09-04-21:05/ae_pointnet,Adam,1000epochs,b50,lr0.0005/model_best.pth.tar',
                        help='Use Pretrained Model for Encoder and Decoder')
    #   parser.add_argument('--pretrained_enc_dec',
    #                       default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/09-12-21:00/ae_pointnet,Adam,1000epochs,b24,lr0.001/model_best.pth.tar',
    #                       help='Use Pretrained Model for testing or resuming training')

    parser.add_argument('--pretrained_G',
                        default='/media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_1dim/999810_G.pth',
                        # 997920 24570  /home/sarmad/Desktop/GANs/Self-Attention-GAN-master/models/sagan_celeb   /media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_1dim/999810_G.pth
                        help='Use Pretrained Generator')  # /media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_16 dim # /home/sarmad/Desktop/GANs/Self-Attention-GAN-master/models/sagan_celeb/999810_G.pth
    parser.add_argument('--pretrained_D',
                        default='/media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_1dim/999810_D.pth',
                        # 997920
                        help='Use Pretrained Discriminator')
    parser.add_argument('--pretrained_Actor',
                        default='/home/ymkim/ShapeCompletion/pointShapeComplete/pytorch_models/DDPG_RLGAN_actor.pth',
                        # 997920
                        help='Use Pretrained Actor')
    parser.add_argument('--pretrained_Critic',
                        default='/home/ymkim/ShapeCompletion/pointShapeComplete/pytorch_models/DDPG_RLGAN_critic.pth',
                        # 997920
                        help='Use Pretrained Critic')
    parser.add_argument('--test_only', default=False, help='Only Test the pre-trained Agent')

    # Data Loader settings
    parser.add_argument('-d', '--data', metavar='DIR',

                        default='',  # Add Path to Complete Train Data set
                        help='Path to Complete Point Cloud Data Set')
    parser.add_argument('--dataIncomplete', metavar='DIR',
                        default='',  # Add Path to Incomplete Validation Data set
                        help='Path to Complete Point Cloud Data Set')
    parser.add_argument('--dataIncompLarge', metavar='DIR',
                        default='',  # Add PAth to Incomplete Test Data set
                        help='Path to Incomplete Point Cloud Data Set')
    parser.add_argument('-s', '--split_value', default=0.9, help='Ratio of train and test data split')
    parser.add_argument('-n', '--dataName', metavar='Data Set Name', default='shapenet', choices=dataset_names)

    # Arguments for Torch Data Loader
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('-w', '--workers', type=int, default=8, help='Set the number of workers')

    # Hyper parameters for RL
    parser.add_argument('--attempts', default=5, type=int)  # Number of tries to give to RL Agent
    parser.add_argument("--policy_name", default="DDPG")  # Policy name TD3 OurDDPG
    parser.add_argument("--env_name", default="RLGAN")  # Policy name TD3 OurDDPG
    parser.add_argument("--state_dim", default=128, type=int)  # State Dimesnions #TODO equal to GFV dims
    parser.add_argument("--max_action", default=10, type=int)  # For Normal Distribution 2.5 is feasible ?

    parser.add_argument("--start_timesteps", default=1e4,  # 1e4
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--save_models", default=True)  # Save Pytorch Models?
    parser.add_argument("--batch_size_actor", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_episodes_steps", default=5, type=int)  # Frequency of delayed policy updates

    # Model Hype-Parameter
    parser.add_argument('--image_size', default=32, type=int)  # TODO original value 64
    parser.add_argument('--z_dim', type=int, default=1)  #
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)

    # Model Settings
    parser.add_argument('-nt', '--net_name', default='auto_encoder', help='Chose The name of your network',
                        choices=['auto_encoder', 'shape_completion'])
    parser.add_argument('--model_generator', default='self_gen_net', help='Chose Your Generator Model Here',
                        choices=['self_gen_net'])
    parser.add_argument('--model_discriminator', default='self_disc_net', help='Chose Your Discriminator Model Here',
                        choices=['self_disc_net'])
    parser.add_argument('--model_encoder', default='encoder_pointnet', help='Chose Your Encoder Model Here',
                        choices=['encoder_pointnet'])
    parser.add_argument('--model_decoder', default='decoder_sonet', help='Chose Your Decoder Model Here',
                        choices=['decoder_sonet'])

    # Visualizer Settings
    parser.add_argument('--name', type=str, default='train',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1001, help='window id of the web display')
    parser.add_argument('--print_freq', type=int, default=40, help='Print Frequency')
    parser.add_argument('--port_id', type=int, default=8102, help='Port id for browser')

    # Setting for Decoder
    # parser.add_argument('--output_pc_num', type=int, default=1280, help='# of output points')
    parser.add_argument('--output_fc_pc_num', type=int, default=256, help='# of fc decoder output points')
    parser.add_argument('--output_conv_pc_num', type=int, default=4096, help='# of conv decoder output points')
    parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
    parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
    parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

    # GPU settings
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0, 1. -1 is no GPU')

    return parser.parse_args()


def main(args,vis_Valid,vis_Valida):
    """ Transforms/ Data Augmentation Tec """
    co_transforms = pc_transforms.Compose([
        #  pc_transforms.Delete(num_points=1466)
          pc_transforms.Jitter_PC(sigma=0.01,clip=0.05),
         # pc_transforms.Scale(low=0.9,high=1.1),
        #   pc_transforms.Shift(low=-0.1,high=0.1),
        #  pc_transforms.Random_Rotate(),
        #  pc_transforms.Random_Rotate_90(),

        # pc_transforms.Rotate_90(args,axis='x',angle=-1.0),# 1.0,2,3,4
        # pc_transforms.Rotate_90(args, axis='z', angle=2.0),
        # pc_transforms.Rotate_90(args, axis='y', angle=2.0),
        # pc_transforms.Rotate_90(args, axis='shape_complete') TODO this is essential for angela data set
    ])

    input_transforms = transforms.Compose([

        pc_transforms.ArrayToTensor(),
        #   transforms.Normalize(mean=[0.5,0.5],std=[1,1])
    ])

    target_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor(),
        #  transforms.Normalize(mean=[0.5, 0.5], std=[1, 1])
    ])

    """-----------------------------------------------Data Loader----------------------------------------------------"""

    if (args.net_name == 'auto_encoder'):
        [train_dataset, valid_dataset] = Datasets.__dict__[args.dataName](input_root=args.data,
                                                                          target_root=None,
                                                                          split=args.split_value,
                                                                          net_name=args.net_name,
                                                                          input_transforms=input_transforms,
                                                                          target_transforms=target_transforms,
                                                                          co_transforms=co_transforms)
        [test_dataset,_] = Datasets.__dict__[args.dataName](input_root=args.dataIncomplete,
                                                                          target_root=None,
                                                                          split=1.0,
                                                                          net_name=args.net_name,
                                                                          input_transforms=input_transforms,
                                                                          target_transforms=target_transforms,
                                                                          co_transforms=co_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)


    """----------------Model Settings-----------------------------------------------"""

    print('Encoder Model: {0}, Decoder Model : {1}'.format(args.model_encoder,args.model_decoder))
    print('GAN Model Generator:{0} & Discriminator : {1} '.format(args.model_generator,args.model_discriminator))


    network_data_AE = torch.load(args.pretrained_enc_dec)

    network_data_G = torch.load(args.pretrained_G)

    network_data_D = torch.load(args.pretrained_D)

    model_encoder = models.__dict__[args.model_encoder](args, num_points=2048, global_feat=True,
                                                        data=network_data_AE, calc_loss=False).cuda()
    model_decoder = models.__dict__[args.model_decoder](args, data=network_data_AE).cuda()

    model_G = models.__dict__[args.model_generator](args, data=network_data_G).cuda()

    model_D = models.__dict__[args.model_discriminator](args, data=network_data_D).cuda()



    params = get_n_params(model_encoder)
    print('| Number of Encoder parameters [' + str(params) + ']...')

    params = get_n_params(model_decoder)
    print('| Number of Decoder parameters [' + str(params) + ']...')



    chamfer = ChamferLoss(args)
    nll = NLL()
    mse = MSE(reduction = 'elementwise_mean')
    norm = Norm(dims=args.z_dim)

    epoch = 0


    test_loss = trainRL(train_loader, valid_loader,test_loader, model_encoder, model_decoder, model_G,model_D, epoch, args, chamfer,nll, mse, norm, vis_Valid,
                     vis_Valida)
    print('Average Loss :{}'.format(test_loss))


if __name__ == '__main__':
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss

    torch.cuda.set_device(args.gpu_id)
    print('Using TITAN XP GPU # :', torch.cuda.current_device())

    print(args)

    """-------------------------------------------------Visualer Initialization-------------------------------------"""

    visualizer = Visualizer(args)

    args.display_id = args.display_id + 10
    args.name = 'Validation'
    vis_Valid = Visualizer(args)
    vis_Valida = []
    args.display_id = args.display_id + 10

    for i in range(1, 15):
        vis_Valida.append(Visualizer(args))
        args.display_id = args.display_id + 10


    main(args,vis_Valid,vis_Valida)
