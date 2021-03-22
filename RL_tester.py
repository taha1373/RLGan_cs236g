import torch
import torch.utils.data
import torch.nn.parallel

from EnvClass import Env

import os

from utils import RL_dataloader, display_env
from RL import TD3


class Tester(object):
    """RL tester"""

    def __init__(self, args, test_loader, model_encoder, model_decoder, model_g, model_d,
                 model_classifier):
        """
        initialize RL trainer

        Parameters
        ----------
        args : dict
            args dictionary look at :func: '~gan_main.parse_args'
        test_loader : torch.utils.data.dataloader
            torch data loader of test dataset
        model_encoder : torch.nn.Module
            encoder model
        model_decoder : torch.nn.Module
            decoder model
        model_g : torch.nn.Module
            generator model
        model_d : torch.nn.Module
            discriminator model
        """
        self.device = args.device

        self.test_loader = RL_dataloader(test_loader)

        self.batch_size = args.batch_size
        self.save_models = args.save_models
        self.max_episodes_steps = args.max_episodes_steps

        self.z_dim = args.z_dim
        self.max_action = args.max_action

        self.encoder = model_encoder
        self.decoder = model_decoder
        self.G = model_g
        self.D = model_d
        self.model_classifier = model_classifier

        self.env = Env(args, self.G, self.D, self.model_classifier, self.decoder)

        self.state_dim = args.state_dim
        self.action_dim = args.z_dim
        self.max_action = args.max_action

        self.policy = TD3(self.device, self.state_dim, self.action_dim, self.max_action)

        self.model_path = os.path.join(args.model_dir, 'RL_train')
        self.save_path = os.path.join(args.result_dir, 'RL_test')
        os.makedirs(self.save_path, exist_ok=True)

        self.policy.load(args.model_name, directory=self.model_path)

    def evaluate(self):
        """
        evaluate RL
        """
        # name of result for saving
        episode_num = 0
        while True:

            # if loader present create until data finished, else random generation of input data
            print('input loader')
            try:
                state_t, episode_target = self.test_loader.next_data()
                state = self.env.set_state(state_t)
                done = False
                episode_return = 0
            except:
                break

            while not done:
                # ================== Run RL ================== #
                with torch.no_grad():
                    action = self.policy.select_action(state)
                    action_t = torch.tensor(action).to(self.device).unsqueeze(dim=0)
                    # Perform action
                    next_state, reward, done = self.env(action_t, episode_target)

                state = next_state
                episode_return += reward

            print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_return))
            episode_num += 1

            self.env.reset()

            with torch.no_grad():
                out_state = torch.FloatTensor(state.reshape((1, 1, -1))).to(self.device)
                state_image = self.decoder(state_t)
                out_state_image = self.decoder(out_state)
            display_env(state_image, out_state_image, episode_return,
                        os.path.join(self.save_path, "episode_{}".format(episode_num)),
                        target=episode_target.detach().cpu().numpy())

            yield state_image, out_state_image, episode_return
