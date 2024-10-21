import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MambaDreams.models.mambacore import StackedMamba
from MambaDreams.models.vae import VariationalAutoEncoder
from MambaDreams.custom_functions.utils import STMNsampler
import pdb
import csv
from mamba_ssm import Mamba2 as Mamba
import math




# class NeuralWorldModel(nn.Module):
#     def __init__(self, num_frames_per_step, action_dims, image_n, hidden_state_size, image_latent_size_sqrt, reward_prediction_logits_num=41):
#         super(NeuralWorldModel, self).__init__()

#         assert hidden_state_size % num_frames_per_step == 0, "Hidden state size must be divisible by number of frames per step"

#         self.image_n = image_n
#         self.action_dims = action_dims
#         self.action_size = np.prod(action_dims)

#         self.frames_per_obs = num_frames_per_step

#         self.image_n = image_n

#         self.hidden_state_size = hidden_state_size



#         self.per_image_discrete_latent_size_sqrt = image_latent_size_sqrt
#         self.seq_obs_latent = self.per_image_discrete_latent_size_sqrt**2 * self.frames_per_obs

#         self.autoencoder = NeuralAutoEncoder(num_frames_per_step, self.image_n, hidden_state_size, self.per_image_discrete_latent_size_sqrt)

#         #self.state_predictor = NeuralRecurrentDynamicsModel(self.hidden_state_size, self.seq_obs_latent, self.action_size, self.frames_per_obs, self.per_image_discrete_latent_size_sqrt)
#         self.seq_model = NeuralSeqModel(self.hidden_state_size, self.seq_obs_latent, self.action_size, self.frames_per_obs, self.per_image_discrete_latent_size_sqrt)
#         self.rep_model = NeuralRepModel(self.hidden_state_size, self.seq_obs_latent, self.frames_per_obs, self.per_image_discrete_latent_size_sqrt)

#         self.reward_model = NeuralControlCritic(self.hidden_state_size, self.frames_per_obs, self.frames_per_obs, internal_h_state_multiplier=8, reward_prediction_logits_num=reward_prediction_logits_num)



#     def encode_obs(self, obs, hidden_state):
#         batch_dim = obs.shape[0]
#         z, dist = self.autoencoder.encode(obs, hidden_state)
#         z = z.view(batch_dim, self.seq_obs_latent)
#         return z

#     def forward(self, obs, action, hidden_state):
#         batch_dim = obs.shape[0]


#         pred_obs_lat_sample, pred_obs_lat_dist = self.rep_model.forward(hidden_state)
#         predicted_rewards_logits, predicted_rewards_logits_ema = self.reward_model.forward(hidden_state)
        
#         decoded_obs, obs_lats_sample, obs_lats_dist = self.autoencoder(obs, hidden_state)
#         decoded_obs = decoded_obs.view(batch_dim, self.frames_per_obs, self.image_n, self.image_n)

#         obs_lats_sample = obs_lats_sample.view(batch_dim, self.seq_obs_latent)
        
        
                
#         hidden_state = self.seq_model.forward(obs_lats_sample, hidden_state, action)


#         return decoded_obs, pred_obs_lat_sample, obs_lats_sample, hidden_state, predicted_rewards_logits, predicted_rewards_logits_ema, obs_lats_dist, pred_obs_lat_dist    


#     def imagine_forward(self, hidden_state, obs_latent, action):
#         batch_size = hidden_state.shape[0]

#         predicted_reward_logits = self.reward_model.forward(hidden_state, ema_forward=False)
#         predicted_reward_logits = predicted_reward_logits.view(batch_size*self.frames_per_obs, self.reward_model.reward_prediction_logits_num)

#         predicted_reward = logits_to_reward(predicted_reward_logits).view(batch_size, self.frames_per_obs)
         

#         pred_obs_lat_sample, _ = self.rep_model.forward(hidden_state)

#         hidden_state = self.seq_model.forward(obs_latent, hidden_state, action)

#         return hidden_state, pred_obs_lat_sample, predicted_reward


def least_power_of_2(value):
    if value <= 0:
        return 1
    
    exponent = math.ceil(math.log(value, 2))
    return 2 ** exponent

class LastTokenSelector(nn.Module):
    def forward(self, x):
        return x[:, -1, :]

class AddUniformBase(nn.Module):
    def forward(self, x):
        return (0.99 * x) + (0.01*(1.0/x.shape[-1]))
    
class WorldModel(nn.Module):
    def __init__(self, action_dims, image_side_size, image_latent_category_size):
        super(WorldModel, self).__init__()

        self.action_dims = action_dims
        self.action_size = np.prod(action_dims)
        self.image_latent_category_size = image_latent_category_size
        self.image_latent_size = image_latent_category_size**2
        self.reward_size = 1
        self.image_side_size = image_side_size

        self.raw_hidden_size = self.image_latent_size + self.action_size + self.reward_size

        self.hidden_size = least_power_of_2(self.raw_hidden_size)

        self.state_pad_size = self.hidden_size - self.raw_hidden_size
        
        self.vae = VariationalAutoEncoder(image_side_size, image_latent_category_size)

        self.predictor = StackedMamba(self.hidden_size, 8)

        self.image_predict_last_dist = nn.Sequential(
            Mamba(self.hidden_size),
            nn.Linear(self.hidden_size, self.image_latent_size),
            nn.Unflatten(-1, (self.image_latent_category_size, self.image_latent_category_size)),
            nn.Softmax(dim=-1),
            AddUniformBase(),
            nn.Flatten(start_dim=2),
        )

        self.reward_predictor = nn.Sequential(
            Mamba(self.hidden_size),
            nn.Linear(self.hidden_size, self.reward_size),
        )

        self.image_lat_sampler = STMNsampler()


    def encode_obs(self, obs):
        batch_size = obs.shape[0]
        seq_length = obs.shape[1]

        obs = obs.reshape(batch_size*seq_length, 1, self.image_side_size, self.image_side_size)
        decoded_obs, latent_samples, latent_distribution = self.vae(obs)

        latent_samples = latent_samples.view(batch_size, seq_length, self.image_latent_size)
        latent_distribution = latent_distribution.view(batch_size, seq_length, self.image_latent_size)
        decoded_obs = decoded_obs.view(batch_size, seq_length, self.image_side_size, self.image_side_size)


        return decoded_obs, latent_samples, latent_distribution




    def forward(self, obs_lats, actions, rewards):
        batch_size = obs_lats.shape[0]
        seq_length = obs_lats.shape[1]


        latent_samples = obs_lats.view(batch_size, seq_length, self.image_latent_size)
        #latent_distribution = latent_distribution.view(batch_size, seq_length, *latent_distribution.shape[1:])

        actions = torch.unsqueeze(actions, dim=-1)
        states = torch.cat([latent_samples, actions, rewards], dim=-1)
        # pads states
        states = torch.cat([states, torch.zeros(batch_size, seq_length, self.state_pad_size+1).to(states.device)], dim=-1)

        hidden_state = self.predictor(states)

        # takes last token
        predicted_image_lat_dists = self.image_predict_last_dist(hidden_state)

        predicted_next_image_lat_samples = self.image_lat_sampler(predicted_image_lat_dists.view(batch_size*seq_length*self.image_latent_category_size, self.image_latent_category_size)).view(batch_size, seq_length, self.image_latent_size)

        predicted_next_decoded_obs = self.vae.decode(predicted_next_image_lat_samples.view(batch_size*seq_length, self.image_latent_size)).view(batch_size, seq_length, self.image_side_size, self.image_side_size)
        
        predicted_rewards = self.reward_predictor(hidden_state)

        return predicted_next_decoded_obs, predicted_image_lat_dists, predicted_rewards, hidden_state



    def sample_obs_from_lat_dist(self, lat_dist):
        batch_size = lat_dist.shape[0]
        seq_length = lat_dist.shape[1]

        image_lat_samples = self.image_lat_sampler(lat_dist.view(batch_size*seq_length*self.image_latent_category_size, self.image_latent_category_size)).view(batch_size, seq_length, self.image_latent_size)


        return self.vae.decode(image_lat_samples.view(batch_size*seq_length, self.image_latent_size)).view(batch_size, seq_length, self.image_side_size, self.image_side_size)









