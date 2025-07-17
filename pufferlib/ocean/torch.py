from types import SimpleNamespace
from typing import Any, Tuple

from gymnasium import spaces

from torch import nn
import torch
from torch.distributions.normal import Normal
from torch import nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy
from pufferlib.models import Convolutional as Conv
Recurrent = pufferlib.models.LSTMWrapper
from pufferlib.pytorch import layer_init, _nativize_dtype, nativize_tensor
import numpy as np

import einops

class Boids(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(4, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
        )
        self.action_vec = tuple(env.single_action_space.nvec)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, sum(self.action_vec)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        batch, n, = observations.shape
        return self.network(observations.reshape(batch, n//4, 4)).max(dim=1)[0]

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden).split(self.action_vec, dim=1)
        return action, value

class NMMO3LSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)

class NMMO3(nn.Module):
    def __init__(self, env, hidden_size=512, output_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        #self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.factors = np.array([4, 4, 17, 5, 3, 5, 5, 5, 7, 4])
        offsets = torch.tensor([0] + list(np.cumsum(self.factors)[:-1])).view(1, -1, 1, 1)
        self.register_buffer('offsets', offsets)
        self.cum_facs = np.cumsum(self.factors)

        self.multihot_dim = self.factors.sum()
        self.is_continuous = False

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.multihot_dim, 128, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(128, 128, 3, stride=1)),
            nn.Flatten(),
        )

        self.player_discrete_encoder = nn.Sequential(
            nn.Embedding(128, 32),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(1817, hidden_size)),
            nn.ReLU(),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def forward(self, x, state=None):
        hidden = self.encode_observations(x)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        batch = observations.shape[0]
        ob_map = observations[:, :11*15*10].view(batch, 11, 15, 10)
        ob_player = observations[:, 11*15*10:-10]
        ob_reward = observations[:, -10:]

        batch = ob_map.shape[0]
        map_buf = torch.zeros(batch, 59, 11, 15, dtype=torch.float32, device=observations.device)
        codes = ob_map.permute(0, 3, 1, 2) + self.offsets
        map_buf.scatter_(1, codes, 1)
        ob_map = self.map_2d(map_buf)

        player_discrete = self.player_discrete_encoder(ob_player.int())

        obs = torch.cat([ob_map, player_discrete, ob_player.to(ob_map.dtype), ob_reward], dim=1)
        obs = self.proj(obs)
        return obs

    def decode_actions(self, flat_hidden):
        flat_hidden = self.layer_norm(flat_hidden)
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class Terraform(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.local_net_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.global_net_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.net_1d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(5, hidden_size)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size + cnn_channels*5, hidden_size)),
            nn.ReLU(),
        )
        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        # breakpoint()
        obs_2d = observations[:, :242].reshape(-1, 2, 11, 11).float()
        obs_1d = observations[:, 242:247].reshape(-1, 5).float()
        location_2d = observations[:, 247:].reshape(-1,2, 6, 6).float()
        hidden_local_2d = self.local_net_2d(obs_2d)
        hidden_global_2d = self.global_net_2d(location_2d)
        hidden_1d = self.net_1d(obs_1d)
        hidden = torch.cat([hidden_local_2d, hidden_global_2d, hidden_1d], dim=1)
        return self.proj(hidden)

    def decode_actions(self, hidden):
        action = self.actor(hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        #action = [head(hidden) for head in self.actor]
        value = self.value(hidden)
        return action, value


class G2048(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(1, cnn_channels, 2, stride=1)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 2, stride=1)),
            nn.Flatten(),
            nn.GELU(),
            pufferlib.pytorch.layer_init(
            nn.Linear(128, hidden_size), std=0.01),
        )

        self.decoder = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward(self, x, state=None):
        return self.forward_eval(x, state)

    def encode_observations(self, observations, state=None):
        #observations = F.one_hot(observations.long(), 16).view(-1, 16, 4, 4).float()
        observations = observations.float().view(-1, 1, 4, 4)
        return self.cnn(observations)

    def decode_actions(self, hidden):
        action = self.decoder(hidden)
        value = self.value(hidden)
        return action, value

class Snake(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        encode_dim = cnn_channels

        '''
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(8, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(encode_dim, hidden_size)),
            nn.ReLU(),
        )
 
        '''
        self.encoder= torch.nn.Sequential(
            nn.Linear(8*np.prod(env.single_observation_space.shape), hidden_size),
            nn.GELU(),
        )
        self.decoder = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        #observations = F.one_hot(observations.long(), 8).permute(0, 3, 1, 2).float()
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).view(-1, 11*11*8).float()
        return self.encoder(observations)

    def decode_actions(self, hidden):
        action = self.decoder(hidden)
        value = self.value(hidden)
        return action, value

'''
class Snake(pufferlib.models.Default):
    def __init__(self, env, hidden_size=128):
        super().__init__()

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).view(-1, 11*11*8).float()
        super().encode_observations(observations, state)
'''

class Grid(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(32, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
            nn.ReLU(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            num_actions = env.single_action_space.n
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_actions), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        hidden = observations.view(-1, 11, 11).long()
        hidden = F.one_hot(hidden, 32).permute(0, 3, 1, 2).float()
        hidden = self.network(hidden)
        return hidden

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden)
            return action, value

class Go(nn.Module):
    def __init__(self, env, cnn_channels=64, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        # 3 categories 2 boards. 
        # categories = player, opponent, empty
        # boards = current, previous
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride = 1)),
            nn.Flatten(),
        )

        obs_size = env.single_observation_space.shape[0]
        self.grid_size = int(np.sqrt((obs_size-2)/2))
        output_size = self.grid_size - 4
        cnn_flat_size = cnn_channels * output_size * output_size
        
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(2,32))
        
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(cnn_flat_size + 32, hidden_size))

        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)
   
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        grid_size = int(np.sqrt((observations.shape[1] - 2) / 2))
        full_board = grid_size * grid_size 
        black_board = observations[:, :full_board].view(-1,1, grid_size,grid_size).float()
        white_board = observations[:, full_board:-2].view(-1,1, grid_size, grid_size).float()
        board_features = torch.cat([black_board, white_board],dim=1)
        flat_feature1 = observations[:, -2].unsqueeze(1).float()
        flat_feature2 = observations[:, -1].unsqueeze(1).float()
        # Pass board through cnn
        cnn_features = self.cnn(board_features)
        # Pass extra feature
        flat_features = torch.cat([flat_feature1, flat_feature2],dim=1)
        flat_features = self.flat(flat_features)
        # pass all features
        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(features))

        return features

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden)
        return action, value
    
class MOBA(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(16 + 3, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
        )
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(26, 128))
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(128+cnn_channels, hidden_size))

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        cnn_features = observations[:, :-26].view(-1, 11, 11, 4).long()
        map_features = F.one_hot(cnn_features[:, :, :, 0], 16).permute(0, 3, 1, 2).float()
        extra_map_features = (cnn_features[:, :, :, -3:].float() / 255).permute(0, 3, 1, 2)
        cnn_features = torch.cat([map_features, extra_map_features], dim=1)
        #print('observations 2d: ', map_features[0].cpu().numpy().tolist())
        cnn_features = self.cnn(cnn_features)
        #print('cnn features: ', cnn_features[0].detach().cpu().numpy().tolist())

        flat_features = observations[:, -26:].float() / 255.0
        #print('observations 1d: ', flat_features[0, 0])
        flat_features = self.flat(flat_features)
        #print('flat features: ', flat_features[0].detach().cpu().numpy().tolist())

        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(F.relu(features)))
        #print('features: ', features[0].detach().cpu().numpy().tolist())
        return features

    def decode_actions(self, flat_hidden):
        #print('lstm: ', flat_hidden[0].detach().cpu().numpy().tolist())
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

            #argmax_samples = [torch.argmax(a, dim=1).detach().cpu().numpy().tolist() for a in action]
            #print('argmax samples: ', argmax_samples)

            return action, value

class TrashPickup(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(5, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = observations.view(-1, 5, 11, 11).float()
        return self.network(observations)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class TowerClimbLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size = 256, hidden_size = 256):
        super().__init__(env, policy, input_size, hidden_size)

class TowerClimb(nn.Module):
    def __init__(self, env, cnn_channels=16, hidden_size = 256, **kwargs):
        self.hidden_size = hidden_size
        self.is_continuous = False
        super().__init__()
        self.network = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(1, cnn_channels, 3, stride = 1)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(cnn_channels, cnn_channels, 3, stride=1)),
                nn.Flatten()       
        )
        cnn_flat_size = cnn_channels * 1 * 1 * 5

        # Process player obs
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(3,16))

        # combine
        self.proj = pufferlib.pytorch.layer_init(
                nn.Linear(cnn_flat_size + 16, hidden_size))
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        board_state = observations[:,:225]
        player_info = observations[:, -3:] 
        board_features = board_state.view(-1, 1, 5,5,9).float()
        cnn_features = self.network(board_features)
        flat_features = self.flat(player_info.float())
        
        features = torch.cat([cnn_features,flat_features],dim = 1)
        features = self.proj(features)
        return features
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        
        return action, value


class ImpulseWarsLSTM(Recurrent):
    def __init__(self, env: pufferlib.PufferEnv, policy: nn.Module, input_size: int = 512, hidden_size: int = 512):
        super().__init__(env, policy, input_size, hidden_size)


class ImpulseWarsPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.PufferEnv,
        cnn_channels: int = 64,
        weapon_type_embedding_dims: int = 2,
        input_size: int = 512,
        hidden_size: int = 512,
        batch_size: int = 131_072,
        num_drones: int = 2,
        continuous: bool = False,
        is_training: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.is_continuous = continuous

        self.numDrones = num_drones
        self.isTraining = is_training
        from pufferlib.ocean.impulse_wars import binding
        self.obsInfo = SimpleNamespace(**binding.get_consts(self.numDrones))

        self.discreteFactors = np.array(
            [self.obsInfo.wallTypes] * self.obsInfo.numNearWallObs
            + [self.obsInfo.wallTypes + 1] * self.obsInfo.numFloatingWallObs
            + [self.numDrones + 1] * self.obsInfo.numProjectileObs,
        )
        discreteOffsets = torch.tensor([0] + list(np.cumsum(self.discreteFactors)[:-1]), device=device).view(
            1, -1
        )
        self.register_buffer("discreteOffsets", discreteOffsets, persistent=False)
        self.discreteMultihotDim = self.discreteFactors.sum()

        multihotBuffer = torch.zeros(batch_size, self.discreteMultihotDim, device=device)
        self.register_buffer("multihotOutput", multihotBuffer, persistent=False)

        # most of the observation is a 2D array of bytes, but the end
        # contains around 200 floats; this allows us to treat the end
        # of the observation as a float array
        _, *self.dtype = _nativize_dtype(
            np.dtype((np.uint8, (self.obsInfo.continuousObsBytes,))),
            np.dtype((np.float32, (self.obsInfo.continuousObsSize,))),
        )
        self.dtype = tuple(self.dtype)

        self.weaponTypeEmbedding = nn.Embedding(self.obsInfo.weaponTypes, weapon_type_embedding_dims)

        # each byte in the map observation contains 4 values:
        # - 2 bits for wall type
        # - 1 bit for is floating wall
        # - 1 bit for is weapon pickup
        # - 3 bits for drone index
        self.register_buffer(
            "unpackMask",
            torch.tensor([0x60, 0x10, 0x08, 0x07], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer("unpackShift", torch.tensor([5, 4, 3, 0], dtype=torch.uint8), persistent=False)

        self.mapObsInputChannels = (self.obsInfo.wallTypes + 1) + 1 + 1 + self.numDrones
        self.mapCNN = nn.Sequential(
            layer_init(
                nn.Conv2d(
                    self.mapObsInputChannels,
                    cnn_channels,
                    kernel_size=5,
                    stride=3,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnnOutputSize = self._computeCNNShape()

        featuresSize = (
            cnnOutputSize
            + (self.obsInfo.numNearWallObs * (self.obsInfo.wallTypes + self.obsInfo.nearWallPosObsSize))
            + (
                self.obsInfo.numFloatingWallObs
                * (self.obsInfo.wallTypes + 1 + self.obsInfo.floatingWallInfoObsSize)
            )
            + (
                self.obsInfo.numWeaponPickupObs
                * (weapon_type_embedding_dims + self.obsInfo.weaponPickupPosObsSize)
            )
            + (
                self.obsInfo.numProjectileObs
                * (weapon_type_embedding_dims + self.obsInfo.projectileInfoObsSize + self.numDrones + 1)
            )
            + ((self.numDrones - 1) * (weapon_type_embedding_dims + self.obsInfo.enemyDroneObsSize))
            + (self.obsInfo.droneObsSize + weapon_type_embedding_dims)
            + self.obsInfo.miscObsSize
        )

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(featuresSize, input_size)),
            nn.ReLU(),
        )

        if self.is_continuous:
            self.actorMean = layer_init(nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.actorLogStd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))
        else:
            self.actionDim = env.single_action_space.nvec.tolist()
            self.actor = layer_init(nn.Linear(hidden_size, sum(self.actionDim)), std=0.01)

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, obs: torch.Tensor, state = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_observations(obs)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def unpack(self, batchSize: int, obs: torch.Tensor) -> torch.Tensor:
        # prepare map obs to be unpacked
        mapObs = obs[:, : self.obsInfo.mapObsSize].reshape((batchSize, -1, 1))
        # unpack wall types, weapon pickup types, and drone indexes
        mapObs = (mapObs & self.unpackMask) >> self.unpackShift
        # reshape so channels are first, required for torch conv2d
        return mapObs.permute(0, 2, 1).reshape(
            (batchSize, 4, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns)
        )

    def encode_observations(self, obs: torch.Tensor, state: Any = None) -> torch.Tensor:
        batchSize = obs.shape[0]

        mapObs = self.unpack(batchSize, obs)

        # one hot encode wall types
        wallTypeObs = mapObs[:, 0, :, :].long()
        wallTypes = F.one_hot(wallTypeObs, self.obsInfo.wallTypes + 1).permute(0, 3, 1, 2).float()

        # unsqueeze floating wall booleans (is wall a floating wall)
        floatingWallObs = mapObs[:, 1, :, :].unsqueeze(1)

        # unsqueeze map pickup booleans (does map tile contain a weapon pickup)
        mapPickupObs = mapObs[:, 2, :, :].unsqueeze(1)

        # one hot drone indexes
        droneIndexObs = mapObs[:, 3, :, :].long()
        droneIndexes = F.one_hot(droneIndexObs, self.numDrones).permute(0, 3, 1, 2).float()

        # combine all map observations and feed through CNN
        mapObs = torch.cat((wallTypes, floatingWallObs, mapPickupObs, droneIndexes), dim=1)
        map = self.mapCNN(mapObs)

        # process discrete observations
        multihotInput = (
            obs[:, self.obsInfo.nearWallTypesObsOffset : self.obsInfo.projectileTypesObsOffset]
            + self.discreteOffsets
        )
        multihotOutput = self.multihotOutput[:batchSize].zero_()
        multihotOutput.scatter_(1, multihotInput.long(), 1)

        weaponTypeObs = obs[:, self.obsInfo.projectileTypesObsOffset : self.obsInfo.discreteObsSize].int()
        weaponTypes = self.weaponTypeEmbedding(weaponTypeObs).float()
        weaponTypes = torch.flatten(weaponTypes, start_dim=1, end_dim=-1)

        # process continuous observations
        continuousObs = nativize_tensor(obs[:, self.obsInfo.continuousObsOffset :], self.dtype)
        # combine all observations and feed through final linear encoder
        features = torch.cat((map, multihotOutput, weaponTypes, continuousObs), dim=-1)

        return self.encoder(features)

    def decode_actions(self, hidden: torch.Tensor):
        if self.is_continuous:
            actionMean = self.actorMean(hidden)
            if self.isTraining:
                actionLogStd = self.actorLogStd.expand_as(actionMean)
                actionStd = torch.exp(actionLogStd)
                action = Normal(actionMean, actionStd)
            else:
                action = actionMean
        else:
            action = self.actor(hidden)
            action = torch.split(action, self.actionDim, dim=1)

        value = self.critic(hidden)

        return action, value

    def _computeCNNShape(self) -> int:
        mapSpace = spaces.Box(
            low=0,
            high=1,
            shape=(self.mapObsInputChannels, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns),
            dtype=np.float32,
        )

        with torch.no_grad():
            t = torch.as_tensor(mapSpace.sample()[None])
            return self.mapCNN(t).shape[1]

class GPUDrive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(6, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )
        max_road_objects = 13
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_road_objects, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )
        max_partner_objects = 7
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_partner_objects, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )

        '''
        self.post_mask_road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size)),
        )
        self.post_mask_partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size)),
        )
        '''
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3*input_size,  hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)
    
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)
   
    def encode_observations(self, observations, state=None):
        ego_dim = 6
        partner_dim = 63 * 7
        road_dim = 200*7
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim+partner_dim]
        road_obs = observations[:, ego_dim+partner_dim:ego_dim+partner_dim+road_dim]
        
        partner_objects = partner_obs.view(-1, 63, 7)
        road_objects = road_obs.view(-1, 200, 7)
        road_continuous = road_objects[:, :, :6]  # First 6 features
        road_categorical = road_objects[:, :, 6]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, 200, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        
        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        
        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value

class Tetris(nn.Module):
    def __init__(
        self, 
        env, 
        cnn_channels=32,
        input_size=128,
        hidden_size=128,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn_channels =  cnn_channels   
        self.n_cols = env.n_cols
        self.n_rows = env.n_rows
        self.scalar_input_size = (6 + 7 * (env.deck_size + 1))
        self.flat_conv_size = cnn_channels * 3 * 10
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        self.conv_grid = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(2, cnn_channels, kernel_size=(5, 3), stride=(2,1), padding=(2,1))),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(5, 3), stride=(2,1), padding=(2,1))),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(5, 5), stride=(2,1), padding=(2,2))),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(self.flat_conv_size, input_size)),
        )

        self.fc_scalar = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.scalar_input_size, input_size)),
            nn.ReLU(),
        )

        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(2 * input_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 7), std=0.01),
            nn.Flatten()
        )

        self.value_fn = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1)),
            nn.ReLU(),
        )

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations) 
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        grid_info = observations[:, 0:(self.n_cols * self.n_rows)].view(B, self.n_rows, self.n_cols)  # (B, n_rows, n_cols)
        grid_info = torch.stack([(grid_info == 1).float(), (grid_info == 2).float()], dim=1)  # (B, 2, n_rows, n_cols)
        scalar_info = observations[:, (self.n_cols * self.n_rows):(self.n_cols * self.n_rows + self.scalar_input_size)].float()

        grid_feat = self.conv_grid(grid_info)  # (B, input_size)
        scalar_feat = self.fc_scalar(scalar_info)  # (B, input_size)

        combined = torch.cat([grid_feat, scalar_feat], dim=-1)  # (B, 2 * input_size)
        features = self.proj(combined)  # (B, hidden_size)
        return features

    def decode_actions(self, hidden):
        action = self.actor(hidden)  # (B, 4 * n_cols)
        value = self.value_fn(hidden)  # (B, 1)
        return action, value

class Drone(nn.Module):
    ''' Drone policy. Flattens obs and applies a linear layer.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            self.encoder = torch.nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space.shape), hidden_size),
                nn.GELU(),
            )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, sum(self.action_nvec)), std=0.01)
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values

class FastSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape
        qkv = self.qkv_proj(x)                      # → (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)              # each (B, S, D)
        # reshape to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)

        q_h = q.half()
        k_h = k.half()
        v_h = v.half()

        out_h = F.scaled_dot_product_attention(
            q_h, k_h, v_h,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False,
        )

        out = out_h.float()
        out = out.reshape(B, S, D)
        return self.out_proj(out)

class PolyTMLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size = 256, hidden_size = 256):
        super().__init__(env, policy, input_size, hidden_size)    

class PolyTM(nn.Module):
    def __init__(
        self, 
        env, 
        hidden_size=256,
        alpha_emb_dim=4,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.is_continuous = False
        
        self.num_observations = env.num_obs
        self.problem_size = env.problem_size

        self.work_tape_size = env.work_tape_size
        self.state_tape_size = env.state_tape_size
        self.result_tape_size = env.result_tape_size
        self.max_tape_size = max(self.work_tape_size, self.state_tape_size, self.result_tape_size)

        self.work_observation_window = env.work_observation_window
        self.state_observation_window = env.state_observation_window
        self.result_observation_window = env.result_observation_window
        self.max_observation_window = max(self.work_observation_window, self.state_observation_window, self.result_observation_window)

        self.num_heads = env.num_heads
        self.tape_alphabet = env.tape_alphabet

        self.num_work_heads = env.num_work_heads
        self.num_state_heads = env.num_state_heads
        self.num_result_heads = env.num_result_heads
        self.max_num_heads = max(self.num_work_heads, self.num_state_heads, self.num_result_heads)

        self.alpha_emb_dim = alpha_emb_dim

        self.obs_sections = self._calculate_sections()

        self.alpha_embd = nn.Sequential(nn.Linear(1, self.alpha_emb_dim), nn.GELU())
        self.sec_embd = nn.Sequential(nn.Linear(1, self.alpha_emb_dim), nn.GELU())
        self.head_idx_embd = nn.Sequential(nn.Linear(1, self.alpha_emb_dim), nn.GELU())
        self.tape_idx_embd = nn.Sequential(nn.Linear(1, self.alpha_emb_dim), nn.GELU())

        self.alpha_prob = nn.Parameter(torch.randn(1) * 1.0 + 1.0)
        
        self.alpha_work = nn.Parameter(torch.randn(1) * 1.0 + 1.0)
        self.beta_work = nn.Parameter(torch.randn(1) * 1.0 + 1.0)
        self.gamma_work = nn.Parameter(torch.randn(1) * 1.0 + 1.0)

        self.alpha_state = nn.Parameter(torch.randn(1) * 1.0 + 1.0)
        self.beta_state = nn.Parameter(torch.randn(1) * 1.0 + 1.0)
        self.gamma_state = nn.Parameter(torch.randn(1) * 1.0 + 1.0)

        self.alpha_result = nn.Parameter(torch.randn(1) * 1.0 + 1.0)
        self.beta_result = nn.Parameter(torch.randn(1) * 1.0 + 1.0)
        self.gamma_result = nn.Parameter(torch.randn(1) * 1.0 + 1.0)

        self.norm = nn.LayerNorm(4 * self.alpha_emb_dim)

        self.pool_mlp = nn.Sequential(
            nn.Linear(self.alpha_emb_dim, self.alpha_emb_dim),
            nn.GELU(),
            nn.Linear(self.alpha_emb_dim, 1),
        )

        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.alpha_emb_dim * 4, hidden_size), std=0.01),
            nn.GELU()
        )


        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)


        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=0.01)

        self._precompute_static_indices()

    def _precompute_static_indices(self):
        device = next(self.parameters()).device

        self.register_buffer('problem_sec_base', torch.zeros(1))
        self.register_buffer('work_sec_base', torch.ones(1))
        self.register_buffer('state_sec_base', 2 * torch.ones(1))
        self.register_buffer('result_sec_base', 3 * torch.ones(1))

        self.register_buffer('sec_base', torch.cat([self.problem_sec_base, self.work_sec_base, self.state_sec_base, self.result_sec_base], dim=0).unsqueeze_(0).unsqueeze_(-1))

        work_head_idx = torch.arange(0, self.num_work_heads)
        self.register_buffer('work_head_base', work_head_idx.float())
        
        state_head_idx = torch.arange(0, self.num_state_heads)
        self.register_buffer('state_head_base', state_head_idx.float())
        
        result_head_idx = torch.arange(0, self.num_result_heads)
        self.register_buffer('result_head_base', result_head_idx.float())

        self.register_buffer('head_base', torch.cat([self.work_head_base, self.state_head_base, self.result_head_base], dim=0).unsqueeze_(0))

    def _calculate_sections(self):
        sections = {}
        start = 0
        
        # Problem section
        sections['problem'] = (start, start + self.problem_size)
        start += self.problem_size

        #work head idx section
        sections['work_head_idx'] = (start, start + self.num_work_heads)
        start += self.num_work_heads
        
        #state head idx section
        sections['state_head_idx'] = (start, start + self.num_state_heads)
        start += self.num_state_heads

        #result head idx section
        sections['result_head_idx'] = (start, start + self.num_result_heads)
        start += self.num_result_heads

        # Work head sections
        work_size = 2 * self.work_observation_window + 1
        sections['work_heads'] = []
        for i in range(self.num_work_heads):
            sections['work_heads'].append((start, start + work_size))
            start += work_size
            
        # State head sections
        state_size = 2 * self.state_observation_window + 1
        sections['state_heads'] = []
        for i in range(self.num_state_heads):
            sections['state_heads'].append((start, start + state_size))
            start += state_size
            
        # Result head sections
        result_size = 2 * self.result_observation_window + 1
        sections['result_heads'] = []
        for i in range(self.num_result_heads):
            sections['result_heads'].append((start, start + result_size))
            start += result_size
            
        return sections
    
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations) 
        actions, value = self.decode_actions(hidden)
        
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def _efficient_emb_comp(self, actual_obs, head_ind, batch_sz, device):
        actual_obs = actual_obs.unsqueeze(-1)
        emb_obs = self.alpha_embd(actual_obs)

        prob_emb_obs = emb_obs[:, :self.problem_size, :]
        
        work_start = self.problem_size
        work_end = work_start + self.num_work_heads * (2 * self.work_observation_window + 1)
        work_emb_obs = emb_obs[:, work_start:work_end, :]
        
        state_start = work_end
        state_end = state_start + self.num_state_heads * (2 * self.state_observation_window + 1)
        state_emb_obs = emb_obs[:, state_start:state_end, :]
        
        result_start = state_end
        result_end = result_start + self.num_result_heads * (2 * self.result_observation_window + 1)
        result_emb_obs = emb_obs[:, result_start:result_end, :]


        emb_obs_weight = self.pool_mlp(emb_obs).squeeze(-1)

        prob_weights = torch.logsumexp(emb_obs_weight[:, :self.problem_size], dim=-1, keepdim=True)
        prob_weights = torch.exp(emb_obs_weight[:, :self.problem_size] - prob_weights)
        prob_pooled = (prob_emb_obs * prob_weights.unsqueeze(-1)).sum(dim=1)
        
        work_weights = torch.logsumexp(emb_obs_weight[:, self.problem_size:work_end], dim=-1, keepdim=True)
        work_weights = torch.exp(emb_obs_weight[:, self.problem_size:work_end] - work_weights)
        work_pooled = (work_emb_obs * work_weights.unsqueeze(-1)).sum(dim=1)
        work_head_weights = work_weights.view(batch_sz, self.num_work_heads, 2 * self.work_observation_window + 1).sum(dim=2)
        
        state_weights = torch.logsumexp(emb_obs_weight[:, work_end:state_end], dim=-1, keepdim=True)
        state_weights = torch.exp(emb_obs_weight[:, work_end:state_end] - state_weights)
        state_pooled = (state_emb_obs * state_weights.unsqueeze(-1)).sum(dim=1)
        state_head_weights = state_weights.view(batch_sz, self.num_state_heads, 2 * self.state_observation_window + 1).sum(dim=2)

        result_weights = torch.logsumexp(emb_obs_weight[:, state_end:result_end], dim=-1, keepdim=True)
        result_weights = torch.exp(emb_obs_weight[:, state_end:result_end] - result_weights)
        result_pooled = (result_emb_obs * result_weights.unsqueeze(-1)).sum(dim=1)
        result_head_weights = result_weights.view(batch_sz, self.num_result_heads, 2 * self.result_observation_window + 1).sum(dim=2)

        sec_emb = self.sec_embd(self.sec_base.expand(batch_sz, -1, -1))
        prob_sec_emb = sec_emb[:, 0, :].squeeze_(1)
        work_sec_emb = sec_emb[:, 1, :].squeeze_(1)
        state_sec_emb = sec_emb[:, 2, :].squeeze_(1)
        result_sec_emb = sec_emb[:, 3, :].squeeze_(1)

        tmp_head_weights = torch.cat([work_head_weights, state_head_weights, result_head_weights], dim=1)

        tmp_head_emb = torch.cat([
            i.sum(dim=1, keepdim=True) for i in torch.split((self.head_base.expand(batch_sz, -1) * tmp_head_weights), (self.num_work_heads, self.num_state_heads, self.num_result_heads), dim=1)
        ], dim=0)

        head_emb = self.head_idx_embd(tmp_head_emb)

        work_head_emb = head_emb[:batch_sz, :]
        state_head_emb = head_emb[batch_sz:2*batch_sz, :]
        result_head_emb = head_emb[2*batch_sz:3*batch_sz, :]

        tmp_tape_emb = torch.cat([
            i.sum(dim=1, keepdim=True) for i in torch.split((head_ind * tmp_head_weights), (self.num_work_heads, self.num_state_heads, self.num_result_heads), dim=1)
        ], dim=0)

        tape_emb = self.tape_idx_embd(tmp_tape_emb)

        work_tape_emb = tape_emb[:batch_sz, :]
        state_tape_emb = tape_emb[batch_sz:2*batch_sz, :]
        result_tape_emb = tape_emb[2*batch_sz:3*batch_sz, :]

        prob_pooled += self.alpha_prob * prob_sec_emb

        work_pooled += self.alpha_work * work_sec_emb + self.beta_work * work_head_emb + self.gamma_work * work_tape_emb

        state_pooled += self.alpha_state * state_sec_emb + self.beta_state * state_head_emb + self.gamma_state * state_tape_emb

        result_pooled += self.alpha_result * result_sec_emb + self.beta_result * result_head_emb + self.gamma_result * result_tape_emb

        emb_obs = torch.cat([prob_pooled, work_pooled, state_pooled, result_pooled], dim=1).squeeze(-1)

        return emb_obs


    def encode_observations(self, observations, state=None):
        # emb = self.embed(observations.sum(dim=1))
        # features = self.proj(observations.float())
        batch_sz = observations.shape[0]
        device = observations.device

        observations = observations.float()
        actual_observations, head_indices = observations[:, self.num_heads:], observations[:, :self.num_heads]

        emb_obs = self._efficient_emb_comp(actual_observations, head_indices, batch_sz, device)
        emb_obs = self.norm(emb_obs)

        features = self.proj(emb_obs)

        return features

    def decode_actions(self, hidden):
        action = self.actor(hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        
        value = self.value_fn(hidden)
        
        return action, value
