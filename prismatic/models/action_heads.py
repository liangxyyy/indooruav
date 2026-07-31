"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

# 把diffusion timestep t 编码成为一个向量。模型不能直接理解数字 42，所以要变成 LLM hidden dim 的 embedding。
class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb

# 一个带残差连接的小 MLP block
class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x

# 由多个 MLPResNetBlock 组成
class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

# 动作位置 hidden states → 连续动作。
# actions_hidden_states: (B, NUM_ACTIONS_CHUNK × ACTION_DIM, hidden_dim)
# 每一步动作的 7 个 hidden states 合起来，预测这一整步的动作。
class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_action_branches=1,
        use_cond_action_tokens=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_branches = num_action_branches
        self.use_cond_action_tokens = use_cond_action_tokens
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim if use_cond_action_tokens else input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim if use_cond_action_tokens else action_dim * num_action_branches,
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        if self.use_cond_action_tokens:
            return self.model(actions_hidden_states)
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        if self.num_action_branches > 1:
            # (B, T, K, action_dim): each future timestep has K candidate actions.
            action = action.reshape(batch_size, NUM_ACTIONS_CHUNK, self.num_action_branches, self.action_dim)
        return action


class GaussianActionHead(nn.Module):
    """Continuous Gaussian policy head with one diagonal distribution per (time, branch)."""

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_action_branches=1,
        use_cond_action_tokens=False,
        log_std_min=-5.0,
        log_std_max=1.0,
        initial_log_std=-0.5,
    ):
        super().__init__()
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if not log_std_min < initial_log_std < log_std_max:
            raise ValueError("initial_log_std must lie strictly between log_std_min and log_std_max")

        self.action_dim = action_dim
        self.num_action_branches = num_action_branches
        self.use_cond_action_tokens = use_cond_action_tokens
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim if use_cond_action_tokens else input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=(
                2 * action_dim
                if use_cond_action_tokens
                else 2 * action_dim * num_action_branches
            ),
        )
        self._initialize_log_std_output(initial_log_std)

    def _initialize_log_std_output(self, initial_log_std):
        fraction = (initial_log_std - self.log_std_min) / (self.log_std_max - self.log_std_min)
        raw_value = math.atanh(2.0 * fraction - 1.0)
        output_branches = 1 if self.use_cond_action_tokens else self.num_action_branches
        with torch.no_grad():
            for branch_idx in range(output_branches):
                start = branch_idx * 2 * self.action_dim + self.action_dim
                end = start + self.action_dim
                self.model.fc2.weight[start:end].zero_()
                self.model.fc2.bias[start:end].fill_(raw_value)

    def _bound_log_std(self, raw_log_std):
        fraction = 0.5 * (torch.tanh(raw_log_std) + 1.0)
        return self.log_std_min + fraction * (self.log_std_max - self.log_std_min)

    def predict_distribution(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        if self.use_cond_action_tokens:
            parameters = self.model(actions_hidden_states)
        else:
            rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
            parameters = self.model(rearranged)
            if self.num_action_branches > 1:
                parameters = parameters.reshape(
                    batch_size,
                    NUM_ACTIONS_CHUNK,
                    self.num_action_branches,
                    2 * self.action_dim,
                )
        mean, raw_log_std = torch.split(parameters, self.action_dim, dim=-1)
        return mean, self._bound_log_std(raw_log_std)

    def predict_action(self, actions_hidden_states):
        mean, _ = self.predict_distribution(actions_hidden_states)
        return mean

    def sample_action(self, actions_hidden_states):
        mean, log_std = self.predict_distribution(actions_hidden_states)
        return mean + torch.exp(log_std) * torch.randn_like(mean)


# 这是diffusion的噪声预测器
class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred
