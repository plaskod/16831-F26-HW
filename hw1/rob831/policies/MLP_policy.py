import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 action_chunk_size=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        if action_chunk_size < 1 or (discrete and action_chunk_size != 1):
            raise ValueError("Action chunks require a positive horizon and continuous actions")
        self.action_chunk_size = action_chunk_size
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim * self.action_chunk_size,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        with torch.no_grad():
            prediction = self(ptu.from_numpy(observation.astype(np.float32)))
            if self.discrete:
                prediction = prediction.argmax(dim=-1)
            elif self.action_chunk_size > 1:
                # Replan on every call; never cache or execute future predictions.
                prediction = prediction.reshape(-1, self.action_chunk_size, self.ac_dim)[:, 0]
        return ptu.to_numpy(prediction)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if self.discrete:
            return self.logits_na(observation)
        return self.mean_net(observation)


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        observations = ptu.from_numpy(observations)
        predictions = self(observations)
        if self.discrete:
            targets = torch.as_tensor(actions, dtype=torch.long,
                                      device=observations.device).reshape(-1)
            loss = F.cross_entropy(predictions, targets)
        elif self.action_chunk_size == 1:
            loss = self.loss(predictions, ptu.from_numpy(actions))
        else:
            targets, mask = actions
            targets, mask = ptu.from_numpy(targets), ptu.from_numpy(mask)
            predictions = predictions.reshape(-1, self.action_chunk_size, self.ac_dim)
            per_offset_mse = (predictions - targets).square().mean(dim=-1)
            # All starting observations are retained. Missing tail labels have zero
            # weight; each observation contributes its mean over valid offsets.
            loss = ((per_offset_mse * mask).sum(dim=1) / mask.sum(dim=1)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Training Loss': ptu.to_numpy(loss)}
