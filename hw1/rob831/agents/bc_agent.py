from rob831.infrastructure.replay_buffer import ReplayBuffer, ActionChunkReplayBuffer
from rob831.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent


class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicySL(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            action_chunk_size=self.agent_params.get('action_chunk_size', 1),
        )

        # replay buffer
        horizon = self.agent_params.get('action_chunk_size', 1)
        self.replay_buffer = (ReplayBuffer(self.agent_params['max_replay_buffer_size'])
                              if horizon == 1 else ActionChunkReplayBuffer(
                                  self.agent_params['max_replay_buffer_size'], horizon))

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self.actor.update(ob_no, ac_na)  # HW1: you will modify this
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)  # HW1: you will modify this

    def save(self, path):
        return self.actor.save(path)