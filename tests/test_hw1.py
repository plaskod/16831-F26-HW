import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rob831.infrastructure import pytorch_util as ptu, utils
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.rl_trainer import RL_Trainer
from rob831.policies.MLP_policy import MLPPolicySL


class CounterEnv:
    def __init__(self, horizon=3):
        self.horizon = horizon

    def reset(self):
        self.t = 0
        return np.array([0.], dtype=np.float32)

    def step(self, action):
        self.t += 1
        return np.array([self.t], dtype=np.float32), float(self.t), self.t >= self.horizon, {}


class ConstantPolicy:
    def get_action(self, observations):
        return np.full((1 if observations.ndim == 1 else len(observations), 1), 2., dtype=np.float32)


class HomeworkTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        ptu.init_gpu(use_gpu=False)

    def test_mlp_depth_and_output(self):
        for depth in (0, 1, 3):
            model = ptu.build_mlp(4, 2, depth, 8, output_activation='sigmoid')
            result = model(torch.ones(5, 4))
            self.assertEqual(result.shape, (5, 2))
            self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in model), depth + 1)
            self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_supervised_learning_and_action_shapes(self):
        policy = MLPPolicySL(2, 3, 0, 8, learning_rate=.05)
        observations = np.random.randn(64, 3).astype(np.float32)
        actions = observations[:, :2] * 2 - .5
        before = np.mean((policy.get_action(observations) - actions) ** 2)
        for _ in range(150):
            log = policy.update(observations, actions)
        after = np.mean((policy.get_action(observations) - actions) ** 2)
        self.assertLess(after, before * .001)
        self.assertTrue(np.isfinite(log['Training Loss']))
        self.assertEqual(policy.get_action(observations[0]).shape, (1, 2))
        self.assertEqual(policy.get_action(observations).shape, (64, 2))

    def test_discrete_policy(self):
        policy = MLPPolicySL(2, 1, 0, 8, discrete=True, learning_rate=.1)
        observations = np.array([[-1.], [1.]], dtype=np.float32)
        for _ in range(50):
            policy.update(observations, np.array([0, 1]))
        np.testing.assert_array_equal(policy.get_action(observations), [0, 1])

    def test_rollout_termination_and_collection(self):
        for horizon, limit, expected in ((3, 10, 3), (10, 2, 2)):
            path = utils.sample_trajectory(CounterEnv(horizon), ConstantPolicy(), limit)
            self.assertEqual(len(path['reward']), expected)
            np.testing.assert_array_equal(path['terminal'], [0.] * (expected - 1) + [1.])
            np.testing.assert_array_equal(path['next_observation'], path['observation'] + 1)
        paths, steps = utils.sample_trajectories(CounterEnv(), ConstantPolicy(), 7, 10)
        self.assertEqual((len(paths), steps), (3, 9))
        self.assertEqual(len(utils.sample_n_trajectories(CounterEnv(), ConstantPolicy(), 2, 10)), 2)

    def test_replay_alignment_capacity_and_oversized_batch(self):
        ids = np.arange(10, dtype=np.float32)
        path = utils.Path(ids[:, None], [], (ids + 10)[:, None], ids + 20, (ids + 30)[:, None], ids + 40)
        buffer = ReplayBuffer(max_size=6)
        buffer.add_rollouts([path])
        for size in (4, 12):
            obs, actions, rewards, next_obs, terminals = buffer.sample_random_data(size)
            self.assertEqual(len(obs), size)
            self.assertTrue((obs >= 4).all())
            np.testing.assert_array_equal(actions, obs + 10)
            np.testing.assert_array_equal(rewards, obs[:, 0] + 20)
            np.testing.assert_array_equal(next_obs, obs + 30)
            np.testing.assert_array_equal(terminals, obs[:, 0] + 40)
        with self.assertRaises(ValueError):
            ReplayBuffer().sample_random_data(1)

    def test_initial_data_and_dagger_aggregation(self):
        trainer = RL_Trainer.__new__(RL_Trainer)
        trainer.env = CounterEnv()
        trainer.params = {'ep_len': 3}
        trainer.log_video = False
        initial = [utils.sample_trajectory(CounterEnv(), ConstantPolicy(), 3)]
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'expert.pkl'
            filename.write_bytes(pickle.dumps(initial))
            paths, steps, video = trainer.collect_training_trajectories(0, filename, None, 3)
            self.assertEqual(steps, 0)
            self.assertIsNone(video)
            buffer = ReplayBuffer()
            buffer.add_rollouts(paths)
            paths, steps, _ = trainer.collect_training_trajectories(1, filename, ConstantPolicy(), 4)
            before = [p['observation'].copy() for p in paths]
            expert = SimpleNamespace(get_action=lambda obs: obs + 10)
            relabeled = trainer.do_relabel_with_expert(expert, paths)
            for path, observations in zip(relabeled, before):
                np.testing.assert_array_equal(path['observation'], observations)
                np.testing.assert_array_equal(path['action'], observations + 10)
            buffer.add_rollouts(relabeled)
            self.assertEqual(steps, 6)
            self.assertEqual(len(buffer), 9)
            np.testing.assert_array_equal(buffer.acs[:3], initial[0]['action'])


if __name__ == '__main__':
    unittest.main()
