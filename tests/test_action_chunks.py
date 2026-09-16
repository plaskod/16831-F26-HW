import unittest
import numpy as np
import torch
from rob831.infrastructure import pytorch_util as ptu, utils
from rob831.infrastructure.replay_buffer import ActionChunkReplayBuffer
from rob831.policies.MLP_policy import MLPPolicySL


class ChunkTests(unittest.TestCase):
    def setUp(self):
        ptu.init_gpu(False)
        np.random.seed(1)
        torch.manual_seed(1)

    def test_k1_matches_default_and_manual_adam(self):
        default = MLPPolicySL(2, 3, 3, 8)
        torch.manual_seed(1)
        explicit = MLPPolicySL(2, 3, 3, 8, action_chunk_size=1)
        x = np.random.randn(10, 3).astype('float32')
        y = np.random.randn(10, 2).astype('float32')
        for _ in range(3):
            default.update(x, y)
            pred = explicit(torch.from_numpy(x))
            loss = torch.nn.functional.mse_loss(pred, torch.from_numpy(y))
            explicit.optimizer.zero_grad()
            loss.backward()
            explicit.optimizer.step()
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        np.testing.assert_array_equal(default.get_action(x), explicit.get_action(x))

    def test_boundaries_masks_alignment_and_capacity(self):
        def path(offset):
            x = np.arange(4, dtype='float32') + offset
            return utils.Path(x[:, None], [], x[:, None], x, x[:, None], [0, 1, 0, 1])
        b = ActionChunkReplayBuffer(6, 4)
        b.add_rollouts([path(0)])
        np.testing.assert_array_equal(b.acs[0, :, 0], [0, 1, 0, 0])
        np.testing.assert_array_equal(b.action_masks, [[1,1,0,0], [1,0,0,0], [1,1,0,0], [1,0,0,0]])
        b.add_rollouts([path(10)])
        self.assertEqual(len(b), 6)
        self.assertEqual(len(b.action_masks), 6)
        for size in (6, 10):
            obs, (targets, mask), *_ = b.sample_random_data(size)
            np.testing.assert_array_equal(obs[:,0], targets[:,0,0])
            np.testing.assert_array_equal(mask[:,0], 1)
        self.assertEqual(b.sample_recent_data(2)[1][0].shape, (2,4,1))

    def test_only_first_action_and_replanning(self):
        policy = MLPPolicySL(1, 1, 0, 8, action_chunk_size=4)
        with torch.no_grad():
            policy.mean_net[0].weight.copy_(torch.tensor([[2.], [10.], [20.], [30.]]))
            policy.mean_net[0].bias.zero_()
        for obs in (1., 3., -2.):
            np.testing.assert_array_equal(policy.get_action(np.array([obs], dtype='float32')), [[2*obs]])

    def test_masked_loss_and_future_gradients(self):
        policy = MLPPolicySL(1, 1, 0, 8, action_chunk_size=4)
        with torch.no_grad():
            policy.mean_net[0].weight.zero_()
            policy.mean_net[0].bias.zero_()
        targets = np.array([[[1.], [3.], [999.], [999.]]], dtype='float32')
        mask = np.array([[1,1,0,0]], dtype='float32')
        result = policy.update(np.ones((1,1), dtype='float32'), (targets, mask))
        self.assertAlmostEqual(float(result['Training Loss']), 5.)
        grad = policy.mean_net[0].weight.grad[:,0]
        torch.testing.assert_close(grad, torch.tensor([-1., -3., 0., 0.]))

    def test_hidden_extractor_identical_across_horizons(self):
        models = []
        for horizon in (1,2,4,8,16):
            torch.manual_seed(5)
            models.append(MLPPolicySL(2, 3, 3, 8, action_chunk_size=horizon))
        for model in models[1:]:
            for first, other in zip(models[0].mean_net[:-2].parameters(), model.mean_net[:-2].parameters()):
                torch.testing.assert_close(first, other, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
