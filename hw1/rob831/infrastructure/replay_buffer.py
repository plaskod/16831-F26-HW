from rob831.infrastructure.utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):
        if batch_size <= 0 or len(self) == 0:
            raise ValueError("Sampling requires a positive batch size and a nonempty buffer")
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.rews.shape[0]
                == self.next_obs.shape[0]
                == self.terminals.shape[0]
        )

        if batch_size <= len(self):
            indices = np.random.permutation(len(self))[:batch_size]
        else:
            indices = np.random.choice(len(self), size=batch_size, replace=True)
        return (self.obs[indices], self.acs[indices], self.rews[indices],
                self.next_obs[indices], self.terminals[indices])


    def sample_recent_data(self, batch_size=1):
        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )


class ActionChunkReplayBuffer(ReplayBuffer):
    """Trajectory-local future expert actions, with masked episode tails.

    All starting observations are retained for every K. The sample action field
    is (targets [B,K,A], validity [B,K]); no label crosses a terminal or path end.
    """
    def __init__(self, max_size, action_chunk_size):
        super().__init__(max_size)
        self.action_chunk_size = action_chunk_size
        self.action_masks = None

    def add_rollouts(self, paths, concat_rew=True):
        chunk_paths, masks = [], []
        for path in paths:
            actions = np.asarray(path['action'])
            n = len(actions)
            if actions.ndim != 2 or n == 0 or not np.isfinite(actions).all():
                raise ValueError('Expected a nonempty trajectory of finite continuous expert actions')
            ends = np.minimum.accumulate(np.where(
                np.asarray(path['terminal']).astype(bool), np.arange(n), n - 1)[::-1])[::-1]
            indices = np.arange(n)[:, None] + np.arange(self.action_chunk_size)[None, :]
            valid = indices <= ends[:, None]
            targets = actions[np.minimum(indices, n - 1)].copy()
            targets[~valid] = 0
            chunk_paths.append(dict(path, action=targets))
            masks.append(valid.astype(np.float32))
        super().add_rollouts(chunk_paths, concat_rew)
        new_masks = np.concatenate(masks)
        self.action_masks = (new_masks if self.action_masks is None else
                             np.concatenate([self.action_masks, new_masks]))[-self.max_size:]

    def sample_random_data(self, batch_size):
        if batch_size <= 0 or len(self) == 0:
            raise ValueError('Sampling requires a positive batch size and a nonempty buffer')
        indices = (np.random.permutation(len(self))[:batch_size] if batch_size <= len(self)
                   else np.random.choice(len(self), batch_size, replace=True))
        return (self.obs[indices], (self.acs[indices], self.action_masks[indices]),
                self.rews[indices], self.next_obs[indices], self.terminals[indices])

    def sample_recent_data(self, batch_size=1):
        return (self.obs[-batch_size:], (self.acs[-batch_size:], self.action_masks[-batch_size:]),
                self.rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:])
