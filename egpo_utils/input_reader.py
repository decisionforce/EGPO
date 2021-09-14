from ray.rllib.offline.input_reader import InputReader
from ray.rllib.policy.sample_batch import SampleBatch
import json
import numpy as np


class InputReader(InputReader):

    def __init__(self, data_set_path=None):
        super(InputReader, self).__init__()
        assert data_set_path is not None
        with open(data_set_path, "r") as f:
            self.data = json.load(f)
        self.data_len = len(self.data)
        np.random.shuffle(self.data)
        self.count = 0

    def next(self) -> SampleBatch:
        if self.count == self.data_len:
            np.random.shuffle(self.data)
            self.count = 0
        index = self.count
        dp = self.data[index]
        # o,a,d,r,i
        batch = SampleBatch({SampleBatch.OBS: [dp[SampleBatch.OBS]],
                             SampleBatch.ACTIONS: [dp[SampleBatch.ACTIONS]],
                             SampleBatch.DONES: [dp[SampleBatch.DONES]],
                             SampleBatch.REWARDS: [dp[SampleBatch.REWARDS]],
                             SampleBatch.NEXT_OBS: [dp[SampleBatch.NEXT_OBS]],
                             # SampleBatch.INFOS: [dp[SampleBatch.INFOS]]
                             })
        self.count += 1
        return batch
