#!/opt/conda/bin python
import torch

import numpy as np

from tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
 
import pybullet as pb
import pybullet_utils.bullet_client as bc

def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
    """Returns a tensordict containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_speed": 8,
                    "max_torque": 2.0,
                    "dt": 0.05,
                    "g": g,
                    "m": 1.0,
                    "l": 1.0,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

class KukaBaseEnv(EnvBase):
    
    def __init__(self, id, state_size, action_size, device = "cpu"):
        super(KukaBaseEnv, self).__init__()
        self.dtype = np.float32
        self.to(device)# type: ignore
        
        self.client = bc.BulletClient(connection_mode=pb.DIRECT)
        self.client_id = id
        self.objects = []
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.state = np.zeros((self.state_size, 1), dtype=self.dtype)
        
        self.action_spec = BoundedTensorSpec(low=-1, high=1, device=self.device, shape=torch.Size([self.action_size]))
        
        observation_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.state_size]))
        self.observation_spec = CompositeSpec(observation=observation_spec)

        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]))
        
        oid = self.client.createCollisionShape(shapeType=pb.GEOM_SPHERE)
        self.client.createMultiBody(0, 0)  

    def _reset(self, tensordict, **kwargs):
        out_tensordict = TensorDict({}, batch_size=torch.Size())
        
        
        pos = self.client.getBasePositionAndOrientation(0)
        
        pos = np.array(pos[0])
        # self.state = np.zeros((self.state_size, 1), dtype=self.dtype)
        out_tensordict.set("observation", torch.tensor(pos.flatten(), device=self.device))

        return out_tensordict
    
    
    def _step(self, tensordict):
        action = tensordict["action"]
        action = action.cpu().numpy().reshape((self.action_size, 1))
        
        reward = np.array(0)
        
        out_tensordict = TensorDict({"observation": torch.tensor(self.state.astype(self.dtype).flatten(), device=self.device),
                                     "reward": torch.tensor(reward.astype(np.float32), device=self.device),
                                     "done": False}, batch_size=torch.Size())

        return out_tensordict
    
    def _set_seed(self, seed):
        pass
    
    gen_params = staticmethod(gen_params)
        
        
