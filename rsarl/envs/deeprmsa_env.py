
from rsarl.envs import Env

class DeepRMSAEnv(Env):
    """DeepRMSAv2 Environment class. 

    paper: DeepRMSA: A Deep Reinforcement Learning Framework for Routing, 
        Modulation and Spectrum Assignment in Elastic Optical Networks
        (https://ieeexplore.ieee.org/document/8738827)

    """

    def __init__(self, net, requester, episode_step=None):
      super().__init__(net, requester, episode_step)


    def compute_reward(self, action) -> float:
      """Compute reward. 

      Returns:
        float: reward
      
      """
      is_assignable = self.is_assignable(action)
      if is_assignable:
        return 1.0
      else:
        return -1.0

