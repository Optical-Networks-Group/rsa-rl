
from rsarl.data import Observation

class Env(object):
    """Environment class. 

    Args:
      net: Network. 
      requester: Request. 
      episode_step (int): The number of steps os each spisode. 

    Attributes:
      net: Network. 
      requester: Request. 
      episode_step (int): The number of steps os each spisode. 
      n_step (int): Step counter. 

    """

    def __init__(self, net, requester, episode_step=None):
      # check parameter is valid or not
      assert net is not None, ""
      assert requester is not None, ""

      self.net = net
      self.requester = requester
      # current observation
      self.last_obs = None
      # the number of steps of each episode 
      # if episode-step is None, then　contnuing task
      # otherwise, episodic task
      self.episode_step = episode_step
      self.n_step = 0
      

    def assign_path(self, act):
      """Assign path. 
      
      """
      self.net.assign_path(
        act.path, 
        act.slot_idx, 
        act.n_slot, 
        act.duration)


    def is_assignable(self, act) -> bool:
      """Check target action is assignable or not

      Returns:
        bool: assignable(True) or not(False)
      
      """
      if act is None:
        return False

      return self.net.is_assignable(
        act.path,
        act.slot_idx,
        act.n_slot)


    def compute_reward(self, action) -> float:
      """Compute reward. 

      Returns:
        float: reward
      
      """
      raise NotImplementedError


    def is_terminate(self) -> bool:
      # Continuing task
      if self.episode_step is None:
        return False
      
      # Episodic task
      if self.n_step >= self.episode_step:
        return True
      else:
        return False


    def step(self, action):
      """Execute target action. 

      Returns:
        Observation: Observation object (request, network). 
      
      """
      self.n_step += 1

      # reward
      reward = self.compute_reward(action)

      # assign path
      is_assignable = self.is_assignable(action)
      if is_assignable:
        self.assign_path(action)

      # Spend time until next request
      time_interval = self.requester.time_interval()
      self.net.spend_time(time_interval)

      # Generate next path request
      req = self.requester.request()

      # Generate next observation
      self.last_obs = Observation(request=req, net=self.net)
      
      # check eposode end
      done = self.is_terminate()

      info = {}
      info["is_success"] = is_assignable
      return self.last_obs, reward, done, info
    

    def reset(self):
      """Reset network status, especially utilizations of both slot and time. 

      Returns:
        Observation: Observation object (request, network). 
      
      """
      # initialize step count
      self.n_step = 0
      # initialize network utilization
      self.net.init_graph()
      # gen next request
      self.requester.init()
      req = self.requester.request()
      # Generate next observation
      self.last_obs = Observation(request=req, net=self.net)
      return self.last_obs

    def seed(self, s):
      self.requester.seed(s)

    def close(self):
      pass
