
import os
import time
import numpy as np
from rsarl.logger import RSADB
from torch.utils.tensorboard import SummaryWriter

def create_db(
    exp_name: str,
    db_path: str,
    is_overwrite: bool,
):
    db = None
    # saver
    db = RSADB(exp_name, db_path)
    if is_overwrite:
        db.delete_experiment_info()

    return db


class Logger():

    def __init__(
        self,
        exp_name,
        save_agent=False,
        # db params
        db_name = "rsa-rl.db",
        save_experience=False,
        is_overwrite=False,
        # tb params
        use_tensorboard=False, 
    ):

        self.exp_name = exp_name
        # agent
        self.save_agent = save_agent
        # prepare db
        self.save_experience = save_experience
        if save_experience:
            self.db = create_db(
                exp_name,
                db_name,
                is_overwrite,
            )

        self.n_steps = 1
        self.min_bp = 100.0

        # tensorboard
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=f"./tb-logs/{self.exp_name}")


    def print_log(self, bps: list, utils: list, rewards: list):
        print('####################################################')
        for i, (bp, util, rwd) in enumerate(zip(bps, utils, rewards)):
            print(f'[{i}-th ENV]Blocking Probability:{bp}')
            print(f'[{i}-th ENV]Mean Resource Utilization:{util}')
            print(f'[{i}-th ENV]Total Rewards:{rwd}')


    def save_experiment(self, env, agent, hparam):
        assert self.save_experience
        self.db.save_experiment(env, agent, hparam)


    def save_drl_agent(self, agent):
        assert hasattr(agent, "drl")
        dirname = os.path.join("trained-agent", f"{self.n_steps}-{self.exp_name}")  
        agent.drl.save(dirname)


    def record_db(self, experiences: dict, bps: list, utils: list, rewards: list):
        for i, (bp, util, rwd) in enumerate(zip(bps, utils, rewards)):
            self.db.save_evaluation(i, self.n_steps, bp, util, rwd)

        # save experiences
        if np.min(bps) < self.min_bp:
            # self.min_bp = np.min(bps)
            # exp_id = int(np.argmin(bps))
            exp_id = 0
            # save each experience
            self.db.save_or_update_experience(experiences[exp_id])


    def record_tb(self, agent, bps: list, utils: list, rewards: list):

        assert (self.writer is not None) and self.use_tensorboard
        now = time.time()

        for i, (bp, util, rwd) in enumerate(zip(bps, utils, rewards)):
            self.writer.add_scalar(f"env-bp/env{i}", bp, self.n_steps, now)
            self.writer.add_scalar(f"env-util/env{i}", util, self.n_steps, now)
            self.writer.add_scalar(f"env-reward/env{i}", rwd, self.n_steps, now)

        # for DRL algorithms
        if hasattr(agent, "drl"):        
            for stat, value in agent.drl.get_statistics():
                self.writer.add_scalar(f"agent/{stat}", value, self.n_steps, now)

        self.writer.flush()


    def __call__(self, agent, experiences: dict, blocking_probs: list, avg_utils: list, total_rewards: list):

        # save evaluation
        if self.save_experience:
            self.record_db(experiences, blocking_probs, avg_utils, total_rewards)
        # tb
        if self.use_tensorboard:
            self.record_tb(agent, blocking_probs, avg_utils, total_rewards)
        # drl-agent
        if self.save_agent:
            self.save_drl_agent(agent)
        # print
        self.print_log(blocking_probs, avg_utils, total_rewards)
        # count up
        self.n_steps += 1

