

import sys
import json
import datetime
from typing import NamedTuple
from collections import defaultdict
from networkx.readwrite.json_graph import adjacency_graph

from rsarl.utils import str_to_list
from rsarl.data import Observation, Action, Request, DBExperience, DBExperiment, DBEvaluation
from rsarl.agents import Agent
from rsarl.logger import SqliteDB

class RSADB(SqliteDB):

    def __init__(self, exp_name="", db_name="rsa-rl.db"):
        super(RSADB, self).__init__(db_name)
        # unique experiment name
        self.exp_name = exp_name
        # create tables if not exist
        self.create_experiment_table()
        self.create_evaluation_table()
        self.create_experience_table()

    def delete_experiment_info(self):
        tables = ["experiments", "evaluations", "experiences"]
        for table in tables:
            sql = f""" delete from {table} where experiment_name = "{self.exp_name}" """
            self.delete(sql)

    def create_experiment_table(self):
        sql = """
                CREATE TABLE IF NOT EXISTS 
                experiments(
                    experiment_name STRING PRIMARY KEY,
                    environment_name STRING,
                    network_name STRING,
                    requester_name STRING,
                    agent_name STRING,
                    hyper_parameters STRING,
                    created_at TIMESTAMP
                )
              """
        self.create_table(sql)

    def create_evaluation_table(self):
        sql = """
                CREATE TABLE IF NOT EXISTS 
                evaluations(
                    experiment_name STRING,
                    env_id INTEGER,
                    batch INTEGER,
                    blocking_prob REAL,
                    slot_utilization REAL,
                    total_reward READ,
                    PRIMARY KEY (experiment_name, env_id, batch)
                )
            """
        self.create_table(sql)

    def create_experience_table(self):
        sql = """
                CREATE TABLE IF NOT EXISTS 
                experiences(
                    experiment_name STRING,
                    request_id INTEGER,
                    source INTEGER,
                    destination INTEGER,
                    bandwidth INTEGER,
                    duration REAL,
                    path STRING,
                    slot_index INTEGER,
                    n_slot INTEGER,
                    is_success BOOL,
                    reward INTEGER,
                    network STRING,
                    slot_utilization REAL,
                    PRIMARY KEY (experiment_name, request_id)
                )
            """
        self.create_table(sql)


    def _insert(self, table_name: str, row: NamedTuple):
        sql = self._get_insert_sql(table_name, row)
        self.insert(sql, row)
    
    def _get_insert_sql(self, table: str, row: NamedTuple) -> str:
        col_names = row._fields
        columns = ', '.join(col_names)
        placeholders = ', '.join('?' * len(col_names))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return sql

    def _experience_exist(self, target_exp_name: str) -> bool:
        sql = f"""
            select count(*)
            from experiences 
            where experiment_name = "{target_exp_name}"
            """
        for row in self.select(sql):
            count = row[0]

        return True if count > 0 else False
        

    def save_experiment(self, env, agent, hyper_params: dict):
        exp = DBExperiment(
            experiment_name = self.exp_name,
            environment_name = env.__class__.__name__,
            network_name = env.net.name,
            requester_name = env.requester.__class__.__name__,
            agent_name = agent.__class__.__name__,
            hyper_parameters = json.dumps(hyper_params),
            created_at = datetime.datetime.now()
        )
        self._insert("experiments", exp)

    def save_evaluation(self, env_id: int, batch: int, bp: float, util: float, rw: float):

        evaluation = DBEvaluation(
            experiment_name = self.exp_name,
            env_id = env_id,
            batch = batch, 
            blocking_prob = bp, 
            slot_utilization = util, 
            total_reward = rw
        )
        self._insert("evaluations", evaluation)

    def save_experience(self, experiences: list):
        db_exp_list = [DBExperience(
            experiment_name=self.exp_name, **exp._asdict()) for exp in experiences]
        sql = self._get_insert_sql("experiences", db_exp_list[0])
        self.many_execute(sql, db_exp_list)

    def update_experience(self, experiences: list):
        sql = ''' 
            update experiences
            set source = ? ,
                destination = ? ,
                bandwidth = ?,
                duration = ?,
                path = ?,
                slot_index = ?,
                n_slot = ?,
                is_success = ?,
                reward = ?,
                network = ?,
                slot_utilization = ?
                WHERE experiment_name = ?
                and request_id = ?
            '''
        db_exp_list = [(
            exp.source, 
            exp.destination, 
            exp.bandwidth,
            exp.duration,
            exp.path,
            exp.slot_index,
            exp.n_slot,
            exp.is_success,
            exp.reward,
            exp.network,
            exp.slot_utilization,
            self.exp_name,
            exp.request_id) for exp in experiences]

        self.many_execute(sql, db_exp_list)

    def save_or_update_experience(self, experiences: list):
        if self._experience_exist(self.exp_name):
            self.update_experience(experiences)
        else:
            self.save_experience(experiences)


    def get_experiment_names(self) -> list:
        sql = f"""
            select experiment_name
            from experiments 
            """
        exp_list = []
        for row in self.select(sql):
            exp_list.append(row[0])

        return exp_list
    

    def get_experiment_settings(self, target_exp_name: str):
        sql = f"""
            select environment_name, 
                network_name, 
                agent_name, 
                requester_name, 
                hyper_parameters
            from experiments
            where experiment_name = "{target_exp_name}"
        """
        
        for env, net, agent, requester, hparams in self.select(sql):
            return env, net, agent, requester, json.loads(hparams)


    def get_batches(self, target_exp_name: str):
        sql = f"""
                select distinct batch  
                from evaluations
                where experiment_name = "{target_exp_name}"
            """
        batch_list = []
        for batch in self.select(sql):
            batch_list.append(batch[0])

        return batch_list


    def get_bp_per_batch(self, target_exp_name: str, batches: list):
        bp_per_batch = defaultdict(lambda: [])
        for batch in batches:
            sql = f"""
                select blocking_prob
                from evaluations
                where experiment_name = "{target_exp_name}"
                and batch = {batch}
                """
            for bp in self.select(sql):
                bp_per_batch[batch].append(bp)
        
        return bp_per_batch


    def get_act_history(self, target_exp_name:str, req_id: int):
        sql = f"""
            select source, destination, bandwidth, duration, path, slot_index, n_slot, network
            from experiences 
            where experiment_name = "{target_exp_name}"
            and request_id = {req_id}
            """
        db_row = self.select(sql)[0]
        path = str_to_list(db_row[4]) if db_row[4] is not None else None
        act = Action(path=path, slot_idx=db_row[5], n_slot=db_row[6], duration=db_row[3])
        req = Request(source=db_row[0], destination=db_row[1], bandwidth=db_row[2], duration=db_row[3])
        G = adjacency_graph(json.loads(db_row[7]))
        return act, req, G

