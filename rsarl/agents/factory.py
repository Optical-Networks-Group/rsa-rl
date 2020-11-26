
import importlib

KSPAgentDict = {
    "ff": "rsarl.agents.ksp_agents:KSP_FF_Agent",
    "random": "rsarl.agents.ksp_agents:KSP_RANDOM_Agent",
    "entropy": "rsarl.agents.ksp_agents:KSP_EntropyAgent",
}


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class KSPAgentFactory:

    @staticmethod
    def names():
        return KSPAgentDict.keys()


    @staticmethod
    def create(name: str, k: int):
        entry_point = KSPAgentDict[name]
        cls = load(entry_point)
        agent = cls(k)
        return agent

