
import importlib

TopologyDict = {
    "arpa": "rsarl.networks.topology:ARPA",
    "clara": "rsarl.networks.topology:CLARA",
    "cost266": "rsarl.networks.topology:COST266",
    "german": "rsarl.networks.topology:GERMAN",
    "italian": "rsarl.networks.topology:ITALIAN",
    "janet": "rsarl.networks.topology:JANET",
    "jpn12": "rsarl.networks.topology:JPN12",
    "jpn25": "rsarl.networks.topology:JPN25",
    "jpn48": "rsarl.networks.topology:JPN48",
    "lattice3x3": "rsarl.networks.topology:LATTICE3x3",
    "nsf": "rsarl.networks.topology:NSF",
    "rnp": "rsarl.networks.topology:RNP",
    "verizon": "rsarl.networks.topology:VERIZON",
}


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class TopologyFactory:

    @staticmethod
    def names():
        return TopologyDict.keys()


    @staticmethod
    def create(name: str):
        entry_point = TopologyDict[name]
        cls = load(entry_point)
        topo = cls()
        return topo

