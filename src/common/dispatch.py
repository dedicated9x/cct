from src.tasks.flowers.module import FlowersModule
from src.tasks.gsn1.module import Gsn1BaseModule

def modulename2cls(name):
    dict_globalname2obj = globals().copy()
    try:
        cls = dict_globalname2obj[name]
    except:
        raise NotImplementedError
    return cls