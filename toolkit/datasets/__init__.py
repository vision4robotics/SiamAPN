from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .visdrone1 import VISDRONED2018Dataset
from .v4r import V4RDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):


        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        
        if 'UAV10' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20l' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'VISDRONED2018' in name:
            dataset = VISDRONED2018Dataset(**kwargs)
        elif 'V4RFlight112' in name:
            dataset = V4RDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

