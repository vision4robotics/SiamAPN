from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from .dtb import DTBDataset
from .uavdt import UAVDTDataset
from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .visdrone import VISDRONEDDataset
from .visdrone1 import VISDRONED2018Dataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAV10fps' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20l' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'VISDRONED2018' in name:
            dataset = VISDRONED2018Dataset(**kwargs)

        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

