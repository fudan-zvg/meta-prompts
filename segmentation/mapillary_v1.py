from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import mmcv
import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES

@DATASETS.register_module()
class MapillaryDataset_v1(CustomDataset):
    """Mapillary dataset.
    """
    CLASSES = ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
               'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area',
               'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
               'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
               'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation',
               'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
               'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
               'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
               'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat',
               'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer',
               'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled')

    PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
               [180, 165, 180], [90, 120, 150], [
                   102, 102, 156], [128, 64, 255],
               [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
               [230, 150, 140], [128, 64, 128], [
                   110, 110, 110], [244, 35, 232],
               [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
               [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
               [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
               [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
               [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
               [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
               [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
               [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
               [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
               [119, 11, 32], [150, 0, 255], [
                   0, 60, 100], [0, 0, 142], [0, 0, 90],
               [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70],
               [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(MapillaryDataset_v1, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)


@PIPELINES.register_module()
class MapillaryHack(object):
    """map MV 65 class to 19 class like Cityscapes."""
    def __init__(self):
        self.map = [[13, 24, 41], [2, 15], [17], [6], [3],
                    [45, 47], [48], [50], [30], [29], [27], [19], [20, 21, 22],
                    [55], [61], [54], [58], [57], [52]]

        self.others = [i for i in range(66)]
        for i in self.map:
            for j in i:
                if j in self.others:
                    self.others.remove(j)

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        gt_map = results['gt_semantic_seg']
        # others -> 255
        new_gt_map = np.zeros_like(gt_map)

        for value in self.others:
            new_gt_map[gt_map == value] = 255

        for index, map in enumerate(self.map):
            for value in map:
                new_gt_map[gt_map == value] = index

        results['gt_semantic_seg'] = new_gt_map

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str