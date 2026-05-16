from .grasp_data import GraspDatasetBase
import glob
import os

from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class JacquardDataset(GraspDatasetBase):
    def __init__(self, file_path, ds_rotate=0, **kwargs):
        super(JacquardDataset, self).__init__(**kwargs)

        # 递归查找所有子文件夹中的 *_grasps.txt 文件
        self.grasp_files = glob.glob(os.path.join(file_path, '**', '*_grasps.txt'), recursive=True)
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No grasp files found in {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + \
                               self.grasp_files[:int(self.length * ds_rotate)]

        # 根据 grasp 文件路径推断 depth 和 rgb 文件
        self.depth_files = [f.replace('_grasps.txt', '_perfect_depth.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('_grasps.txt', '_RGB.png') for f in self.grasp_files]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
        return rgb_img.img
