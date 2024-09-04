import os
import numpy as np
import random
from utils.kitti_dataset import KittiDataset
import utils.kitti_aug_utils as augUtils
import utils.kitti_bev_utils as bev_utils
import utils.config as cnf

import torch
import torch.nn.functional as F

import cv2

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class KittiYOLODataset(KittiDataset):

    def __init__(self, root_dir, split='train', mode ='TRAIN', folder=None, data_aug=True, multiscale=False):
        super().__init__(root_dir=root_dir, split=split, folder=folder)
        self.split = split
        self.multiscale = multiscale
        self.data_aug = data_aug
        self.img_size = cnf.BEV_WIDTH
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        self.sample_id_list = []

        if mode == 'TRAIN':
            self.preprocess_yolo_training_data()
        else:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list] #测试集不管

        print('Load %s samples from %s' % (mode, self.imageset_dir))
        print('Done: total %s samples %d' % (mode, len(self.sample_id_list)))

    def preprocess_yolo_training_data(self):
        """
        Discard samples which don't have current training class objects, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        for idx in range(0, self.num_samples):
            sample_id = int(self.image_idx_list[idx])
            objects = self.get_label(sample_id) #获取标签，t对应的是目标框中心的位置
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects) #labels还是三维的坐标哈
            if not noObjectLabels:
                labels[:, 1:] = augUtils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_pc_range(labels[i, 1:4]) is True:
                        valid_list.append(labels[i,0]) #确定符合范围的标签

            if len(valid_list):
                self.sample_id_list.append(sample_id) #如果符合要求，就写入sample_id_list，可能之前发现的train不足是因为里面的标签不符合要求

    def check_pc_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def get_lss_matrix(self,calib):
        """
        Calculate the lidar2image, cam_intrinsic, and camera2lidar matrices.
        """
        # 1. 雷达坐标系转图像坐标系的矩阵
        R0_rect=np.append(calib.R0,np.array([0,0,0]).reshape(1,3),axis=0)
        R0_rect=np.append(R0_rect,np.array([0,0,0,1]).reshape(4,1),axis=1)
        Tr_velo_to_cam=np.append(calib.V2C,np.array([0,0,0,1]).reshape(1,4),axis=0)
        lidar2image=np.dot(calib.P,np.dot(R0_rect,Tr_velo_to_cam))
        # 2. 相机的内参矩阵
        cam_intrinsic = np.array([
            [calib.f_u, 0, calib.c_u],
            [0, calib.f_v, calib.c_v],
            [0, 0, 1]
        ])

        # 3. 相机坐标系转雷达坐标系的矩阵
        camera2lidar = calib.C2V
        camera2lidar = np.append(camera2lidar, np.array([0, 0, 0, 1]).reshape(1, 4), axis=0)

        return lidar2image, cam_intrinsic, camera2lidar

    def __getitem__(self, index):
        
        data_aug_calib_dict={}
        lss_calib_dict={}
        sample_id = int(self.sample_id_list[index])

        if self.mode in ['TRAIN', 'EVAL']:
            lidarData = self.get_lidar(sample_id)    
            objects = self.get_label(sample_id)   
            calib = self.get_calib(sample_id)
            lidar2image,cam_intrinsic,camera2lidar=self.get_lss_matrix(calib)
            lss_calib_dict['lidar2image']=lidar2image
            lss_calib_dict['cam_intrinsic']=cam_intrinsic
            lss_calib_dict['camera2lidar']=camera2lidar
            imageData=self.get_image(sample_id) #已经读取到图像数据了
            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)
    
            if not noObjectLabels:
                labels[:, 1:] = augUtils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P)  # convert rect cam to velo cord

            if self.data_aug and self.mode == 'TRAIN':
                lidarData, lidar_aug_matrix, labels[:, 1:]= augUtils.complex_yolo_pc_augmentation(lidarData, labels[:, 1:], True) #针对雷达的数据增强
                data_aug_calib_dict['lidar_aug_matrix']=lidar_aug_matrix
                imageData, image_aug_matrix=augUtils.complex_yolo_img_augmentation(imageData) #对图像进行增强，包括旋转和缩放
                image_aug_matrix_4x4=np.eye(4)
                image_aug_matrix_4x4[:3,:3]=image_aug_matrix
                data_aug_calib_dict['image_aug_matrix']=image_aug_matrix_4x4
            else:#不做数据增强时矩阵为单位矩阵
                data_aug_calib_dict['lidar_aug_matrix']=np.eye(4)
                data_aug_calib_dict['image_aug_matrix']=np.eye(4)
                
            b = bev_utils.removePoints(lidarData, cnf.boundary)
            #TODO：LSS特征图生成和训练
            rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            #TODO：Attention: rgb_map为特征图
            target = bev_utils.build_yolo_target(labels)
            img_file = os.path.join(self.image_path, '%06d.png' % sample_id)
            #imageData = cv2.resize(imageData, (self.img_size,self.img_size), interpolation=cv2.INTER_NEAREST)
            ntargets = 0
            for i, t in enumerate(target):
                if t.sum(0):
                    ntargets += 1            
            targets = torch.zeros((ntargets, 8))
            for i, t in enumerate(target):
                if t.sum(0):
                    targets[i, 1:] = torch.from_numpy(t)
                    
            #这里还有一个对于BEV特征的随机翻转，直接对雷达提取了BEV
            rgb_map = torch.from_numpy(rgb_map).type(torch.FloatTensor)
            imageData = torch.from_numpy(imageData).type(torch.FloatTensor).permute(2,0,1)
            b= torch.from_numpy(b).type(torch.FloatTensor)
            # if self.data_aug:
            #     if np.random.random() < 0.5:
            #         use_horisontal_flip=True
            #         rgb_map, targets = self.horisontal_flip(rgb_map, targets)
            #         data_aug_calib_dict['use_horsiontal_flip']=use_horisontal_flip
            #     else:
            #         use_horisontal_flip=False
            #         data_aug_calib_dict['use_horsiontal_flip']=use_horisontal_flip
                    
            return img_file, rgb_map, imageData, b, targets, data_aug_calib_dict, lss_calib_dict #rgb_map是雷达提取的bev特征图，imageData是图像特征图，b是经过位置筛选后的点云

        else:
            lidarData = self.get_lidar(sample_id)
            b = bev_utils.removePoints(lidarData, cnf.boundary)
            rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            img_file = os.path.join(self.image_path, '%06d.png' % sample_id)
            return img_file, rgb_map, imageData, b

    def collate_fn(self, batch):
        paths, rgb_maps, imageDatasets, points, targets, data_aug_calib_dicts,lss_calib_dicts = list(zip(*batch))
        # Remove empty placeholder targets
        image_aug_list=[]
        lidar_aug_list=[]
        lidar2image_list=[]
        cam_intrinsic_list=[]
        camera2lidar_list=[]

        for lss_calib_dict in lss_calib_dicts:
            lidar2image,cam_intrinsic,camera2lidar=lss_calib_dict['lidar2image'],lss_calib_dict['cam_intrinsic'],lss_calib_dict['camera2lidar']
            lidar2image_list.append(lidar2image)
            cam_intrinsic_list.append(cam_intrinsic)
            camera2lidar_list.append(camera2lidar)
        lidar2image_matrix=torch.tensor(lidar2image_list).unsqueeze(1).to(torch.float32)
        cam_intrinsic_matrix=torch.tensor(cam_intrinsic_list).unsqueeze(1).to(torch.float32)
        camera2lidar_matrix=torch.tensor(camera2lidar_list).unsqueeze(1).to(torch.float32)
        lss_calib_matrix_batch={'lidar2image':lidar2image_matrix,'cam_intrinsic':cam_intrinsic_matrix,'camera2lidar':camera2lidar_matrix}
        targets = [boxes for boxes in targets if boxes is not None]
        points = [point for point in points if points is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        rgb_maps = torch.stack([resize(rgb_map, self.img_size) for rgb_map in rgb_maps])
        imageData= torch.stack([resize(imageData, self.img_size) for imageData in imageDatasets])
        self.batch_count += 1      
        for data_aug_calib_dict in data_aug_calib_dicts:
            image_aug=data_aug_calib_dict['image_aug_matrix']
            lidar_aug=data_aug_calib_dict['lidar_aug_matrix']
            image_aug_list.append(image_aug)
            lidar_aug_list.append(lidar_aug)
        image_aug_matrix=torch.tensor(image_aug_list).unsqueeze(1).to(torch.float32)
        lidar_aug_matrix=torch.tensor(lidar_aug_list).to(torch.float32)
        return paths, rgb_maps, imageData, points, targets, image_aug_matrix, lidar_aug_matrix,lss_calib_matrix_batch
        
    def horisontal_flip(self, images, targets):
        images = torch.flip(images, [-1])
        targets[:, 2] = 1 - targets[:, 2] # horizontal flip
        targets[:, 6] = - targets[:, 6] # yaw angle flip

        return images, targets

    def __len__(self):
        return len(self.sample_id_list)