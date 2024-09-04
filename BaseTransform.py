from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]

def boolmask2idx(mask):
    # A utility function, workaround for ONNX not supporting 'nonzero'
    return torch.nonzero(mask).squeeze(1).tolist()

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        use_points='lidar', 
        depth_input='scalar',
        height_expand=True,
        add_depth_features=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.use_points = use_points
        assert use_points in ['radar', 'lidar']
        self.depth_input=depth_input
        assert depth_input in ['scalar', 'one-hot']
        self.height_expand = height_expand
        self.add_depth_features = add_depth_features

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        #final = torch.cat(x.unbind(dim=2), 1)
        final =torch.mean(x,dim=2)
        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        radar, 
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        mats_dict = {
            'intrin_mats': camera_intrinsics, 
            'ida_mats': img_aug_matrix, 
            'bda_mat': lidar_aug_matrix,
            'sensor2ego_mats': camera2ego, 
        }
        x = self.get_cam_feats(img, mats_dict)

        use_depth = False
        if type(x) == tuple:
            x, depth = x 
            use_depth = True
        
        x = self.bev_pool(geom, x)

        if use_depth:
            return x, depth 
        else:
            return x

class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,#[batch, 1, c, h, w]
        points,#列表，里面都是Tensor
        #矩阵维度形式：[batch,1,4,4]或[batch,1,3,3]
        lidar2image,#[batch,1,4,4]
        cam_intrinsic,#[batch,1,4,4]
        camera2lidar,#[batch,1,4,4]
        img_aug_matrix,#[batch,1,3,3]
        lidar_aug_matrix,#[batch,4,4]
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        if self.height_expand:
            for b in range(len(points)):
                points_repeated = points[b].repeat_interleave(8, dim=0)
                points_repeated[:, 2] = torch.arange(0.25, 2.25, 0.25).repeat(points[b].shape[0])
                points[b] = points_repeated

        batch_size = len(points) #小心points维度
        depth_in_channels = 1 if self.depth_input=='scalar' else self.D
        if self.add_depth_features:
            depth_in_channels += points[0].shape[1]

        depth = torch.zeros(batch_size, img.shape[1], depth_in_channels, *self.image_size, device=points[0].device)#第二个维度的含义是有几个相机

        for b in range(batch_size):
            cur_coords = points[b][:, :3] #shape of points:[batch size, num points, 3+?]
            cur_img_aug_matrix = img_aug_matrix[b] #shape of transform matrix:[batch size, num_images, 4, 4]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b] #shape of cur_lidar2image:[num_images,3,3]

            # inverse aug，别忘了把翻转去了
            cur_coords -= cur_lidar_aug_matrix[:3, 3] #cur_coord维度：[num points, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            ) #给雷达坐标系下的点去增强，cur_coords维度：[3,num_points]

            # lidar2image，雷达坐标系下的点转移到图像/相机坐标系（图像还是相机我不确定，写的时候要重点盯一下对应，nuscene数据集的相机不只有1个，但是kitti只有1个，要注意）
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1) 
            # get 2d coords
            dist = cur_coords[:, 2, :] #怎么能3个维度呢？
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] #获取2D坐标

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)#[1,num_points,2]

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            ) #交换x和y，并判断点是否在图像上，对于KITTI需要考虑是否要交换的问题
            #shape of on_img:[num_cameras, y, x]
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]

                if self.depth_input == 'scalar':
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

                if self.add_depth_features:
                    depth[b, c, -points[b].shape[-1]:, masked_coords[:, 0], masked_coords[:, 1]] = points[b][boolmask2idx(on_img[c])].transpose(0,1)

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)

        use_depth = False
        if type(x) == tuple:
            x, depth = x 
            use_depth = True
        
        x = self.bev_pool(geom, x)

        if use_depth:
            return x, depth 
        else:
            return x