#  Copyright (c) 6.2021. Yinyu Nie
#  License: MIT
import numpy as np
from external.tsdf_fusion import fusion
from utils.virtualhome.vhome_utils import get_cam_intrinsics, get_cam_extrinsics, pc_from_dep_by_frame
import os
from skimage import measure
import trimesh
from utils.tools import get_box_corners, voxel_faces


def vis_crops(scene_tsdf, scene_vox, scene_color, color_const, vol_bnds, dataset_config, tsdf_plyfile, voxel_plyfile):
    # write TSDF to mesh
    verts, faces, norms, vals = measure.marching_cubes(scene_tsdf, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts * dataset_config.voxel_size + vol_bnds[:, 0]
    rgb_vals = scene_color[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / color_const)
    colors_g = np.floor((rgb_vals - colors_b * color_const) / 256)
    colors_r = rgb_vals - colors_b * color_const - colors_g * 256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)
    fusion.meshwrite(tsdf_plyfile, verts, faces, norms, colors)

    # write voxel to mesh.
    voxel_vectors = np.array(
        [[dataset_config.voxel_size, 0., 0.], [0., dataset_config.voxel_size, 0.],
         [0., 0., dataset_config.voxel_size]]) * 9 / 20

    voxel_idx = np.array(scene_vox.nonzero()).T
    voxel_points = (voxel_idx + 0.5) * dataset_config.voxel_size + vol_bnds[:, 0]
    rgb_vals = scene_color[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]]
    colors_b = np.floor(rgb_vals / color_const)
    colors_g = np.floor((rgb_vals - colors_b * color_const) / 256)
    colors_r = rgb_vals - colors_b * color_const - colors_g * 256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    all_points = []
    all_faces = []
    point_colors = []
    for point, color in zip(voxel_points, colors):
        corner_pnts = get_box_corners(point, voxel_vectors)
        new_faces = voxel_faces + len(all_points)
        all_points += corner_pnts
        all_faces.append(new_faces)
        point_colors.append(np.tile(color, (8, 1)))
    scene = trimesh.Trimesh(vertices=all_points, faces=np.vstack(all_faces), vertex_colors=np.vstack(point_colors))
    scene.export(voxel_plyfile)

def get_metda_info(comm, frame_ids, image_width, image_height, far_clip):
    _, cam_data = comm.camera_data(frame_ids)
    cam_Ks = []
    cam2world_RTs = []
    vol_bnds = np.zeros((3, 2))
    valid_frame_ids = []
    for idx, frame_id in enumerate(frame_ids):
        '''Recover camera params'''
        per_cam_data = cam_data[idx]
        projection_matrix = np.asarray(per_cam_data['projection_matrix']).reshape([4, 4], order='F')
        world2camera_gl = np.asarray(per_cam_data['world_to_camera_matrix']).reshape([4, 4], order='F')

        cam_K = get_cam_intrinsics(projection_matrix, im_width=image_width, im_height=image_height)['cam_K']
        cam2world_RT = get_cam_extrinsics(world2camera_gl)

        '''Load Depth'''
        (ok_img, depth_map) = comm.camera_image(frame_id, mode='depth', image_width=image_width,
                                                image_height=image_height)
        depth_map = depth_map[0][..., 0]
        depth_map[depth_map > far_clip] = 0

        '''Read volume boundary'''
        view_frust_pts = fusion.get_view_frustum(depth_map, cam_K, cam2world_RT)

        if len(valid_frame_ids) == 0:
            vol_bnds[:, 0] = np.amin(view_frust_pts, axis=1)
            vol_bnds[:, 1] = np.amax(view_frust_pts, axis=1)
        else:
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

        '''Store data'''
        cam_Ks.append(cam_K)
        cam2world_RTs.append(cam2world_RT)
        valid_frame_ids.append(frame_id)

    return cam_Ks, cam2world_RTs, valid_frame_ids, vol_bnds

def export_TSDF(comm, valid_frame_ids, cam_Ks, cam2world_RT, vol_bnds, dataset_config):
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=dataset_config.voxel_size, use_gpu=True)
    for idx, frame_id in enumerate(valid_frame_ids):
        '''Load Depth'''
        (ok_img, depth_map) = comm.camera_image(frame_id, mode='depth', image_width=dataset_config.im_size[0],
                                                image_height=dataset_config.im_size[1])
        depth_map = depth_map[0][..., 0]
        depth_map[depth_map > dataset_config.far_clip] = 0

        '''Load RGB image'''
        (ok_img, rgb_imgs) = comm.camera_image(frame_id, mode='normal', image_width=dataset_config.im_size[0],
                                                image_height=dataset_config.im_size[1])

        rgb_img = rgb_imgs[0][..., [2, 1, 0]]

        tsdf_vol.integrate(rgb_img, depth_map, cam_Ks[idx], cam2world_RT[idx], obs_weight=1.)

    return tsdf_vol

def get_scene_voxels_w_point_clouds(comm, valid_frame_ids, cam_Ks, cam2world_RTs, scene_vox, scene_color,
                                    color_const, vol_bnds, dataset_config):
    point_list_canonical = []
    color_intensities = []
    camera_poses_vis = []
    for idx, frame_id in enumerate(valid_frame_ids):
        per_frame_scene_vox = np.zeros_like(scene_vox)
        '''Load Depth'''
        (ok_img, depth_map) = comm.camera_image(frame_id, mode='depth', image_width=dataset_config.im_size[0],
                                                image_height=dataset_config.im_size[1])
        depth_map = depth_map[0][..., 0]

        '''Load cam2world cam_RT'''
        point_canonical, color_indices = pc_from_dep_by_frame(depth_map, cam_Ks[idx], cam2world_RTs[idx],
                                                           far_clip=dataset_config.far_clip,
                                                           sample_rate=dataset_config.pixel_sample_rate)
        cam_param = {'cam_RT': cam2world_RTs[idx],
                     'cam_K': cam_Ks[idx]}

        points_v = np.uint16((point_canonical - vol_bnds[:, 0]) / dataset_config.voxel_size)
        per_frame_scene_vox[points_v[:, 0], points_v[:, 1], points_v[:, 2]] = True
        frame_points = per_frame_scene_vox.nonzero()
        rgb_vals = scene_color[frame_points]
        colors_b = np.floor(rgb_vals / color_const)
        colors_g = np.floor((rgb_vals - colors_b * color_const) / 256)
        colors_r = rgb_vals - colors_b * color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        frame_points = np.array(frame_points).T * dataset_config.voxel_size + vol_bnds[:, 0]

        scene_vox += per_frame_scene_vox
        point_list_canonical.append(frame_points)
        color_intensities.append(colors)
        camera_poses_vis.append(cam_param)

    return scene_vox, point_list_canonical, color_intensities, camera_poses_vis

def read_frames(comm, frame_ids, dataset_config, if_vis=False, replace=True):
    '''
    Get frame information from Unity assets.
    @param comm: communication handle from Unity
    @param frame_ids: frame ids for scanning.
    @param dataset_config: constant config params.
    @param if_vis: if visualize the output.
    @return:
    '''
    '''export meta info'''
    cam_Ks, cam2world_RTs, valid_frame_ids, vol_bnds = get_metda_info(comm, frame_ids, dataset_config.im_size[0],
                                                                      dataset_config.im_size[1],
                                                                      dataset_config.far_clip)

    '''export TSDF'''
    cam_Ks = np.array(cam_Ks)
    cam2world_RTs = np.array(cam2world_RTs)
    tsdf_vol = export_TSDF(comm, valid_frame_ids, cam_Ks, cam2world_RTs, vol_bnds, dataset_config)
    scene_tsdf, scene_color = tsdf_vol.get_volume()

    '''Get scene voxels and point clouds'''
    scene_vox = np.zeros_like(scene_tsdf, dtype=np.bool)
    scene_vox, point_list_canonical, color_intensities, camera_poses_vis = get_scene_voxels_w_point_clouds(comm,
                                                                                                           valid_frame_ids,
                                                                                                           cam_Ks,
                                                                                                           cam2world_RTs,
                                                                                                           scene_vox,
                                                                                                           scene_color,
                                                                                                           tsdf_vol._color_const,
                                                                                                           vol_bnds,
                                                                                                           dataset_config)

    '''Read point cloud with colors'''
    points_world = {'pc': point_list_canonical, 'cam': camera_poses_vis, 'color': color_intensities}

    '''Visualize results'''
    tsdf_plyfile = './temp/tsdf_mesh_vhome.ply'
    voxel_plyfile = './temp/voxel_mesh_vhome.ply'
    points_file = './temp/points_vhome.npz'
    if if_vis:
        if (not os.path.exists(tsdf_plyfile) or not os.path.exists(voxel_plyfile)) or replace:
            vis_crops(scene_tsdf, scene_vox, scene_color, tsdf_vol._color_const, vol_bnds, dataset_config, tsdf_plyfile,
                      voxel_plyfile)
        if (not os.path.exists(points_file)) or replace:
            np.savez(points_file, points=np.vstack(point_list_canonical),
                     colors=np.vstack(color_intensities))

    return points_world, vol_bnds[:, 0]



