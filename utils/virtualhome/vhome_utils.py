#  Copyright (c) 6.2021. Yinyu Nie
#  License: MIT
from copy import deepcopy
import numpy as np
from scipy.optimize import lsq_linear
import quaternion
import math
from utils.tools import get_box_corners
from utils.virtualhome import dataset_config
import itertools

property_action_pairs = {
    'CAN_OPEN': ['OPEN', 'CLOSE'],
    'GRABBABLE': ['GRAB'],
    'HAS_SWITCH': ['SWITCHON', 'SWITCHOFF'],
    'SITTABLE': ['SIT', 'STANDUP'],
    'SURFACES': ['PUT', 'PUTBACK'],
    'CONTAINERS': ['PUTIN']
}

command_template = {
    'Walk':'<char0> [Walk] <{0:s}> ({1:d})',
    'Find':'<char0> [Find] <{0:s}> ({1:d})',
    'Grab':'<char0> [Grab] <{0:s}> ({1:d})',
    'Open':'<char0> [Open] <{0:s}> ({1:d})',
    'PutIn':'<char0> [PutIn] <{0:s}> ({1:d}) <{2:s}> ({3:d})',
    'Close':'<char0> [Close] <{0:s}> ({1:d})',
    'SwitchOn':'<char0> [SwitchOn] <{0:s}> ({1:d})',
    'SwitchOff':'<char0> [SwitchOff] <{0:s}> ({1:d})',
    'Sit':'<char0> [Sit] <{0:s}> ({1:d})',
    'StandUp':'<char0> [StandUp]',
    'Put':'<char0> [Put] <{0:s}> ({1:d}) <{2:s}> ({3:d})',
    'PutBack':'<char0> [PutBack] <{0:s}> ({1:d}) <{2:s}> ({3:d})'
}

def category_mapping(class_names_raw, return_cateory_names=False):
    class_labels_raw = dataset_config.class_labels_raw
    category_mapping_list = dataset_config.category_mapping
    category_ids = [category_mapping_list[class_labels_raw.index(name_raw)] for name_raw in class_names_raw]
    if return_cateory_names:
        cateory_names = [dataset_config.category_labels[idx] for idx in category_ids]
    else:
        cateory_names = None
    return category_ids, cateory_names

def class_mapping(class_names_raw, return_class_names=False):
    class_labels_raw = dataset_config.class_labels_raw
    class_mapping_list = dataset_config.class_mapping
    class_ids = [class_mapping_list[class_labels_raw.index(name_raw)] for name_raw in class_names_raw]
    if return_class_names:
        class_names = [dataset_config.class_labels[idx] for idx in class_ids]
    else:
        class_names = None
    return class_ids, class_names

def edges_with_node_id(edges, node_ids):
    return [edge_id for edge_id, edge in enumerate(edges) if edge['from_id'] in node_ids or edge['to_id'] in node_ids]


def get_nodes_in_room(nodes, edges, room_node):
    '''
    Get all nodes and edges in a room
    @param nodes: all nodes in this scene.
    @param edges: all edges in this scene.
    @param room_node: room node
    @return:
    '''
    '''Get bbox of room_node'''
    rm_centroid, rm_size, rm_R_mat = get_box_prop(room_node)
    rm_bbox = {'centroid': rm_centroid, 'size': rm_size, 'R_mat': rm_R_mat}

    '''Get all nodes in this bounding box (including itself)'''
    node_candidates = []
    for node in nodes:
        centroid, size, R_mat = get_box_prop(node)
        in_box = check_in_box(centroid, rm_bbox)
        if in_box:
            node_candidates.append(node)
    node_candidate_ids = [node['id'] for node in node_candidates]

    '''Update edges'''
    edge_candidates = []
    for edge in edges:
        if edge['from_id'] in node_candidate_ids and edge['to_id'] in node_candidate_ids:
            edge_candidates.append(edge)

    # [{'from_id': 128, 'to_id': 109, 'relation_type': 'INSIDE'}]
    '''Add doors to edges'''
    for node in node_candidates:
        if node['class_name'] != 'door':
            continue

        existing_edges = [edge for edge in edge_candidates if
                          edge['from_id'] == node['id'] and edge['to_id'] == room_node[
                              'id'] and 'relation_type' == 'INSIDE']
        if not len(existing_edges):
            door_in_room_edge = {'from_id': node['id'], 'to_id': room_node['id'], 'relation_type': 'INSIDE'}
            edge_candidates.append(door_in_room_edge)

    return node_candidates, edge_candidates


def rel_nodes_edges(graph, node_id):
    """
    give related nodes and edges dependent on node_id in graph.
    """
    nodes = graph['nodes']
    edges = graph['edges']
    input_edges = deepcopy(edges)
    root_node_ids = {node_id}
    output_edges = []

    while root_node_ids and input_edges:
        '''get all edges connecting to the root nodes'''
        rel_edge_ids = edges_with_node_id(input_edges, root_node_ids)
        rel_edges = [input_edges[idx] for idx in rel_edge_ids]
        output_edges += rel_edges
        '''get new root nodes'''
        rel_edge_pairs = [[edge['from_id'], edge['to_id']] for edge in rel_edges]
        root_node_ids = set(sum(rel_edge_pairs, [])) - root_node_ids

        '''get all edges not connecting to the root nodes'''
        left_edge_ids = set(range(len(input_edges))) - set(rel_edge_ids)
        input_edges = [input_edges[idx] for idx in left_edge_ids]

    out_edge_pairs = [[edge['from_id'], edge['to_id']] for edge in output_edges]
    output_node_ids = set(sum(out_edge_pairs, []))
    output_nodes = [node for node in nodes if node['id'] in output_node_ids]
    # verify there are no missing nodes
    assert set([node['id'] for node in output_nodes]) == output_node_ids
    return output_nodes, output_edges

def close_doors(nodes):
    for node in nodes:
        if node['class_name'] == 'door':
            node['states'] = ['CLOSED']
    return nodes

def open_doors(nodes):
    for node in nodes:
        if node['class_name'] == 'door':
            node['states'] = ['OPEN']
    return nodes

def refine_room_bbox(room_node, nodes_in_room):
    rm_centroid, rm_size, rm_R_mat = get_box_prop(room_node)
    layout_nodes = [node for node in nodes_in_room if node['category'] in ['Walls', 'Ceiling', 'Floor', 'Floors']]
    all_corners = []
    for layout_node in layout_nodes:
        centroid, size, R_mat = get_box_prop(layout_node)
        vectors = np.diag(size / 2.).dot(R_mat)
        all_corners += get_box_corners(centroid, vectors)
    all_corners = np.array(all_corners)
    coeffs = all_corners.dot(rm_R_mat.T)
    centroid_coeffs = (coeffs.max(0) + coeffs.min(0)) / 2.
    new_centroid = centroid_coeffs.dot(rm_R_mat)
    size_coeffs = coeffs.max(0) - coeffs.min(0)
    new_size = np.abs(size_coeffs.dot(rm_R_mat))
    return {'centroid': new_centroid, 'size': new_size, 'R_mat': rm_R_mat}

def clean_nodes_in_room(nodes_in_room, edges_in_room, room_node):
    '''Re-select all nodes and edges in this room.'''
    '''Refine room bbox'''
    room_bbox = refine_room_bbox(room_node, nodes_in_room)

    '''Re-select all nodes in nodes_in_room'''
    nodes_update = []
    remove_node_ids = []
    for node in nodes_in_room:
        centroid = get_box_prop(node)[0]
        in_box = check_in_box(centroid, room_bbox)
        if not in_box and node['category'] != 'Doors':
            remove_node_ids.append(node['id'])
        else:
            nodes_update.append(node)

    edges_update = []
    for edge in edges_in_room:
        if edge['from_id'] in remove_node_ids or edge['to_id'] in remove_node_ids:
            continue
        edges_update.append(edge)

    return nodes_update, edges_update, room_bbox

def remove_objects(nodes, edges, class_labels, level='class', mode='include'):
    nodes_update = []
    remove_node_ids = []
    for node in nodes:
        if level == 'class' and mode == 'include':
            if node['class_name'] not in class_labels:
                remove_node_ids.append(node['id'])
            else:
                nodes_update.append(node)
        elif level == 'class' and mode == 'exclude':
            if node['class_name'] in class_labels:
                remove_node_ids.append(node['id'])
            else:
                nodes_update.append(node)
        elif level == 'category' and mode == 'include':
            if node['category'] not in class_labels:
                remove_node_ids.append(node['id'])
            else:
                nodes_update.append(node)
        elif level == 'category' and mode == 'exclude':
            if node['category'] in class_labels:
                remove_node_ids.append(node['id'])
            else:
                nodes_update.append(node)
        else:
            raise NameError('Cannot identify the level and mode.')

    edges_update = []
    for edge in edges:
        if edge['from_id'] in remove_node_ids or edge['to_id'] in remove_node_ids:
            continue
        edges_update.append(edge)

    return nodes_update, edges_update

def correct_door_bbox(nodes_for_det, nodes_update):
    '''Get the correct bounding box of doors'''
    doorjamb_nodes = []
    for node in nodes_update:
        if node['class_name'] == 'doorjamb':
            doorjamb_nodes.append(node)

    doorjamb_centroids = np.array([node['bounding_box']['center'] for node in doorjamb_nodes])

    nodes_for_det_updated = deepcopy(nodes_for_det)
    for node in nodes_for_det_updated:
        if node['class_name'] == 'door':
            offsets = doorjamb_centroids - node['bounding_box']['center']
            jamb_id = np.argmin(np.linalg.norm(offsets, axis=1))
            node['bounding_box'] = doorjamb_nodes[jamb_id]['bounding_box']
            node['obj_transform'] = doorjamb_nodes[jamb_id]['obj_transform']

    return nodes_for_det_updated

def get_box_prop(node):
    obj_transform = node['obj_transform']
    bounding_box = node['bounding_box']
    quant = obj_transform['rotation']
    quant = np.quaternion(quant[0], quant[1], quant[2], quant[3])
    R_mat = -quaternion.as_rotation_matrix(quant)
    R_mat[2] = np.cross(R_mat[0], R_mat[1])
    size = np.abs(R_mat.dot(bounding_box['size']))
    centroid = bounding_box['center']

    # correct these wrongly labelled bboxes
    if np.argmax(np.abs(R_mat[:, 1])) == 0:
        R_mat = np.array([R_mat[2], -R_mat[0], -R_mat[1]])
        size = np.array([size[2], size[0], size[1]])
    return centroid, size, R_mat

def sample_points_in_box(box, step_len=1, padding=0):
    '''Sample points in a node bounding box'''
    centroid = box['centroid']
    size = box['size'] + padding
    R_mat = box['R_mat']

    '''sample points'''
    vectors = np.diag(size / 2.).dot(R_mat)
    bbox_corner = centroid - vectors[0] - vectors[1] - vectors[2]

    cx, cy, cz = np.meshgrid(np.arange(step_len, size[0], step_len),
                             np.arange(step_len, size[1], step_len),
                             np.arange(step_len, size[2], step_len), indexing='ij')
    cxyz = np.array([cx, cy, cz]).reshape(3, -1).T
    cxyz = cxyz[:, np.newaxis]
    R_mat = np.tile(R_mat, (cxyz.shape[0], 1, 1))
    points_in_cube = np.matmul(cxyz, R_mat) + bbox_corner
    return points_in_cube

def check_in_box(points, box_prop):
    '''Check if a point located in a box'''
    centroid = np.array(box_prop['centroid'])
    size = np.array(box_prop['size'])
    R_mat = np.array(box_prop['R_mat'])

    offsets = points - centroid
    offsets_proj = np.abs(offsets.dot(R_mat.T))

    return np.min(offsets_proj <= size/2., axis=-1)

def filter_cam_locs(cam_locs, nodes):
    '''filter out the cam locs that are in nodes' bboxes'''
    inbox_vec = np.zeros(shape=(cam_locs.shape[:-1]), dtype=np.bool)
    for node in nodes:
        centroid, size, R_mat = get_box_prop(node)
        inbox_vec += check_in_box(cam_locs, {'centroid': centroid, 'size': size, 'R_mat': R_mat})
    out_cam_locs = cam_locs[~inbox_vec[:, 0]]
    return out_cam_locs

def generate_cameras(room_node, room_bbox, all_nodes, dataset_config):
    '''
    Generate cameras in room node.
    @param room_node: root node of this room.
    @param room_bbox: room bounding box.
    @param all_nodes: all nodes in this room.
    @param dataset_config: config params to generate data..
    @return:
    '''
    '''Clean room node and wall nodes'''
    object_nodes = []
    for node in all_nodes:
        if node['id'] == room_node['id']:
            continue
        if node['category'] in ['Walls'] or node['class_name'] in ['wall']:
            continue
        object_nodes.append(node)

    '''Generate camera location in room but out of object bounding boxes'''
    cam_locs = sample_points_in_box(room_bbox, dataset_config.cam_loc_sample_step, padding=dataset_config.cam_range_padding)
    cam_locs = filter_cam_locs(cam_locs, object_nodes)

    '''Generate camera orientations'''
    # For Unity/OpenGL, pitch in [-90, 90], yaw in [-180, 180], roll in [-180, 180]
    # we fix roll to zeros
    angle_step = dataset_config.cam_angle_sample_step

    pitch_yaw_pairs = np.array(
        np.meshgrid(np.linspace(-90, 90, 180 // angle_step + 2)[1:-1], np.arange(-180., 180., angle_step)))
    pitch_yaw_pairs = pitch_yaw_pairs.reshape(2, -1).T
    pitch_yaw_pairs = np.vstack([pitch_yaw_pairs, np.array([[-90., 0.], [90., 0.]])])

    return cam_locs, pitch_yaw_pairs

def get_cam_intrinsics(projection_matrix, im_width, im_height):
    '''Get the camera intrinsic params from OPENGL projection_matrix.'''
    z_near = projection_matrix[2, 3] / (projection_matrix[2, 2] - 1)
    z_far = projection_matrix[2, 3] / (projection_matrix[2, 2] + 1)

    c_x = im_width * (1 - projection_matrix[0, 2]) / 2.
    c_x = np.around(c_x).astype(np.uint32)

    c_y = (projection_matrix[1, 2] + 1) * im_height / 2
    c_y = np.around(c_y).astype(np.uint32)

    f_x = projection_matrix[0, 0] / 2 * im_width
    f_y = projection_matrix[1, 1] / 2 * im_height

    f_xy = projection_matrix[0, 1] * im_width / -2.

    fov_x = math.atan(0.5 * im_width / f_x) * 2
    fov_y = math.atan(0.5 * im_height / f_y) * 2

    # Intrinsic param
    cam_K = np.array([[f_x, f_xy, c_x], [0, f_y, c_y], [0, 0, 1]])

    return {'z_near': z_near, 'z_far': z_far, 'cam_K': cam_K, 'fov_x': fov_x, 'fov_y': fov_y}

def get_cam_extrinsics(world2camera_gl):
    cam2world_RT = np.linalg.inv(world2camera_gl)
    cam2world_RT[:3, :3] *= -1
    cam2world_RT[:, 0] *= -1
    return cam2world_RT

def pc_from_dep_by_frame(depth_map, cam_K, cam2world_RT, rgb_img=None, far_clip=15., sample_rate=1):
    '''
    get point cloud from a single depth map.
    @param depth_map: a single depth map.
    @param cam_K: camera intrinsic matrix.
    @param cam2world_RT: camera extrinsic matrix transforming cam to world.
    @param rgb_img: color image
    @param far_clip: clip depth values
    @param sample_rate: for sampling pixels from image plane
    :return:
    '''
    img_height, img_width = depth_map.shape
    u, v = np.meshgrid(range(0, img_width, sample_rate), range(0, img_height, sample_rate))
    u = u.reshape([1, -1])[0]
    v = v.reshape([1, -1])[0]

    z = depth_map[v, u]

    # remove invalid pixels
    valid_indices = np.argwhere(np.logical_and(z < far_clip, z > 0.)).T[0]

    if isinstance(rgb_img, np.ndarray) and rgb_img.shape[:2] == depth_map.shape[:2]:
        color_indices = rgb_img[v, u][valid_indices]
    else:
        color_indices = []

    z = z[valid_indices]
    u = u[valid_indices]
    v = v[valid_indices]

    # calculate coordinates
    x = (u - cam_K[0][2]) * z / cam_K[0][0]
    y = (v - cam_K[1][2]) * z / cam_K[1][1]

    point_cam = np.vstack([x, y, z]).T

    point_canonical = point_cam.dot(cam2world_RT[:3, :3].T) + cam2world_RT[:3, -1]

    return point_canonical, color_indices

def get_nodes_for_det(nodes):
    nodes_for_vis = []
    for node in nodes:
        node_vis = deepcopy(node)
        del node_vis['obj_transform']
        del node_vis['bounding_box']
        centroid, size, R_mat = get_box_prop(node)
        node_vis['R_mat'] = R_mat
        node_vis['size'] = np.array(size)
        node_vis['centroid'] = np.array(centroid)
        nodes_for_vis.append(node_vis)
    return nodes_for_vis

def get_switch_cmd(interact_node):
    '''Walk to a switchable node, switch on/off it.'''
    walk_1 = command_template['Walk'].format(interact_node['class_name'], interact_node['id'])
    find_1 = command_template['Find'].format(interact_node['class_name'], interact_node['id'])
    switch_on = command_template['SwitchOn'].format(interact_node['class_name'], interact_node['id'])
    switch_off = command_template['SwitchOff'].format(interact_node['class_name'], interact_node['id'])

    init_states = interact_node['states']
    if ['ON'] in init_states:
        command_script = [walk_1, find_1, switch_off, switch_on]
    else:
        command_script = [walk_1, find_1, switch_on, switch_off]

    return command_script

def get_sit_cmd(interact_node):
    '''Walk to a sittable node, sit on it and standup'''
    find_1 = command_template['Find'].format(interact_node['class_name'], interact_node['id'])
    sit_1 = command_template['Sit'].format(interact_node['class_name'], interact_node['id'])
    standup_1 = command_template['StandUp']

    command_script = [find_1, sit_1, standup_1]
    return command_script

def get_open_close_cmd(interact_node):
    '''Walk to an openable node, open and close it'''
    # walk, find and close it.
    walk_1 = command_template['Walk'].format(interact_node['class_name'], interact_node['id'])
    find_1 = command_template['Find'].format(interact_node['class_name'], interact_node['id'])
    open_1 = command_template['Open'].format(interact_node['class_name'], interact_node['id'])
    close_1 = command_template['Close'].format(interact_node['class_name'], interact_node['id'])

    init_states = interact_node['states']

    if 'CLOSED' in init_states:
        command_script = [walk_1, find_1, open_1, close_1]
    else:
        command_script = [walk_1, find_1, close_1, open_1]

    return command_script

def get_put_in_cmd(interact_node, grabbale_nodes):
    '''Walk to a grabbale node, find and grab it and put into this container'''
    selected_id = np.random.randint(len(grabbale_nodes))
    grabbale_node = grabbale_nodes[selected_id]

    # grab some object
    walk_1 = command_template['Walk'].format(grabbale_node['class_name'], grabbale_node['id'])
    find_1 = command_template['Find'].format(grabbale_node['class_name'], grabbale_node['id'])
    grab_1 = command_template['Grab'].format(grabbale_node['class_name'], grabbale_node['id'])

    # put into this container
    walk_2 = command_template['Walk'].format(interact_node['class_name'], interact_node['id'])
    find_2 = command_template['Find'].format(interact_node['class_name'], interact_node['id'])
    open_2 = command_template['Open'].format(interact_node['class_name'], interact_node['id'])
    putin_2 = command_template['PutIn'].format(grabbale_node['class_name'], grabbale_node['id'],
                                               interact_node['class_name'], interact_node['id'])
    close_2 = command_template['Close'].format(interact_node['class_name'], interact_node['id'])

    open_3 = command_template['Open'].format(interact_node['class_name'], interact_node['id'])

    init_states = interact_node['states']
    if 'CLOSED' in init_states:
        command_script = [walk_1, find_1, grab_1, walk_2, find_2, open_2, putin_2, close_2]
    else:
        command_script = [walk_1, find_1, grab_1, walk_2, find_2, putin_2, close_2, open_3]

    return command_script

def get_surface_cmd(interact_node, grabbale_nodes):
    '''Walk to a grabbale node, find and grab it and put to the surface'''
    selected_id = np.random.randint(len(grabbale_nodes))
    grabbale_node = grabbale_nodes[selected_id]

    # grab some object
    walk_1 = command_template['Walk'].format(grabbale_node['class_name'], grabbale_node['id'])
    find_1 = command_template['Find'].format(grabbale_node['class_name'], grabbale_node['id'])
    grab_1 = command_template['Grab'].format(grabbale_node['class_name'], grabbale_node['id'])

    # put onto the surface
    # walk_2 = command_template['Walk'].format(interact_node['class_name'], interact_node['id'])
    find_2 = command_template['Find'].format(interact_node['class_name'], interact_node['id'])
    put_2 = command_template['Put'].format(grabbale_node['class_name'], grabbale_node['id'],
                                           interact_node['class_name'], interact_node['id'])

    # grab and put back
    grab_3 = command_template['Grab'].format(grabbale_node['class_name'], grabbale_node['id'])
    put_3 = command_template['PutBack'].format(grabbale_node['class_name'], grabbale_node['id'],
                                               interact_node['class_name'], interact_node['id'])

    command_script = [walk_1, find_1, grab_1, find_2, put_2, grab_3, put_3]
    return command_script

def clean_det_objects(comm, scene_id, scene_graph, room_node, nodes_in_room, edges_in_room, nodes_for_det, edges_for_det, dataset_config):
    '''
    Drop out the nodes that cannot actually interactable in Unity.
    @param comm: communication from Unity.
    @param scene_id: current scene id.
    @param scene_graph: current scene graph.
    @param room_node: the room root node.
    @param nodes_in_room: all object nodes in room bbox.
    @param edges_in_room: all edges in room bbox.
    @param nodes_for_det: nodes for detection and cleaning.
    @param edges_for_det: egdes of nodes for detections and cleaning.
    @param dataset_config: config class
    @return:
    '''
    node_ids_interact = [node['id'] for node in nodes_for_det]

    '''First get all grabbale objects as candidates to put on/in nodes_for_det.'''
    grabbale_nodes = []
    for obj_node in nodes_in_room:
        if obj_node['category'] in ['Decor', 'Furniture']:
            # rule out some big-size objects
            continue
        if obj_node['class_name'] in ['bananas', 'wallphone']:
            # some specific classes are not grabbable (unknow reasons).
            continue
        if ('GRABBABLE' in obj_node['properties']) and (obj_node['id'] not in node_ids_interact):
            # object should be grabbale and not be the targets.
            possible_ids = [edge['to_id'] for edge in edges_in_room if
                             edge['from_id'] == obj_node['id'] and edge['relation_type'] == 'INSIDE']
            container_nodes = [node for node in nodes_in_room if (node['id'] in possible_ids) and (node['category'] != 'Rooms')]
            states_of_containers = sum([node['states'] for node in container_nodes], [])
            if 'CLOSED' in states_of_containers:
                # grabbale objects can not be inside some closed container.
                continue
            # Verify the object is really grabale or not.
            walk_1 = command_template['Walk'].format(obj_node['class_name'], obj_node['id'])
            find_1 = command_template['Find'].format(obj_node['class_name'], obj_node['id'])
            grab_1 = command_template['Grab'].format(obj_node['class_name'], obj_node['id'])
            command_script = [walk_1, find_1, grab_1]
            comm.reset(scene_id)
            success = comm.expand_scene(scene_graph)
            assert success[0]
            comm.add_character('Chars/Female2', initial_room=room_node['class_name'])
            success, _ = comm.render_script(command_script, skip_execution=True, image_synthesis=[], recording=False,
                                            skip_animation=True)
            if success:
                grabbale_nodes.append(obj_node)

    if not len(grabbale_nodes):
        print('There is no grabbale object in this room.')

    comm.reset(scene_id)
    success = comm.expand_scene(scene_graph)
    assert success[0]
    comm.add_character('Chars/Female2', initial_room=room_node['class_name'])

    # update nodes
    interactable_nodes = []
    interactable_node_cmds = []
    for interact_node in nodes_for_det:
        avail_props = set(interact_node['properties']).intersection(dataset_config.object_props)

        # for kitchencabinet with a door, you cannot put objects in it without open the door.
        if interact_node['class_name'] in ['kitchencabinet']:
            if {'SURFACES', 'CAN_OPEN', 'CONTAINERS'}.issubset(set(interact_node['properties'])):
                avail_props = avail_props - {'SURFACES'}

        interactable_props = []
        interactable_cmds = []
        for node_prop in avail_props:
            if node_prop == 'SITTABLE':
                command = get_sit_cmd(interact_node)
            elif node_prop == 'SURFACES':
                command = get_surface_cmd(interact_node, grabbale_nodes)
            elif node_prop == 'CAN_OPEN':
                if 'CONTAINERS' not in interact_node['properties']:
                    command = get_open_close_cmd(interact_node)
                else:
                    command = get_put_in_cmd(interact_node, grabbale_nodes)
            elif node_prop == 'HAS_SWITCH':
                command = get_switch_cmd(interact_node)
            else:
                raise NotImplementedError('Not defined property in dataset_config.')
            success, _ = comm.render_script(command, skip_execution=True, image_synthesis=[], recording=False,
                                            skip_animation=True)
            if success:
                interactable_props.append(node_prop)
                interactable_cmds.append(command)
            else:
                print(command, 'is not a valid command.')
        if len(interactable_props):
            new_interact_node = deepcopy(interact_node)
            new_interact_node['properties'] = interactable_props
            interactable_nodes.append(new_interact_node)
            interactable_node_cmds.append(interactable_cmds)

    interactable_node_ids = [node['id'] for node in interactable_nodes]
    # update edges
    edges_update = []
    for edge in edges_for_det:
        if (edge['from_id'] not in interactable_node_ids) or (edge['to_id'] not in interactable_node_ids):
            continue
        edges_update.append(edge)

    return interactable_nodes, edges_update, grabbale_nodes, interactable_node_cmds

def target_func(x, A, b):
    v = A.dot(x) - b
    return sum(v**2)

def target_func_der(x, A, b):
    return 2 * ((A.T).dot(A)).dot(x) - 2 * (A.T).dot(b)

def get_cond_prob_matrix(nodes, labels, typename):
    prob_by_label = np.ones(len(labels))
    cond_prob_matrix = np.zeros(shape=(len(labels), len(nodes)))
    if typename == 'properties':
        for node_idx, node in enumerate(nodes):
            for prop in node['properties']:
                prop_index = labels.index(prop)
                cond_prob_matrix[prop_index, node_idx] = 1
        return cond_prob_matrix, prob_by_label
    elif typename == 'classnames':
        for node_idx, node in enumerate(nodes):
            class_index = labels.index(node['class_name'])
            cond_prob_matrix[class_index, node_idx] = 1
        return cond_prob_matrix, prob_by_label
    else:
        raise NotImplementedError


def generate_programs(nodes_for_det, interactable_node_cmds, dataset_config):
    '''
    Synthesize programs of human interactions.
    @param interactable_node_cmds: cmd for each interactable node
    @param dataset_config: config files to generate programs
    @return programs: can be used to drive Unity to synthesize human motions.
    '''
    '''Get a probability for each object to balance interaction type'''
    all_interaction_types = list(set(sum([node['properties'] for node in nodes_for_det], [])))
    all_obj_classes = list(set([node['class_name'] for node in nodes_for_det]))

    cond_prob_matrix_int, prob_by_int = get_cond_prob_matrix(nodes_for_det, all_interaction_types, typename='properties')
    cond_prob_matrix_cls, prob_by_cls = get_cond_prob_matrix(nodes_for_det, all_obj_classes,  typename='classnames')

    cond_prob_matrix = np.vstack([cond_prob_matrix_int, cond_prob_matrix_cls])
    prob_vec = np.hstack([prob_by_int, prob_by_cls])

    lower_bnd = 0.1
    res = lsq_linear(cond_prob_matrix, prob_vec, bounds=(lower_bnd * np.ones(len(nodes_for_det)), np.ones(len(nodes_for_det))))
    inst_prob = res.x
    inst_prob = inst_prob / sum(inst_prob)

    '''Get interaction permutations'''
    max_n_seq = min(dataset_config.n_seq_per_room, np.math.factorial(len(nodes_for_det)))

    if max_n_seq == dataset_config.n_seq_per_room:
        interact_lists = []
        for _ in range(max_n_seq):
            if len(nodes_for_det) > dataset_config.n_inst_per_room:
                instance_ids = np.random.choice(len(nodes_for_det), dataset_config.n_inst_per_room, replace=False,
                                                p=inst_prob)
            else:
                instance_ids = np.random.permutation(len(nodes_for_det))
            interact_lists.append(tuple(instance_ids.tolist()))
    else:
        if len(nodes_for_det) > dataset_config.n_inst_per_room:
            all_combs = itertools.combinations(range(len(nodes_for_det)), dataset_config.n_inst_per_room)
            interact_lists = []
            for comb in all_combs:
                interact_lists.append(tuple(comb))
        else:
            interact_lists = list(itertools.permutations(range(len(nodes_for_det))))

    interact_lists = list(set(interact_lists))

    all_command_sequences = []
    interact_sequences = []
    for seq in interact_lists:
        cmd_seqs = list(itertools.product(*[interactable_node_cmds[idx] for idx in seq]))
        # generate texts
        for cmd_seq in cmd_seqs:
            all_command_sequences.append(sum(cmd_seq, []))
            interact_sequences.append(seq)

    return all_command_sequences, interact_sequences