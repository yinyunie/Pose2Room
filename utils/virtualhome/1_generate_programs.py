#  Generate programs from the interactable nodes
#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import argparse
import numpy as np
import random
from external.virtualhome.simulation.unity_simulator import comm_unity
from utils.virtualhome.vhome_utils import get_nodes_in_room, open_doors, remove_objects, clean_det_objects, \
    correct_door_bbox, get_nodes_for_det, generate_programs, clean_nodes_in_room
from utils.virtualhome import dataset_config
from utils.tools import write_json
from copy import deepcopy
from utils.tools import ndarray2list
import subprocess
import time

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Generate programs in a room.')
    parser.add_argument('--scene-id', type=int, default=3,
                        help='Give a scene id in [0-7].')
    parser.add_argument('--room-id', type=int, default=0,
                        help='Give a scene id in [0-N].')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(dataset_config.random_seed)
    random.seed(dataset_config.random_seed)

    '''Start Unity and continue'''
    unity_laucher=subprocess.Popen(dataset_config.unity_lauch_cmd)
    time.sleep(5)

    '''Build scene communication'''
    comm = comm_unity.UnityCommunication(timeout_wait=300)

    '''Reset the scene'''
    comm.reset(args.scene_id)

    '''Get graph'''
    s, graph = comm.environment_graph()
    nodes = graph['nodes']
    edges = graph['edges']
    # check if has redundant nodes
    all_node_ids = [node['id'] for node in nodes]
    assert len(set(all_node_ids)) == len(all_node_ids)

    '''open doors to prohibit ambiguous cases'''
    nodes = open_doors(nodes)

    '''Get room nodes'''
    room_nodes = []
    for node in nodes:
        if node['category'] == 'Rooms':
            room_nodes.append(node)

    # make sure the room id for visualize not exceed the maximal room count.
    if args.room_id >= len(room_nodes):
        unity_laucher.kill()
        raise IndexError('Room id exceeds the maximal room count.')
    room_node = room_nodes[args.room_id]

    '''Parse all nodes dependent on this room'''
    nodes_in_room, edges_in_room = get_nodes_in_room(nodes, edges, room_node)

    '''Refine nodes in room, get all nodes in refined room bbox'''
    nodes_in_room, edges_in_room, room_bbox = clean_nodes_in_room(nodes_in_room, edges_in_room, room_node)

    '''update graph'''
    graph_update = {}
    graph_update['nodes'] = nodes
    graph_update['edges'] = edges
    success = comm.expand_scene(graph_update)
    assert success[0]

    '''get the object bbox for detection with target labels'''
    nodes_for_det, edges_for_det = remove_objects(nodes_in_room, edges_in_room, dataset_config.class_labels_raw,
                                                  level='class', mode='include')

    if not len(nodes_for_det):
        print('No available objects in room.')
        unity_laucher.kill()
        sys.exit(0)

    '''clean nodes_for_det that are not interactable'''
    nodes_for_det, edges_for_det, grabbale_nodes, interactable_node_cmds = clean_det_objects(comm, args.scene_id,
                                                                                             graph_update, room_node,
                                                                                             nodes_in_room,
                                                                                             edges_in_room,
                                                                                             nodes_for_det,
                                                                                             edges_for_det,
                                                                                             dataset_config)
    if not len(nodes_for_det):
        print('No available objects in room.')
        unity_laucher.kill()
        sys.exit(0)

    comm.reset(args.scene_id)
    success = comm.expand_scene(graph_update)
    assert success[0]

    '''correct bboxes for doors'''
    nodes_for_det = correct_door_bbox(nodes_for_det, nodes_in_room)
    nodes_for_det = get_nodes_for_det(nodes_for_det)

    '''write out all results for animation'''
    output_scene_path = dataset_config.script_bbox_path.joinpath(str(args.scene_id))
    if not output_scene_path.is_dir():
        output_scene_path.mkdir()

    # generate programs for this room
    script_file = output_scene_path.joinpath('script_' + str(args.room_id) + '.json')
    all_command_scripts, interact_sequences = generate_programs(nodes_for_det, interactable_node_cmds, dataset_config)
    write_json(script_file, {'scripts': all_command_scripts, 'instance_ids': ndarray2list(interact_sequences)})

    # write object bbox
    bbox_file = output_scene_path.joinpath('bbox_' + str(args.room_id) + '.json')
    nodes_for_det_save = ndarray2list(deepcopy(nodes_for_det))
    write_json(bbox_file, nodes_for_det_save)

    # write room bbox
    room_bbox_file = output_scene_path.joinpath('room_bbox_' + str(args.room_id) + '.json')
    room_bbox_save = ndarray2list(deepcopy(room_bbox))
    write_json(room_bbox_file, {'room_bbox': room_bbox_save, 'room_type':room_node['class_name']})

    '''Finish unity'''
    unity_laucher.kill()