#  Get pose motion skeletons from program scrips
#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import numpy as np
import random
from utils.virtualhome import dataset_config
from utils.tools import read_json, write_json
from external.virtualhome.simulation.unity_simulator import comm_unity
from utils.virtualhome.vhome_utils import open_doors
import os
import subprocess
import time
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_skeleton_from_scripts(comm, scene_id, room_id, room_node, init_graph, script_file, dataset_config, unity_laucher):
    script_data = read_json(script_file)
    programs_in_room = script_data['scripts']
    instance_lists = script_data['instance_ids']
    for script_idx, program_script in enumerate(programs_in_room):
        # write instance ids
        out_script_path = dataset_config.recording_path.joinpath(str(scene_id), str(room_id), str(script_idx))
        if not out_script_path.exists():
            os.makedirs(out_script_path)
        write_json(out_script_path.joinpath('instance_ids.json'), instance_lists[script_idx])

        for character_name in dataset_config.character_names:
            output_path = out_script_path.joinpath(character_name.split('/')[1])

            '''Pass those already created skeletons'''
            skeleton_file = output_path.joinpath('script', '0', 'pd_script.txt')
            if skeleton_file.is_file():
                print('File exists: %s. Continue.' % (skeleton_file))
                continue

            script_mark = '%d %d %d %d\n' % (
            scene_id, room_id, script_idx, dataset_config.character_names.index(character_name))

            '''Pass those failure cases'''
            if dataset_config.failed_script_log.is_file():
                with open(dataset_config.failed_script_log, 'r') as file:
                    failure_scripts = file.readlines()
                if script_mark in failure_scripts:
                    print('File in failure log: %s. Continue.' % (skeleton_file))
                    continue

            if not output_path.is_dir():
                os.makedirs(output_path)

            try:
                with time_limit(20):
                    # initialize scene before each script
                    comm.reset(scene_id)
                    success = comm.expand_scene(init_graph)
                    assert success[0]
                    comm.add_character(character_name, initial_room=room_node['class_name'])
                    success, message = comm.render_script(program_script,
                                                          image_width=dataset_config.im_size[0],
                                                          image_height=dataset_config.im_size[1],
                                                          recording=True,
                                                          frame_rate=dataset_config.frame_rate,
                                                          image_synthesis=[],
                                                          camera_mode=["PERSON_FROM_BACK"],
                                                          save_pose_data=True,
                                                          output_folder=str(output_path.absolute()),
                                                          skip_animation=False)
                if not success:
                    print('Can not generate data in %s.' % (output_path))
                    print('Please check scene_id: %d, room_id: %d, script_id: %d, character: %s.' % (scene_id, room_id, script_idx, character_name))
                    with open(dataset_config.failed_script_log, 'a') as file:
                        file.write(script_mark)
                    continue
            except TimeoutException as e:
                print('Time is out.')
                print('Can not generate data in %s.' % (output_path))
                print('Please check scene_id: %d, room_id: %d, script_id: %d, character: %s.' % (
                scene_id, room_id, script_idx, character_name))

                with open(dataset_config.failed_script_log, 'a') as file:
                    file.write(script_mark)

                '''Restart Unity and continue'''
                unity_laucher.kill()
                time.sleep(5)
                unity_laucher = subprocess.Popen(dataset_config.unity_lauch_cmd)
                time.sleep(5)

                '''Rebuild scene communication'''
                comm = comm_unity.UnityCommunication(timeout_wait=300)


if __name__ == '__main__':
    '''Load Unity communication'''
    np.random.seed(dataset_config.random_seed)
    random.seed(dataset_config.random_seed)

    '''Start Unity and continue'''
    unity_laucher=subprocess.Popen(dataset_config.unity_lauch_cmd)
    time.sleep(5)

    '''Build scene communication'''
    comm = comm_unity.UnityCommunication(timeout_wait=300)

    '''Read skeletons'''
    all_room_script_files = list(dataset_config.script_bbox_path.glob('./*/script_*.json'))

    '''Write motion skeletons'''
    for script_file in all_room_script_files:
        print('Processing: %s.' % (script_file))
        '''Get scene id and room id'''
        scene_id = int(script_file.parent.name)
        room_id = int('.'.join(script_file.name.split('.')[:-1]).split('_')[-1])
        '''Reset the scene'''
        comm.reset(scene_id)
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
        assert room_id <= len(room_nodes) + 1
        room_node = room_nodes[room_id]
        '''Update graph'''
        graph_update = {}
        graph_update['nodes'] = nodes
        graph_update['edges'] = edges

        get_skeleton_from_scripts(comm, scene_id, room_id, room_node, graph_update, script_file, dataset_config,
                                  unity_laucher)

    unity_laucher.kill()
    time.sleep(5)
