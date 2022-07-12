#  Copyright (c) 6.2021. Yinyu Nie
#  License: MIT

from configs.dataset_config import Dataset_Config

dataset_config = Dataset_Config('virtualhome')

joint_names = ['Hips', 'LeftUpperLeg', 'RightUpperLeg', 'LeftLowerLeg', 'RightLowerLeg', 'LeftFoot',
               'RightFoot', 'Spine', 'Chest', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder',
               'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm', 'LeftHand',
               'RightHand', 'LeftToes', 'RightToes', 'LeftEye', 'RightEye', 'Jaw', 'LeftThumbProximal',
               'LeftThumbIntermediate', 'LeftThumbDistal', 'LeftIndexProximal',
               'LeftIndexIntermediate', 'LeftIndexDistal', 'LeftMiddleProximal',
               'LeftMiddleIntermediate', 'LeftMiddleDistal', 'LeftRingProximal',
               'LeftRingIntermediate', 'LeftRingDistal', 'LeftLittleProximal',
               'LeftLittleIntermediate', 'LeftLittleDistal', 'RightThumbProximal',
               'RightThumbIntermediate', 'RightThumbDistal', 'RightIndexProximal',
               'RightIndexIntermediate', 'RightIndexDistal', 'RightMiddleProximal',
               'RightMiddleIntermediate', 'RightMiddleDistal', 'RightRingProximal',
               'RightRingIntermediate', 'RightRingDistal', 'RightLittleProximal',
               'RightLittleIntermediate', 'RightLittleDistal', 'UpperChest', 'LastBone']

LIMBS = [(0, 1), (1, 3), (3, 5), (5, 19), (0, 2), (2, 4), (4, 6), (6, 20),
         # HIPS -> UPPER LEG -> LOWER LEG -> FOOT -> TOES (for left and right legs)
         (0, 7), (7, 8), (8, 9), (9, 10),  # HIPS -> SPINE -> CHEST -> NECK -> HEAD
         (10, 21), (10, 22),  # HEAD -> EYE (for left and right eyes)
         (8, 11), (11, 13), (13, 15), (15, 17), (8, 12), (12, 14), (14, 16), (16, 18),
         # CHEST -> SHOULDER -> ARM -> FOREARM -> HAND (for left and right hands)
         (17, 24), (24, 25), (25, 26), (17, 27), (27, 28), (28, 29), (17, 30), (30, 31), (31, 32), (17, 33), (33, 34),
         (34, 35), (17, 36), (36, 37), (37, 38),
         # HAND -> PROXIMAL -> INTERMEDIATE -> DISTAL (for five fingers in left and right hands)
         (18, 39), (39, 40), (40, 41), (18, 42), (42, 43), (43, 44), (18, 45), (45, 46), (46, 47), (18, 48), (48, 49),
         (49, 50), (18, 51), (51, 52), (52, 53)]

# 'Jaw' (23), 'UpperChest' (54) and 'LastBone'(55) are not in the limb list.

valid_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28,
                   29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

# limb_nodes = list(set([item for sets in LIMBS for item in sets ]))
# node_mapping = {old:new for new, old in enumerate(limb_nodes)}
# LIMBS_new = [(node_mapping[item[0]], node_mapping[item[1]]) for item in LIMBS]

__all__ = ['dataset_config']
