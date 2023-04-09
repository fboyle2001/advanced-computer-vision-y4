from typing import Optional
from dataclasses import dataclass

import numpy as np
import cv2

# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
joint_colours = np.linspace(0, 179, 26)[:-1]
joint_colours = np.array([(round(h), 255, 255) for h in joint_colours]).astype(np.uint8)
joint_colours = joint_colours[np.newaxis, ...]
joint_colours = cv2.cvtColor(joint_colours, cv2.COLOR_HSV2BGR)
joint_colours = joint_colours.squeeze().astype(int)
joint_colours = [tuple([int(x) for x in joint_colour]) for joint_colour in joint_colours]

@dataclass
class Joint:
    parent: Optional["Joint"]
    x: float
    y: float
    confidence: float
    
    def is_empty(self):
        return self.x == 0 and self.y == 0 and self.confidence == 0

joint_connections = {
    "nose": None,
    "neck": "nose",
    "right_shoulder": "neck",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_shoulder": "neck",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "mid_hip": "neck",
    "right_hip": "mid_hip",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "left_hip": "mid_hip",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "right_eye": "nose",
    "left_eye": "nose",
    "right_ear": "right_eye",
    "left_ear": "left_eye",
    "left_big_toe": "left_ankle",
    "left_small_toe": "left_big_toe",
    "left_heel": "left_ankle",
    "right_big_toe": "right_ankle",
    "right_small_toe": "right_big_toe",
    "right_heel": "right_ankle"
}

@dataclass
class PoseKeypoints:
    nose: Joint
    neck: Joint
    right_shoulder: Joint
    right_elbow: Joint
    right_wrist: Joint
    left_shoulder: Joint
    left_elbow: Joint
    left_wrist: Joint
    mid_hip: Joint
    right_hip: Joint
    right_knee: Joint
    right_ankle: Joint
    left_hip: Joint
    left_knee: Joint
    left_ankle: Joint
    right_eye: Joint
    left_eye: Joint
    right_ear: Joint
    left_ear: Joint
    left_big_toe: Joint
    left_small_toe: Joint
    left_heel: Joint
    right_big_toe: Joint
    right_small_toe: Joint
    right_heel: Joint
    
    def get_empty_joints(self):
        return {joint_name: getattr(self, joint_name) for joint_name in self.__annotations__.keys() if getattr(self, joint_name).is_empty()}
    
    def overlay_pose(self, frame):
        mapping = list(PoseKeypoints.__dict__["__annotations__"].keys())
        pose_overlayed_frame = frame.copy()
        height, width, channels = pose_overlayed_frame.shape

        for i, joint_name in enumerate(self.__annotations__.keys()):
            joint = getattr(self, joint_name)

            if joint.is_empty(): 
                continue
            
            joint_pos = (round(joint.x * width), round(joint.y * height))
            cv2.circle(pose_overlayed_frame, joint_pos, radius=1, color=joint_colours[i], thickness=2)
            cv2.putText(pose_overlayed_frame, mapping[i], (round(joint.x * width), round(joint.y * (height - 0.02))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, joint_colours[i], 1, cv2.LINE_AA)
            
            if joint.parent is not None and not joint.parent.is_empty():
                parent_pos = (round(joint.parent.x * width), round(joint.parent.y * height))
                cv2.line(pose_overlayed_frame, joint_pos, parent_pos, joint_colours[i], thickness=1)
        
        return pose_overlayed_frame
    
    def head_visibility_score(self):
        # 0 = not visible, 1 = completely visible
        parts = {"nose", "left_eye", "left_ear", "right_eye", "right_ear"}
        empty_parts = set(self.get_empty_joints().keys())
        visible_parts = parts - empty_parts
        return len(visible_parts) / len(parts)
    
    def torso_arms_visibility_score(self):
        # 0 = not visible, 1 = completely visible
        parts = {"neck", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"}
        empty_parts = set(self.get_empty_joints().keys())
        visible_parts = parts - empty_parts
        return len(visible_parts) / len(parts)
    
    def hip_visibility_score(self):
        # 0 = not visible, 1 = completely visible
        parts = {"mid_hip", "left_hip", "right_hip"}
        empty_parts = set(self.get_empty_joints().keys())
        visible_parts = parts - empty_parts
        return len(visible_parts) / len(parts)
    
    def legs_visibility_score(self):
        # 0 = not visible, 1 = completely visible
        parts = {"left_knee", "left_ankle", "right_knee", "right_ankle"}
        empty_parts = set(self.get_empty_joints().keys())
        visible_parts = parts - empty_parts
        return len(visible_parts) / len(parts)
    
    def feet_visibility_score(self):
        # 0 = not visible, 1 = completely visible
        parts = {"left_ankle", "left_heel", "left_big_toe", "left_small_toe", "right_ankle", "right_heel", "right_big_toe", "right_small_toe"}
        empty_parts = set(self.get_empty_joints().keys())
        visible_parts = parts - empty_parts
        return len(visible_parts) / len(parts)
    
    def knee_angles(self):
        left_ratio = None
        right_ratio = None
        
        # we can estimate if someone is sitting by looking at the angle of their knee to the hip
        # note left_hip > left_knee but left_hip.y < left_knee.y due to the way coords are done in these images
        
        if not self.left_hip.is_empty() and not self.left_knee.is_empty():
            left_opp = -self.left_hip.y + self.left_knee.y
            left_adj = abs(self.left_hip.x - self.left_knee.x)
            
            left_ratio = np.rad2deg(np.arctan2(left_opp, left_adj))
            
        if not self.right_hip.is_empty() and not self.right_knee.is_empty():
            right_opp = -self.right_hip.y + self.right_knee.y
            right_adj = abs(self.right_hip.x - self.right_knee.x)
            
            right_ratio = np.rad2deg(np.arctan2(right_opp, right_adj))
        
        return left_ratio, right_ratio
    
    def is_facing_camera(self):
        possible_pairs = ["eye", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "big_toe", "small_toe", "heel"]
        
        left_dominated = 0
        right_dominated = 0
        
        for pair_name in possible_pairs:
            left = getattr(self, f"left_{pair_name}")
            right = getattr(self, f"right_{pair_name}")
            
            if not left.is_empty() and not right.is_empty():
                is_left_dominated = left.x > right.x
                
                if is_left_dominated:
                    left_dominated += 1
                else:
                    right_dominated += 1
                
                # print(pair_name, left.x, right.x)
        
        return left_dominated > right_dominated
    
    def classify_general_pose(self):
        head_score = self.head_visibility_score()
        ta_score = self.torso_arms_visibility_score()
        hip_score = self.hip_visibility_score()
        legs_score = self.legs_visibility_score()
        feet_score = self.feet_visibility_score()
        
        if head_score >= 0.6 and ta_score <= 0.45 and hip_score + legs_score + feet_score == 0:
            return "Head Only"
        
        # hip_score >= 0 redundant
        if head_score >= 0.4 and ta_score >= 0.5 and hip_score >= 0 and legs_score + feet_score == 0: 
            return "Half Body"
        
        # feet_score >= 0 redundant
        if head_score >= 0.4 and ta_score >= 0.7 and hip_score >= 0.5 and legs_score >= 0.5 and feet_score >= 0:
            return "Full Body"
        
        return "Other"
        
    def is_sitting(self):
        left_angle, right_angle = self.knee_angles()
        
        if left_angle is None and right_angle is None:
            return "Other"
        
        if left_angle is not None:
            if left_angle < 0:
                return "Other"
        
        if right_angle is not None:
            if right_angle < 0:
                return "Other"
            
        angle_threshold = 50
        
        left_sitting = None if left_angle is None else left_angle < angle_threshold
        right_sitting = None if right_angle is None else right_angle < angle_threshold
        
        if left_sitting is not None and right_sitting is not None:
            # As long as one says they are sitting, we take it to be that they are sitting
            return "Sitting" if left_sitting or right_sitting else "Standing"
        
        if left_sitting is not None:
            return "Sitting" if left_sitting else "Standing"
        
        if right_sitting is not None:
            return "Sitting" if right_sitting else "Standing"
    
    def classify(self, enforce_facing=False):
        if not self.is_facing_camera() and enforce_facing:
            return "Other"
        
        general_pose = self.classify_general_pose()
        
        if general_pose != "Full Body":
            return general_pose
        
        position = self.is_sitting()
        
        if position == "Other":
            return "Other"
        
        return f"Full Body {position}"
    
    @staticmethod
    def load_keypoints(raw_kps):
        mapping = list(PoseKeypoints.__dict__["__annotations__"].keys())
        points_as_dict = {}
        
        for i, part in enumerate(mapping):
            if joint_connections[part] is None:
                points_as_dict[part] = Joint(None, raw_kps[3 * i], raw_kps[3 * i + 1], raw_kps[3 * i + 2])
            else:
                points_as_dict[part] = Joint(points_as_dict[joint_connections[part]], raw_kps[3 * i], raw_kps[3 * i + 1], raw_kps[3 * i + 2])
        
        return PoseKeypoints(**points_as_dict)