"""Living-room MuJoCo scene used by the interactive room demo."""

from __future__ import annotations

import numpy as np


ROOM_XML = r'''
<mujoco model="objecthunter_living_room">
  <option timestep="0.0166667" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.35 0.35 0.35" specular="0.16 0.16 0.16"/>
    <rgba haze="0.12 0.16 0.22 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.22 0.34 0.48" rgb2="0.025 0.035 0.055" width="512" height="3072"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.18 0.12 0.08" rgb2="0.11 0.07 0.045" mark="edge" markrgb="0.28 0.18 0.11" width="512" height="512"/>
    <texture name="rug_tex" type="2d" builtin="checker" rgb1="0.18 0.24 0.30" rgb2="0.12 0.16 0.22" mark="edge" markrgb="0.28 0.34 0.40" width="256" height="256"/>
    <texture name="food_tex" type="2d" file="assets/meshes/Textures/food_colormap.png"/>
    <texture name="platformer_tex" type="2d" file="assets/meshes/Textures/platformer_colormap.png"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="7 7" texuniform="true" reflectance="0.12"/>
    <material name="rug_mat" texture="rug_tex" texrepeat="5 4" texuniform="true" reflectance="0.02"/>
    <material name="wall_mat" rgba="0.68 0.64 0.57 1" reflectance="0.02"/>
    <material name="wood_mat" rgba="0.28 0.12 0.045 1" reflectance="0.12" shininess="0.35"/>
    <material name="wood_light" rgba="0.46 0.24 0.10 1" reflectance="0.08"/>
    <material name="sofa_mat" rgba="0.10 0.22 0.30 1" reflectance="0.08" shininess="0.30"/>
    <material name="sofa_light" rgba="0.18 0.34 0.43 1" reflectance="0.06"/>
    <material name="screen_mat" rgba="0.015 0.02 0.025 1" reflectance="0.35" shininess="0.75"/>
    <material name="metal_mat" rgba="0.45 0.48 0.52 1" reflectance="0.58" shininess="0.80" specular="0.8"/>
    <material name="plant_mat" rgba="0.06 0.32 0.12 1"/>
    <material name="pot_mat" rgba="0.52 0.20 0.09 1" reflectance="0.05"/>
    <material name="frame_mat" rgba="0.08 0.05 0.035 1"/>
    <material name="art_mat" rgba="0.82 0.46 0.22 1" reflectance="0.04"/>
    <material name="bookmark_mat" rgba="0.86 0.10 0.08 1" reflectance="0.04"/>
    <material name="mug_mat" rgba="0.08 0.36 0.95 1" reflectance="0.20" shininess="0.65" specular="0.55"/>
    <material name="bottle_mat" rgba="0.08 0.75 0.26 1" reflectance="0.15" shininess="0.55" specular="0.40"/>
    <material name="book_mat" texture="food_tex" texrepeat="1 1" texuniform="true"/>
    <material name="key_mat" rgba="0.72 0.74 0.78 1" reflectance="0.65" shininess="0.80" specular="0.85"/>
    <mesh name="mug_mesh" file="assets/meshes/mug.obj"/>
    <mesh name="bottle_mesh" file="assets/meshes/bottle.obj"/>
    <mesh name="book_mesh" file="assets/meshes/book.obj"/>
    <mesh name="key_mesh" file="assets/meshes/key.obj"/>
  </asset>
  <worldbody>
    <light pos="-2 -3 5" dir="0.2 0.35 -1" directional="true" diffuse="0.75 0.78 0.9" specular="0.25 0.25 0.25"/>
    <light pos="3 1 3" dir="-0.35 -0.1 -1" directional="true" diffuse="0.45 0.33 0.22"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>
    <geom name="back_wall" type="box" pos="0 4.0 2.5" size="4.5 0.08 2.5" material="wall_mat"/>
    <geom name="left_wall" type="box" pos="-4.0 0 2.5" size="0.08 4.0 2.5" material="wall_mat"/>
    <geom name="right_wall" type="box" pos="4.0 0 2.5" size="0.08 4.0 2.5" material="wall_mat"/>
    <geom name="rug" type="box" pos="0 1.0 0.018" size="2.65 1.75 0.018" material="rug_mat"/>

    <!-- Sofa -->
    <body name="sofa" pos="0 2.45 0">
      <geom name="sofa_base" type="box" pos="0 0 0.38" size="1.75 0.60 0.38" material="sofa_mat"/>
      <geom name="sofa_seat" type="box" pos="0 -0.10 0.78" size="1.58 0.48 0.12" material="sofa_light"/>
      <geom name="sofa_back" type="box" pos="0 0.43 1.25" size="1.62 0.16 0.62" material="sofa_mat"/>
      <geom name="sofa_arm_l" type="box" pos="-1.58 0.02 0.82" size="0.18 0.58 0.48" material="sofa_mat"/>
      <geom name="sofa_arm_r" type="box" pos="1.58 0.02 0.82" size="0.18 0.58 0.48" material="sofa_mat"/>
      <geom name="sofa_leg_l" type="box" pos="-1.35 -0.42 0.09" size="0.10 0.10 0.09" material="wood_mat"/>
      <geom name="sofa_leg_r" type="box" pos="1.35 -0.42 0.09" size="0.10 0.10 0.09" material="wood_mat"/>
    </body>

    <!-- Coffee table -->
    <body name="coffee_table" pos="0 0.80 0">
      <geom type="box" pos="0 0 0.56" size="1.35 0.76 0.09" material="wood_light"/>
      <geom type="box" pos="-1.10 -0.56 0.28" size="0.09 0.09 0.28" material="wood_mat"/>
      <geom type="box" pos="1.10 -0.56 0.28" size="0.09 0.09 0.28" material="wood_mat"/>
      <geom type="box" pos="-1.10 0.56 0.28" size="0.09 0.09 0.28" material="wood_mat"/>
      <geom type="box" pos="1.10 0.56 0.28" size="0.09 0.09 0.28" material="wood_mat"/>
    </body>

    <!-- TV console and screen -->
    <body name="tv_console" pos="0 3.55 0">
      <geom type="box" pos="0 0 0.35" size="1.65 0.32 0.35" material="wood_mat"/>
      <geom type="box" pos="0 0 1.55" size="1.45 0.08 0.78" material="frame_mat"/>
      <geom type="box" pos="0 -0.085 1.55" size="1.30 0.012 0.62" material="screen_mat"/>
      <geom type="box" pos="-1.15 0 0.80" size="0.16 0.24 0.16" material="metal_mat"/>
      <geom type="box" pos="1.15 0 0.80" size="0.16 0.24 0.16" material="metal_mat"/>
    </body>

    <!-- Bookshelf -->
    <body name="bookshelf" pos="-3.05 3.25 0">
      <geom type="box" pos="0 0 1.55" size="0.78 0.28 1.55" material="wood_mat"/>
      <geom type="box" pos="0 -0.33 0.35" size="0.72 0.06 0.06" material="wood_light"/>
      <geom type="box" pos="0 -0.33 0.95" size="0.72 0.06 0.06" material="wood_light"/>
      <geom type="box" pos="0 -0.33 1.55" size="0.72 0.06 0.06" material="wood_light"/>
      <geom type="box" pos="0 -0.33 2.15" size="0.72 0.06 0.06" material="wood_light"/>
      <geom type="box" pos="-0.40 -0.37 0.66" size="0.09 0.025 0.23" material="art_mat"/>
      <geom type="box" pos="-0.18 -0.37 0.66" size="0.08 0.025 0.25" material="sofa_light"/>
      <geom type="box" pos="0.05 -0.37 0.66" size="0.09 0.025 0.22" material="bookmark_mat"/>
      <geom type="box" pos="0.32 -0.37 1.26" size="0.10 0.025 0.24" material="art_mat"/>
      <geom type="box" pos="0.52 -0.37 1.26" size="0.08 0.025 0.20" material="sofa_light"/>
    </body>

    <!-- Lamp -->
    <body name="floor_lamp" pos="-2.65 1.25 0">
      <geom type="cylinder" pos="0 0 0.05" size="0.38 0.05" material="metal_mat"/>
      <geom type="cylinder" pos="0 0 1.35" size="0.035 1.3" material="metal_mat"/>
      <geom type="cylinder" pos="0 0 2.55" size="0.32 0.28" material="art_mat"/>
      <geom type="sphere" pos="0 0 2.28" size="0.10" material="screen_mat"/>
    </body>

    <!-- Plant -->
    <body name="plant" pos="3.0 2.35 0">
      <geom type="cylinder" pos="0 0 0.28" size="0.32 0.28" material="pot_mat"/>
      <geom type="cylinder" pos="0 0 1.05" size="0.045 0.55" material="plant_mat"/>
      <geom type="ellipsoid" pos="-0.25 0 1.55" size="0.38 0.12 0.10" material="plant_mat"/>
      <geom type="ellipsoid" pos="0.25 0.05 1.72" size="0.38 0.12 0.10" material="plant_mat"/>
      <geom type="ellipsoid" pos="0 0.22 1.92" size="0.32 0.12 0.10" material="plant_mat"/>
      <geom type="ellipsoid" pos="0 -0.20 1.80" size="0.34 0.12 0.10" material="plant_mat"/>
    </body>

    <!-- Wall art and target props -->
    <geom type="box" pos="-1.95 3.86 2.45" size="0.65 0.025 0.50" material="frame_mat"/>
    <geom type="box" pos="-1.95 3.82 2.45" size="0.54 0.015 0.39" material="art_mat"/>
    <body name="mug" mocap="true" pos="-0.45 0.82 0.755">
      <geom type="mesh" mesh="mug_mesh" material="mug_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="bottle" mocap="true" pos="0.22 0.86 0.755">
      <geom type="mesh" mesh="bottle_mesh" material="bottle_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="book" mocap="true" pos="0.48 1.08 0.755">
      <geom type="mesh" mesh="book_mesh" material="book_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="keys" mocap="true" pos="-0.10 1.12 0.755">
      <geom type="mesh" mesh="key_mesh" material="key_mat" contype="0" conaffinity="0"/>
    </body>
    <camera name="room_camera" pos="5.2 -6.2 3.4" quat="0.86 0.22 0.32 0.34" fovy="55"/>
  </worldbody>
</mujoco>
'''

ROOM_TARGETS = {
    "mug": np.array([-0.45, 0.82], dtype=float),
    "bottle": np.array([0.22, 0.86], dtype=float),
    "book": np.array([0.48, 1.08], dtype=float),
    "keys": np.array([-0.10, 1.12], dtype=float),
}

ROOM_EMBEDDINGS = {
    "mug": np.array([1.0, 0.0, 0.0], dtype=float),
    "bottle": np.array([0.0, 1.0, 0.0], dtype=float),
    "book": np.array([0.0, 0.0, 1.0], dtype=float),
    "keys": np.array([0.7, 0.7, 0.0], dtype=float),
}
