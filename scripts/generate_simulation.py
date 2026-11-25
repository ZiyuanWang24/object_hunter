"""Render a MuJoCo tabletop scene and export virtual-camera ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

FPS = 30
WIDTH, HEIGHT = 960, 540
SCENE_XML = r'''
<mujoco model="objecthunter_tabletop">
  <option timestep="0.0166667" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="960" offheight="540"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.35 0.35 0.35" specular="0.18 0.18 0.18"/>
  </visual>
  <asset>
    <texture name="food_tex" type="2d" file="assets/meshes/Textures/food_colormap.png"/>
    <texture name="platformer_tex" type="2d" file="assets/meshes/Textures/platformer_colormap.png"/>
    <material name="floor_mat" rgba="0.055 0.07 0.09 1" reflectance="0.12"/>
    <material name="wood_mat" rgba="0.38 0.20 0.08 1" reflectance="0.08" shininess="0.22"/>
    <material name="edge_mat" rgba="0.16 0.07 0.025 1" reflectance="0.06"/>
    <material name="wood_highlight" rgba="0.52 0.29 0.11 1" reflectance="0.07"/>
    <material name="ceramic_mat" rgba="0.08 0.36 0.95 1" reflectance="0.18" shininess="0.55" specular="0.45"/>
    <material name="ceramic_dark" rgba="0.025 0.08 0.22 1" reflectance="0.12"/>
    <material name="coffee_mat" rgba="0.055 0.018 0.008 1" reflectance="0.08"/>
    <material name="bottle_mat" rgba="0.08 0.75 0.26 1" reflectance="0.16" shininess="0.45" specular="0.35"/>
    <material name="bottle_dark" rgba="0.025 0.22 0.07 1" reflectance="0.12"/>
    <material name="label_mat" rgba="0.72 0.92 0.72 1" reflectance="0.03"/>
    <material name="book_mat" rgba="0.72 0.12 0.70 1" reflectance="0.12" shininess="0.30"/>
    <material name="book_edge" rgba="0.42 0.045 0.38 1" reflectance="0.08"/>
    <material name="key_mat" rgba="0.72 0.74 0.78 1" reflectance="0.65" shininess="0.75" specular="0.8"/>
    <material name="key_dark" rgba="0.25 0.27 0.31 1" reflectance="0.35" shininess="0.55"/>
    <material name="paper_mat" rgba="0.88 0.84 0.70 1" reflectance="0.02"/>
    <material name="bookmark_mat" rgba="0.86 0.10 0.08 1" reflectance="0.04"/>
    <material name="mug_mesh_mat" rgba="0.08 0.36 0.95 1" reflectance="0.20" shininess="0.65" specular="0.55"/>
    <material name="bottle_mesh_mat" rgba="0.08 0.75 0.26 1" reflectance="0.15" shininess="0.55" specular="0.40"/>
    <material name="book_mesh_mat" texture="food_tex" texrepeat="1 1" texuniform="true" rgba="1 1 1 1" reflectance="0.08"/>
    <material name="key_mesh_mat" rgba="0.72 0.74 0.78 1" reflectance="0.65" shininess="0.80" specular="0.85"/>
    <mesh name="mug_mesh" file="assets/meshes/mug.obj"/>
    <mesh name="bottle_mesh" file="assets/meshes/bottle.obj"/>
    <mesh name="book_mesh" file="assets/meshes/book.obj"/>
    <mesh name="key_mesh" file="assets/meshes/key.obj"/>
  </asset>
  <worldbody>
    <light pos="-1.5 -1.8 3.0" dir="0.3 0.35 -1" diffuse="0.75 0.78 0.85" specular="0.25 0.25 0.25"/>
    <light pos="1.7 0.8 2.4" dir="-0.45 -0.25 -1" diffuse="0.42 0.34 0.26" specular="0.15 0.12 0.1"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="floor_mat"/>
    <body name="desk" pos="0 0 0.72">
      <geom name="desk_top" type="box" size="0.72 0.50 0.035" material="wood_mat"/>
      <geom name="desk_inset" type="box" pos="0 0 0.038" size="0.66 0.44 0.004" material="wood_highlight"/>
      <geom name="desk_edge" type="box" pos="0 0 -0.07" size="0.75 0.53 0.035" material="edge_mat"/>
      <geom name="desk_leg_front_left" type="box" pos="-0.62 -0.40 -0.38" size="0.055 0.055 0.34" material="edge_mat"/>
      <geom name="desk_leg_front_right" type="box" pos="0.62 -0.40 -0.38" size="0.055 0.055 0.34" material="edge_mat"/>
      <geom name="desk_leg_back_left" type="box" pos="-0.62 0.40 -0.38" size="0.055 0.055 0.34" material="edge_mat"/>
      <geom name="desk_leg_back_right" type="box" pos="0.62 0.40 -0.38" size="0.055 0.055 0.34" material="edge_mat"/>
    </body>
    <body name="cup" mocap="true" pos="-0.30 -0.15 0.755">
      <geom name="cup_mesh_geom" type="mesh" mesh="mug_mesh" material="mug_mesh_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="bottle" mocap="true" pos="0.28 -0.05 0.755">
      <geom name="bottle_mesh_geom" type="mesh" mesh="bottle_mesh" material="bottle_mesh_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="book" mocap="true" pos="0.30 0.27 0.755">
      <geom name="book_mesh_geom" type="mesh" mesh="book_mesh" material="book_mesh_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="keys" mocap="true" pos="-0.28 0.27 0.755">
      <geom name="key_mesh_geom" type="mesh" mesh="key_mesh" material="key_mesh_mat" contype="0" conaffinity="0"/>
    </body>
    <camera name="rgb_camera" pos="0 -0.05 2.65" quat="1 0 0 0" fovy="48"/>
  </worldbody>
</mujoco>
'''

WORLD_OFFSET = np.array([0.72, 0.50], dtype=float)
EMBEDDINGS = {"cup": [1.0, 0.0, 0.0], "bottle": [0.0, 1.0, 0.0], "book": [0.0, 0.0, 1.0], "keys": [0.7, 0.7, 0.0]}
COLORS_RGB = {"cup": np.array([20, 92, 242]), "bottle": np.array([20, 191, 66]), "book": np.array([184, 31, 178]), "keys": np.array([184, 189, 199])}


def state(name: str, frame: int) -> tuple[np.ndarray, bool, float]:
    t = frame / FPS
    if name == "cup":
        xy = np.array([-0.30 + 0.13 * np.sin(t * 0.55), -0.15 + 0.07 * np.cos(t * 0.45)])
        visible = not (6.0 <= t < 8.7)
        z = 0.755 if visible else 0.70
    elif name == "bottle":
        xy, visible, z = np.array([0.28, -0.05]), True, 0.755
    elif name == "book":
        xy, visible, z = np.array([0.30, 0.27]), True, 0.755
    else:
        xy, visible, z = np.array([-0.28, 0.27]), True, 0.755
    return xy, visible, z


def mocap_id(model: mujoco.MjModel, name: str) -> int:
    return int(model.body(name).mocapid[0])


def color_bbox(rgb: np.ndarray, target: np.ndarray) -> tuple[list[int], list[int]] | None:
    distance = np.linalg.norm(rgb.astype(np.float32) - target.astype(np.float32), axis=2)
    mask = (distance < 105).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return [x1, y1, x2, y2], [int(np.median(xs[ys >= y2 - max(2, (y2 - y1) // 5)])), y2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assets/simulation_scene.mp4")
    parser.add_argument("--ground-truth", default="assets/simulation_ground_truth.jsonl")
    parser.add_argument("--xml", default="assets/mujoco_tabletop.xml")
    parser.add_argument("--seconds", type=float, default=16.0)
    args = parser.parse_args()

    output, truth_path, xml_path = map(Path, (args.output, args.ground_truth, args.xml))
    output.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(SCENE_XML.strip() + "\n", encoding="utf-8")

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    names = ["cup", "bottle", "book", "keys"]
    ids = {name: mocap_id(model, name) for name in names}
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open MuJoCo video writer")

    with truth_path.open("w", encoding="utf-8") as handle:
        for frame_index in range(int(args.seconds * FPS)):
            visible_state = {}
            for name in names:
                xy, visible, z = state(name, frame_index)
                mid = ids[name]
                data.mocap_pos[mid] = [float(xy[0]), float(xy[1]), z]
                visible_state[name] = visible
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="rgb_camera")
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            objects = []
            for name in names:
                if not visible_state[name]:
                    continue
                bbox_contact = color_bbox(rgb, COLORS_RGB[name])
                if bbox_contact is None:
                    continue
                xy, _, _ = state(name, frame_index)
                world = (xy + WORLD_OFFSET).tolist()
                bbox, contact = bbox_contact
                objects.append({"class_name": name, "confidence": 0.99, "bbox": bbox, "contact_pixel": contact, "world_position": world, "embedding": EMBEDDINGS[name]})
            timestamp = frame_index / FPS
            cv2.putText(bgr, f"MUJOCO VIRTUAL RGB CAMERA  t={timestamp:05.2f}s", (20, HEIGHT - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 240, 245), 2, cv2.LINE_AA)
            cv2.putText(bgr, "desk: cup / bottle / book / keys", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 240), 1, cv2.LINE_AA)
            writer.write(bgr)
            handle.write(json.dumps({"frame": frame_index, "timestamp": timestamp, "objects": objects}) + "\n")
    writer.release()
    print(f"Wrote MuJoCo scene: {output}")
    print(f"Wrote ground truth: {truth_path}")
    print(f"Wrote model XML: {xml_path}")


if __name__ == "__main__":
    main()
