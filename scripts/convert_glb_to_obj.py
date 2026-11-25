"""Convert the static, textured subset of a GLB file to OBJ for MuJoCo.

This deliberately handles the small static Kenney props used by the demo:
POSITION, NORMAL, TEXCOORD_0 and indexed triangle primitives.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


COMPONENT_TYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Unsupported GLB header in {path}")
    cursor = 12
    document = None
    binary = b""
    while cursor < length:
        chunk_length, chunk_type = struct.unpack_from("<I4s", raw, cursor)
        chunk = raw[cursor + 8:cursor + 8 + chunk_length]
        if chunk_type == b"JSON":
            document = json.loads(chunk.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            binary = chunk
        cursor += 8 + chunk_length
    if document is None:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    return document, binary


def read_accessor(document: dict, binary: bytes, accessor_index: int) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_TYPES[accessor["componentType"]])
    width = TYPE_WIDTH[accessor["type"]]
    count = int(accessor["count"])
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dtype.itemsize * width))
    if stride == dtype.itemsize * width:
        values = np.frombuffer(binary, dtype=dtype, count=count * width, offset=start).reshape(count, width)
    else:
        values = np.vstack([
            np.frombuffer(binary, dtype=dtype, count=width, offset=start + index * stride)
            for index in range(count)
        ])
    return values.astype(np.float32, copy=False)


def convert(input_path: Path, output_path: Path, texture_name: str, target_height: float) -> None:
    document, binary = read_glb(input_path)
    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    texcoords: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive["attributes"]
            if "POSITION" not in attributes:
                continue
            position = read_accessor(document, binary, attributes["POSITION"])
            # glTF is Y-up; MuJoCo uses Z-up.
            position = np.column_stack((position[:, 0], -position[:, 2], position[:, 1]))
            scale = target_height / max(float(position[:, 2].max() - position[:, 2].min()), 1e-6)
            position *= scale
            position[:, 0] -= (position[:, 0].min() + position[:, 0].max()) / 2.0
            position[:, 1] -= (position[:, 1].min() + position[:, 1].max()) / 2.0
            position[:, 2] -= position[:, 2].min()
            base = sum(len(item) for item in vertices)
            vertices.append(position)
            if "NORMAL" in attributes:
                normal = read_accessor(document, binary, attributes["NORMAL"])
                normal = np.column_stack((normal[:, 0], -normal[:, 2], normal[:, 1]))
                normals.append(normal)
            else:
                normals.append(np.zeros_like(position))
            if "TEXCOORD_0" in attributes:
                texcoords.append(read_accessor(document, binary, attributes["TEXCOORD_0"]))
            else:
                texcoords.append(np.zeros((len(position), 2), dtype=np.float32))
            if "indices" in primitive:
                index_values = read_accessor(document, binary, primitive["indices"]).reshape(-1).astype(np.int64)
            else:
                index_values = np.arange(len(position), dtype=np.int64)
            if len(index_values) % 3:
                index_values = index_values[: len(index_values) - len(index_values) % 3]
            faces.append(index_values.reshape(-1, 3) + base + 1)

    if not vertices:
        raise ValueError(f"No POSITION mesh primitives found in {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    material_path = output_path.with_suffix(".mtl")
    with output_path.open("w", encoding="utf-8") as obj:
        obj.write(f"mtllib {material_path.name}\nusemtl prop_material\n")
        for array in vertices:
            for x, y, z in array:
                obj.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for array in texcoords:
            for u, v in array:
                obj.write(f"vt {u:.7f} {1.0 - v:.7f}\n")
        for array in normals:
            for x, y, z in array:
                obj.write(f"vn {x:.7f} {y:.7f} {z:.7f}\n")
        vertex_offset = tex_offset = normal_offset = 0
        for vertices_part, faces_part in zip(vertices, faces):
            count = len(vertices_part)
            for face in faces_part:
                refs = []
                for index in face:
                    local = int(index) - 1 - vertex_offset
                    refs.append(f"{int(index)}/{tex_offset + local + 1}/{normal_offset + local + 1}")
                obj.write("f " + " ".join(refs) + "\n")
            vertex_offset += count
            tex_offset += count
            normal_offset += count
    material_path.write_text(
        "newmtl prop_material\nKd 1 1 1\nKa 0 0 0\nKs 0.25 0.25 0.25\nNs 64\n"
        f"map_Kd {texture_name}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--texture", default="colormap.png")
    parser.add_argument("--height", type=float, required=True)
    args = parser.parse_args()
    convert(args.input, args.output, args.texture, args.height)
    print(f"Converted {args.input.name} -> {args.output}")


if __name__ == "__main__":
    main()
