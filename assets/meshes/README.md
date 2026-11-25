# ObjectHunter simulation meshes

These are small, static visual assets used only by the MuJoCo virtual scene.

- `mug.obj`, `book.obj`: converted from Kenney Food Kit GLB assets. License: CC0 1.0. The original Kenney license is included as `Kenney_LICENSE.txt`.
- `key.obj`: converted from a Kenney Platformer asset. License: CC0 1.0.
- `bottle.obj`: converted from the Khronos glTF Sample Models WaterBottle asset. License: CC0 1.0.

The conversion is reproducible with `scripts/convert_glb_to_obj.py`. MuJoCo loads OBJ meshes and uses the repository's scene materials for reliable rendering across platforms.
