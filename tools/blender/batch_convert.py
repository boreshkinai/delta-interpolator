import bpy
import sys


# How to use this script:
# Create the following batch script inside the directory with the .bvh files:
#
# FOR %%f IN (*.bvh) DO "c:\program files\blender foundation\blender 2.92\blender.exe" -b --python "C:\Users\antonios.valkanas\Documents\deeppose\.tools\blender\batch_convert.py" -- "%%f"
# echo done
#
# You may need to replace the blender.exe path and the absolute path to this script.
# The converted files will be in the same directory as the original files.

#Get command line arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "—"
bvh_in = argv[0]
fbx_out = argv[0].split(".")[0] + ".fbx"

# Import the BVH file
# See https://docs.blender.org/api/current/bpy.ops.import_anim.html#bpy.ops.import_anim.bvh
bpy.ops.import_anim.bvh(filepath=bvh_in, filter_glob="*.bvh", global_scale=0.01, frame_start=1, update_scene_fps=True, use_cyclic=False, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')

# Export as FBX
# See https://docs.blender.org/api/current/bpy.ops.export_scene.html
bpy.ops.export_scene.fbx(filepath=fbx_out, axis_forward='Y', axis_up='-Z', global_scale=0.01, use_selection=True)