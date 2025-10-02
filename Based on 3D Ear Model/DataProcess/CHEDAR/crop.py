# 对chedar数据集进行裁剪的代码
# 需要配置source_dir = 输入路径（人头建模路径）
# target_dir = 输出路径（人耳模型路径）
# min_bound、max_bound 裁剪的空间（使用三维坐标系中两个点指定，使用默认给出的裁剪空间即可）

import open3d as o3d
import numpy as np
import os
import glob

def crop_and_visualize_mesh(input_file, output_dir, min_bound, max_bound):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(input_file)

    # Create an axis-aligned bounding box
    bounding_box = mesh.get_axis_aligned_bounding_box()
    bounding_box.min_bound = min_bound  # Set the minimum boundary
    bounding_box.max_bound = max_bound  # Set the maximum boundary

    # Crop the mesh
    cropped_mesh = mesh.crop(bounding_box)

    # Save the cropped mesh
    output_file = os.path.join(output_dir, "fsj-right.ply")
    o3d.io.write_triangle_mesh(output_file, cropped_mesh)

    # Optionally, visualize the cropped mesh
    # o3d.visualization.draw_geometries([cropped_mesh])

# Define the directory containing the chader*.ply files and the target directory
# source_dir = r"D:\hrtf\myHead\heads"
# target_dir = r"D:\hrtf\myHead\cropped_chader"

source_dir=r"E:\grad_project\my_head_3D"
target_dir = r"E:\grad_project\crop_3D"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Define the bounding box's minimum and maximum boundaries
# min_bound = [-0.028, -0.05, -0.032]
# max_bound = [0.018, 0.2, 0.037]

#
# min_bound = [-0.05, -0.2, -0.38]
# max_bound = [-0.02, -0.128, -0.34]

min_bound = [0.08, -0.19, -0.436]
max_bound = [0.15, -0.125, -0.4]
# Find all chader*.ply files in the directory

for input_file in glob.glob(os.path.join(source_dir, "f*.ply")):
    crop_and_visualize_mesh(input_file, target_dir, min_bound, max_bound)

