# 人耳模型体素化代码
# 调用方式示例： 输入：人耳模型文件夹路径，自动遍历所有ply（stl）文件并执行体素化
# 应对不同数据集需要修改的地方： process_ply_to_npy_centered(file_path, voxel_size=0.0028, max_dimension=32) 方法中的voxel_size，因为不同数据集的标准单位不同，
#   例如对于CHEDAR数据集这个voxel_size参数应该设置为0.28（不太记得了，需要多试试并绘制体素化结果确认）
# directory_path = r"/data1/fyw/hrtf/dataset/chedar/cropped_chader/"
# process_directory(directory_path)


# import numpy as np
# import trimesh
#
# VOXEL_SIZE = 10 # mm
#
# def voxel_center(xyz):
#     assert(len(xyz) == 3)
#     x, y, z = [i * VOXEL_SIZE + VOXEL_SIZE / 2.0 for i in xyz]
#
#     return (x, y, z)
#
# def point2voxelID(xyz):
#     assert(len(xyz) == 3)
#     x, y, z = [int(i / VOXEL_SIZE) for i in xyz]
#     return (x, y, z)
#
# def point_in_voxel(xyz, voxel_xyz):
#     voxel_c = voxel_center(voxel_xyz)
#     voxel_bounday = [[axis - VOXEL_SIZE / 2, axis + VOXEL_SIZE / 2] for axis in voxel_c]
#
#     flag = True
#     for axis, boundary in enumerate(voxel_bounday):
#         if xyz[axis] < boundary[0] or xyz[axis] > boundary[1]:
#             flag = False
#
#     return flag
#
# def calc_intersect(voxel_xyz, face):
#     assert(len(voxel_xyz) == 3)
#     assert face.shape == (3, 3)
#
#     voxel_c = voxel_center(voxel_xyz)
#     voxel_bounday = [[axis - VOXEL_SIZE / 2, axis + VOXEL_SIZE / 2] for axis in voxel_c]
#
#     for i, v1 in enumerate(face):
#         v2 = face[(i+1) % len(face)]
#
#         v1_to_v2 = v2 - v1
#         for axis, boundary in enumerate(voxel_bounday):
#
#             # Calculate the intersection between the line of v1_to_v2 and each boundary
#             if v1_to_v2[axis] == 0.0:
#                 continue
#
#             for boundary_side in boundary:
#                 t = (boundary_side - v1[axis]) / v1_to_v2[axis]
#                 intersect = v1 + t * v1_to_v2
#                 axis1 = (axis + 1) % 3
#                 axis2 = (axis + 2) % 3
#
#                 if t >= 0 and t <= 1 \
#                     and intersect[axis1] >= voxel_bounday[axis1][0] \
#                     and intersect[axis1] <= voxel_bounday[axis1][1] \
#                     and intersect[axis2] >= voxel_bounday[axis2][0] \
#                     and intersect[axis2] <= voxel_bounday[axis2][1]:
#
#                     return True
#
#     # Special case: the triangle lies inside voxel
#     if point_in_voxel(face[0], voxel_xyz) and point_in_voxel(face[1], voxel_xyz) and point_in_voxel(face[2], voxel_xyz):
#         return True
#     return False
#
# def transform_mesh(mesh):
#     max_cord = np.amax(mesh.vertices, axis=0)
#     min_cord = np.amin(mesh.vertices, axis=0)
#
#     min_x, min_y, min_z = min_cord
#     max_x, max_y, max_z = max_cord
#     print("Size of mesh (m):", (max_x, max_y, max_z))
#     print("Transformed size (per voxel):", point2voxelID(max_cord * 1e3))
#
#     # check if the cordinate start from positive
#     assert round(min_x) == 0 and round(min_y) == 0 and round(min_z) == 0
#
#     voxel_space_size = point2voxelID(max_cord * 1e3) + (4,)
#     print(voxel_space_size)
#     voxel_space = np.zeros(voxel_space_size)
#
#     faces = mesh.faces
#     print("Number of faces:", len(faces))
#
#     sum_size = np.zeros((3,))
#     for face_idx, face in enumerate(faces):
#         face_vertices = mesh.vertices[face]
#
#         mesh_max = np.amax(face_vertices, axis=0)
#         mesh_min = np.amin(face_vertices, axis=0)
#
#         max_voxel = point2voxelID(mesh_max * 10e3)
#         min_voxel = point2voxelID(mesh_min * 10e3)
#         sum_size += np.array(max_voxel) - np.array(min_voxel)
#
#         for ix in range(min_voxel[0], max_voxel[0] + 1):
#             for iy in range(min_voxel[1], max_voxel[1] + 1):
#                 for iz in range(min_voxel[2], max_voxel[2] + 1):
#
#                     # Calculate if the block has intersection with the face
#                     if calc_intersect((ix, iy, iz), face_vertices):
#                         color = np.copy(mesh.visual.face_colors[face_idx])
#                         voxel_space[ix, iy, iz] = np.concatenate([[1], color[:3]])
#
#     print("Mean covered voxel size: ", sum_size / len(faces))
#
#     return voxel_space

# import trimesh
# if __name__ == '__main__':
#     mesh = trimesh.load(r"D:\hrtf\chader\chedar_0001.ply")
#
#     voxel = transform_mesh(mesh)
#     print(voxel.shape)
#     np.save('voxel10.npy', voxel)

# import open3d as o3d
# import numpy as np
#
# # 读取网格并创建体素网格
# mesh = o3d.io.read_triangle_mesh(r"C:\Users\freeway\Desktop\HRTF_dataset\HUBTUB\3D head meshes\pp1_3DheadMesh.ply")
# mesh.compute_vertex_normals()
# voxel_size = 0.002
# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
#
# # 获取体素网格的边界盒
# bounding_box = voxel_grid.get_axis_aligned_bounding_box()
# min_bound = bounding_box.get_min_bound()
# max_bound = bounding_box.get_max_bound()
#
# print(min_bound)
# print(max_bound)
#
# # 计算体素网格的维度
# dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
# print(dims)
# # Check if any dimension exceeds 300
# if np.any(dims > 300):
#     raise ValueError("One or more dimensions exceed the limit of 300.")
#
# # Expand the matrix to 300x300x300
# expanded_dims = np.array([300, 300, 300])
# voxel_matrix = np.zeros(expanded_dims, dtype=np.uint8)
#
# # Iterate over voxels and fill the NumPy array
# for voxel in voxel_grid.get_voxels():
#     voxel_index = np.floor((voxel.grid_index - min_bound) / voxel_size).astype(int)
#     if np.all(voxel_index < dims):  # Ensure the index is within the original dims
#         voxel_matrix[tuple(voxel_index)] = 1
#
# # Print the shape of the voxel matrix
# print(voxel_matrix.shape)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# # Your voxel_matrix
# # voxel_matrix = ...
#
# # Create a figure for plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Extract the x, y, z coordinates of the voxels
# x, y, z = np.indices(np.array(voxel_matrix.shape) + 1)
# voxels = (voxel_matrix == 1)  # Voxels to draw
#
# # Customize the colors and transparency
# colors = np.empty(voxels.shape, dtype=object)
# colors[voxels] = 'blue'  # Change 'blue' to your preferred color
#
# # Plotting the voxels
# ax.voxels(x, y, z, voxels, facecolors=colors, edgecolor='k')
#
# # Setting the labels and title
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('Voxel Grid Visualization')
#
# # Show the plot
# plt.show()
# 保存 NumPy 数组到文件
# np.save("voxel_matrix.npy", voxel_matrix)

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_voxel_matrix(voxel_matrix):
    # 获取体素矩阵中所有非零体素的坐标
    x, y, z = voxel_matrix.nonzero()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red', marker='s')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 调整视角
    ax.view_init(elev=20., azim=-35)

    plt.show()

def process_ply_to_npy(file_path, voxel_size=0.010, max_dimension=32):
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()

    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    # Get bounding box
    bounding_box = voxel_grid.get_axis_aligned_bounding_box()
    min_bound = bounding_box.get_min_bound()
    max_bound = bounding_box.get_max_bound()
    # Compute dimensions
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    if np.any(dims > max_dimension):
        raise ValueError("Dimension exceeds the limit of 300.")

    # Expand the matrix
    voxel_matrix = np.zeros((max_dimension, max_dimension, max_dimension), dtype=np.uint8)

    # Fill the matrix
    for voxel in voxel_grid.get_voxels():
        voxel_index = np.floor((voxel.grid_index - min_bound) / voxel_size).astype(int)
        if np.all(voxel_index < dims):
            voxel_matrix[tuple(voxel_index)] = 1

    # Save to .npy file
    npy_file_path = os.path.splitext(file_path)[0] + '.npy'
    visualize_voxel_matrix(voxel_matrix)
    np.save(npy_file_path, voxel_matrix)

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.ply'):
            file_path = os.path.join(directory_path, filename)
            try:
                process_ply_to_npy_centered(file_path)
                # print(f"Processed {file_path}")
            except ValueError as e:
                print(f"Error processing {filename}: {e}")

check = 0

def process_ply_to_npy_centered(file_path, voxel_size=0.0028, max_dimension=32):
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()

    # Get vertices
    vertices = np.asarray(mesh.vertices)

    # Find the vertex closest to y=0 in XZ plane
    distances_to_y_axis = np.sqrt(vertices[:, 0]**2 + vertices[:, 2]**2)
    closest_idx = np.argmin(distances_to_y_axis)
    closest_point_to_y0 = vertices[closest_idx]

    # Calculate bounding box of the mesh
    bounding_box_min = vertices.min(axis=0)
    bounding_box_max = vertices.max(axis=0)

    # Centering: move the closest point to the center of the voxel grid
    mesh_center_in_voxels = (closest_point_to_y0 - bounding_box_min) / voxel_size
    desired_center = np.array([max_dimension // 2, max_dimension // 2, max_dimension // 2])
    offset_in_voxels = desired_center - mesh_center_in_voxels

    # Adjust all vertices based on the calculated offset
    adjusted_vertices = (vertices - bounding_box_min) / voxel_size + offset_in_voxels

    # Ensure adjusted vertices are within bounds
    adjusted_vertices = np.clip(adjusted_vertices, 0, max_dimension - 1).astype(int)

    # Initialize the voxel matrix
    voxel_matrix = np.zeros((max_dimension, max_dimension, max_dimension), dtype=np.uint8)

    # Fill the matrix
    for voxel_index in adjusted_vertices:
        if np.all(voxel_index < max_dimension):  # Double-check to avoid index out of bounds
            voxel_matrix[tuple(voxel_index)] = 1

    # Save to .npy file
    npy_file_path = os.path.splitext(file_path)[0] + '.npy'
    np.save(npy_file_path, voxel_matrix)
    print(f"Saved centered voxel grid to {npy_file_path}")

    return voxel_matrix



def process_ply_to_npy_self(file_path, voxel_size=2.8, max_dimension=32):
    global check
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    # Get vertices
    vertices = np.asarray(mesh.vertices)

    # Compute the bounding box
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)


    # Compute dimensions based on the bounding box and voxel size
    dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    # print("Dimensions:", dims)

    # Check if dimensions exceed the limit
    if np.any(dims > max_dimension):
        raise ValueError("Dimension exceeds the limit of 32.")
    else:
        check = check + 1
        # print(check)
    # Initialize the voxel matrix
    voxel_matrix = np.zeros(dims, dtype=np.uint8)

    # Fill the matrix
    for vertex in vertices:
        voxel_index = np.floor((vertex - min_bound) / voxel_size).astype(int)
        # Ensure the index is within the bounds before setting it to 1
        if np.all(voxel_index < dims):
            voxel_matrix[tuple(voxel_index)] = 1

    # Adjust the voxel matrix to max_dimension if its size is smaller
    if np.any(voxel_matrix.shape) < max_dimension:
        padded_voxel_matrix = np.zeros((max_dimension, max_dimension, max_dimension), dtype=np.uint8)
        padded_voxel_matrix[:voxel_matrix.shape[0], :voxel_matrix.shape[1], :voxel_matrix.shape[2]] = voxel_matrix
        voxel_matrix = padded_voxel_matrix

    # print("Voxel Matrix Shape:", voxel_matrix.shape)

    # Save to .npy file
    npy_file_path = os.path.splitext(file_path)[0] + '.npy'
    np.save(npy_file_path, voxel_matrix)
    # 可视化
    print(f"Saved voxel grid to {npy_file_path}")
    # visualize_voxel_matrix(voxel_matrix)
    npy_file_path = os.path.splitext(file_path)[0] + '.npy'
    visualize_voxel_matrix(voxel_matrix)
    print(npy_file_path)
    np.save(npy_file_path, voxel_matrix)
    return voxel_matrix

# Example usage
directory_path = r"E:\grad_project\mirror"
process_directory(directory_path)
