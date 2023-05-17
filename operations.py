# Importing external libraries
import math
import numpy as np
import open3d as o3d
import os
import time
import statistics
import cv2


def generate_list(file_format, dir, exclude='text_to_exclude', include=''):
    """
    Function that generates the list of file with a specific extension in a folder

    Parameters
    ----------
    file_format : TYPE str
        DESCRIPTION.The extension to look for
    dir : TYPE str
        DESCRIPTION. The folder to look into
    exclude : TYPE str, optional
        DESCRIPTION. An optional parameter to exclude files that include some
        text in their nameThe default is 'text_to_exclude'.
    include : TYPE str, optional
        DESCRIPTION. An optional parameter to specifically include files with
        some text in their nameThe default is ''.

    Returns
    -------
    file_list : TYPE list
        DESCRIPTION.The list of detected files

    """

    file_list = []
    for file in os.listdir(dir):
        fileloc = os.path.join(dir, file)
        if file.endswith(file_format):
            if exclude not in file:
                if include in file:
                    file_list.append(fileloc)
    return file_list


def new_dir(dir_path):
    """
    Simple function to verify if a directory exists and if not creating it

    Parameters
    ----------
    dir_path : TYPE str
        DESCRIPTION.The path to check

    Returns
    -------
    None.

    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def find_substring(substring, folder):
    """
    Function that finds a file with a specific substring in a folder, and return its path

    Parameters
    ----------
    substring : TYPE str
        DESCRIPTION.Substring to be looked for
    folder : TYPE str
        DESCRIPTION.Input folder

    Returns
    -------
    path : TYPE
        DESCRIPTION.

    """
    for file in os.listdir(folder):
        if substring in file:
            path = os.path.join(folder, file)
    return path

def find_substring_new_path(substring, new_path, folder):
    """
    Function that finds a file with a specific substring in a folder, and move it to a new location
    @ parameters:
        substring -- substring to be looked for (string)
        new_path -- new_path where to move the found file (string)
        folder -- input folder (string)
    """
    # rename and place in right folder

    for file in os.listdir(folder):
        if substring in file:
            os.rename(os.path.join(folder, file), new_path)

def find_substring_delete(substring, folder):
    for file in os.listdir(folder):
        if substring in file:
            os.remove(os.path.join(folder, file))

"""
======================================================================================
ALGEBRA
======================================================================================
"""
# definition of rotation matrices, useful for camera operations
def rot_x_matrix(angle):
    matrix = np.asarray([[1, 0, 0, 0],
                         [0, math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                         [0, math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                         [0, 0, 0, 1]])
    return matrix


def rot_y_matrix(angle):
    matrix = np.asarray([[math.cos(math.radians(angle)), 0, math.sin(math.radians(angle)), 0],
                         [0, 1, 0, 0],
                         [-math.sin(math.radians(angle)), 0, math.cos(math.radians(angle)), 0],
                         [0, 0, 0, 1]])
    return matrix


def rot_z_matrix(angle):
    matrix = np.asarray([[math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0, 0],
                         [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    return matrix


def front_mat():
    matrix = rot_x_matrix(-90)
    inv_matrix = rot_x_matrix(90)
    return matrix, inv_matrix


def back_mat():
    matrix1 = rot_x_matrix(-90)
    matrix2 = rot_y_matrix(180)
    final_matrix = matrix2 @ matrix1
    inv_matrix1 = rot_y_matrix(-180)
    inv_matrix2 = rot_x_matrix(90)
    final_inv_matrix = inv_matrix2 @ inv_matrix1
    return final_matrix, final_inv_matrix


def right_mat():
    matrix1 = rot_x_matrix(-90)
    matrix2 = rot_y_matrix(-90)
    final_matrix = matrix2 @ matrix1
    inv_matrix1 = rot_y_matrix(90)
    inv_matrix2 = rot_x_matrix(90)
    final_inv_matrix = inv_matrix2 @ inv_matrix1
    return final_matrix, final_inv_matrix


def left_mat():
    matrix1 = rot_x_matrix(-90)
    matrix2 = rot_y_matrix(90)
    final_matrix = matrix2 @ matrix1
    inv_matrix1 = rot_y_matrix(-90)
    inv_matrix2 = rot_x_matrix(90)
    final_inv_matrix = inv_matrix2 @ inv_matrix1
    return final_matrix, final_inv_matrix


def iso2_mat():
    matrix1 = rot_x_matrix(60)
    matrix2 = rot_y_matrix(-20)
    matrix3 = rot_z_matrix(190)
    final_matrix1 = matrix3 @ matrix2
    final_matrix = final_matrix1 @ matrix1
    inv_matrix1 = rot_z_matrix(-190)
    inv_matrix2 = rot_y_matrix(20)
    inv_matrix3 = rot_x_matrix(-60)
    final_inv_matrix1 = inv_matrix3 @ inv_matrix2
    final_inv_matrix = final_inv_matrix1 @ inv_matrix1
    return final_matrix, final_inv_matrix


def iso1_mat():
    matrix1 = rot_x_matrix(-60)
    matrix2 = rot_y_matrix(-20)
    matrix3 = rot_z_matrix(-10)
    final_matrix1 = matrix3 @ matrix2
    final_matrix = final_matrix1 @ matrix1
    inv_matrix1 = rot_z_matrix(10)
    inv_matrix2 = rot_y_matrix(20)
    inv_matrix3 = rot_x_matrix(60)
    final_inv_matrix1 = inv_matrix3 @ inv_matrix2
    final_inv_matrix = final_inv_matrix1 @ inv_matrix1
    return final_matrix, final_inv_matrix


def name_to_matrix(orientation):
    if orientation == 'iso_front':
        trans_init, inv_trans = iso1_mat()
    elif orientation == 'iso_back':
        trans_init, inv_trans = iso2_mat()
    elif orientation == 'left':
        trans_init, inv_trans = left_mat()
    elif orientation == 'right':
        trans_init, inv_trans = right_mat()
    elif orientation == 'front':
        trans_init, inv_trans = front_mat()
    elif orientation == 'back':
        trans_init, inv_trans = back_mat()

    return trans_init, inv_trans


"""
======================================================================================
IO OPERATIONS 
======================================================================================
"""

def read_points(pcd_load):
    """
    Function to read a point cloud assets
        @param CloudAsset: the name of the asset
        @param zoom: rendering option: zoom level
        @param p_size: rendering option: point size
        @return: /
    """

    def compute_density(center, dim_z):
        density = []
        pt1 = [center[0] - 2, center[1] - 2, center[2] - dim_z / 2]
        pt2 = [center[0] + 2, center[1] + 2, center[2] + dim_z / 2]
        pt3 = [center[0] - 2, center[1] + 2, center[2] + dim_z / 2]
        np_points = [pt1, pt2, pt3]
        points = o3d.utility.Vector3dVector(np_points)

        crop_box = o3d.geometry.AxisAlignedBoundingBox
        crop_box = crop_box.create_from_points(points)

        point_cloud_crop = pcd_load.crop(crop_box)
        dist = point_cloud_crop.compute_nearest_neighbor_distance()

        if dist:
            density = statistics.mean(dist)
        return density

    # Compute basic properties
    bound = pcd_load.get_axis_aligned_bounding_box()
    center = bound.get_center()
    dim = bound.get_extent()
    points_list = np.asarray(pcd_load.points)
    n_points = len(points_list[:, 1])
    density = compute_density(center, dim[2])
    # if the density was computed on an empty zone, switch the computation zone
    if not density:
        center_x_bis = center[0] - dim[0] / 4
        center_bis = [center_x_bis,center[1], center[2]]
        density = compute_density(center_bis, dim[2])

    return bound, center, dim, points_list, n_points, density


"""
======================================================================================
RENDERING ORTHOS 
======================================================================================
"""

def rgb2gray(img_rgb):
    return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])

def crop_empty_areas(img_path):
    img = cv2.imread(img_path)
    gray = rgb2gray(img)
    gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

    crop_img = img[y:y + h, x:x + w]

    cv2.imwrite(img_path, crop_img)


def load_cloud(cloud_path):
    pcd_load = o3d.io.read_point_cloud(cloud_path)
    return pcd_load


def basic_vis_creation(pcd_load, orientation, p_size=1, back_color=[1, 1, 1]):
    """A function that creates the basic environment for creating things with open3D
            @ parameters :
                pcd_load -- a point cloud loaded into open3D
                orientation -- orientation of the camera; can be 'top', ...
                p_size -- size of points
                back_color -- background color
    """
    if orientation != 'top':
        trans_init, inv_trans = name_to_matrix(orientation)
        pcd_load.transform(trans_init)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd_load)
    opt = vis.get_render_option()
    opt.point_size = p_size
    opt.background_color = np.asarray(back_color)
    ctr = vis.get_view_control()

    return vis, opt, ctr

def render_cloud_rgb_simple(pcd_load, output_path, orientation):
    """A function to render a point cloud with Open3D engine
        @ parameters :
            pcd_load -- output of the loaf_cloud function
            orientation -- the point of view, choose between iso_front / iso_back / left / right / front / back (str)
    """

    vis, opt, ctr = basic_vis_creation(pcd_load, orientation)

    ctr.change_field_of_view(step=-90)
    opt.point_size = 1.3

    vis.poll_events()
    vis.update_renderer()
    time.sleep(1)
    vis.capture_screen_image(output_path, do_render=True)

    # remove white parts
    crop_empty_areas(output_path)


def render_cloud_rgb_zoom(pcd_load, output_path, orientation, style='orthogonal', zoom=0):
    """A function to render a point cloud with Open3D engine
    @ parameters :
        pcd_load -- output of the loaf_cloud function
        orientation -- the point of view, choose between iso_front / iso_back / left / right / front / back (str)
        style -- choose between orthogonal or perspective (str)"""

    vis, opt, ctr = basic_vis_creation(pcd_load, orientation)

    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    new_param = param
    ex = new_param.extrinsic.copy()
    ex[2, 3] = ex[2, 3] - zoom

    new_param.extrinsic = ex
    ctr.convert_from_pinhole_camera_parameters(new_param)
    if style == 'orthogonal':
        ctr.change_field_of_view(step=-90)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(1)
    vis.capture_screen_image(output_path, do_render=True)

    # remove white parts
    crop_empty_areas(output_path)


def render_cloud_rgb_ortho_res(pcd_load, output_path, orientation, p_size, res, pix_x=1920, pix_y=1055):

    def h_fov(pix_x, pix_y):
        d = pix_y / 2 / math.tan(30 * math.pi / 180)
        x = math.atan(pix_x / 2 / d)
        return x

    def inter(d, fov):
        inter = 2 * d * math.tan(fov) - 2 * center[2] * math.tan(fov)
        return inter

    def d_for_res(res, fov):
        required_int = res*pix_x
        required_d = (required_int + 2 * center[2] * math.tan(fov)) /2 /math.tan(fov)
        return required_d

    def zoom_factor(required_d, inter):
        zoom_factor = inter / (required_d * (2 * math.tan(fov)) - 2 * center[2] * math.tan(fov))
        return zoom_factor

    # basic environment
    vis, opt, ctr = basic_vis_creation(pcd_load, orientation)

    # Get extrinsic camera parameters
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    new_param = param
    ex = new_param.extrinsic.copy()

    bound = pcd_load.get_axis_aligned_bounding_box()
    center = bound.get_center()

    fov = h_fov(pix_x, pix_y)
    inter_init = inter(ex[2, 3], fov)
    d_zoom = d_for_res(res, fov)

    zoom = zoom_factor(d_zoom, inter_init)
    struc = [math.ceil(zoom), math.ceil(zoom)]

    step_h = inter(d_zoom, fov)
    step_v = step_h * pix_y / pix_x
    range_h = step_h * struc[1]
    range_v = step_v * struc[0]

    width = struc[1] * pix_x
    height = struc[0] * pix_y

    #create matrix of the final recomposed image
    final_img = np.zeros((height, width, 3), np.uint8)

    for j in range(1, struc[0] + 1):
        for i in range(1, struc[1] + 1):
            if i == 1 and j == 1:
                param = vis.get_view_control().convert_to_pinhole_camera_parameters()
                a = range_h / 2 - step_h / 2
                b = range_v / 2 - step_v / 2
                c = -(ex[2, 3] - d_zoom)
            elif i == 1:
                ctr.convert_from_pinhole_camera_parameters(param)
                a = (struc[1] - 1) * step_h
                b = - step_v
                c = 0
            else:
                ctr.convert_from_pinhole_camera_parameters(param)
                a = - step_h
                b = 0
                c = 0
            new_param = param
            ex = new_param.extrinsic.copy()

            ex[0, 3] += a
            ex[1, 3] += b
            ex[2, 3] += c

            new_param.extrinsic = ex
            ctr.convert_from_pinhole_camera_parameters(new_param)
            ctr.change_field_of_view(step=-90)
            vis.poll_events()
            vis.update_renderer()

            img_name = output_path[:-4] + str(j) + str(i) + '.png'
            vis.capture_screen_image(img_name, True)
            im = cv2.imread(img_name)

            start_pixel_h = (i - 1) * pix_x
            stop_pixel_h = start_pixel_h + pix_x
            start_pixel_v = (j - 1) * pix_y
            stop_pixel_v = start_pixel_v + pix_y

            final_img[start_pixel_v:stop_pixel_v, start_pixel_h:stop_pixel_h, :] = im

            # remove_temp_img
            os.remove(img_name)
            cv2.imwrite(output_path, final_img)

    # Eliminate white areas
    image_processing.crop_empty_areas(output_path)



def render_cloud_rgb_ortho_allviews(pcd_load, output_folder, p_size, res):
    orientations = ['top', 'left', 'right', 'front', 'back', 'iso_front', 'iso_back']
    for num, orientation in enumerate(orientations):
        output_path = os.path.join(output_folder, orientation + '.png')
        render_cloud_rgb_ortho_res(pcd_load, output_path, orientation, p_size, res)