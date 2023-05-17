import os
import shutil

from PySide6 import QtWidgets, QtGui, QtCore
import open3d as o3d
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import numpy as np

from qt_material import apply_stylesheet

import resources as res
import widgets as wid
from pointify_engine import process

# PARAMETERS
POINT_LIM = 1_000_000 # the limit of point above which performing a subsampling operation for advanced computations
VOXEL_DS = 0.025 # When the point cloud is to dense, this gives the minimum spatial distance to keep between two points
MIN_RANSAC_FACTOR = 350 # (Total number of points / MIN_RANSAC_FACTOR) gives the minimum amount of points to define a
# Ransac detection
RANSAC_DIST = 0.03 # maximum distance for a point to be considered belonging to a plane

# Floor detection
BIN_HEIGHT = 0.1

# Geometric parameters
SPHERE_FACTOR = 10
SPHERE_MIN = 0.019
SPHERE_MAX = 1

# Planar analysis
LIM_POINTS = 1_000_000

# choose outputs options
slices = False  # generate slices in all directions
xray_exterior_views = False  # generate exterior xray views
normal_exterior_views = True  # generate exterior views
line_render_exterior_views = True
h_planar_views = False  # generate views of each detected horizontal plane
simple_floorplan = False
advanced_floorplan = False

class DiaOptions(QtWidgets.QDialog):
    """
    Dialog that allows to choose some processing parameters
    """
    def __init__(self, parent=None):

        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'parameters'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        # combobox
        self.list_items = ['.las', '.laz', '.ply', '.e57']
        self.comboBox.addItems(self.list_items)
        self.lineEdit.setText(str(LIM_POINTS))

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

class NokPointCloud:
    def __init__(self):
        self.name = ''
        self.path = ''

        self.pc_load = None
        self.bound_pc_path = ''
        self.sub_pc_path = ''
        self.folder = ''
        self.processed_data_dir = ''
        self.img_dir = ''
        self.ransac_done = False
        self.sub_sampled = False
        self.view_names = []
        self.view_paths = []

        # basic properties
        self.bound, self.bound_points, self.center, self.dim, self.density, self.n_points = 0, 0, 0, 0, 0, 0
        self.n_points = 0
        self.n_points_sub = 0

        # ransac properties
        self.n_planes = 0

        # render properties
        self.res = 0

    def update_dirs(self):
        self.location_dir, self.file = os.path.split(self.path)

    def do_preprocess(self):
        self.pc_load = o3d.io.read_point_cloud(self.path)

        self.bound_pc_path = os.path.join(self.processed_data_dir, "pc_limits.ply")
        self.bound, self.bound_points, self.center, self.dim, self.density, self.n_points = process.compute_basic_properties(self.pc_load,
                                                                                               save_bound_pc=True,
                                                                                               output_path_bound_pc=self.bound_pc_path)

        print(f'The point cloud density is: {self.density:.3f}')

    def standard_images(self):
        self.res = round(self.density * 4, 3) * 1000
        process.raster_all_bound(self.path, self.res / 1000, self.bound_pc_path, xray=False)

        # create new images paths
        path_top = os.path.join(self.img_dir, 'top.tif')
        path_right = os.path.join(self.img_dir, 'right.tif')
        path_front = os.path.join(self.img_dir, 'front.tif')
        path_back = os.path.join(self.img_dir, 'back.tif')
        path_left = os.path.join(self.img_dir, 'left.tif')

        self.view_names.extend(['top', 'right', 'front', 'back', 'left'])
        self.view_paths.extend([path_top, path_right, path_front, path_back, path_left])

        # relocate image files
        img_list = process.generate_list('.tif', self.location_dir)
        os.rename(img_list[0], path_right)
        os.rename(img_list[1], path_back)
        os.rename(img_list[2], path_top)
        os.rename(img_list[3], path_left)
        os.rename(img_list[4], path_front)

        # rotate right view (CloudCompare output is tilted by 90°)
        # read the images
        im_front = Image.open(path_front)
        im_back = Image.open(path_back)
        im_left = Image.open(path_left)

        # rotate image by 90 degrees and mirror if needed
        angle = 90
        # process front image
        out_f = im_front.rotate(angle, expand=True)
        out_f_mir = ImageOps.mirror(out_f)
        out_f_mir.save(path_front)
        # process back image
        out_b = im_back.rotate(angle, expand=True)
        out_b.save(path_back)
        # process left image
        out_l_mir = ImageOps.mirror(im_left)
        out_l_mir.save(path_left)

    def do_orient(self):
        R = process.preproc_align_cloud(self.path, self.ransac_obj_folder, self.ransac_cloud_folder)
        print(f'The point cloud has been rotated with {R} matrix...')

        transformed_path = process.find_substring('TRANSFORMED', self.location_dir)
        _, trans_file = os.path.split(transformed_path)
        new_path = os.path.join(self.processed_data_dir, trans_file)
        # move transformed file
        os.rename(transformed_path, new_path)

        self.path = new_path
        self.update_dirs()
        self.pc_load = o3d.io.read_point_cloud(self.path)
        self.bound, self.bound_points, self.center, self.dim, _, _ = process.compute_basic_properties(self.pc_load,
                                                                                               save_bound_pc=True,
                                                                                               output_path_bound_pc=self.bound_pc_path)
        if self.sub_sampled:
            self.sub_pc_path

    def do_ransac(self, min_factor = MIN_RANSAC_FACTOR):
        self.sub_sampled = False
        # create RANSAC directories
        self.ransac_cloud_folder = os.path.join(self.processed_data_dir, 'RANSAC_pc')
        process.new_dir(self.ransac_cloud_folder)

        self.ransac_obj_folder = os.path.join(self.processed_data_dir, 'RANSAC_meshes')
        process.new_dir(self.ransac_obj_folder)

        # subsampling the point cloud if needed
        self.sub_pc_path = os.path.join(self.processed_data_dir,
                                        'subsampled.ply')  # path to the subsampled version of the point cloud

        if self.n_points > POINT_LIM:  # if to many points --> Subsample
            sub = self.pc_load.voxel_down_sample(VOXEL_DS)
            o3d.io.write_point_cloud(self.sub_pc_path, sub)
            self.sub_sampled = True
        else:
            shutil.copyfile(self.path, self.sub_pc_path)

        if self.sub_sampled:
            _, _, _, _, _, self.n_points_sub = process.compute_basic_properties(sub)
            print(f'The subsampled point cloud has {self.n_points_sub} points')

        # fixing RANSAC Parameters
        if not self.sub_sampled:
            points = self.n_points
        else:
            points = self.n_points_sub
        min_points = points / min_factor
        print(f'here are the minimum points {min_points}')

        self.n_planes = process.preproc_ransac_short(self.sub_pc_path, min_points, RANSAC_DIST, self.ransac_obj_folder,
                                                self.ransac_cloud_folder)

        self.ransac_done = True


class AboutDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('What is this app about?')
        self.setFixedSize(300,300)
        self.layout = QtWidgets.QVBoxLayout()

        about_text = QtWidgets.QLabel('This app was made by Buildwise, to analyze roofs and their deformation.')
        about_text.setWordWrap(True)

        logos1 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_buildwise2.png'))
        w = self.width()
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos1.setPixmap(pixmap)

        logos2 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_pointify.png'))
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos2.setPixmap(pixmap)

        self.layout.addWidget(about_text)
        self.layout.addWidget(logos1, alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(logos2, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)


class CroppinWindow(QtWidgets.QMainWindow):
    """
    Main Window class for the Nok-out application.
    """

    def __init__(self, parent=None):
        """
        Function to initialize the class
        :param parent:
        """
        super(CroppinWindow, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = 'croppin'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        print(uifile)
        wid.loadUi(uifile, self)

        self.image_array = []
        self.image_path = ''
        self.image_loaded = False
        self.current_view = 'top'
        self.Nokclouds = []
        self.nb_roi = 0

        self.pl_or, self.lim_points = 'all', LIM_POINTS

        self.legend_created = False
        self.legend_shown = False

        # add actions to action group
        ag = QtGui.QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.actionCrop)
        ag.addAction(self.actionHand_selector)

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)
        self.selmod = self.treeView.selectionModel()

        # initialize status
        self.update_progress(nb=100, text="Status: Choose point cloud!")

        # Add icons to buttons
        self.add_icon(res.find('img/load.png'), self.actionLoad)
        self.add_icon(res.find('img/magic.png'), self.actionDetectPlanes)
        self.add_icon(res.find('img/crop.png'), self.actionCrop)
        self.add_icon(res.find('img/hand.png'), self.actionHand_selector)
        self.add_icon(res.find('img/show.png'), self.actionShowData)
        self.add_icon(res.find('img/save.png'), self.actionSaveData)

        self.viewer = wid.PhotoViewer(self)
        self.horizontalLayout_2.addWidget(self.viewer)

        # create connections (signals)
        self.create_connections()

    def update_progress(self, nb=None, text=''):
        self.label_status.setText(text)
        if nb is not None:
            self.progressBar.setProperty("value", nb)

            # hide progress bar when 100%
            if nb >= 100:
                self.progressBar.setVisible(False)
            elif self.progressBar.isHidden():
                self.progressBar.setVisible(True)

    def reset_parameters(self):
        """
        Reset all model parameters (image and categories)
        """

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # clean graphicscene
        # self.viewer.clean_scene()

        # clean combobox
        # self.comboBox_cat.clear()

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QtGui.QIcon(img_source))

    def create_connections(self):
        """
        Link signals to slots
        """
        self.actionLoad.triggered.connect(self.get_pointcloud)
        self.actionDetectPlanes.triggered.connect(self.go_segment)
        self.actionCrop.triggered.connect(self.go_crop)
        self.actionInfo.triggered.connect(self.show_info)
        self.actionShowData.triggered.connect(self.toggle_legend)

        self.comboBox_viewpoint.currentIndexChanged.connect(self.on_img_combo_change)
        self.viewer.endDrawing_rect.connect(self.perform_crop)

        self.selmod.selectionChanged.connect(self.on_tree_change)

    def show_info(self):
        dialog = AboutDialog()
        if dialog.exec_():
            pass

    def toggle_legend(self):
        print('Legend toggled')
        if not self.legend_created:
            fig, ax = plt.subplots()
            data = np.clip(np.random.randn(10, 10) * 100, -SPAN, SPAN)

            colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 1, 0), (1, 0, 0)]
            cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors, N=256)
            cax = ax.imshow(data, cmap=cmap)

            # Add colorbar, make sure to specify tick locations to match desired ticklabels
            n_colors = 12
            ticks = np.linspace(-SPAN, SPAN, n_colors + 1, endpoint=True)
            cbar = fig.colorbar(cax, ticks=ticks, extend='both')
            ax.remove()

            folder = self.app_dir
            self.legend_path = os.path.join(folder, 'plot_legend.png')
            plt.savefig(self.legend_path, bbox_inches='tight')

            self.legend_created = True

        if not self.legend_shown:
            self.legend_label = QtWidgets.QLabel()
            self.legend_label.setStyleSheet("background-color: white")
            self.legend_label.setPixmap(QtGui.QPixmap(self.legend_path))
            self.horizontalLayout_2.addWidget(self.legend_label)
            self.legend_shown = True
        else:
            self.legend_label.setStyleSheet("")
            self.legend_label.clear()
            self.horizontalLayout_2.removeWidget(self.legend_label)
            self.legend_shown = False

    def go_crop(self):
        if self.actionCrop.isChecked():
            self.viewer.rect = True
            self.viewer.toggleDragMode()

    def perform_crop(self):
        # switch back to hand tool
        self.hand_pan()

        # get coordinates and crop cloud
        coords = self.viewer.crop_coords
        start_x = int(coords[0].x())*self.current_cloud.res / 1000
        start_y = int(coords[0].y())*self.current_cloud.res / 1000
        end_x = int(coords[1].x())*self.current_cloud.res / 1000
        end_y = int(coords[1].y())*self.current_cloud.res / 1000

        # crop the point cloud
        bound = self.current_cloud.pc_load.get_axis_aligned_bounding_box()
        center = bound.get_center()
        dim = bound.get_extent()

        if self.current_view == 'top':
            pt1 = [center[0] - dim[0]/2 + start_x, center[1] + dim[1]/2 - start_y, center[2] - dim[2] / 2]
            pt2 = [pt1[0] + (end_x-start_x), pt1[1] - (end_y-start_y), center[2] + dim[2] / 2]
            np_points = [pt1, pt2]
            points = o3d.utility.Vector3dVector(np_points)
            orientation = "top"

        elif self.current_view == 'front':
            pt1 = [center[0] - dim[0] / 2 + start_x, center[1] - dim[1] / 2, center[2] + dim[2] / 2 - start_y]
            pt2 = [pt1[0] + (end_x - start_x), center[1] + dim[1] / 2, pt1[2] - (end_y-start_y)]
            np_points = [pt1, pt2]
            points = o3d.utility.Vector3dVector(np_points)
            orientation = "front"

        elif self.current_view == 'right':
            pt1 = [center[0] - dim[0] / 2, center[1] - dim[1] / 2 + start_x, center[2] + dim[2] / 2 - start_y]
            pt2 = [center[0] + dim[0] / 2, pt1[1] + (end_x - start_x), pt1[2] - (end_y-start_y)]
            np_points = [pt1, pt2]
            points = o3d.utility.Vector3dVector(np_points)
            orientation = "front"

        crop_box = o3d.geometry.AxisAlignedBoundingBox
        crop_box = crop_box.create_from_points(points)
        point_cloud_crop = self.current_cloud.pc_load.crop(crop_box)
        self.nb_roi += 1
        roi_path = os.path.join(self.current_cloud.processed_data_dir, f"roi{self.nb_roi}.ply")
        o3d.io.write_point_cloud(roi_path, point_cloud_crop)

        # create new point cloud
        self.create_point_cloud_object(roi_path, f'roi{self.nb_roi}', orient=False, ransac=False)
        process.basic_vis_creation(point_cloud_crop, orientation)

        self.viewer.clean_scene()

    def hand_pan(self):
        # switch back to hand tool
        self.actionHand_selector.setChecked(True)
        self.viewer.rect = False
        self.viewer.toggleDragMode()

    def go_segment(self):
        """
        Launch the plane segmentation
        """
        # create new renders
        self.config_options()
        self.current_cloud.planarity_images(self.pl_or, self.lim_points)
        self.on_tree_change()

    def get_pointcloud(self):
        """
        Get the point cloud path from the user
        :return:
        """
        try:
            pc = QtWidgets.QFileDialog.getOpenFileName(self, u"Ouverture de fichiers","", "Point clouds (*.ply *.las)")
            print(f'the following point cloud will be loaded {pc[0]}')
        except:
            pass
        if pc[0] != '':
            # load and show new image
            self.load_main_pointcloud(pc[0])

    def load_main_pointcloud(self, path):
        """
        Load the new point cloud and reset the model
        :param path:
        :return:
        """
        original_dir, _ = os.path.split(path)

        # create specific folder for the app outputs
        self.app_dir = os.path.join(original_dir, 'NokOut')
        process.new_dir(self.app_dir)

        self.create_point_cloud_object(path, 'Original_point_cloud')

    def create_point_cloud_object(self, path, name, orient=True, ransac=True):
        cloud = NokPointCloud()
        self.Nokclouds.append(cloud)  # note: self.Nokclouds[0] is always the original point cloud

        self.Nokclouds[-1].path = path
        self.Nokclouds[-1].update_dirs()
        self.Nokclouds[-1].name = name

        self.Nokclouds[-1].folder = os.path.join(self.app_dir, self.Nokclouds[-1].name)
        self.Nokclouds[-1].processed_data_dir = os.path.join(self.Nokclouds[-1].folder, 'processes')
        self.Nokclouds[-1].img_dir = os.path.join(self.Nokclouds[-1].folder, 'images')

        process.new_dir(self.Nokclouds[-1].folder)
        process.new_dir(self.Nokclouds[-1].processed_data_dir)
        process.new_dir(self.Nokclouds[-1].img_dir)
        self.process_pointcloud(self.Nokclouds[-1], orient=orient, ransac=ransac)

        # add element to treeview
        self.current_cloud = self.Nokclouds[-1]
        self.add_item_in_tree(self.model, self.Nokclouds[-1].name) # signal a tree change

        # load image
        self.comboBox_viewpoint.setEnabled(True)
        self.image_loaded = True

        self.comboBox_viewpoint.clear()
        self.comboBox_viewpoint.addItems(self.current_cloud.view_names)
        self.on_img_combo_change()

        nb_pc = len(self.Nokclouds)
        build_idx = self.model.index(nb_pc-1,0)
        self.selmod.setCurrentIndex(build_idx, QtCore.QItemSelectionModel.Select)
        self.treeView.expandAll()

        # enable action(s)
        self.actionCrop.setEnabled(True)
        self.actionDetectPlanes.setEnabled(True)
        self.actionHand_selector.setEnabled(True)

    def on_tree_change(self):
        print('CHANGED!')
        indexes = self.treeView.selectedIndexes()
        sel_item = self.model.itemFromIndex(indexes[0])
        print(indexes[0])

        for cloud in self.Nokclouds:
            if cloud.name == sel_item.text():
                self.current_cloud = cloud
                print('Current cloud name: ', self.current_cloud.name)

                self.comboBox_viewpoint.clear()
                self.comboBox_viewpoint.addItems(self.current_cloud.view_names)


    def process_pointcloud(self, pc, orient = True, ransac = True):
        # 1. BASIC DATA ____________________________________________________________________________________________________
        # read full high definition point cloud (using open3d)
        print('Reading the point cloud!')
        pc.do_preprocess()

        # 2. RANSAC DETECTION __________________________________________________________________________________
        if ransac:
            print('Launching RANSAC detection...')
            pc.do_ransac()

        # 3. ORIENT THE POINT CLOUD PERPENDICULAR TO AXIS_______________________________________________________
        if orient:
            print('Orienting the point cloud perpendicular to the axes...')
            pc.do_orient()

        # 4. GENERATE BASIC VIEWS_______________________________________________________
        print('Launching RGB render/exterior views creation...')
        pc.standard_images()

    def on_img_combo_change(self):
        self.actionCrop.setEnabled(True)
        i = self.comboBox_viewpoint.currentIndex()
        if i < 0:
            i = 0
        self.current_view = self.current_cloud.view_names[i]
        print(i)
        print(self.current_view)
        if self.current_view == 'back' or self.current_view == 'left':
            self.actionCrop.setEnabled(False)

        img_paths = self.current_cloud.view_paths
        if self.image_loaded:
            self.viewer.setPhoto(QtGui.QPixmap(img_paths[i]))

        if 'planarity' in self.current_view:
            self.actionShowData.setEnabled(True)
            self.actionCrop.setEnabled(False)
            if not self.legend_shown:
                print('bla')
                self.toggle_legend()
        else:
            self.actionShowData.setEnabled(False)
            if self.legend_shown:
                print(self.legend_shown)
                print('go')
                self.toggle_legend()

    def config_options(self):
        dialog = DiaOptions()
        dialog.setWindowTitle("Choose parameters for planarity processing")

        if dialog.exec_():
            i = dialog.comboBox.currentIndex()
            self.lim_points = float(dialog.lineEdit.text())
            self.pl_or = dialog.hidden_list[i]

    def add_item_in_tree(self, parent, line):
        item = QtGui.QStandardItem(line)
        parent.appendRow(item)

def main(argv=None):
    """
    Creates the main window for the application and begins the \
    QApplication if necessary.

    :param      argv | [, ..] || None

    :return      error code
    """

    # Define installation path
    install_folder = os.path.dirname(__file__)

    app = None
    extra = {

        # Density Scale
        'density_scale': '-2',
    }

    # create the application if necessary
    if (not QtWidgets.QApplication.instance()):
        app = QtWidgets.QApplication(argv)
        #app.setStyle('Fusion')
        apply_stylesheet(app, theme='light_teal.xml',extra=extra)

    # create the main window

    window = CroppinWindow()
    window.show()

    # run the application if necessary
    if (app):
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))