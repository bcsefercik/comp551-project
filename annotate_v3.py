import open3d as o3d
import numpy as np
import pickle as pkl
import time
from matplotlib import pyplot as plt

class Annotator:
    '''
        All the operations of annotaion is going to be integrated
        to this class (e.g. background aggregation, background subtraction, reconstruction, ...)
    '''
    def __init__(self, ):
        self.metrics     = None
        self.bg_samples  = None

    def set_bg(self, samples):
        '''
            set bg samples of the annotator
                
            Note: some preprocessing steps on bg samples can be added here (e.g. adding aggregating frames)
        '''
        self.bg_samples = samples

    def o3d_classify(self, bg_cloud, target_cloud, metric='distance_changed', removal_th=0.02, clip_depth=True, max_depth=1):
        '''
            Classifies 3D points into arm and background points
        '''
        dists = target_cloud.compute_point_cloud_distance(bg_cloud)
        dists = np.asarray(dists)
        mask  = dists > removal_th

        if clip_depth:
            points = np.asarray(target_cloud.points)
            mask   = np.logical_and(points[:,2] < max_depth ,  mask)
        
        ind      = np.where(mask == True)[0]

        return ind



def surface_reconstruct(pcd, visualize=False):

    print("Compute the normal of the point cloud")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))
    #pcd.orient_normals_consistent_tangent_plane(100)

    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    
    if visualize:
        print('visualizing mesh.')
        o3d.visualization.draw_geometries([mesh])

    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)


    if visualize:
        print('visualizing estimated densities.')
        o3d.visualization.draw_geometries([density_mesh])

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.08)
    mesh.remove_vertices_by_mask(vertices_to_remove)


    if visualize:
        print('visualizing mesh after removing low dencity verticies.')
        o3d.visualization.draw_geometries([mesh])

    mesh.compute_vertex_normals()
    
    pcd_new = mesh.sample_points_uniformly(number_of_points=200000)

    if visualize:
        print('visualizing the sampled point cloud.')
        o3d.visualization.draw_geometries([mesh])

    return pcd_new


visualize       = False
remove_outliers = False
reconstruct     = True

pcd_path  = './_gitignore/pcd_files/bg_data/bg_1.pcd'
pcd_obj_p = o3d.io.read_point_cloud(pcd_path)

pcd_path  = './_gitignore/pcd_files/data_samples/data_30.pcd'
pcd_obj_c = o3d.io.read_point_cloud(pcd_path)

ann       = Annotator()

start = time.time()
arm_ind  = ann.o3d_classify(pcd_obj_p ,pcd_obj_c)
end = time.time()
print("elapsed time: ", end - start)

arm_cloud = pcd_obj_c.select_by_index(arm_ind)
bg_cloud  = pcd_obj_c.select_by_index(arm_ind, invert=True)


if remove_outliers:
    del pcd_obj_c,pcd_obj_p
    cl, ind        = arm_cloud.remove_statistical_outlier(nb_neighbors=40,std_ratio=2.0)
    arm_cloud_down = arm_cloud.select_by_index(ind)

    cl, ind        = bg_cloud.remove_statistical_outlier(nb_neighbors=40,std_ratio=2.0)
    bg_cloud_donw  = arm_cloud.select_by_index(ind)
    

if reconstruct:
    print("Reconstructing the arm points.")
    new_pcd = surface_reconstruct(arm_cloud)
    

if visualize:
    print("visualizing the arm points.")
    o3d.visualization.draw_geometries([arm_cloud])

    print("visualizing the background points.")
    o3d.visualization.draw_geometries([bg_cloud])

o3d.io.write_point_cloud("arm_cloud.pcd", arm_cloud, write_ascii=False, compressed=False, print_progress=True)
o3d.io.write_point_cloud("bg_cloud.pcd", bg_cloud, write_ascii=False, compressed=False, print_progress=True)

if reconstruct:
    print("Writing reconstructed arm points.")
    o3d.io.write_point_cloud("rec_arm_cloud.pcd", new_pcd, write_ascii=False, compressed=False, print_progress=True)

