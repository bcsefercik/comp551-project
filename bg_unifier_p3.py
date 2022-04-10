

"""
Given a folder with multiple pcd files it loads them and combine them to generate
one pcd file
"""
import sys
from os import listdir
from os.path import isfile, join
import open3d as o3d
import numpy as np
# from pcd_bg_extractor import write_pcd

def write_pcd(filename,  pointcloud, overwrite, viewpoint=None,
              mode='binary_stripped'):
    """
    Writes a sensor_msgs::PointCloud2 to a .pcd file.
    :param filename - the pcd file to write
    :param pointcloud - sensor_msgs::PointCloud2 to write to a file
    :param overwrite - if True, allow overwriting existing files
    :param viewpoint - the camera viewpoint, (x,y,z,qw,qx,qy,qz)
    :param mode - the writing mode: 'ascii' for human readable, 'binary' for
                  a straight dump of the binary data, 'binary_stripped'
                  to strip out data padding before writing (saves space but it slow)
    """
    assert isinstance(pointcloud, PointCloud2)
    if mode not in ['ascii', 'binary', 'binary_stripped']:
        raise Exception("Mode must be 'binary' or 'ascii'")
    if not overwrite and os.path.isfile(filename):
        raise Exception("File exists.")
    try:
        with open(filename, "w") as f:
            f.write("VERSION .7\n")
            _size =  {}
            _type = {}
            _count = {}
            _offsets = {}
            _fields = []
            _size['_'] = 1
            _count['_'] = 1
            _type['_'] = 'U'
            offset = 0
            for field in pointcloud.fields:
                if field.offset != offset:
                    # some padding
                    _fields.extend(['_']*(field.offset - offset))
                isinstance(field, PointField)
                _size[field.name], _type[field.name] = datatype_to_size_type(field.datatype)
                _count[field.name] = field.count
                _offsets[field.name] = field.offset
                _fields.append(field.name)
                offset = field.offset + _size[field.name] * _count[field.name]
            if pointcloud.point_step != offset:
                _fields.extend(['_']*(pointcloud.point_step - offset))


            if mode != 'binary':
                #remove padding fields
                while True:
                    try:
                        _fields.remove('_')
                    except:
                        break

            #_fields = _count.keys()
            _fields_str =  reduce(lambda a, b: a +  ' ' + b,
                                  map(lambda x: "{%s}" % x,
                                      _fields))

            f.write("FIELDS ")
            f.write(reduce(lambda a, b: a + ' ' + b,
                           _fields))
            f.write("\n")
            f.write("SIZE ")
            f.write(_fields_str.format(**_size))
            f.write("\n")
            f.write("TYPE ")
            f.write(_fields_str.format(**_type))
            f.write("\n")
            f.write("COUNT ")
            f.write(_fields_str.format(**_count))

            f.write("\n")
            f.write("WIDTH %s" % pointcloud.width)
            f.write("\n")
            f.write("HEIGHT %s" % pointcloud.height)
            f.write("\n")

            if viewpoint is None:
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            else:
                try:
                    assert len(viewpoint) == 7
                except:
                    raise Exception("viewpoint argument must be  tuple "
                                    "(x y z qx qy qz qw)")
                f.write("VIEWPOINT {} {} {} {} {} {} {}\n".format(*viewpoint))

            f.write("POINTS %d\n" % (pointcloud.width * pointcloud.height))
            if mode == "binary":
                #TODO: check for row padding.
                f.write("DATA binary\n")
                f.write(bytearray(pointcloud.data))
            elif mode ==  "binary_stripped":
                f.write("DATA binary\n")
                if pointcloud.point_step == sum([v[0]*v[1] for v in zip(_size.values(),
                                                                        _count.values())]): #danger, assumes ordering
                    # ok to just blast it all out; TODO: this assumes row step has no padding
                    f.write(bytearray(pointcloud.data))
                else:
                    # strip all the data padding
                    _field_step = {}
                    for field in _fields:
                        _field_step[field] = _size[field] * _count[field]
                    out =  bytearray(sum(_field_step.values())*pointcloud.width*pointcloud.height)
                    b = 0
                    for v in range(pointcloud.height):
                        offset = pointcloud.row_step * v
                        for u in range(pointcloud.width):
                            for field in _fields:
                                out[b:b+_field_step[field]] = pointcloud.data[offset+_offsets[field]:
                                                                              offset+_offsets[field]+_field_step[field]]
                                b += _field_step[field]
                            offset += pointcloud.point_step
                    f.write(out)
            else:
                f.write("DATA ascii\n")
                for p in pc2.read_points(pointcloud,  _fields):
                    for i, field in enumerate(_fields):
                        f.write("%f " % p[i])
                    f.write("\n")

    except (IOError, e):
        raise Exception("Can't write to %s: %s" %  (filename, e.message))


common_path = 'tmp'
if len(sys.argv) > 1:
    common_path = sys.argv[1]

path = common_path + '/bg'
file_names = [f for f in listdir(path) if isfile(join(path, f)) and f[-4:] == ".pcd"]
# file_names = ['BG_1.pcd', 'BG_5.pcd','BG_9.pcd']
n_initial  = 0
n_next     = 0



def load_pcd(file_name):
    return o3d.io.read_point_cloud(file_name)

def count_points(pcd_obj):
    return np.asarray(pcd_obj.points).shape[0]

def update_frame(curr_background, next_background, removal_th=0.01):

    dists                   = np.asarray(next_background.compute_point_cloud_distance(curr_background))
    point_mask              = dists > removal_th
    point_idx               = np.where(point_mask == True)[0]
    next_background_refined = next_background.select_by_index(point_idx)
    pcd_combined = curr_background + next_background_refined
    del curr_background
    del next_background
    del next_background_refined
    return pcd_combined


def find_added_percentage(n_initial,n_final):
    n_added    = n_final - n_initial
    percentage = (n_added * 100) / (n_initial)
    return percentage

background  = load_pcd( path + '/' + file_names[0])
n_next      = count_points(background)
percentages = [100]
for i in range(1,len(file_names)):
    n_initial  = n_next
    next_frame = load_pcd(path + '/' + file_names[i])
    background = update_frame(background, next_frame)
    n_next     = count_points(background)
    print("Total Number of Points: ", n_next)
    percentages.append(find_added_percentage(n_initial,n_next))
    print("Currently: ", i, '/', str(len(file_names)-1) )
print(percentages)
apply_downsampling = False

if apply_downsampling == True:
    print("Starting downsampling")
    voxel_size = 0.02
    print("Voxel size: ", voxel_size)
    background = background.voxel_down_sample(voxel_size=voxel_size)
    print("Final Number of Points: ",count_points(background))



output_file = common_path + '/combined_bg.pcd'
o3d.io.write_point_cloud(output_file, background, write_ascii=False, compressed=False, print_progress=True)
