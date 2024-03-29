import argparse
import pdb

import sensor_msgs.point_cloud2 as pc2
import rospy
import pickle
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def datatype_to_size_type(datatype):
    """
    Takes a UINT8 datatype field from a PointFields and returns the size in
    bytes and a char for its type ('I': int, 'U': unsigned, 'F': float)
    """
    # might be nicer to look it up in message definition but quicker just to
    # do this.
    if datatype in [2, 3, 4]:
        t = "U"
    elif datatype in [1, 3, 5]:
        t = "I"
    elif datatype in [7, 8]:
        t = "F"
    else:
        raise Exception("Unknown datatype in PointField")

    if datatype < 3:
        s = 1
    elif datatype < 5:
        s = 2
    elif datatype < 8:
        s = 4
    elif datatype < 9:
        s = 8
    else:
        raise Exception("Unknown datatype in PointField")

    return s, t


def size_type_to_datatype(size, type):
    """
    Given a .pcd size/type pair, return a sensor_msgs/PointField datatype
    """
    if type == "F":
        if size == 4:
            return 7
        if size == 8:
            return 8
    if type == "I":
        if size == 1:
            return 1
        if size == 2:
            return 3
        if size == 4:
            return 5
    if type == "U":
        if size == 1:
            return 2
        if size == 2:
            return 4
        if size == 4:
            return 6
    raise Exception("Unknown size/type pair in .pcd")


def write_pcd(filename, pointcloud, overwrite, viewpoint=None, mode="binary_stripped"):
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
    if mode not in ["ascii", "binary", "binary_stripped"]:
        raise Exception("Mode must be 'binary' or 'ascii'")
    if not overwrite and os.path.isfile(filename):
        raise Exception("File exists.")
    try:
        with open(filename, "w") as f:
            f.write("VERSION .7\n")
            _size = {}
            _type = {}
            _count = {}
            _offsets = {}
            _fields = []
            _size["_"] = 1
            _count["_"] = 1
            _type["_"] = "U"
            offset = 0
            for field in pointcloud.fields:
                if field.offset != offset:
                    # some padding
                    _fields.extend(["_"] * (field.offset - offset))
                isinstance(field, PointField)
                _size[field.name], _type[field.name] = datatype_to_size_type(
                    field.datatype
                )
                _count[field.name] = field.count
                _offsets[field.name] = field.offset
                _fields.append(field.name)
                offset = field.offset + _size[field.name] * _count[field.name]
            if pointcloud.point_step != offset:
                _fields.extend(["_"] * (pointcloud.point_step - offset))

            if mode != "binary":
                # remove padding fields
                while True:
                    try:
                        _fields.remove("_")
                    except:
                        break

            # _fields = _count.keys()
            _fields_str = reduce(
                lambda a, b: a + " " + b, map(lambda x: "{%s}" % x, _fields)
            )

            f.write("FIELDS ")
            f.write(reduce(lambda a, b: a + " " + b, _fields))
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
                    raise Exception(
                        "viewpoint argument must be  tuple " "(x y z qx qy qz qw)"
                    )
                f.write("VIEWPOINT {} {} {} {} {} {} {}\n".format(*viewpoint))

            f.write("POINTS %d\n" % (pointcloud.width * pointcloud.height))
            if mode == "binary":
                # TODO: check for row padding.
                f.write("DATA binary\n")
                f.write(bytearray(pointcloud.data))
            elif mode == "binary_stripped":
                f.write("DATA binary\n")
                if pointcloud.point_step == sum(
                    [v[0] * v[1] for v in zip(_size.values(), _count.values())]
                ):  # danger, assumes ordering
                    # ok to just blast it all out; TODO: this assumes row step has no padding
                    f.write(bytearray(pointcloud.data))
                else:
                    # strip all the data padding
                    _field_step = {}
                    for field in _fields:
                        _field_step[field] = _size[field] * _count[field]
                    out = bytearray(
                        sum(_field_step.values()) * pointcloud.width * pointcloud.height
                    )
                    b = 0
                    for v in range(pointcloud.height):
                        offset = pointcloud.row_step * v
                        for u in range(pointcloud.width):
                            for field in _fields:
                                out[b : b + _field_step[field]] = pointcloud.data[
                                    offset
                                    + _offsets[field] : offset
                                    + _offsets[field]
                                    + _field_step[field]
                                ]
                                b += _field_step[field]
                            offset += pointcloud.point_step
                    f.write(out)
            else:
                f.write("DATA ascii\n")
                for p in pc2.read_points(pointcloud, _fields):
                    for i, field in enumerate(_fields):
                        f.write("%f " % p[i])
                    f.write("\n")

    except (IOError, e):
        raise Exception("Can't write to %s: %s" % (filename, e.message))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--infile", default="tmp/ee.pickle", type=str)
    parser.add_argument("--outfolder", default="tmp", type=str)
    parser.add_argument("--startid", default=1, type=int)

    args = parser.parse_args()
    input_path = args.infile
    input_file = open(input_path, "rb")

    output_folder = args.outfolder + "/"

    i = args.startid - 1
    while True:
        try:
            i += 1
            point_cloud, ee_pose, robot2ee_pose = pickle.load(input_file)
            # if i %100 != 0:
            #    continue
            pcd_full = output_folder + str(i) + ".pcd"
            ee_full = output_folder + str(i)
            robot2ee_full = output_folder + str(i) + "_robot2ee_pose"
            # pdb.set_trace()
            print("Output: ", pcd_full)
            pcl_obj = write_pcd(pcd_full, point_cloud, True)
            ee_pose = np.array(
                [
                    ee_pose.pose.position.x,
                    ee_pose.pose.position.y,
                    ee_pose.pose.position.z,
                    ee_pose.pose.orientation.x,
                    ee_pose.pose.orientation.y,
                    ee_pose.pose.orientation.z,
                    ee_pose.pose.orientation.w,
                ],
                dtype=np.float32,
            )
            robot2ee_pose = np.array(
                [
                    robot2ee_pose.pose.position.x,
                    robot2ee_pose.pose.position.y,
                    robot2ee_pose.pose.position.z,
                    robot2ee_pose.pose.orientation.x,
                    robot2ee_pose.pose.orientation.y,
                    robot2ee_pose.pose.orientation.z,
                    robot2ee_pose.pose.orientation.w,
                ],
                dtype=np.float32,
            )
            np.save(ee_full, ee_pose)
            np.save(robot2ee_full, robot2ee_pose)
            del point_cloud
        except EOFError:
            print("Done...")
            break
