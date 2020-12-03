"""
Process the bag file to extract its content
to given files into more readable format
COMP551 TEAM
"""

import rosbag

def main():

    bag_dir  = '_gitignore/bag_files/'
    bag_name = '001.bag'
    bag_full = bag_dir + bag_name

    print('Processing: ', bag_full)
    bag    = rosbag.Bag(bag_full)
    topics = bag.get_type_and_topic_info()[1].keys()
    types  = []
    for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
        types.append(bag.get_type_and_topic_info()[1].values()[i][0])
    print(topics)
    print(types)


main()
