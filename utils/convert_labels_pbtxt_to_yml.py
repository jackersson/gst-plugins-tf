"""
Converts StringIntLabelMap proto text file to IntString yml format

Label Maps Format
    class_id (int): class_name (str)

Example
        1: person
        2: bicycle
        3: car

Usage:
    python3 convert_labels_pbtxt_to_yml.py -f mscoco_label_map.pbtxt
"""

import os
import tensorflow as tf
import argparse
import yaml

from google.protobuf import text_format
from .protos import string_int_label_map_pb2


def get_filename(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def load_labels_pbtxt(path: str) -> dict:
    """Loads label map proto.
    :param path: path to StringIntLabelMap proto text file.
    :rtype: StringIntLabelMapProto
    """
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)

    result = {}
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if (item.id == 0 and item.name != 'background' and
                item.display_name != 'background'):
            raise ValueError(
                'Label map id 0 is reserved for the background label')
        result[item.id] = item.display_name
    return result


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, default='.', help="Path to *.pbtxt")
args = vars(ap.parse_args())

pbtxt_filename = os.path.abspath(args["filename"])
assert os.path.isfile(pbtxt_filename), f"Invalid filename: {pbtxt_filename}"

out_filename = os.path.abspath(f"{get_filename(pbtxt_filename)}.yml")

labels = load_labels_pbtxt(pbtxt_filename)
print(f"Loaded {len(labels)} labels")
with open(out_filename, 'w') as f:
    yaml.dump(labels, f, default_flow_style=False)

print(f"Saved to {out_filename}")
