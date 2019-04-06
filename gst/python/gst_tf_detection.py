"""
Usage
    export GST_PLUGIN_PATH=$PWD

    # Log detections
    GST_DEBUG=python:4 gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! gst_tf_detection config=cfg.yml ! videoconvert ! gtksink sync=False

    # Draw detections
    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! gst_tf_detection config=cfg.yml ! videoconvert ! gst_detection_overlay ! videoconvert ! gtksink sync=False

"""

import os
import logging
import timeit
import traceback
import time
import cv2
import tensorflow as tf
from typing import List, Tuple
import yaml
import numpy as np

from pygst_utils import gst_buffer_with_pad_to_ndarray, Gst, GObject
from pygst_utils.gst_objects_info_meta import gst_meta_write


def is_gpu(device: str) -> bool:
    return "gpu" in device.lower()


def create_config(device: str = '/device:CPU:0',
                  per_process_gpu_memory_fraction: float = 0.0,
                  log_device_placement: bool = False) -> tf.ConfigProto:

    if is_gpu(device):
        config = tf.ConfigProto(log_device_placement=log_device_placement)
        if per_process_gpu_memory_fraction > 0.0:
            config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        else:
            config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(
            log_device_placement=log_device_placement, device_count={'GPU': 0})

    return config


def parse_graph_def(model_path: str) -> tf.GraphDef:
    model_path = os.path.abspath(model_path)
    assert os.path.isfile(model_path), "Invalid filename {}".format(model_path)
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def import_graph(graph_def: tf.GraphDef, device: str, name: str="") -> tf.Graph:
    with tf.device(device):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name=name)
            return graph


def load_labels_from_file(filename: str) -> dict:
    assert os.path.isfile(filename), "Invalid filename {}".format(filename)
    labels = {}
    with open(filename, 'r') as f:
        for line in f:
            try:
                items = line.strip().split(":")
                # print(items)
                label_id, label_name = items[:2]
                labels[int(label_id)] = label_name[1:]
            except Exception as e:
                print(e, items)
    return labels


def load_config(filename: str) -> dict:
    filename = os.path.abspath(filename)
    assert os.path.isfile(filename), "Invalid filename {}".format(filename)

    with open(filename, 'r') as stream:
        try:
            data = yaml.load(stream, Loader=yaml.Loader)
            return data
        except yaml.YAMLError as exc:
            raise OSError('Parsing error. Filename: {}'.format(filename))


class TfObjectDetectionModel(object):

    def __init__(self, weights: str,
                 threshold: float = 0.5,
                 device: str = '/device:CPU:0',
                 per_process_gpu_memory_fraction: float = 0.0,
                 log_device_placement=False,
                 labels: List[str] = None,
                 input_shape: Tuple[int, int]=(300, 300)):

        # TODO Docs
        graph_def = parse_graph_def(weights)
        config = create_config(device,
                               log_device_placement=log_device_placement,
                               per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        graph = import_graph(graph_def, device)

        print(f"Model {weights} placed on {device}")

        self.session = tf.Session(graph=graph, config=config)

        # Taken from official website
        # https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
        self.input = graph.get_tensor_by_name("image_tensor:0")
        self.input_shape = input_shape or (300, 300)
        self.input_shape = tuple(self.input_shape)

        # print([n.name for n in graph.as_graph_def().node][:10])
        # print("Shape : ", self.input.shape)

        # Taken from official website
        # https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
        output_names = ["detection_classes:0",
                        "detection_boxes:0",
                        "detection_scores:0"]
        self.output = [graph.get_tensor_by_name(name) for name in output_names]

        self.threshold = threshold
        self.labels = labels or {}

        self._box_scaler = None

    def process_single(self, image: np.ndarray) -> List[dict]:
        return self._process(np.expand_dims(self._preprocess(image), 0), image.shape[:2][::-1])[0]

    def process_batch(self, images: List[np.ndarray]) -> List[dict]:
        images_ = np.stack([self._preprocess(image) for image in images])
        return self._process(images_, images[0].shape[:2][::-1])

    def _process(self, images: np.ndarray, initial_shape: Tuple[int, int]) -> List[dict]:
        classes, boxes, scores = self.session.run(self.output,
                                                  feed_dict={self.input: images})

        # _, h, w = images.shape[:3]
        w, h = initial_shape
        box_scaler = np.array([h, w, h, w])

        num_detections = len(classes)
        objects = [[] for _ in range(num_detections)]
        for i in range(num_detections):
            for class_id, box, score in zip(classes[i], boxes[i], scores[i]):
                if class_id not in self.labels or \
                        score < self.threshold:
                    continue

                ymin, xmin, ymax, xmax = list(map(lambda x: int(x), box * box_scaler))
                object_info = {'confidence': float(score),
                               'bounding_box': [xmin, ymin, xmax - xmin, ymax - ymin],
                               'class_name': self.labels[class_id]}

                objects[i].append(object_info)
        return objects

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.input_shape, interpolation=cv2.INTER_NEAREST)

    def __del__(self):
        """ Releases model when object deleted """
        if self.session is not None:
            self.session.close()


def tf_object_detection_model_from_file(filename: str) -> TfObjectDetectionModel:
    """
    :param filename: filename to model config
    """
    return tf_object_detection_model_from_config(load_config(filename))


def tf_object_detection_model_from_config(config: dict) -> TfObjectDetectionModel:
    """
    :param config: model config
    """
    labels = load_labels_from_file(config['labels'])

    return TfObjectDetectionModel(weights=config['weights'],
                                  threshold=config.get('threshold', 0.5),
                                  device=config.get('device', "/device:CPU:0"),
                                  per_process_gpu_memory_fraction=config.get('per_process_gpu_memory_fraction', 0.0),
                                  labels=labels,
                                  log_device_placement=config.get("log_device_placemenent", False),
                                  input_shape=config.get('input_shape', (300, 300)))


class GstTfDetectionPluginPy(Gst.Element):

    # Metadata Explanation:
    # http://lifestyletransfer.com/how-to-create-simple-blurfilter-with-gstreamer-in-python-using-opencv/

    GST_PLUGIN_NAME = 'gst_tf_detection'

    __gstmetadata__ = ("Name",
                       "Transform",
                       "Description",
                       "Author")

    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       Gst.Caps.from_string("video/x-raw,format={RGB}"))

    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        Gst.Caps.from_string("video/x-raw,format={RGB}"))

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "model": (GObject.TYPE_PYOBJECT,
                  "model",
                  "Contains model that implements process(any_data)",
                  GObject.ParamFlags.READWRITE),

        "config": (str,
                   "str property",
                   "A property that contains str",
                   None,  # default
                   GObject.ParamFlags.READWRITE
                   ),
    }

    def __init__(self):
        super(GstTfDetectionPluginPy, self).__init__()

        # Explained:
        # http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/

        # Explanation how to init Pads
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/pads.html
        self.sinkpad = Gst.Pad.new_from_template(self._sinktemplate, 'sink')

        # Set chain function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/chainfn.html
        self.sinkpad.set_chain_function_full(self.chainfunc, None)

        # Set event function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/eventfn.html
        self.sinkpad.set_event_function_full(self.eventfunc, None)
        self.add_pad(self.sinkpad)

        self.srcpad = Gst.Pad.new_from_template(self._srctemplate, 'src')

        # Set event function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/eventfn.html
        self.srcpad.set_event_function_full(self.srceventfunc, None)

        # Set query function
        # https://gstreamer.freedesktop.org/documentation/plugin-development/basics/queryfn.html
        self.srcpad.set_query_function_full(self.srcqueryfunc, None)
        self.add_pad(self.srcpad)

        self.model = None
        self.config = None
        self._channels = 3  # RGB -> 3 channels

    def chainfunc(self, pad: Gst.Pad, parent, buffer: Gst.Buffer) -> Gst.FlowReturn:

        if self.model is None:
            return self.srcpad.push(buffer)

        try:
            # Convert Gst.Buffer to np.ndarray
            image = gst_buffer_with_pad_to_ndarray(buffer, pad, self._channels)

            # model inference
            objects = self.model.process_single(image)
            Gst.info(str(objects))

            # write objects to as Gst.Buffer's metadata
            gst_meta_write(buffer, objects)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            return Gst.FlowReturn.ERROR

        return self.srcpad.push(buffer)

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'model':
            return self.model
        if prop.name == 'config':
            return self.config
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'model':
            self.model = value
        elif prop.name == "config":
            self.model = tf_object_detection_model_from_file(value)
            self.config = value
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def eventfunc(self, pad: Gst.Pad, parent, event: Gst.Event) -> bool:
        """ Forwards event to SRC (DOWNSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadEventFunction

        :param parent: GstTfDetectionPluginPy
        """
        return self.srcpad.push_event(event)

    def srcqueryfunc(self, pad: Gst.Pad, parent, query: Gst.Query) -> bool:
        """ Forwards query bacj to SINK (UPSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadQueryFunction

        :param parent: GstTfDetectionPluginPy
        """
        return self.sinkpad.query(query)

    def srceventfunc(self, pad: Gst.Pad, parent, event: Gst.Event) -> bool:
        """ Forwards event back to SINK (UPSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadEventFunction

        :param parent: GstTfDetectionPluginPy
        """
        return self.sinkpad.push_event(event)


# Required for registering plugin dynamically
# Explained:
# http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstTfDetectionPluginPy)
__gstelementfactory__ = (GstTfDetectionPluginPy.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstTfDetectionPluginPy)
