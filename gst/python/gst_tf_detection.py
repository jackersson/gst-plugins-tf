"""
Usage
    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
    export GST_DEBUG=python:4

    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
        gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! autovideosink
"""

import os
import logging
import cv2
import trafaret as t
import tensorflow as tf
import typing as typ
import yaml
import numpy as np

from gstreamer import Gst, GObject, GstBase, GstVideo
import gstreamer.utils as utils
from gstreamer.gst_objects_info_meta import gst_meta_write


def _get_log_level() -> int:
    return int(os.getenv("GST_PYTHON_LOG_LEVEL", logging.DEBUG / 10)) * 10


log = logging.getLogger('gst_python')
log.setLevel(_get_log_level())


def _is_gpu_available() -> bool:
    """Check is GPU available or not"""
    try:
        from tensorflow.python.client import device_lib
        return any(d.device_type == 'GPU' for d in device_lib.list_local_devices())
    except ImportError:
        return os.path.isfile('/usr/local/cuda/version.txt')


def _parse_device(device: str) -> str:
    """Parse device on which run model
    Device value examples:
        - GPU|CPU
        - CPU
        - CPU:0
        - GPU
        - GPU:0
        - GPU:1
    For device name "GPU|CPU": use GPU if available else use CPU
    """

    result = device
    if device == 'GPU|CPU':
        result = 'GPU' if _is_gpu_available() else 'CPU'

    if 'GPU' in device and not _is_gpu_available():
        raise ValueError('Specified "{}" device but GPU not available'.format(device))

    return result if ':' in result else f'{result}:0'


def is_gpu(device: str) -> bool:
    return "gpu" in device.lower()


def create_config(device: str = 'CPU', *,
                  per_process_gpu_memory_fraction: float = 0.0,
                  log_device_placement: bool = False) -> tf.ConfigProto:
    """Creates tf.ConfigProto for specifi device"""
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    if is_gpu(device):
        if per_process_gpu_memory_fraction > 0.0:
            config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        else:
            config.gpu_options.allow_growth = True
    else:
        config.device_count = {'GPU': 0}

    return config


def parse_graph_def(model_path: str) -> tf.GraphDef:
    """Parse graph from file"""
    if not os.path.isfile(model_path):
        raise ValueError(f"Invalid filename {model_path}")
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def import_graph(graph_def: tf.GraphDef, device: str, name: str = "") -> tf.Graph:
    """Imports graph and places on specified device"""
    with tf.device(f"/device:{device}"):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name=name)
            return graph


def load_labels_from_file(filename: str) -> dict:
    """Parses labels from file

    File example:
        1: person
        2: tv monitor
        ...
    """
    if not os.path.isfile(filename):
        raise ValueError(f"Invalid filename {filename}")
    labels = {}
    with open(filename, 'r') as f:
        for line in f:
            try:
                label_id, label_name = line.split(":")[:2]
                labels[int(label_id)] = label_name.strip()
            except Exception as e:
                log.error(f"Error loading {items}: {e}")
    return labels


def load_config(filename: str) -> dict:
    if not os.path.isfile(filename):
        raise ValueError(f"Invalid filename {filename}")

    with open(filename, 'r') as stream:
        try:
            data = yaml.load(stream, Loader=yaml.Loader)
            return data
        except yaml.YAMLError as exc:
            raise OSError(f'Parsing error. Filename: {filename}')


class TfObjectDetectionModel(object):
    """Implementation for TF Object Detection API Inference"""

    # model's input tensor name from official website
    # https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    input_tensors = {
        'images': "image_tensor:0"
    }

    # model's output tensors names from official website
    # https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    output_tensors = {
        "labels": "detection_classes:0",
        "boxes": "detection_boxes:0",
        "scores": "detection_scores:0"
    }

    # default configuration
    _config_schema = t.Dict({
        # path or name of the frozen weights file (*.pb)
        t.Key('weights', default="data/models/*.pb"): t.String(min_length=4),
        t.Key('width', default=300): t.Int(gt=0),  # input tensor width
        t.Key('height', default=300): t.Int(gt=0),  # output tensor width
        t.Key('threshold', default=0.5): t.Float(gte=0.0, lte=1.0),  # confidence threshold for detected objects
        # labels dict or file
        t.Key('labels', default={1: 'person'}): t.Or(t.Dict({}, allow_extra='*'), t.String(min_length=4)),
        # device to execute graph
        t.Key('device', default='GPU|CPU'): t.Regexp(r'GPU\|CPU|CPU(?:\:0)?|GPU(?:\:\d)?') >> _parse_device,
        t.Key('log_device_placement', default=False): t.Bool,  # TF specific
        t.Key('per_process_gpu_memory_fraction', default=0.0): t.Float(gte=0.0, lte=1.0),  # TF specific
    }, allow_extra='*')

    def __init__(self, **kwargs):

        # validate config
        try:
            self.config = self._config_schema.check(kwargs or {})
        except t.DataError as err:
            raise ValueError(
                'Wrong model configuration for {}: {}'.format(self, err))

        self._session = None  # tf.Session
        self._inputs = None   # typ.Dict[str, tf.Tensor]
        self._outputs = None  # typ.Dict[str, tf.Tensor]
        self._labels = None   # typ.Dict[int, str]

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return '<{}>'.format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def log(self) -> logging.Logger:
        return log

    def startup(self):

        self.log.info("Starting %s ...", self)

        self._labels = self.config['labels'] if isinstance(
            self.config['labels'], dict) else load_labels_from_file(self.config['labels'])

        if len(self._labels) <= 0:
            raise ValueError(f"Labels can't be empty {self._labels}")

        tf_config = create_config(self.config['device'],
                                  log_device_placement=self.config['log_device_placement'],
                                  per_process_gpu_memory_fraction=self.config['per_process_gpu_memory_fraction'])

        graph = import_graph(parse_graph_def(self.config['weights']), self.config['device'])

        self.log.debug("Model (%s) placed on %s", self.config['weights'], self.config['device'])

        self._session = tf.Session(graph=graph, config=tf_config)

        self._inputs = {alias: graph.get_tensor_by_name(name) for alias, name in self.input_tensors.items()}
        self._outputs = {alias: graph.get_tensor_by_name(name) for alias, name in self.output_tensors.items()}

        # warm up
        self.log.info("Warming up %s ...", self)
        self.process_single(np.zeros((2, 2, 3), dtype=np.uint8))

    def shutdown(self):
        """ Releases model when object deleted """
        self.log.info("Shutdown %s ...", self)

        if self._session is None:
            return

        try:
            self._session.close()
            self._session = None
        except tf.OpError as err:
            self.log.error('%s close TF session error: %s. Skipping...', self, err)

        self.log.info("%s Destroyed successfully", self)

    def process_single(self, image: np.ndarray) -> typ.List[dict]:
        return self.process_batch([image])[0]

    def process_batch(self, images: typ.List[np.ndarray]) -> typ.List[dict]:
        preprocessed = np.stack([self._preprocess(image) for image in images])

        result = self._session.run(self._outputs, feed_dict={self._inputs['images']: preprocessed})

        detections_per_image = []

        for image, scores, boxes, labels in zip(images, result['scores'], result['boxes'], result['labels']):
            detections = []
            for score, box, label in zip(scores, boxes, labels):

                class_name = self._labels.get(int(label))
                if not class_name or score < self.config['threshold']:
                    continue

                # scale boxes wrt initial image size
                ymin, xmin, ymax, xmax = (np.tile(image.shape[:2], 2) * box).astype(np.int32).tolist()

                width, height = xmax - xmin, ymax - ymin

                detections.append({
                    'confidence': float(score),
                    'bounding_box': [xmin, ymin, width, height],
                    'class_name': class_name,
                })

            detections_per_image.append(detections)

        return detections_per_image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.config['width'], self.config['height']), interpolation=cv2.INTER_NEAREST)


def from_config_file(filename: str) -> TfObjectDetectionModel:
    """
    :param filename: filename to model config
    """
    return TfObjectDetectionModel(**load_config(filename))


class GstTfDetectionPluginPy(GstBase.BaseTransform):

    # Metadata Explanation:
    # http://lifestyletransfer.com/how-to-create-simple-blurfilter-with-gstreamer-in-python-using-opencv/

    GST_PLUGIN_NAME = 'gst_tf_detection'

    __gstmetadata__ = ("Name",
                       "Transform",
                       "Description",
                       "Author")

    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       Gst.Caps.from_string("video/x-raw,format=RGB"))

    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        Gst.Caps.from_string("video/x-raw,format=RGB"))

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "model": (GObject.TYPE_PYOBJECT,
                  "model",
                  "Contains model TfObjectDetectionModel",
                  GObject.ParamFlags.READWRITE),

        "config": (str,
                   "Path to config file",
                   "Contains path to config *.yml supported by TfObjectDetectionModel",
                   None,  # default
                   GObject.ParamFlags.READWRITE
                   ),
    }

    def __init__(self):
        super().__init__()

        self.model = None
        self.config = None        

    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:

        if self.model is None:
            Gst.warning(f"No model speficied for {self}. Plugin working in passthrough mode")
            return Gst.FlowReturn.OK

        try:
            # Convert Gst.Buffer to np.ndarray
            image = utils.gst_buffer_with_caps_to_ndarray(buffer, self.sinkpad.get_current_caps())

            # model inference
            objects = self.model.process_single(image)

            Gst.debug(f"Frame id ({buffer.pts // buffer.duration}). Detected {str(objects)}")
           
            # write objects to as Gst.Buffer's metadata
            # Explained: http://lifestyletransfer.com/how-to-add-metadata-to-gstreamer-buffer-in-python/
            gst_meta_write(buffer, objects)
        except Exception as err:
            logging.error("Error %s: %s", self, err)

        return Gst.FlowReturn.OK

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'model':
            return self.model
        if prop.name == 'config':
            return self.config
        else:
            raise AttributeError('Unknown property %s' % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'model':
            self._do_set_model(value)
        elif prop.name == "config":
            self._do_set_model(from_config_file(value))
            self.config = value
            Gst.info(f"Model's config updated from {self.config}")
        else:
            raise AttributeError('Unknown property %s' % prop.name)

    def _do_set_model(self, model: TfObjectDetectionModel):

        # stop previous instance
        if self.model:
            self.model.shutdown()
            self.model = None

        self.model = model

        # start new instance
        if self.model:
            self.model.startup()

    def __exit__(self, exc_type, exc_val, exc_tb):

        Gst.info(f"Shutdown {self}")

        if self.model:
            self.model.shutdown()

        sGst.info(f"Destroyed {self}")


# Required for registering plugin dynamically
# Explained: http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstTfDetectionPluginPy)
__gstelementfactory__ = (GstTfDetectionPluginPy.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstTfDetectionPluginPy)
