"""
Usage
    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
    export GST_DEBUG=python:4

    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
        gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! \
        gst_detection_overlay ! videoconvert ! autovideosink
"""

import os
import logging
import random
import typing as typ
import cairo

from gstreamer import Gst, GObject, GstBase
from gstreamer import map_gst_buffer
import gstreamer.utils as utils
from gstreamer.gst_objects_info_meta import gst_meta_get


def _get_log_level() -> int:
    return int(os.getenv("GST_PYTHON_LOG_LEVEL", logging.DEBUG / 10)) * 10


log = logging.getLogger('gst_python')
log.setLevel(_get_log_level())


class ColorPicker:
    """Generates random colors"""

    def __init__(self):
        self._color_by_id = {}

    def get(self, idx: typ.Any):
        if idx not in self._color_by_id:
            self._color_by_id[idx] = self.generate_color()
        return self._color_by_id[idx]

    def generate_color(self, low=0, high=1):
        return random.uniform(low, high), random.uniform(low, high), random.uniform(low, high)


class ObjectsOverlayCairo:
    """Draws objects on video frame"""

    def __init__(self, line_thickness_scaler: float = 0.0025,
                 font_size_scaler: float = 0.01,
                 font_family: str = 'Sans',
                 font_slant: cairo.FontSlant = cairo.FONT_SLANT_NORMAL,
                 font_weight: cairo.FontWeight = cairo.FONT_WEIGHT_BOLD,
                 text_color: typ.Tuple[int, int, int] = [255, 255, 255],
                 colors: ColorPicker = None):

        self.line_thickness_scaler = line_thickness_scaler
        self.font_size_scaler = font_size_scaler
        self.font_family = font_family
        self.font_slant = font_slant
        self.font_weight = font_weight

        self.text_color = [float(x) / max(text_color) for x in text_color]
        self.colors = colors or ColorPicker()

    @property
    def log(self) -> logging.Logger:
        return log

    def draw(self, buffer: Gst.Buffer, width: int, height: int, objects: typ.List[dict]) -> bool:
        """Draws objects on video buffer"""
        try:
            stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_RGB24, width)
            surface = cairo.ImageSurface.create_for_data(buffer,
                                                         cairo.FORMAT_RGB24,
                                                         width, height,
                                                         stride)
            context = cairo.Context(surface)
        except Exception as err:
            logging.error("Failed to create cairo surface for buffer %s. %s", err, self)
            return False

        try:

            context.select_font_face(self.font_family, self.font_slant, self.font_weight)

            diagonal = (width**2 + height**2)**0.5
            context.set_font_size(int(diagonal * self.font_size_scaler))
            context.set_line_width(int(diagonal * self.line_thickness_scaler))

            for obj in objects:

                # set color by class_name
                r, g, b = self.colors.get(obj["class_name"])
                context.set_source_rgb(r, g, b)

                # draw bounding box
                l, t, w, h = obj['bounding_box']
                context.rectangle(l, t, w, h)
                context.stroke()

                # tableu for additional info
                text = "{}".format(obj["class_name"])
                _, _, text_w, text_h, _, _ = context.text_extents(text)

                tableu_height = text_h
                context.rectangle(l, t - tableu_height, w, tableu_height)
                context.fill()

                # draw class name
                r, g, b = self.text_color
                context.set_source_rgb(r, g, b)
                context.move_to(l, t)
                context.show_text(text)

        except Exception as e:
            logging.error("Failed cairo render %s. %s", err, self)
            return False

        return True


class GstDetectionOverlay(GstBase.BaseTransform):

    # Metadata Explanation:
    # http://lifestyletransfer.com/how-to-create-simple-blurfilter-with-gstreamer-in-python-using-opencv/

    GST_PLUGIN_NAME = 'gst_detection_overlay'

    __gstmetadata__ = ("Name",
                       "Transform",
                       "Description",
                       "Author")

    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       Gst.Caps.from_string("video/x-raw,format={RGBx}"))

    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        Gst.Caps.from_string("video/x-raw,format={RGBx}"))

    __gsttemplates__ = (_srctemplate, _sinktemplate)

    # Explanation: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#GObject.GObject.__gproperties__
    # Example: https://python-gtk-3-tutorial.readthedocs.io/en/latest/objects.html#properties
    __gproperties__ = {
        "model": (GObject.TYPE_PYOBJECT,
                  "ObjectsOverlayCairo",
                  "Contains model that implements ObjectsOverlayCairo",
                  GObject.ParamFlags.READWRITE),
    }

    def __init__(self):
        super().__init__()

        self.model = ObjectsOverlayCairo()

    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:

        if self.model is None:
            Gst.warning(f"No model speficied for {self}. Plugin working in passthrough mode")
            return Gst.FlowReturn.OK

        try:
            objects = gst_meta_get(buffer)

            if objects:
                width, height = utils.get_buffer_size_from_gst_caps(self.sinkpad.get_current_caps())

                # Do drawing
                with map_gst_buffer(buffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
                    self.model.draw(mapped, width, height, objects)

        except Exception as err:
            Gst.error(f"Error {self}: {err}")
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == 'model':
            return self.model
        else:
            raise AttributeError('unknown property %s' % prop.name)

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == 'model':
            self.model = value
        else:
            raise AttributeError('unknown property %s' % prop.name)


# Required for registering plugin dynamically
# Explained: http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstDetectionOverlay)
__gstelementfactory__ = (GstDetectionOverlay.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstDetectionOverlay)
