"""
Usage
    export GST_PLUGIN_PATH=$PWD

    gst-launch-1.0 videotestsrc ! gstplugin_py ! videoconvert ! autovideosink

"""

import os
import logging
import timeit
import traceback
import time
import random
import cv2
import tensorflow as tf
from typing import List, Tuple
import yaml
import cairo
import numpy as np

from pygst_utils import get_buffer_size, map_gst_buffer, Gst, GObject
from pygst_utils.gst_objects_info_meta import gst_meta_get


class ColorPicker:

    def __init__(self):
        self._color_by_id = {}

    def get(self, idx):
        if idx not in self._color_by_id:
            self._color_by_id[idx] = self.generate_color()
        return self._color_by_id[idx]

    def generate_color(self, low=0, high=1):
        return random.uniform(low, high), random.uniform(low, high), random.uniform(low, high)


class ObjectsOverlayCairo:

    def __init__(self, line_thickness_scaler: float = 0.0025,
                 font_size_scaler: float = 0.01,
                 font_family: str = 'Sans',
                 font_slant: cairo.FontSlant = cairo.FONT_SLANT_NORMAL,
                 font_weight: cairo.FontWeight = cairo.FONT_WEIGHT_BOLD,
                 text_color: Tuple[int, int, int]=[255, 255, 255],
                 colors: ColorPicker = None):

        self.line_thickness_scaler = line_thickness_scaler
        self.font_size_scaler = font_size_scaler
        self.font_family = font_family
        self.font_slant = font_slant
        self.font_weight = font_weight

        self.text_color = [float(x) / max(text_color) for x in text_color]
        self.colors = colors or ColorPicker()

    def draw(self, buffer: Gst.Buffer, width: int, height: int, objects: List[dict]) -> bool:
        try:
            stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_RGB24, width)
            surface = cairo.ImageSurface.create_for_data(buffer,
                                                         cairo.FORMAT_RGB24,
                                                         width, height,
                                                         stride)
            context = cairo.Context(surface)
        except Exception as e:
            logging.error(e)
            logging.error("Failed to create cairo surface for buffer")
            return False

        try:

            context.select_font_face(self.font_family, self.font_slant, self.font_weight)

            diagonal = (width**2 + height**2)**0.5
            context.set_font_size(int(diagonal * self.font_size_scaler))
            context.set_line_width(int(diagonal * self.line_thickness_scaler))

            for obj in objects:

                r, g, b = self.colors.get(obj["class_name"])
                context.set_source_rgb(r, g, b)

                l, t, w, h = obj['bounding_box']
                context.rectangle(l, t, w, h)
                context.stroke()

                text = "{}".format(obj["class_name"])
                _, _, text_w, text_h, _, _ = context.text_extents(text)

                tableu_height = text_h
                context.rectangle(l, t - tableu_height, w, tableu_height)
                context.fill()

                r, g, b = self.text_color
                context.set_source_rgb(r, g, b)
                context.move_to(l, t)
                context.show_text(text)

        except Exception as e:
            logging.error(e)
            logging.error("Failed cairo render")
            traceback.print_exc()
            return False

        return True


class GstDetectionOverlay(Gst.Element):

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
        super(GstDetectionOverlay, self).__init__()

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

        self.model = ObjectsOverlayCairo()

    def chainfunc(self, pad: Gst.Pad, parent, buffer: Gst.Buffer) -> Gst.FlowReturn:
        """
        :param parent: GstDetectionOverlay
        """
        # Get Buffer Width/Height
        success, (width, height) = get_buffer_size(
            self.srcpad.get_current_caps())

        if not success:
            # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.FlowReturn
            return Gst.FlowReturn.ERROR

        try:
            objects = gst_meta_get(buffer)
            if objects:
                # Do Buffer processing
                with map_gst_buffer(buffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
                    if self.model:
                        self.model.draw(mapped, width, height, objects)

        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            return Gst.FlowReturn.ERROR

        return self.srcpad.push(buffer)

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

    def eventfunc(self, pad, parent, event):
        """ Forwards event to SRC (DOWNSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadEventFunction
        """
        return self.srcpad.push_event(event)

    def srcqueryfunc(self, pad, object, query):
        """ Forwards query bacj to SINK (UPSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadQueryFunction
        """
        return self.sinkpad.query(query)

    def srceventfunc(self, pad, parent, event):
        """ Forwards event back to SINK (UPSTREAM)
            https://lazka.github.io/pgi-docs/Gst-1.0/callbacks.html#Gst.PadEventFunction
        """
        return self.sinkpad.push_event(event)


# Required for registering plugin dynamically
# Explained:
# http://lifestyletransfer.com/how-to-write-gstreamer-plugin-with-python/
GObject.type_register(GstDetectionOverlay)
__gstelementfactory__ = (GstDetectionOverlay.GST_PLUGIN_NAME,
                         Gst.Rank.NONE, GstDetectionOverlay)
