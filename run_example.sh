#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=5
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
export GST_DEBUG=python:4

gst-launch-1.0 filesrc location=data/videos/trailer.mp4 ! decodebin ! videoconvert ! \
 gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! \
 gst_detection_overlay ! videoconvert ! autovideosink