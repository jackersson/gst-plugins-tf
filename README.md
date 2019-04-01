# gst-plugins-tf

### Usage

    export GST_PLUGIN_PATH=$PWD
    
    
    GST_DEBUG=python:4 gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! gst_tf_detection config=cfg.yml ! videoconvert ! gtksink sync=False
    
    
    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! gst_tf_detection config=cfg.yml ! videoconvert ! gst_detection_overlay ! videoconvert ! gtksink sync=False

### Utils
 - [convert_labels_pbtxt_to_yml](https://github.com/jackersson/gst-plugins-tf/blob/master/utils/convert_labels_pbtxt_to_yml.py)
 
 
