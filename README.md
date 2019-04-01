# gst-plugins-tf

- Allows to inject [Tensorflow Models Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to Gstreamer Pipeline
- [Labels](https://github.com/tensorflow/models/tree/master/research/object_detection/data)

### Usage

    export GST_PLUGIN_PATH=$PWD
    
#### gst_tf_detection
    GST_DEBUG=python:4 gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! gst_tf_detection config=cfg.yml ! videoconvert ! gtksink sync=False
   
##### Parameters
 - config: path to filename of [Config Format](https://github.com/jackersson/gst-plugins-tf/blob/master/docs/tf_object_detection_model_config.md) 
    
#### gst_detection_overlay    
    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB ! gst_tf_detection config=cfg.yml ! videoconvert ! gst_detection_overlay ! videoconvert ! gtksink sync=False

### Utils
 - [convert_labels_pbtxt_to_yml](https://github.com/jackersson/gst-plugins-tf/blob/master/utils/convert_labels_pbtxt_to_yml.py)
       
       python3 convert_labels_pbtxt_to_yml.py -f mscoco_label_map.pbtxt
 
 
