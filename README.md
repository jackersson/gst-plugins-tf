# gst-plugins-tf

- Allows to inject [Tensorflow Models Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) into Gstreamer Pipeline in Python
- [COCO Labels](https://github.com/tensorflow/models/tree/master/research/object_detection/data)

## Installation
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade wheel pip setuptools
pip install --upgrade --requirement requirements.txt

export GOOGLE_APPLICATION_CREDENTIALS=$PWD/credentials/gs_viewer.json
dvc pull
```

### Install Tensorflow
- Tested on TF-GPU==1.5
#### TF-CPU
```bash
pip install tensorflow==1.15
```

```bash
pip install tensorflow-gpu==1.15
```

## Usage

### Run example
```bash
./run_example.sh
```

### To enable plugins implemented in **gst/python**
```bash
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/venv/lib/gstreamer-1.0/:$PWD/gst/
```

### Plugins
#### gst_tf_detection
    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGB ! gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! gtksink sync=False

##### Parameters
 - **config**: path to filename of [Config Format](https://github.com/jackersson/gst-plugins-tf/blob/master/docs/tf_object_detection_model_config.md)
 - **model**: instance of object [TFObjectDetectionModel](https://github.com/jackersson/gst-plugins-tf/blob/master/gst/python/gst_tf_detection.py#L90)

#### gst_detection_overlay
    gst-launch-1.0 filesrc location=video.mp4 ! decodebin ! videoconvert ! \
    video/x-raw,format=RGB ! gst_tf_detection config=data/tf_object_api_cfg.yml ! videoconvert ! \
    gst_detection_overlay ! videoconvert ! gtksink sync=False

### Utils
 - [convert_labels_pbtxt_to_yml](https://github.com/jackersson/gst-plugins-tf/blob/master/utils/convert_labels_pbtxt_to_yml.py)

       python convert_labels_pbtxt_to_yml.py -f mscoco_label_map.pbtxt


### Additional
#### Enable/Disable TF logs
```bash
export TF_CPP_MIN_LOG_LEVEL={0,1,2,3,4,5 ...}
```

#### Enable/Disable Gst logs
```bash
export GST_DEBUG=python:{0,1,2,3,4,5 ...}
```

#### Enable/Disable Python logs
```bash
export GST_PYTHON_LOG_LEVEL={0,1,2,3,4,5 ...}
```