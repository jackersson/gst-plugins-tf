
#### Config Format: YAML

#### Config Example
 - [file example](https://github.com/jackersson/gst-plugins-tf/blob/master/data/tf_object_api_cfg.yml)

##### Path to Model's .pb file [Supported Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
    weights: "frozen_inference_graph.pb"

##### Threshold [0.0, 1.0]
    threshold: 0.5

##### Input Shape
    input_shape: [w, h]

##### Log device placement

    log_device_placement: true or false

##### GPU Options:

   - [explained](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto#L36)

         per_process_gpu_memory_fraction: 0.0

    device: "GPU|CPU", "GPU", "CPU", "GPU:0", ..

#### Labels [Format](https://github.com/jackersson/gst-plugins-tf/blob/master/docs/label_map_format.md)
    labels: mscoco_label_map.yml or {1: "person", ..}


