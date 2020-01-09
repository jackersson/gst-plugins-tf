import pytest

import gst.python.gst_tf_detection as gst_tf_detection


def test_tf_object_detection_model():

    config = {
        "weights": "data/models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"
    }

    with gst_tf_detection.TfObjectDetectionModel(**config) as model:
        assert isinstance(model, gst_tf_detection.TfObjectDetectionModel)

    with gst_tf_detection.from_config_file("data/tf_object_api_cfg.yml"):
        assert isinstance(model, gst_tf_detection.TfObjectDetectionModel)


def test_load_labels_from_file():
    labels = gst_tf_detection.load_labels_from_file(
        "data/mscoco_label_map.yml")
    assert isinstance(labels, dict)
    assert all(isinstance(l, int) for l in labels.keys())
    assert all(isinstance(l, str) for l in labels.values())
    assert len(labels.keys()) == 80
