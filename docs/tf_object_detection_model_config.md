
#### Config Format: YAML

#### Config Example
    weights: "frozen_inference_graph.pb"
    threshold: 0.5
    per_process_gpu_memory_fraction: 1.0
    device: "/device:GPU:0"
    labels: mscoco_label_map.yml
    input_shape: [300, 300]