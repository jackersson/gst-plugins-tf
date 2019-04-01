
#### Label Maps Format
    class_id (int): class_name (str)

##### Example
    1: person
    2: bicycle
    3: car
    
#### How to convert from [Tensorflow .pbtxt Labels](https://github.com/tensorflow/models/tree/master/research/object_detection/data)
- [convert_labels_pbtxt_to_yml](https://github.com/jackersson/gst-plugins-tf/blob/master/utils/convert_labels_pbtxt_to_yml.py)
       
       python3 convert_labels_pbtxt_to_yml.py -f mscoco_label_map.pbtxt
 
