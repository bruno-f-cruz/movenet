# Bonsai - MoveNet

## How to install

Bonsai.TensorFlow.MoveNet can be downloaded through the Bonsai package manager. However, in order to use it for either CPU or GPU inference, you need to pair it with a compiled native TensorFlow binary. You can find precompiled binaries for Windows 64-bit at https://www.tensorflow.org/install/lang_c.

To use GPU TensorFlow (highly recommended for live inference), you also need to install the `CUDA Toolkit` and the `cuDNN libraries`. The current package was developed and tested with [CUDA v11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) and [cuDNN 8.2](https://developer.nvidia.com/cudnn). Additionally, make sure you have a CUDA [compatible GPU](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#support-hardware) with the latest NVIDIA drivers.

After downloading the native TensorFlow binary and cuDNN, you can follow these steps to get the required native files into the `Extensions` folder of your local Bonsai install:

1. The easiest way to find your Bonsai install folder is to right-click on the Bonsai shortcut > Properties. The path to the folder will be shown in the "Start in" textbox;
2. Copy `tensorflow.dll` file from either the CPU or GPU [tensorflow release](https://www.tensorflow.org/install/lang_c#download_and_extract) to the `Extensions` folder;
3. If you are using TensorFlow GPU, make sure to add the `cuda/bin` folder of your cuDNN download to the `PATH` environment variable, or copy all DLL files to the `Extensions` folder.

## How to use

The package already includes a pre-trained [MoveNet](https://tfhub.dev/s?q=movenet) .pb file that is loaded when using the `PredictMoveNet` operator. In order to run inference on an incoming image, simply connect a node that provides an `IplImage` type to `PredictMoveNet`. E.g:

![WorkflowExample](./docs/images/workflow_example.svg)

The output of the model (`Pose`) can be indexed using one of the following 17 human `bodypart` keypoints:

```
- nose
- left_eye
- right_eye
- left_ear
- right_ear
- left_shoulder
- right_shoulder
- left_elbow
- right_elbow
- left_wrist
- right_wrist
- left_hip
- right_hip
- left_knee
- right_knee
- left_ankle
- right_ankle
```

## How to download pre-exported networks:

Network files are available from [TensorFlow Hub](https://tfhub.dev/):
- [movenet/singlepose/lightning](https://tfhub.dev/google/movenet/singlepose/lightning/4)
- [movenet/singlepose/thunder](https://tfhub.dev/google/movenet/singlepose/thunder/4)
- [movenet/multipose/lightning](https://tfhub.dev/google/movenet/multipose/lightning/1)

All downloaded network .pb files should be placed inside `src/Externals/Networks` in order to build the project successfully.

## How to export the .pb files

The pre-trained models are provided as .pb files. If you want to export your own .pb files, you can use the following code:

```python

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

## Pick one of the networks
networkurl = r"https://tfhub.dev/google/movenet/singlepose/lightning/4"
#networkurl = r"https://tfhub.dev/google/movenet/singlepose/thunder/4"
#networkurl = r"https://tfhub.dev/google/movenet/multipose/lightning/1"

## Download the network
module = hub.load(networkurl)
networkName = networkurl.split("/")[-3] + "_" + networkurl.split("/")[-2]

## Freeze the network
model = module.signatures['serving_default']
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

## Print the network layers and output/input specs
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                logdir="./frozen_models",
                name=f"frozen_graph{networkName}.pb",
                as_text=False)

```
