# import onnx
# from onnx import helper, TensorProto

# # Define input and output tensors
# input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, None])
# output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, None])

# # Create the HardSwish node
# hardswish_node = helper.make_node("HardSwish", inputs=["input"], outputs=["output"])

# # Build the graph
# graph_def = helper.make_graph(
#     [hardswish_node], "HardSwishGraph", [input_tensor], [output_tensor]
# )

# # Build the model
# model_def = helper.make_model(graph_def, producer_name="onnx-hardswish-example")

# # Save the model
# onnx.save(model_def, "../models/hardswish.onnx")


import sys
import os

# Add the parallel directory to sys.path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PyRuntime import OMExecutionSession
