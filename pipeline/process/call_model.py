import os

import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import grpc
from grpc._cython import cygrpc

MODEL_NAME = os.getenv("MODEL_NAME")


def create_channel(host, port):
    return grpc.insecure_channel(
        target=host if port is None else "%s:%d" % (host, port),
        options=[
            (cygrpc.ChannelArgKey.max_send_message_length, -1),
            (cygrpc.ChannelArgKey.max_receive_message_length, -1),
        ],
    )


CHANNEL = create_channel(os.getenv("URL"), int(os.getenv("PORT")))


def collaborative_filtering_predict(user_id, movie_id):
    #convert to float
    input_1 = np.array([user_id for _ in range(len(movie_id))])
    input_2 = np.array([i for i in movie_id])

    # create PredictRequest
    res = predict_pb2.PredictRequest()
    res.model_spec.name = MODEL_NAME
    res.model_spec.signature_name = "serving_default"
    res.inputs["input_1"].CopyFrom(
        tf.make_tensor_proto(np.float32(input_1), shape=input_1.shape)
    )
    res.inputs["input_2"].CopyFrom(
        tf.make_tensor_proto(np.float32(input_2), shape=input_2.shape)
    )
    sub = prediction_service_pb2_grpc.PredictionServiceStub(CHANNEL)
    print("asdnjkasndasndjas")
    result = sub.Predict(res, timeout=60.0)

    outputs_tensor_proto = result.outputs["lambda"]
    print(outputs_tensor_proto)
    print("outputs")
    shape = [dim.size for dim in outputs_tensor_proto.tensor_shape.dim]
    print(shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape)
    # outputs = np.array(outputs_tensor_proto.float_val)

    print(outputs)
    print("outputs")
    return outputs
