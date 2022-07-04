import os

import numpy as np
import pandas as pd
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import grpc
from grpc._cython import cygrpc

PATH = "./data/"

movies = pd.read_csv(PATH + "movies.csv")
ratings = pd.read_csv(PATH + "ratings.csv")
ratings["userId"] = ratings["userId"].fillna("")
ratings = ratings.drop(ratings[ratings.rating > 5].index)
ratings = ratings.drop(ratings[ratings.rating < 0].index)
ratings = ratings.drop(ratings[ratings.userId == ""].index)

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
    # as input of model it is require user id index
    user_index = ratings.userId.unique().tolist().index(user_id)

    # convert to np
    input_1 = np.array([user_index for _ in range(len(movie_id))])
    input_2 = np.array(movie_id)

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
    result = sub.Predict(res, timeout=60.0)

    outputs_tensor_proto = result.outputs["lambda"]
    shape = [dim.size for dim in outputs_tensor_proto.tensor_shape.dim]
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape)

    return outputs
