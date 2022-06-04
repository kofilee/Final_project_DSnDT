from . import cnn, rnn, dnn

model_list = [
    cnn.CNN(),
    rnn.RNN(),
    dnn.DNN()
]


def get_model(idx):
    return model_list[idx]

