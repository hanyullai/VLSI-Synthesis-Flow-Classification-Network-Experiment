from keras.models import Sequential
from keras.layers import Embedding, Reshape
from keras.layers import Dense, Dropout
from keras.layers import LSTM

def get_model(input_type, output_type):
    model = Sequential()

    if input_type == 'embedding':
        model.add(Embedding(6, 12))
        model.add(Reshape((24,12)))
    elif input_type == 'one-hot':
        model.add(Reshape((24,6)))
    else:
        assert False

    model.add(LSTM(50))

    if output_type == 'reg':
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    elif output_type == 'cls':
        model.add(Dense(7, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        assert False
    
    if input_type == 'embedding':
        model.build(input_shape=(1,24,12))
    elif input_type == 'one-hot':
        model.build(input_shape=(1,24,6))
    elif input_type == 'index':
        model.build(input_shape=(1,24))
    else:
        assert False

    return model