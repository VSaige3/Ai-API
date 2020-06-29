# The meat and potatoes of the Code
# Should probably be made of different sections that we can put together
# IDK why but Model seems better for this.
from keras.models import Model
import numpy as np
from API.ApiMain import DataProcessor
from keras import backend as K
from typing import Tuple, Union
from keras.layers import Conv2D, Lambda, BatchNormalization, Activation, Dense, Dropout, MaxPooling2D
from keras.layers import Input, UpSampling2D, Flatten, Convolution1D, MaxPooling1D


def create_Conv2D(x, filters=(16, 32, 64), size: tuple = (3, 3), cs=8):
    if isinstance(filters, tuple):
        for filter in filters:
            x = Conv2D(filter, size, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=size, padding="same")(x)
    else:
        x = Conv2D(filters, size)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=size)(x)

    x = Flatten()(x)
    x = Dense(cs * 2)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    return x


def create_Conv1D(x, filters=(16, 32, 64), size: Union[tuple, int] = 3, cs=8):
    if isinstance(filters, tuple):
        for filter in filters:
            print(filter)
            x = Convolution1D(filter, size)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling1D()(x)
    else:
        x = Convolution1D(filters, size)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D()(x)

    x = Dense(cs * 2)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    return x


def create_Downsample(x, pool_size: tuple=(2, 2), rate=0.5):
    x = MaxPooling2D(pool_size)(x)
    x = Dropout(rate)(x)
    x = Activation('relu')(x)

    return x


def create_Upsample(x, size: tuple=(2, 2)):
    x = UpSampling2D(size)(x)
    x = Activation('relu')(x)
    return x


def _merge(x):
    a = x[0]
    b = x[1]
    return K.concatenate([a, b], axis=3)


def create_Merge():
    return Lambda(_merge)


def create_network(o_input_sizes: Tuple[tuple, tuple, tuple, tuple], common_size: int=8):
    input_sizes = []
    if common_size <= 0:
        raise ValueError(f"cannot set layer units to size {common_size}")

    for i, x in enumerate(o_input_sizes):
        # if len(x) == 2:
        if False:
            n = (x[0], x[1], x[1])
            input_sizes.append(n)
            print(n)
        else:
            input_sizes.append(x)
    i1 = Input(input_sizes[0], name="geo_input")
    i2 = Input(input_sizes[1], name="demo_input")
    i3 = Input(input_sizes[2], name="econ_input")
    i4 = Input(input_sizes[3], name="viral_data")

    # process geographic data
    x = Dense(common_size * 2, activation='relu')(i1)
    # x = create_Conv2D(x, (4, 8), (2, 1), common_size)

    x = Dense(common_size, activation='relu')(x)

    # process demographic data
    y = Dense(common_size * 2, activation='relu')(i2)
    y = Dense(common_size, activation='relu')(y)

    # process economic data
    z = create_Conv1D(i3, size=2)
    z = Dense(common_size * 4, activation='relu')(z)
    z = Dense(common_size * 2, activation='relu')(z)
    z = Dense(common_size, activation='relu')(z)

    # process viral data
    a = Dense(common_size * 2, activation='relu')(i4)
    a = Dense(common_size, activation='relu')(a)
    # combine geographic and demographic
    xy = create_Merge()([x, y])
    xy = Dense(common_size * 2, activation='relu')(xy)
    xy = Dense(common_size, activation='relu')(xy)

    # merge it all
    merged = create_Merge()([xy, z])
    merged = Dense(common_size * 4, activation='relu')(merged)
    out = create_Merge()([merged, a])
    out = Dense(common_size * 8, activation='relu')(out)
    out = Dense(common_size * 4, activation='relu')(out)
    out = Dense(common_size * 2, activation='relu')(out)
    out = Dense(common_size, activation='sigmoid')(out)
    out = Dense(1, activation='linear')(out)

    return Model(inputs=[i1, i2, i3], outputs=out)

def get_split_from_collecter(c: DataProcessor):
    pass
    #TODO: have to get size from