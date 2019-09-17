from functools import partial

from keras.layers import BatchNormalization, Conv2D, Input, Add, Activation, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from sgf_solver.constants import INPUT_DATA_SHAPE, CHANNELS_AMOUNT, RESIDUAL_BLOCKS, L2_CONST

RegularizedConv2D = partial(Conv2D, data_format='channels_first')
PaddedConv2D = partial(RegularizedConv2D, padding='same', kernel_regularizer=l2(L2_CONST))


def create_model():
    input_ = Input(shape=INPUT_DATA_SHAPE)

    layer = input_
    layer = RegularizedConv2D(CHANNELS_AMOUNT, (1, 1))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    for _ in range(RESIDUAL_BLOCKS):
        res = layer
        layer = PaddedConv2D(CHANNELS_AMOUNT, (3, 3))(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = PaddedConv2D(CHANNELS_AMOUNT, (3, 3))(layer)
        layer = BatchNormalization()(layer)

        layer = Add()([layer, res])
        layer = Activation('relu')(layer)

    value_head = layer
    value_head = RegularizedConv2D(1, (1, 1))(value_head)
    value_head = BatchNormalization()(value_head)
    value_head = Activation("relu")(value_head)
    value_head = Flatten()(value_head)
    value_head = Dense(1)(value_head)
    value_head = Activation("tanh", name="vh")(value_head)

    policy_head = layer
    policy_head = RegularizedConv2D(2, (1, 1))(policy_head)
    policy_head = BatchNormalization()(policy_head)
    policy_head = Activation('relu')(policy_head)
    policy_head = Flatten()(policy_head)
    policy_head = Dense(361)(policy_head)
    policy_head = Activation('softmax', name="ph")(policy_head)

    model = Model(inputs=[input_], outputs=[value_head, policy_head])
    model.compile(
        optimizer=Adam(),
        loss=["mean_squared_error", "categorical_crossentropy"],
        loss_weights=[0.5, 0.5],
        metrics=["accuracy"]
    )
    model.summary()
    return model

create_model()