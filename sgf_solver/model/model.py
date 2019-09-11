from functools import partial
from keras.layers import BatchNormalization, Conv2D, Input, Add, Activation, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

INPUT_DATA_SHAPE = (1, 19, 19)

RESIDUAL_LAYERS = 10
FILTER_SIZE = 32

L2_CONST = 1e-4

RegularizedConv2D = partial(Conv2D, kernel_regularizer=l2(1e-4))
PaddedConv2D = partial(Conv2D, padding='same', kernel_regularizer=l2(1e-4))


def create_model():
    input_ = Input(shape=INPUT_DATA_SHAPE)

    layer = input_
    layer = RegularizedConv2D(FILTER_SIZE, (1, 1))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = PaddedConv2D(FILTER_SIZE, (3, 3))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    for _ in range(RESIDUAL_LAYERS):
        res = layer
        layer = PaddedConv2D(FILTER_SIZE, (3, 3))(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = PaddedConv2D(FILTER_SIZE, (3, 3))(layer)
        layer = BatchNormalization()(layer)

        layer = Add()([layer, res])
        layer = Activation('relu')(layer)

    value_head = layer
    value_head = Dense(FILTER_SIZE, kernel_regularizer=l2(1e-4))(value_head)
    value_head = Activation("relu")(value_head)
    value_head = Flatten()(value_head)
    value_head = Dense(1)(value_head)
    value_head = Activation("tanh", name="vh")(value_head)

    policy_head = layer
    policy_head = RegularizedConv2D(FILTER_SIZE, (1, 1))(policy_head)
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

    return model
