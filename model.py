from keras.layers import BatchNormalization, Conv2D, Input, Add, Activation, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

INPUT_DATA_SHAPE = (6, 19, 19)


def create_model():
    input_ = Input(shape=INPUT_DATA_SHAPE)
    l2const = 1e-4

    layer = input_
    layer = Conv2D(64, (1, 1), kernel_regularizer=l2(l2const))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2const))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    for _ in range(8):
        res = layer
        layer = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2const))(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2const))(layer)
        layer = BatchNormalization()(layer)

        layer = Add()([layer, res])
        layer = Activation('relu')(layer)

    phead = layer
    phead = Conv2D(2, (1, 1), kernel_regularizer=l2(l2const))(phead)
    phead = BatchNormalization()(phead)
    phead = Activation('relu')(phead)
    phead = Flatten()(phead)
    phead = Dense(361)(phead)
    phead = Activation('softmax')(phead)

    model = Model(inputs=[input_], outputs=[phead])
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    import os
    from utils import elementary_data
    import matplotlib.pyplot as plt

    model = create_model()
    model.summary()

    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')

    length = elementary_data['problem'].shape[0]
    x, y = elementary_data['problem'][:], elementary_data['answers'][:].reshape(length, -1)

    history = model.fit(x, y, validation_split=0.2, epochs=5, verbose=1)
    model.save_weights('weights.h5')

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
