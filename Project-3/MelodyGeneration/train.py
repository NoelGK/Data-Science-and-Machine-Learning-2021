import tensorflow.keras as keras


def build_lstm(nodes, output_units, loss, learning_rate):
    input_layer = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(units=nodes[0])(input_layer)
    x = keras.layers.Dropout(0.2)(x)  # to avoid overflow
    output = keras.layers.Dense(output_units, activation='softmax')(x)
    model = keras.Model(input_layer, output)

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model
