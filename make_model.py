from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

def double_conv_block(input, num_filters, activation='relu'):
    x = Conv2D(num_filters, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x

def encoder_block(input, num_filters):
    x = double_conv_block(input, num_filters)
    down = MaxPooling2D((2,2))(x)

    return x, down

def decoder_block(input, skip_connections, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_connections])
    x = double_conv_block(x, num_filters)

    return x

def UNet(input_shape, num_filters=64, num_layers=4):
    inputs = Input(input_shape)

    skip_1, down_1 = encoder_block(inputs, 64)
    skip_2, down_2 = encoder_block(down_1, 128)
    skip_3, down_3 = encoder_block(down_2, 256)
    skip_4, down_4 = encoder_block(down_3, 512)

    bridge = double_conv_block(down_4, 1024)

    up_1 = decoder_block(bridge, skip_4, 512)
    up_2 = decoder_block(up_1, skip_3, 256)
    up_3 = decoder_block(up_2, skip_2, 128)
    up_4 = decoder_block(up_3, skip_1, 64)

    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(up_4)
    model = Model(inputs, outputs)

    return model

if __name__ == '__main__':
    input_shape = (128, 128, 1)
    model = UNet(input_shape)
    model.summary()
