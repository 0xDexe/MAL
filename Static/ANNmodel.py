from keras.layers import Input, Conv2D, Lambda, MaxPool2D, UpSampling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Flatten, Dense, Add, Multiply, BatchNormalization, Dropout, Average

from keras.models import Model


class ResidualAttentionNetwork:

    def __init__(self, input_shape, n_classes, activation, p=1, t=2, r=1):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.activation = activation
        self.p = p
        self.t = t
        self.r = r

    def build_model(self):
        # Initialize a Keras Tensor of input_shape
        input_data = Input(shape=self.input_shape)

        # Initial Layers before Attention Module

        conv_layer_1 = Conv2D(filters=32,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same')(input_data)

        max_pool_layer_1 = MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding='same')(conv_layer_1)

        # Residual Unit then Attentfion Module #1
        x = self.residual_unit(max_pool_layer_1, filters=[32, 64, 128])
        x = self.soft_attention(x, filters=[32, 64, 64])
        x = self.residual_unit(x, filters=[32, 64, 128])
        x = self.soft_attention(x, filters=[32, 64, 64])
        x = self.residual_unit(x, filters=[32, 64, 128])

        # Avg Pooling
        avg_pool_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        # Flatten the data
        flatten_op = Flatten()(avg_pool_layer)

        # FC Layers for prediction

        fully_connected_layer_2 = Dense(128, activation='relu')(flatten_op)
        dropout_layer_2 = Dropout(0.5)(fully_connected_layer_2)
        fully_connected_layer_3 = Dense(64, activation='relu')(dropout_layer_2)
        dropout_layer_3 = Dropout(0.5)(fully_connected_layer_3)
        fully_connected_layer_last = Dense(self.n_classes, activation=self.activation)(dropout_layer_3)

        # Fully constructed model
        model = Model(inputs=input_data, outputs=fully_connected_layer_last)

        return model

    def residual_unit(self, residual_input_data, filters):

        identity_x = residual_input_data

        filter1, filter2, filter3 = filters

        # 1x1 conv
        batch_norm_op_1 = BatchNormalization()(residual_input_data)
        activation_op_1 = Activation('relu')(batch_norm_op_1)
        conv_op_1 = Conv2D(filters=filter1,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='same')(activation_op_1)

        # high capacity kernel

        activation_op_2 = Activation('relu')(conv_op_1)
        kernel1 = Conv2D(filters=filter2,
                         kernel_size=(7, 7),
                         strides=(1, 1),
                         padding='same', activation='relu')(activation_op_2)

        # Layer 3

        activation_op_3 = Activation('relu')(conv_op_1)
        kernel2 = Conv2D(filters=filter2,
                         kernel_size=(7, 7),
                         strides=(1, 1),
                         padding='same', activation='relu')(activation_op_3)
        # concat kernel
        concat = Average()([kernel1, kernel2])

        # skip 1

        if identity_x.shape[-1] != concat.shape[-1]:
            filter_n = concat.shape[-1]

            identity_x = Conv2D(filters=filter_n,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                padding='same')(identity_x)

        skip_1 = Add()([identity_x, concat])

        conv_final = Conv2D(filters=filter3,
                            kernel_size=(5, 5),
                            strides=(1, 1),
                            padding='same', activation='relu')(skip_1)
        # skip 2

        if identity_x.shape[-1] != conv_final.shape[-1]:
            filter_n = conv_final.shape[-1]

            identity_x = Conv2D(filters=filter_n,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                padding='same')(identity_x)

        skip_2 = Add()([identity_x, conv_final])
        activation_op_4 = Activation('relu')(skip_2)
        output = Dropout(0.5)(activation_op_4)
        return output

    def plain_convolution(self, residual_input_data, filters):
        filter1, filter2, filter3 = filters

        # 1x1 conv
        batch_norm_op_1 = BatchNormalization()(residual_input_data)
        conv_op_1 = Conv2D(filters=filter1,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='same', activation='relu')(batch_norm_op_1)
        return conv_op_1

    def RAN_layer(self, attention_input_data, filters):
        # p=1, t=2, r=1

        # Perform Trunk Branch Operation
        trunk_branch_op = self.trunk_branch(trunk_input_data=attention_input_data, filters=filters)

        mask_branch_op = self.soft_attention(mask_input_data=attention_input_data, filters=filters)
        ar_learning_op = self.attention_residual_learning(mask_input=mask_branch_op, trunk_input=trunk_branch_op)

        out = self.plain_convolution(residual_input_data=ar_learning_op, filters=filters)

        return out

    def trunk_branch(self, trunk_input_data, filters):
        t_res_unit_op = trunk_input_data
        for _ in range(2):
            t_res_unit_op = self.residual_unit(t_res_unit_op, filters=filters)

        return t_res_unit_op

    def soft_attention(self, mask_input_data, filters):

        downsampling = MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2),
                                 padding='same')(mask_input_data)

        # ===================================================================================================

        for _ in range(1):
            middleware = self.plain_convolution(residual_input_data=downsampling, filters=filters)

        # ===================================================================================================

        # Upsampling Step Initialization - Top
        upsampling = UpSampling2D(size=(2, 2))(middleware)

        conv_filter = upsampling.shape[-1]

        conv1 = Conv2D(filters=conv_filter,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same')(upsampling)

        sigmoid = Activation('sigmoid')(conv1)

        return sigmoid

    def attention_residual_learning(self, mask_input, trunk_input):
        # Mx = Lambda(lambda x: 1 + x)(mask_input)  # 1 + mask
        return Add()([mask_input, trunk_input])  # M(x) * T(x)
