from tensorflow.keras import layers
import tensorflow as tf

class DenseConnectLayer(layers.Layer):
    def __init__(self, n_channels, growth_rate, drop_rate=0.0, bottleneck=False, **kwargs):
        super(DenseConnectLayer, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.bottleneck = bottleneck

        if self.bottleneck:
            inter_channels = 4 * growth_rate
            self.bn1 = layers.BatchNormalization()
            self.relu1 = layers.Activation('relu')
            self.conv1 = layers.Conv2D(inter_channels, (1, 1), padding='same', use_bias=False)

        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)
        if drop_rate > 0:
            self.dropout = layers.Dropout(drop_rate)
        else:
            self.dropout = None

    def call(self, inputs, training=None):
        x = inputs
        if self.bottleneck:
            x = self.bn1(x, training=training)
            x = self.relu1(x)
            x = self.conv1(x)
            if self.dropout is not None:
                x = self.dropout(x, training=training)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return tf.concat([inputs, x], axis=-1)

    def get_config(self):
        config = super(DenseConnectLayer, self).get_config()
        config.update({
            'n_channels': self.n_channels,
            'growth_rate': self.growth_rate,
            'drop_rate': self.drop_rate,
            'bottleneck': self.bottleneck
        })
        return config