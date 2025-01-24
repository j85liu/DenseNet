from tensorflow.keras import layers, models
from DenseConnectLayer import DenseConnectLayer

def add_dense_block(x, num_layers, growth_rate, drop_rate, bottleneck):
    for _ in range(num_layers):
        x = DenseConnectLayer(x.shape[-1], growth_rate, drop_rate, bottleneck)(x)
    return x

def add_transition(x, reduction=0.5, drop_rate=0.2, last=False, input_size=None, num_blocks=None):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if last:
        # Dynamically calculate global pooling size
        pool_size = input_size // (2 ** num_blocks)
        x = layers.AveragePooling2D((pool_size, pool_size))(x)
        return layers.Flatten()(x)
    else:
        x = layers.Conv2D(int(x.shape[-1] * reduction), (1, 1), padding='same', use_bias=False)(x)
        if drop_rate > 0:
            x = layers.Dropout(drop_rate)(x)
        return layers.AveragePooling2D((2, 2))(x)

def create_densenet(dataset, input_size, depth, num_blocks, growth_rate, bottleneck, compression, num_classes, drop_rate=0.2):
    N = (depth - 4) // (num_blocks * (2 if bottleneck else 1))
    reduction = 0.5 if compression else 1.0

    # Input layer
    input_layer = layers.Input(shape=(input_size, input_size, 3))
    x = layers.Conv2D(2 * growth_rate, (7 if dataset == 'imagenet' else 3, 3), 
                      padding='same', strides=(2 if dataset == 'imagenet' else 1), use_bias=False)(input_layer)

    # Dense Blocks and Transitions
    for i in range(num_blocks):
        x = add_dense_block(x, N, growth_rate, drop_rate, bottleneck)
        x = add_transition(x, reduction=reduction, drop_rate=drop_rate, last=(i == num_blocks - 1), 
                           input_size=input_size, num_blocks=num_blocks)

    # Output layer with dynamic num_classes
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=input_layer, outputs=output_layer)