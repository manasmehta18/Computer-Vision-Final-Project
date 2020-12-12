

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
do_images = False
rgb = False
size = (128, 128)
canny = True
layer_amt = 2
aug_imgs = False


# choose model here
filter_size = 3
pool_size = 2

epochs = 15

print("----------------------")

layer_opts = {
    # ---------------
    # 32x32
    # ---------------
    "1 Layer 32x32":
        lambda: Sequential([
            Conv2D(20, filter_size, activation='relu',
                   input_shape=(size[0], size[1], 3 if rgb else 1)),
            MaxPooling2D(pool_size=pool_size),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax'),
        ]),
    "2 Layer 32x32":
        lambda: Sequential([
            Conv2D(20, filter_size, activation='relu',
                   input_shape=(size[0], size[1], 3 if rgb else 1)),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(60, filter_size, activation='relu'),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax'),
        ]),
    "3 Layer 32x32":
    lambda: Sequential([
        Conv2D(20, filter_size, activation='relu',
               input_shape=(size[0], size[1], 3 if rgb else 1)),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(60, filter_size, activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(60, filter_size, activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax'),
    ]),

    # ---------------
    # 64x64
    # ---------------

    "1 Layer 64x64":
        lambda: Sequential([
            Conv2D(20, filter_size, activation='relu',
                   input_shape=(size[0], size[1], 3 if rgb else 1)),
            MaxPooling2D(pool_size=pool_size),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax'),
        ]),
    "2 Layer 64x64":
        lambda: Sequential([
            Conv2D(20, filter_size, activation='relu',
                   input_shape=(size[0], size[1], 3 if rgb else 1)),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(60, filter_size, activation='relu'),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax'),
        ]),
    "4 Layer 64x64":
    lambda: Sequential([
        Conv2D(20, filter_size, activation='relu',
               input_shape=(size[0], size[1], 3 if rgb else 1)),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(60, filter_size, activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(60, filter_size, activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(60, filter_size, activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax'),
    ]),

    # ---------------
    # 128x128
    # ---------------

    "1 Layer 128x128":
    lambda: Sequential([
        Conv2D(20, filter_size, activation='relu',
               input_shape=(size[0], size[1], 3 if rgb else 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax'),
    ]),
    "2 Layer 128x128":
        lambda: Sequential([
            Conv2D(20, filter_size, activation='relu',
                   input_shape=(size[0], size[1], 3 if rgb else 1)),
            MaxPooling2D(pool_size=pool_size),
            Conv2D(60, filter_size, activation='relu'),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax'),
        ]),
    "4 Layer 128x128":
        lambda: Sequential([
            Conv2D(20, filter_size, activation='relu',
                   input_shape=(size[0], size[1], 3 if rgb else 1)),
            MaxPooling2D(pool_size=4),
            Conv2D(60, filter_size, activation='relu'),
            MaxPooling2D(pool_size=2),
            Conv2D(60, filter_size, activation='relu'),
            MaxPooling2D(pool_size=2),
            Conv2D(60, filter_size, activation='relu'),
            MaxPooling2D(pool_size=pool_size),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax'),
        ]),
}

layer = layer_opts[str(layer_amt) + " Layer " +
                   str(size[0]) + "x" + str(size[1])]

l = layer()
print(l.summary())
