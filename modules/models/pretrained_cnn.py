from tensorflow.keras import Sequential

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation

def pretrained_cnn_module(pretrained_type):

    if pretrained_type.startswith("VGG"):
        if pretrained_type.endswith("16"):
            from tensorflow.keras.applications import vgg16 as module
        elif pretrained_type.endswith("19"):
            from tensorflow.keras.applications import vgg19 as module
    elif pretrained_type.startswith("ResNet"):
        if pretrained_type.endswith("V2"):
            from tensorflow.keras.applications import resnet_v2 as module
        else:
            from tensorflow.keras.applications import resnet as module
    elif pretrained_type.startswith("NASNet"):
        from tensorflow.keras.applications import nasnet as module
    elif pretrained_type.startswith("Xception"):
        from tensorflow.keras.applications import xception as module 
    else:
        raise ValueError("Model type must be a VGG, ResNet or NASNet derivant. See tensorflow.keras.applications for all options")
        
    return module

def pretrained_cnn(config, image_size, n_channels):
    
    pretrained_type = config["pretrained"]["type"]
    
    module = pretrained_cnn_module(pretrained_type)

    ConvNet = getattr(module, pretrained_type)

    input_layer = Input(shape=(image_size, image_size, n_channels))

    convnet = ConvNet(
        include_top=False,
        weights=config["pretrained"]["weights"],
        input_tensor=input_layer,
        pooling=config["pretrained"]["pooling"],
        classes=config["n_classes"]
    )
    
    if config["pretrained"]["frozen"]:
        for layer in convnet.layers:
            layer.trainable = False

    model = Sequential(convnet)
    model.add(Flatten())
    for layer in range(config["pretrained"]["fnn_layers"]):
        model.add(Dense(config["pretrained"]["fnn_units"], activation="relu"))
        model.add(Dropout(config["pretrained"]["dropout"]))
    model.add(Dense(config["n_classes"], activation="softmax"))
    
    return model

def pretrained_cnn_multichannel(config, image_size, n_channels):
    if n_channels == 3:
        return pretrained_cnn(config, image_size, n_channels)
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                     input_shape=(224, 224, 4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(config["pretrained"]["dropout"]))
    model.add(Dense(3, activation='softmax'))
    
    return model
    