from tensorflow.keras import Sequential

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

def pretrained_cnn(config, image_size, n_channels):
    
    pretrained_type = config["pretrained"]["type"]

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
    else:
        raise ValueError("Model type must be a VGG or ResNet derivant. See tensorflow.keras.applications for all options")

    ConvNet = getattr(module, pretrained_type)

    input_layer = Input(shape=(image_size, image_size, n_channels))

    convnet = ConvNet(
        include_top=False,
        weights=config["pretrained"]["weights"],
        input_tensor=input_layer,
        pooling=config["pretrained"]["pooling"],
        classes=config["n_classes"]
    )

    model = Sequential(convnet)
    model.add(Flatten())
    for layer in range(config["pretrained"]["fnn_layers"]):
        model.add(Dense(config["pretrained"]["fnn_units"], activation="relu"))
        model.add(Dropout(config["pretrained"]["dropout"]))
    model.add(Dense(config["n_classes"], activation="softmax"))
    
    return model