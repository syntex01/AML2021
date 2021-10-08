import data
import cnn
import cam
import utils
from tensorflow.keras import backend as K
import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disables GPU computing

###################
# hyper parameters #
###################

RGB = True                  # use three or one color channel (only when creating new model)
IMAGE_SHAPE = (256, 256)    # input image dimensions (only when creating new data)
SPLIT = (0.8, 0.15, 0.05)   # train/test/validation split for the created data folders (only when creating new data)
MODE = "validate"           # new->override model and start new one, continue->continue training on existing model,
                            # validate->only validate existing model
BALANCE = True              # When True underrepresented classes get filled up by copies (only when creating new data)
NO_PLOT = True              # no plotting when true

# default cnn parameter used when not in a w&b sweep
SINGLE_RUN_CONFIG = utils.dotdict(dict(
    activation_conv="relu",
    activation_dense="relu",
    activation_output="softmax",
    batch_size=16,
    conv_sections=5,
    conv_layers_per_section=3,
    conv_window_0=3,
    conv_window_1=3,
    conv_window_2=3,
    conv_window_3=3,
    conv_window_4=3,
    dense_layers=2,
    dense_neurons=4094,
    dense_scaling=0.2,
    dropout=0.35,
    epochs=2,
    filter=64,
    filter_scaling=2**(3/4),
    upsampling=1,
    learning_rate=0.00001,
    loss="categorical_crossentropy",
    optimizer="adam",
    pool_window=2,
    pooling=True,
    gap=True,
    batchnorm=False,
    class_weight_mode="none",
    zca_whitening=False,
    data_name="data_tile",
    model_name="EfficientNetB2_tile_v2",
    accum=1,
    patience=7,
    use_pretrained="EfficientNetB2",
    rescale=1
))
# creates an instance of the data and cnn class
data = data.Data()
cnn = cnn.Cnn(IMAGE_SHAPE, RGB, MODE, NO_PLOT, SINGLE_RUN_CONFIG)

# loads the cnn if so specified and gets the mode from cnn.mode
mode = cnn.load()

# trains and saves the model and creates the training data if not available
if mode == "continue" or mode == "new":
    data.save(f"./{cnn.config.data_name}", SPLIT, cnn.IMAGE_SHAPE, balance=BALANCE)
    cnn.train()
    cnn.save()
    cnn.plot()

# validates the cnn independent of mode
cnn.validate()

K.clear_session()

# visualize a CAM
"""iid = "0a677ac5768f"
x, y, b = data.sample(iid, IMAGE_SHAPE, RGB)
cam = cam.Cam(cnn)
cam.plot(x, y, b, path="problem_4_samples_flowfromdir_cam")
"""