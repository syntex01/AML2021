import data
import cnn
import utils
from tensorflow.keras import backend as K
import wandb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import *
from tensorflow_addons.losses import giou_loss

from sklearn.metrics import classification_report, confusion_matrix
# to compute weights for the class distribution
from sklearn.utils.class_weight import compute_class_weight
#import wandb
#from wandb.keras import WandbCallback
import opt

###################
# hyper parameters #
###################

RGB = True  # use three or one color channel (only when creating new model)
TILE_SHAPE = (256, 256)  # input tile dimensions (only when creating new data)
IMAGE_SHAPE = (2048, 2048)
SAMPLE_NUMBER = 420 # !!! may not be larger than smallest class / 4
MODE = "continue"  # new->override model and start new one, continue->continue training on existing model,
# validate->only validate existing model
BALANCE = True  # When True underrepresented classes get filled up by copies (only when creating new data)
NO_PLOT = True  # no plotting when true

# default cnn parameter used when not in a w&b sweep
TILE_CONFIG = utils.dotdict(dict(
    exclude_wandb=True,
    data_name="data_tile",
    dropout=0.2,
    output_nodes=4,
    optimizer="adam",
    learning_rate=0.0001,
    accum=0,
    loss="categorical_crossentropy",
    model_name="EfficientNetB2_tile_v2",
    use_pretrained="EfficientNetB2",
))

def box_loss(y_true, y_pred):
  # compute the losses for both
  aa = giou_loss(y_true[:,:4], y_pred[:,:4])
  ab = giou_loss(y_true[:,:4], y_pred[:,4:])
  ba = giou_loss(y_true[:,4:], y_pred[:,:4])
  bb = giou_loss(y_true[:,4:], y_pred[:,4:])
  a = aa + bb
  b = ab + ba
  return tf.math.minimum(aa+bb, ab+ba) # element-wise minimum of both losses


BOX_CONFIG = utils.dotdict(dict(
    dropout=0.0,
    learning_rate=0.0001,
    outputs=["label", "box"],
    output_nodes=[4, 8],
    losses=["categorical_crossentropy", box_loss],
    loss_weights=[1, 1],
    model_name="DNN_box_v2",
    dense_layers=4,
    dense_scaling=0.5,
    dense_neurons=256,
    activation_dense="relu",
    activation_output=["softmax", "relu"],
    epochs=100
))

# trains the second model (target model) on tilemaps created by the first model (data model) and retruns the target model
def tile_train(sample_number, epochs, data_model, target_model, tile_shape, image_shape):
    wandb_callback = wandb.keras.WandbCallback(log_weights=True)
    for i in range(epochs):
        random_image_ids = data.sample_ids(sample_number)
        tilex, tiley1, tiley2 = data.to_tilemap(random_image_ids, data_model.model, tile_shape, image_shape, mode="edge")
        tiley = (tiley1, tiley2)

        target_model.history = target_model.fit(
            x=tilex,
            y={BOX_CONFIG.outputs[j]: tiley[j] for j in range(len(BOX_CONFIG.outputs))},
            batch_size=16,
            epochs=1,
            callbacks=[wandb_callback]
        ).history
    return target_model

# create the target model(box_cnn) manually since it is a DNN and can therefore not be created by the CNN class
layer_in = Input(shape=(*tuple(l1 // l2 for l1, l2 in zip(IMAGE_SHAPE, TILE_SHAPE)), 4), name="input")
x = Flatten()(layer_in)

for i in range(BOX_CONFIG.dense_layers):
    x = Dense(int(BOX_CONFIG.dense_neurons * BOX_CONFIG.dense_scaling ** i),
              activation=BOX_CONFIG.activation_dense,
              kernel_initializer="HeNormal")(x)
    x = Dropout(BOX_CONFIG.dropout)(x)
outputs = []
for i in range(len(BOX_CONFIG.outputs)):
  outputs.append( Dense(BOX_CONFIG.output_nodes[i], activation=BOX_CONFIG.activation_output[i], dtype='float32', name=BOX_CONFIG.outputs[i])(x) )
box_cnn = Model(inputs=layer_in, outputs=outputs)
box_cnn.compile(optimizer=Adam(BOX_CONFIG.learning_rate),
                loss={BOX_CONFIG.outputs[i]: BOX_CONFIG.losses[i] for i in range(len(BOX_CONFIG.outputs))},
                loss_weights={BOX_CONFIG.outputs[i]: BOX_CONFIG.loss_weights[i] for i in range(len(BOX_CONFIG.outputs))},
                metrics=["accuracy"])
# print model summary
box_cnn.summary()

# login to W&B
wandb.login()
wandb.init(project='covid19', entity='cov01', config=BOX_CONFIG)

# create instances of the data and cnn class, the latter is for our data model
data = data.Data()
tile_cnn = cnn.Cnn(TILE_SHAPE, RGB, "continue", NO_PLOT, TILE_CONFIG)

# load the previously trained tile cnn
_ = tile_cnn.load()

# train and save the box cnn
box_cnn = tile_train(SAMPLE_NUMBER, BOX_CONFIG.epochs, tile_cnn, box_cnn, TILE_SHAPE, IMAGE_SHAPE)
box_cnn.save(BOX_CONFIG.model_name)

K.clear_session()
