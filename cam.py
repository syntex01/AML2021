import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy as sp
from tensorflow.keras.models import Model
# Package pillow to save the images.
from PIL import Image

# https://arxiv.org/abs/1512.04150
# http://cnnlocalization.csail.mit.edu/
# use global average pooling instead of max pooling in activation-layer!

SLASH = "/"

class Cam:
    def __init__(self, model):
        """
        Initializes the CAM module.

        Parameters
        ----------
        model: keras Model
            The CAM module expects the
            - last layer of the model to be named "output", and
            - the last convolutional layer of the model to be named
              "activation".
            The number of filters in the "activation"-layer has to match
            the number of neurons in the fully connected layer preceding
            "output". The resulting class activation maps (CAMs) will
            then have the same width and height as the "activation"-
            layer. To increase it esp. in deep CNNs, an umsampling
            layer before "activation" may be used.
            Based on: http://cnnlocalization.csail.mit.edu/ (Sep. 2021)
        """
        self.output_weights = model.get_layer("output").get_weights()[0]
        self.cam_model = Model(inputs=model.input, outputs=(model.get_layer("activation").output, model.get_layer("output").output))

    def generate(self, image, label=None, zoom=False):
        """
        Generate a class activation map (CAM).

        Parameters
        ----------
        image: ndarray
            The image for which to generate the CAM.
        label: int, optional
            The index of the class for which to calculate the CAM.
            By default, the predicted class will be used.
        zoom: bool, optional
            If True the resulting CAM will be scaled up to meet the
            input image's dimensions.

        Returns
        -------
        cam: ndarray
          The class activation map.
        """
        activation, prediction = self.cam_model.predict(np.array([image]))
        activation = activation[0]
        prediction = prediction[0]
        if label == None:
            label = np.argmax(prediction)
        #scale = [image.shape[i]/activation.shape[i] for i in range(2)]
        #activation = sp.ndimage.zoom(activation, (scale[0], scale[1], self.output_weights.shape[0]/activation.shape[2]), order=2)
        cam = activation.dot(self.output_weights[:,label])
        if zoom:
            scale = [image.shape[i]/cam.shape[i] for i in range(2)]
            print("image:", image.shape, "cam:", cam.shape, "-> scale:", scale)
            cam = sp.ndimage.zoom(cam, (scale[0], scale[1]), order=2)
            print("zoomed:", cam.shape)
        # clip the CAM
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

    def plot(self, x, y, b, path=None):
        """
        Plot class activation maps.

        Parameters
        ----------
        x: ndarray
            The batch of images for which to plot the cams
        y: ndarray
            The corresponding labels
        b: list
            A list of boxes to plot over the images.
        """
        label = ["atypical", "indeterminate", "negative", "typical"]
        _, pred = self.cam_model.predict(x)
        for i in range(len(x)):
            image = x[i] if x.shape[-1] == 3 else np.squeeze(x[i], -1)

            fig, axs = plt.subplots(2, 2)
            for j in range(4):
                ax_x = [0, 1, 0, 1]
                ax_y = [0, 0, 1, 1]
                ax = axs[ax_x[j], ax_y[j]]
                p = np.argmax(pred[i])
                a = np.argmax(y[i])
                c = '(pa)' if j == p and p == a else '(p)'  if j == p else '(a)'  if j == a else ''
                ax.title.set_text(f"{label[j]} {c}")
                # hide axis ticks
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis='both', which='both', length=0)
                # plot original image with boxes
                ax.imshow(image, cmap="gray", aspect="equal")
                for box in b[i]:
                    ax.add_patch(Rectangle((box["x"], box["y"]), box["width"], box["height"], linewidth=1, edgecolor="r", facecolor="None", alpha=0.6))
                # plot CAM
                camap = self.generate(x[i], label=j, zoom=True)
                camap = ax.imshow(camap, cmap="coolwarm", aspect="equal", alpha=0.6)
            #cax = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0,0.02, ax2.get_position().height])
            #plt.colorbar(camap, cax=cax, orientation="vertical")
            if path != None: plt.savefig(path + f"_{i}.png", dpi=300, format="png")
            plt.show()

    def save(self, x, y, names, path="", zoom=False):
        """
        Save class activation maps.

        Parameters
        ----------
        x: ndarray
            The batch of images for which to plot the cams
        y: ndarray
            The one-hot encoded class labels for which to generate
            the cams.
        names: list of str
            A list of names for the cams.
        path: str, optionalc
            The path under which to save the images.
        zoom: bool, optional
            If True the resulting CAM will be scaled up to meet the
            input image's dimensions.
        """
        for i in range(len(x)):
            image = self.generate(x[i], label=np.argmax(y[i]), zoom=zoom)
            image = Image.fromarray((image*255).astype("uint8"))
            image.save(path + names[i] + ".png", "PNG")
