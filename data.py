import numpy as np
# Package pandas to create a database containing information on the dataset
import pandas as pd
# Package os to search files and directories in the dataset
import os
# Package pydicom to load dicom-images. Some of the images in our dataset contain compressed image self.data.
# To decompress it, an additonal library is necessary. We use the package python-gdcm which pydicom then
# automatically uses.
# https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html?highlight=compressed
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
# Package pillow to save the images.
from PIL import Image
# Package random to shuffle a list in save()
import random
# Package scipy to rescale images in __get_image()
from scipy.ndimage import zoom

import matplotlib.pyplot as plt


from matplotlib.patches import Rectangle

# global configuration
SLASH = "/"


class Data:
    def __get_path(self, study_id, image_id):
        """
        Get the PATH to an image.

        Parameters
        ----------
        study_id: str
          The study ID of the sample.
        image_id: str
          The image ID of the sample.

        Returns
        -------
        path: str
            The path to the image file.
        """
        path_ = self.path + "train" + SLASH + study_id + SLASH
        # in each study directory multiple directories for each series may exist.
        # to get the series of the current image, find the directory containing
        # the current image.
        # Note: The names of the image files differ from their ID by arbitrary
        # leading digits!
        if os.path.isdir(path_):
            for d in os.scandir(path_):
                for f in os.scandir(d.path):
                    if (f.name[-4-len(image_id):-4] == image_id):
                        return f.path
        return ""

    def __get_image(self, image_id, dimension, rgb, mode):
        """
        Load a dicom image.

        Parameters
        ----------
        PATH: str
          The PATH to the image file.
        dimension: tuple
          A tuple containing the desired dimensions of the image. If it is
          None, the original dimensions will be kept.
        rgb: bool
          If true, the image is encoded in the rgb format, otherwise in
          grayscale.

        Returns
        -------
        image: ndarray
          The padded and if necessary downsampled image.
        boxes:
          A list of boxes
        """
        mask = (self.data["image_id"].values == image_id)
        row = self.data.loc[mask].squeeze()

        # load the imageprint
        dcm = dcmread(row.path)
        #info = {k: dcm.get(k) for k in dcm.keys()}
        if dcm.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.70":
          dcm.decompress("gdcm")

        ### source: https://www.kaggle.com/xhlulu/siim-covid-19-convert-to-jpg-256px/comments (24.08.21)
        # VOI LUT (if available by DICOM device) is used to transform raw DICOM self.data to
        # "human-friendly" view
        image = apply_voi_lut(dcm.pixel_array, dcm)
        # depending on this value, X-ray may look inverted - fix that:
        if dcm.PhotometricInterpretation == "MONOCHROME1":
          image = np.amax(image) - image

        # normalize the image
        image = image - np.min(image)
        image = image.astype(float) / np.max(image)

        pad = np.zeros((len(image.shape), 2), dtype=int)
        scale = np.ones(len(image.shape), dtype=float)
        if mode == "pad":
          # compute downsampling factor so that the image fits into the target dimension
          downsample = max([int(image.shape[i] / dimension[i] + 1) for i in range(len(image.shape))])
          for i in range(len(scale)): scale[i] = 1/downsample
          a, b, = image.shape[0], image.shape[1]
          image = image[::downsample, ::downsample]
          # for each axis compute the number of zeros to pad the (possibly smaller) image
          for axis in range(len(pad)):
            total = dimension[axis] - image.shape[axis]
            left = total // 2
            right = total // 2
            if total % 2: right += 1
            pad[axis] = [left, right]
          # pad the image
          image = np.pad(image, pad, "constant", constant_values=70/255)
        elif mode == "scale":
            scale = [dimension[i]/image.shape[i] for i in range(len(image.shape))]
            image = zoom(image, scale)
            scale = np.flip(scale)
        else:
            print(f"ERROR: invalid value \"{mode}\" for parameter mode.")
            return None

        # if desired, convert to rgb
        if rgb:
          image = np.stack((image, image, image), axis=2)
        else:
          image = image.reshape(*image.shape, 1)

        # adjust boxes
        boxes = []
        for box in list(row.boxes):
            boxes.append({ # make deep copies!
              "x": round(box["x"] * scale[0] + pad[1][0]),
              "y": round(box["y"] * scale[1] + pad[0][0]),
              "width": round(box["width"] * scale[0]),
              "height": round(box["height"] * scale[1])
            })

        return image, boxes

    def __intersect(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """
        Calculate the area of intersection of two rectangles.

        Parameters:
        -----------
        x1, y1, w1, h1: float
            The first rectangle's dimensions.
        x2, y2, w2, h2: float
            The second rectangle's dimensions.

        Returns:
        --------
        intersection
            The intersection.
        """
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1+w1, x2+w2)
        iy2 = min(y1+h1, y2+h2)
        iw = max(ix2-ix1, 0)
        ih = max(iy2-iy1, 0)
        return iw * ih

    def __init__(self, path=("siim-covid19-detection" + SLASH), cache=False):
        """
        Create a pandas dataframe containing information on the dataset.

        Parameters
        ----------
        path: str, optional
            The path to where the KAGGLE data directory.
        cache: bool, optional
            If True, the database is cached as a pickle
        """

        self.path = path
        if os.path.isfile("data.pickle") and cache:
            self.data = pd.read_pickle("data.pickle").sample(frac=1)
        else:
            self.data = pd.read_csv(self.path+r"train_image_level.csv").rename(columns={
                "id" : "image_id",
                "StudyInstanceUID" : "study_id"})
            # cut trailing "_image" from the id
            self.data["image_id"] = self.data["image_id"].apply(lambda s : s[:s.index("_")])
            # convert box definitions into python variables
            self.data["boxes"] = self.data["boxes"].apply(lambda s : eval(s if type(s) == str else "[]"))
            # get covid status
            self.data["label"] = self.data["label"].apply(lambda s : 1 if s.lstrip()[0] == 'o' else 0)
            # construct paths to the image files
            self.data["path"] = self.data.apply(lambda r : self.__get_path(r.study_id, r.image_id), axis=1)
            # delete entries without image self.data
            # (this is used for testing the code
            # with access to a subset of the dataset
            self.data = self.data[self.data["path"] != ""]
            # get the image dimensions
            # self.data["dimension"] = self.data.apply(lambda r : (dcmread(r.PATH).pixel_array.shape if r.PATH else ()), axis=1)

            study = pd.read_csv(self.path+r"train_study_level.csv").rename(columns={
              "id" : "study_id",
              "Negative for Pneumonia" : "negative",
              "Typical Appearance" : "typical",
              "Indeterminate Appearance" : "indeterminate",
              "Atypical Appearance" : "atypical"})
            study["study_id"] = study["study_id"].apply(lambda s : s[:s.index("_")])
            self.data = pd.merge(self.data, study, on="study_id")
            # construct class labels
            self.data["class"] = self.data.apply(lambda r : np.array([r.atypical, r.indeterminate, r.negative, r.typical], dtype=float), axis=1)
            # shuffle the data
            self.data = self.data.sample(frac=1)
            if cache: self.data.to_pickle("data.pickle")

        self.class_labels = ["atypical", "indeterminate", "negative", "typical"]
        self.class_atypical      = np.array([1, 0, 0, 0])
        self.class_indeterminate = np.array([0, 1, 0, 0])
        self.class_negative      = np.array([0, 0, 1, 0])
        self.class_typical       = np.array([0, 0, 0, 1])

        # Get the class distribution to balance loss in model.fit() by converting the pandas dataframe to numpy(one-hot)
        # then create array with integers from 0-3 for the classes and compute the class weights via sklearn.
        #y_data = self.data.filter(items=self.class_labels).to_numpy()
        #y_integers = np.argmax(y_data, axis=1)
        #class_distribution_no_dict = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        #self.class_distribution = dict(enumerate(class_distribution_no_dict))


        print("class distribution of the data:")
        print("PNEUMONIA:")
        for label in self.class_labels:
            print(f"  {label}: {sum(self.data[label] == 1) / len(self.data) * 100:.1f}%")
        print(f"{len(self.data)} samples in total")
        print("BOX>=3", sum([len(self.data["boxes"].values.tolist()[i]) > 2 for i in range(len(self.data))]))
        #print("COVID-19:")
        #print(f"  positive: {sum(self.data['label'] == 1) / len(self.data) * 100:.1f}%")
        #print(f"  negative: {sum(self.data['label'] == 0) / len(self.data) * 100:.1f}%")

    def shuffle(self):
        self.data = self.data.sample(frac=1)

    def images(self, mask=None):
        if mask is None:
            return self.data["image_id"].values.tolist()
        else:
            return self.data["image_id"][mask].values.tolist()

    def classes(self, mask=None, onehot=True):
        if mask is None:
            mask = np.ones(len(self.data), dtype=bool)
        classes = np.vstack(self.data[mask]["class"].values)
        return classes if onehot else np.argmax(classes, axis=1)

    def sample(self, image_id, dimension, rgb=False, mode="scale"):
        """
        Get samples from the self.data.

        Parameters
        ----------
        self.data: pd.dataframe
            The information database.
        image_id: str or list of str
            The id(s) of the image(s) to load.
        dimension: (int, int)
            The desired dimensions of the images.
        rgb: bool, optional
            If true, the image(s) will have three color channels instead
            of one.

        Returns
        -------
        x: ndarray
            An array containing the image self.data of shape (len(image_id), *dimension, channels)
            where channels is 3 or 1 depending on rgb.
        y: ndarray
            An array containing the one-hot encoded predictions
        b: list
            A list of boxes.

        """
        def get(data, image_id, dimension, rgb, mode):
            x, b = self.__get_image(image_id, dimension, rgb, mode)
            y = self.data[self.data["image_id"] == image_id]["class"].values[0] #np.array([row.atypical, row.indeterminate, row.negative, row.typical], dtype=float)
            return x, y, b

        if type(image_id) == str:
            print("loading 1 image")
            return get(self.data, image_id, dimension, rgb, mode)
        else:
            x = []
            y = []
            b = []
            print(f"loading {len(image_id)} images")
            for i in range(len(image_id)):
                x_, y_, b_ = get(self.data, image_id[i], dimension, rgb, mode)
                x.append(x_)
                y.append(y_)
                b.append(b_)
                print(f"\r{int((i + 1) / len(image_id) * 100):3d}%", end="")
            print("")
            x = np.array(x)
            y = np.array(y)
            return x, y, b

    def save(self, path, split, dimension, mode="scale", balance=True):
        """
        Save the images in png-format. The created directory structure
        is compatible with tensorflow.keras.preprocessing.image
        .ImageDataGenerator.flow_from_directory()

        Parameters
        ----------
        path: str
            The path where the directory tree containing the images
            will be created.
        split: tuple
            A tuple specifying the relative amount of the data for the
            train-, test- and validation-set, e.g. (0.7, 0.2, 0.1).
        dimension: tuple
            The desired dimensions of the images.
        balance: bool, optional
            If true, for each set all classes with less images than the
            class with most images are stocked up to this number by
            copying images which are already in this class. However, if
            a class is empty it will stay empty.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
            split_names = ["train", "test", "validation"]
            # calculate split sizes
            split = [int(part * len(self.images())) for part in split]
            split = [sum(split[:i+1]) for i in range(len(split))]
            # split the images
            split = [
                self.images()[:split[0]] if split[0] > 0 else [],
                self.images()[split[0]:split[1]] if split[1] > split[0] else [],
                self.images()[split[1]:split[2]] if split[2] > split[1] else []
            ]
            # sort the images into classes
            split_images = [[ [] for j in range(len(self.class_labels))] for i in range(len(split_names))]

            distribution = []
            for i in range(len(split_names)):
                # populate the set
                for k in range(len(split[i])):
                    y = self.data[self.data["image_id"] == split[i][k]]["class"].values[0]
                    split_images[i][np.argmax(y)].extend([split[i][k]])
                distribution.append([len(split_images[i][j]) for j in range(len(self.class_labels))])
                for j in range(len(self.class_labels)):
                    if distribution[i][j] > 0 and balance:
                        fill = max(distribution[i]) - distribution[i][j]
                        while fill > 0:
                            k = min(len(split_images[i][j]), fill)
                            s = random.sample(split_images[i][j], k)
                            split_images[i][j].extend(s)
                            fill -= k
                    else:
                        print(f"  \"{self.class_labels[j]}\" is empty")
            # print summary:
            print("\n###########")
            print("# SUMMARY #")
            print("###########", end="\n")
            for i in range(len(split_names)):
                print(f"{split_names[i]}")
                for j in range(len(self.class_labels)):
                    length = len(split_images[i][j])
                    refill = length - distribution[i][j]
                    print(f"  {self.class_labels[j]}: ", end="")
                    if refill > 0:
                        print(f"{length-refill}+{refill}={length} images")
                    else:
                        print(f"{length} images")
            print("")
            # save the images
            for i in range(len(split_images)):
                num = sum([len(split_images[i][j]) for j in range(len(split_images[i]))])
                if num > 0:
                    # create a directory for the current set
                    os.mkdir(path + SLASH + split_names[i])
                    # create subdirectories for each class
                    for j in range(len(split_images[i])):
                        os.mkdir(path + SLASH + split_names[i] + SLASH + self.class_labels[j])
                        # load the images
                        x, y, _ = self.sample(split_images[i][j], dimension, rgb=False, mode=mode)
                        # go through the images
                        for k in range(len(x)):
                            image = (np.squeeze(x[k], -1)*255).astype("uint8")
                            image = Image.fromarray(image)
                            # save the image in the correct subdirectory
                            # if there are double images, append a verion
                            # number so that all images get saved
                            ver = 0
                            name = ""
                            while True:
                                name = path + SLASH + split_names[i] + SLASH + self.class_labels[np.argmax(y[k])] + SLASH + split_images[i][j][k] + (f"_{ver}" if ver > 0 else "") + ".png"
                                if os.path.isfile(name):
                                    ver += 1
                                else:
                                    break
                            image.save(name, "PNG")

    def save_tiles(self, path, dimension, full_dimension, n=0, sigma=0.5, with_replacement=False, min_intersect=0.33, sum_intersect=True, mode="constant", plot=False):
        """
        Save tiles of the images.

        Parameters:
        -----------
        path: str
            The path where the directory tree containing the images
            will be created.
        dimension: (int, int)
            The target dimension of the tiles.
        full_dimension: (int, int)
            The desired dimensions of the full images which will be
            split into tiles.
        n: int, optional
            The number of tiles to generate per image. By default or if
            it is zero, the whole image will be split into tiles and
            saved.
        sigma: float, optional
            Tiles to save are selected randomly from a normal distribion
            centered in the image's center. Its standart deviation per
            axis is sigma * dimension/2.
        with_replacement: bool, optional
            If it is false, each tile is sampled a single time. Otherwise,
            it may occur multiple times in the dataset and n may be larger
            than the maximum number of unique tiles.
        min_intersect: float, optional
            The tile's label matches the whole image's class label if
            its intersection with bounding boxes is large enough.
            Otherwise it is assigned to be negative.
        sum_intersect: bool, optional
            When checking a tile's area of intersection with bounding
            boxes, this specifies whether the sum of all areas or only
            the largest area of intersection shall be accounted for.
        mode: str, optional
            Padding mode passed to numpy.pad when padding tiles.
        plot: bool, optional
            If True the tiles sampled from an image are plotted
        """
        if os.path.isdir(path): return
        os.mkdir(path)
        for i in range(len(self.class_labels)):
            os.mkdir(path + SLASH + self.class_labels[i])

        image_id = self.data["image_id"].values.tolist()
        grid_dimension = np.array([np.ceil(full_dimension[i] / dimension[i]) for i in range(2)], dtype=int)
        if n <= 0:
            n = np.prod(grid_dimension)
        elif not with_replacement:
            n = np.clip(n, 0, np.prod(grid_dimension))
        offset = np.array([(grid_dimension[i] * dimension[i] - full_dimension[i]) / 2 for i in range(2)])
        for idx, iid in enumerate(image_id):
            #2c297acf1d25
            #09cf9767a7bf
            #2c297acf1d25
            #74077a8e3b7c
            print(f"{idx/len(image_id)*100:.1f}%")

            if plot: fig, ax = plt.subplots(grid_dimension[0], grid_dimension[1])

            # load the current image
            full_image, full_label, boxes = self.sample(iid, full_dimension, rgb=False)
            full_image = np.squeeze(full_image, -1)
            grid_used = np.zeros(grid_dimension, dtype=bool)

            # determine the tile's labels
            grid_labels = np.zeros(grid_dimension, dtype=int)
            for i in range(np.prod(grid_dimension)):
                grid_pos = np.array([i % grid_dimension[0], i // grid_dimension[0]])
                pos = (-offset + grid_pos * dimension).astype(int)
                intersect = np.zeros(len(boxes))
                for j in range(len(boxes)):
                    # calculate intersection
                    intersect[j] = self.__intersect(
                        boxes[j]["x"], boxes[j]["y"], boxes[j]["width"], boxes[j]["height"],
                        pos[1], pos[0], dimension[1], dimension[0]
                    ) / min(np.prod(dimension), boxes[j]["width"]*boxes[j]["height"])
                if sum_intersect:
                    intersect = sum(intersect)
                else:
                    intersect = max(intersect)
                grid_labels[grid_pos[0], grid_pos[1]] = np.argmax(full_label if intersect >= min_intersect else self.class_negative)

            # count number of non-negative tiles
            non_negative = sum(grid_labels.flatten() != np.argmax(self.class_negative))

            for i in range(n):
                # find an unused grid position
                while True:
                    grid_pos = np.array([int(np.random.normal(loc=grid_dimension[j]/2, scale=sigma*grid_dimension[j]/2)) for j in range(2)], dtype=int)
                    grid_pos = np.clip(grid_pos, a_min=0, a_max=grid_dimension-1)
                    if (    (not grid_used[grid_pos[0], grid_pos[1]] or with_replacement)
                        and (grid_labels[grid_pos[0], grid_pos[1]] != np.argmax(self.class_negative) or i >= non_negative)):
                        grid_used[grid_pos[0], grid_pos[1]] = True
                        break
                pos = (-offset + grid_pos * dimension).astype(int)

                # cut out the tile
                l = [max(pos[j], 0) for j in range(len(pos))]
                r = [min(pos[j]+dimension[j], full_dimension[j]) for j in range(len(pos))]
                pl = [abs(min(pos[j], 0)) for j in range(len(pos))]
                pr = [max(pos[j]+dimension[j]-full_dimension[j], 0) for j in range(len(pos))]
                pad = ((pl[0], pr[0]), (pl[1], pr[1]))
                tile_image = full_image[l[0]:r[0], l[1]:r[1]]
                tile_image = np.pad(tile_image, pad, mode)
                tile_label = grid_labels[grid_pos[0], grid_pos[1]]

                # plot bounding box edges
                if plot:
                    colors=["red", "green", "blue"]
                    for j in range(len(boxes)):
                        x = [boxes[j]["x"], boxes[j]["x"], boxes[j]["x"]+boxes[j]["width"], boxes[j]["x"]+boxes[j]["width"]]
                        y = [boxes[j]["y"], boxes[j]["y"]+boxes[j]["height"], boxes[j]["y"], boxes[j]["y"]+boxes[j]["height"]]
                        b = [(y[k] >= pos[0] and y[k] < pos[0]+dimension[0] and x[k] >= pos[1] and x[k] < pos[1]+dimension[1]) for k in range(4)]
                        for k in range(4):
                            if b[k]: ax[grid_pos[0]][grid_pos[1]].scatter(x[k]-pos[1], y[k]-pos[0], color=colors[j], marker="x")

                name = ""
                ver = 0
                while True:
                    name = path + SLASH + self.class_labels[tile_label] + SLASH + iid + "_" + str(grid_pos[0]) + "_" + str(grid_pos[1]) + (f"_{ver}" if ver > 0 else "") + ".png"
                    if os.path.isfile(name):
                        ver += 1
                    else:
                        break
                Image.fromarray((tile_image*255).astype("uint8")).save(name, "PNG")
                if plot:
                    ax[grid_pos[0]][grid_pos[1]].imshow(tile_image, cmap="gray" if tile_label == np.argmax(self.class_negative) else "viridis")
                    plt.setp(ax[grid_pos[0]][grid_pos[1]].get_xticklabels(), visible=False)
                    plt.setp(ax[grid_pos[0]][grid_pos[1]].get_yticklabels(), visible=False)

            if plot: plt.show()

    def to_tilemap(self, image_id, model, dimension, full_dimension, mode="constant"):
        """
        Convert an image into tiles, predict their labels and return
        the assembled tilemap.

        Parameters:
        -----------
        image_id: list of str
            List of image IDs.
        model: Keras Model
            The model which shall be used to predict tile labels. Its
            input shape must match <dimension>.
        dimension: (int, int)
            The target dimension of the tiles.
        full_dimension: (int, int)
            The desired dimensions of the full images which will be
            split into tiles.
        mode: str, optional
            Padding mode passed to numpy.pad when padding tiles.

        Returns:
        --------
        tilex: ndarray
            An array containing the computed tilemaps.
            The shape is (len(image_id),
                          ceil(full_dimension[0] / dimension[0]),
                          ceil(full_dimension[1] / dimension[1]),
                          4)
        tiley: ndarray
            The 12-labels
        """

        grid_dimension = np.array([np.ceil(full_dimension[i] / dimension[i]) for i in range(2)], dtype=int)
        offset = np.array([(grid_dimension[i] * dimension[i] - full_dimension[i]) / 2 for i in range(2)])

        # load the current image
        full_image, full_labels, boxes = self.sample(image_id, full_dimension, rgb=False)

        full_image = np.squeeze(full_image, -1)

        # go through every tile
        tilex = np.zeros((len(image_id), grid_dimension[0], grid_dimension[1], len(self.class_labels)))
        for i in range(len(image_id)):
            # stack the tiles into a numpy array
            tiles = np.zeros((np.prod(grid_dimension), dimension[0], dimension[1], 3))
            for j in range(np.prod(grid_dimension)):
                # first index changes fastest -> Fortran-like indexing
                grid_pos = np.array([j % grid_dimension[0], j // grid_dimension[0]])
                pos = (-offset + grid_pos * dimension).astype(int)
                # cut out the tile
                l = [max(pos[j], 0) for j in range(len(pos))]
                r = [min(pos[j]+dimension[j], full_dimension[j]) for j in range(len(pos))]
                pl = [abs(min(pos[j], 0)) for j in range(len(pos))]
                pr = [max(pos[j]+dimension[j]-full_dimension[j], 0) for j in range(len(pos))]
                pad = ((pl[0], pr[0]), (pl[1], pr[1]))
                tile_image = full_image[i, l[0]:r[0], l[1]:r[1]]
                tile_image = np.pad(tile_image, pad, mode)
                # convert the tile to rgb
                tiles[j] = np.stack((tile_image, tile_image, tile_image), axis=2)
            # predict all tiles' labels at once
            labels = model.predict(tiles)
            # reshape the labels to match the grid's shape (using Fortran-like indexing)
            tilex[i] = labels.reshape((grid_dimension[0], grid_dimension[1], len(self.class_labels)), order="F")

        # tiley
        tiley1 = np.zeros((len(tilex), 4))
        tiley2 = np.zeros((len(tilex), 8))
        for j in range(len(image_id)):
          tiley1[j] = full_labels[j]
          n = len(boxes[j])
          if n > 0:
            tiley2[j][1] = boxes[j][0]["x"] / full_dimension[1]
            tiley2[j][0] = boxes[j][0]["y"] / full_dimension[0]
            tiley2[j][3] = (boxes[j][0]["x"]+boxes[j][0]["width"])  / full_dimension[1]
            tiley2[j][2] = (boxes[j][0]["y"]+boxes[j][0]["height"]) / full_dimension[0]
          if n > 1:
            tiley2[j][5] = boxes[j][1]["x"] / full_dimension[1]
            tiley2[j][4] = boxes[j][1]["y"] / full_dimension[0]
            tiley2[j][7] = (boxes[j][1]["x"]+boxes[j][1]["width"]) / full_dimension[1]
            tiley2[j][6] = (boxes[j][1]["y"]+boxes[j][1]["height"]) / full_dimension[0]
          if any(tiley2[j] > 1):
            print(f"ERROR:{image_id[j]}: wrong normalization")
            if n > 0: print(f"  {boxes[j][0]}")
            if n > 1: print(f"  {boxes[j][1]}")
            print(f"->{tiley2[j]}")
            print(f"  {tiley1[j]}")
        return tilex, tiley1, tiley2

    def sample_ids(self, id_count, balance_classes=True):
        data = self.data

        # scan all not negative images for missing boxes
        data_not_negative = data[data["negative"] != 1]
        blacklist = data_not_negative[data_not_negative["boxes"].str.len() == 0]["image_id"]

        # delete the rows containing missing boxes from the dataframe (~300)
        data_cleaned = data[~data["image_id"].isin(blacklist)]

        sampled_ids = []
        if balance_classes:
            count_per_class = int(id_count / 4)
            for j in data.columns.values[5:9]:
                sampled_ids.extend(list(data_cleaned['image_id'][data[j] == 1].sample(n=count_per_class)))
            return sampled_ids

        else:
            return list(data_cleaned['image_id'].sample(n=id_count))
