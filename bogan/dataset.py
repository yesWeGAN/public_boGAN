import json
import os
import pickle
import warnings
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from datetime import datetime
import label_inference
# Ignore warnings
warnings.filterwarnings("ignore")


class ShopDataset(Dataset):
    """Shop Store dataset."""

    def __init__(self, csv_file, root_dir, transform=None, verbose=False, build_df_from_raw=False,
                 raw_path="/home/frank/tame/raw"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        Options:
            build_df_from_raw: provide path of raw dataset, build df from raw.
        """
        if build_df_from_raw:
            self.build_df_from_raw(raw_path)
        else:
            self.df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.raw_path = raw_path

        self.img_dir = os.path.join(self.root_dir, "1024px_img")
        self.heatmask_dir = os.path.join(self.root_dir, "1024px_heatmask")
        self.binmask_dir = os.path.join(self.root_dir, "1024px_binmask")

        self.small_img_dir = os.path.join(self.root_dir, "256px_img")
        self.small_heatmask_dir = os.path.join(self.root_dir, "256px_heatmask")
        self.small_binmask_dir = os.path.join(self.root_dir, "256px_binmask")
        self.reldense_dir = os.path.join(self.root_dir, "density_rel")
        self.totdense_dir = os.path.join(self.root_dir, "density_total")

        self.verbose = verbose
        self.dummy = os.path.join(self.root_dir, "img")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, big=False):

        # this might make it necessary to adjust this method for multiple items
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if big:
            img_path = self.get_1k_img_path(idx)
        else:
            img_path = self.get_256_img_path(idx)
        image = io.imread(img_path)
        brand = self.df.loc[idx, 'brand']
        categ = self.df.loc[idx, 'category']
        color = self.df.loc[idx, 'color']
        plus = self.df.loc[idx, 'plus']
        gender = self.df.loc[idx, 'gender']
        sample = {'image': image, 'brand': brand, 'category': categ, 'color': color, 'plus': plus, gender: 'gender'}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def store_self(self, storagepath):
        """store the class to output directory"""
        datestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.df.to_csv(os.path.join(storagepath, datestamp + "_dataframe.csv"))

    def build_df_from_raw(self, raw_input_folder):
        """if building from raw-folder, parse store-json-files and concat to self.df"""
        self.df = pd.DataFrame(
            data={'brand': [], 'category': [], 'color': [], 'product_id': [], 'img_id': [],
                  'raw_path': [], 'file': [], 'plus': [], 'gender': []})
        folders = label_inference.get_directory_list(raw_input_folder)
        for folder in folders:
            frame = label_inference.parse_store(os.path.join(raw_input_folder, folder))
            self.df = pd.concat([self.df, frame], ignore_index=True)
        pass

    def show_image_by_idx(self, idx, item="img", big=True):
        """shows element with index idx in notebook
        input: index, item from image, heatmask, binmask (default=image)"""
        img = None

        if item == "img":
            if big:
                img = mpimg.imread(self.get_1k_img_path(idx))
            else:
                img = mpimg.imread(self.get_256_img_path(idx))

        elif item == "heatmask":
            if big:
                img = mpimg.imread(self.get_1k_mask_path(idx, heatmask=True))
            else:
                img = mpimg.imread(self.get_256_mask_path(idx, heatmask=True))

        elif item == "binmask":
            if big:
                img = mpimg.imread(self.get_1k_mask_path(idx, heatmask=True))
            else:
                img = mpimg.imread(self.get_256_mask_path(idx, heatmask=True))

        plt.imshow(img)
        plt.show()
        pass

    def get_raw_img_path(self, idx):
        """returns the path to the raw, original sized image for given idx"""
        return os.path.join(self.raw_path, self.df.loc[idx, 'file'] + ".jpg")

    def get_1k_img_path(self, idx):
        """returns the path to the (1024xY) sized image for given idx. image not padded"""
        return os.path.join(self.img_dir, self.df.loc[idx, 'file'] + ".jpg")

    def get_256_img_path(self, idx):
        """returns the path to the (256x256) sized image for given idx"""
        return os.path.join(self.small_img_dir, self.df.loc[idx, 'file'] + ".jpg")

    def get_1k_mask_path(self, idx, heatmask=False):
        """returns the path to the (1024xY) sized mask for given idx. default: binary mask. mask not padded"""
        if heatmask:
            return os.path.join(self.heatmask_dir, self.df.loc[idx, 'file'] + ".png")
        else:
            return os.path.join(self.binmask_dir, self.df.loc[idx, 'file'] + ".png")

    def get_256_mask_path(self, idx, heatmask=False):
        """returns the path to the (256 x 256) sized mask for given idx. default: binary mask. padded"""
        if heatmask:
            return os.path.join(self.small_heatmask_dir, self.df.loc[idx, 'file'] + ".png")
        else:
            return os.path.join(self.small_binmask_dir, self.df.loc[idx, 'file'] + ".png")

    def get_json_filepath(self, idx):
        """returns the json-filepath for given idx"""
        return os.path.join(self.raw_path, self.df.loc[idx, 'raw_path'].split('/')[0]
                            + "/" + self.df.loc[idx, 'raw_path'].split('/')[0] + ".json")

    def get_basename(self, idx):
        """returns the foldername/basename for given idx"""
        return self.df.loc[idx, 'raw_path'].split('/')[-3]

    def get_json_product(self, idx):
        """returns the json-content for the given idx"""
        with open(self.get_json_filepath(idx)) as jsonin:
            data = json.load(jsonin)
        for product in data["products"]:
            if product["id"] == self.df.loc[idx, 'product_id']:
                return product
        return None

    def get_json_attr(self, idx, attribute="all"):
        """get specific json-content for given idx
        input:    idx, attribute (default=all)
        returns:  dict(dict)"""

        attributes = {"html": "body_html", "title": "title", "store": "vendor",
                      "vendor": "vendor", "tags": "tags", "img": "images", "images": "images"}

        product = self.get_json_product(idx)

        if attribute == "all":
            return product
        else:
            return product[attributes.get(attribute)]

    def get_frame_subset(self, selector_dict):
        """get a subset of the dataframe (deepcopy, does not affect original frame)
        inputs:    dict (key in [category, color, brand, plus, img_id, raw_path, file, product_id, gender]
        returns:   pd.Dataframe """
        copy = self.df.copy(deep=True)
        for key, value in selector_dict.items():
            copy = copy[copy[key] == value]
        return copy

    def get_attr_for_idx(self, idx, attribute):
        """fetch attribute for idx
           input: idx, attribute
           returns: the value of attribute in the dataframe
           optional: shortcuts for attributes: cat:category, col:color, b:brand, p:plus 
           """
        shortcuts = {"cat": "category", "col": "color", "b": "brand", "p": "plus", "id": "product_id"}

        if attribute in shortcuts.keys():
            return self.df.loc[idx, shortcuts.get(attribute)]
        else:
            return self.df.loc[idx, attribute]

    def show_all_product_images_by_idx(self, idx, big=True):
        """prints all images matching product_id of idx
           inputs: idx
           returns: None
           """
        product_id = self.get_attr_for_idx(idx, "product_id")
        product_images = self.df[self.df.product_id == product_id].index.to_list()
        for image in product_images:
            if big:
                img = mpimg.imread(self.get_1k_img_path(image))
            else:
                img = mpimg.imread(self.get_256_img_path(image))
            plt.imshow(img)
            plt.show()

    def stacked_product_images_by_idx(self, idx, padding_token=255):
        """ get all images as numpy arrays
            input: idx
            optional: padding_token
            returns: np.array of shape (N, 1024, 1024, 3)
            images padded with padding_token (default 255)"""

        product_id = self.get_attr_for_idx(idx, "product_id")

        product_images = self.df[self.df.product_id == product_id].filename.to_list()

        collection = []

        for image in product_images:
            # this is the dummy version for testing
            img = mpimg.imread(os.path.join(self.dummy, image + ".jpg"))
            collection.append(img)

        # I NEED TO IMPLEMENT SOME PADDING FOR MISSING IMAGE DIMENSIONS!
        return np.stack(collection)

    def get_relative_dense_vector(self, idx):
        """returns the relative DensePose-vector for the given idx"""
        with open(os.path.join(self.reldense_dir, self.df.loc[idx, 'file'] + ".pkl")) as picklein:
            return pickle.load(picklein)

    def get_total_dense_vector(self, idx):
        """returns the total DensePose-vector for the given idx"""
        with open(os.path.join(self.totdense_dir, self.df.loc[idx, 'file'] + ".pkl")) as picklein:
            return pickle.load(picklein)
