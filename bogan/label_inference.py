import os

from categorymappings import categorymapping
from colormappings import colormapping
import re
import json
import pandas as pd


def get_json(json_basepath):
    """returns the json-content for store basename"""
    try:
        with open(json_basepath) as json_in:
            data = json.load(json_in)
    except FileNotFoundError as e:
        print(e)
        print("Exception occured handling:", json_basepath)
    return data


def infer_color(product):
    """returns the most likely color of product"""
    for color in colormapping.keys():
        if (re.search(color, product['title'], flags=re.IGNORECASE)) is not None:
            return colormapping[color]

    for tag in product['tags']:
        for color in colormapping.keys():
            if (re.search(color, tag, flags=re.IGNORECASE)) is not None:
                return colormapping[color]
    return "other"


def infer_category(product):
    """returns the most likely color of product"""
    for cat in categorymapping.keys():
        if (re.search(cat, product['title'], flags=re.IGNORECASE)) is not None:
            return categorymapping[cat]

    for tag in product['tags']:
        for cat in categorymapping.keys():
            if (re.search(cat, tag, flags=re.IGNORECASE)) is not None:
                return categorymapping[cat]
    return "other"


def infer_plus(product):
    """parse the json file for plus-size items"""
    if (re.search("plus", product['title'], flags=re.IGNORECASE)) is not None:
        return True
    for tag in product['tags']:
        if (re.search("plus", tag, flags=re.IGNORECASE)) is not None:
            return True
    return False


def parse_store(storepath):
    """build a dataframe for a store.
    input:      path to store, structure ./img/, ./json/{storename}.json
    returns:    dataframe holding information, add to self.df via pd.concat[[self.df, returndf]]"""

    img_suf = "img"
    df = pd.DataFrame(
        data={'brand': [], 'category': [], 'color': [], 'product_id': [], 'img_id': [],
              'raw_path': [], 'file': [], 'plus': []})

    imagefolder_listing = os.listdir(os.path.join(storepath, img_suf))
    jay = get_json(os.path.join(storepath, storepath.split('/')[-1] + ".json"))
    for product in jay["products"]:
        color = infer_color(product)
        categ = infer_category(product)
        gender = infer_gender(product['tags'])
        plus = infer_plus(product)      # some plus labels are in the json file
        image_dict, plus_dict = get_imagelist(product, imagefolder_listing)  # some in img-filename
        if image_dict is None:
            image_dict = {}
        for imageid, imagepath in image_dict.items():
            plus_image = bool(plus_dict[imageid])
            plus_image = (plus_image or plus)
            s_row = pd.DataFrame(data={'brand': product['vendor'], 'category': categ, 'color': color,
                                       'product_id': str(product['id']), 'img_id': str(imageid),
                                       'raw_path': storepath.split('/')[-1] + "/img/",
                                       'file': str(product['id'])+"_"+str(imageid), 'plus': plus_image,
                                       'gender': gender}, index=[0])
            df = pd.concat([df, s_row], ignore_index=True)
    return df


def get_imagelist(product, img_dirlist):
    """returns list of all image-paths from a json-product-file, as well as respective plus-size-images"""
    found = False
    image_list = {}
    plus_list = {}

    for image in product['images']:
        # get plus size images
        if ("PLUS" in image['src']) | ("Plus" in image['src']) | ("plus" in image['src']):
            plus_list[image['id']] = True
        else:
            plus_list[image['id']] = False

        for file in img_dirlist:
            if str(image['id']) in file:
                image_list[image['id']] = file
                found = True
    if found:
        return image_list, plus_list
    else:
        return None, None


def get_directory_list(path):
    """returns absolute paths for all folders in dir"""
    dir_list = []
    for fold in os.listdir(path):
        if os.path.isdir(os.path.join(path, fold)):
            dir_list.append(os.path.join(path, fold))
    return dir_list

