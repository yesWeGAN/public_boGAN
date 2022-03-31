import os
import cv2


def resize_and_pad_image(image_path, MAXIMUM_DIM, outputdir):
    """resizes an image to ( MAX_DIM x MAX_DIM ).
    input: path. outputs JPEG image to folder.
    returns None"""
    og_size = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    try:
        resized = resize_img_to_dim_square(og_size, MAXIMUM_DIM)
        new_filepath = os.path.join(outputdir, image_path.split('/')[-1].split(".")[0])
        cv2.imwrite(new_filepath+".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    except AssertionError:
        print("file dim is off on file:", image_path)
    return None


def resize_img_to_dim_square(og_size, desired_dim):
    """input: image and a dimension.
    returns image padded to desired dim.
    throws AssertionError if dim is off!"""
    height = og_size.shape[0]
    width = og_size.shape[1]
    pad_horizontal = False
    pad_vertical = False
    needed_padding = None

    # the bigger dimension gets set to the desired dim
    if height > width:
        new_height = desired_dim
        new_width = int(width * (desired_dim / height))
        if new_width != desired_dim:
            needed_padding = desired_dim - new_width
            pad_horizontal = True

    elif height == width:
        new_height = desired_dim
        new_width = desired_dim

    else:
        new_width = desired_dim
        new_height = int(height * (desired_dim / width))
        if new_height != desired_dim:
            needed_padding = desired_dim - new_height
            pad_vertical = True

    new_shape = (new_width, new_height)

    resized = cv2.resize(og_size, new_shape, interpolation=cv2.INTER_NEAREST)

    if pad_horizontal:
        resized = cv2.copyMakeBorder(resized, 0, 0, needed_padding, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if pad_vertical:
        resized = cv2.copyMakeBorder(resized, needed_padding, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    resized = cv2.resize(resized, (desired_dim, desired_dim), interpolation=cv2.INTER_NEAREST)

    assert resized.shape[0] == desired_dim, "dim is off"
    assert resized.shape[1] == desired_dim, "dim is off"
    return resized


def resize_img_ar_preserve(og_size, desired_dim):
    """input: image and a dimension.
        returns image with max-dim [desired_dim] and preserved aspect-ratio.
        """

    # cv2 is in width, height
    KEEP_ASPECT_RATIO = desired_dim / max(og_size.shape)
    new_height = int(round(og_size.shape[0] * KEEP_ASPECT_RATIO, 0))
    new_width = int(round(og_size.shape[1] * KEEP_ASPECT_RATIO, 0))
    new_shape = (new_width, new_height)

    resized = cv2.resize(og_size, new_shape, interpolation=cv2.INTER_NEAREST)

    assert max(resized.shape) == desired_dim, "dim is off"
    return resized

