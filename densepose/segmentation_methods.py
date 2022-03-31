# methods based on Facebook AI Research (FAIR) DensePose project

import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import pickle
import os


class SegmentationHSC:
    """class to build segmentations using human skintone characteristics (HSC)
    input:      DensePose I(UV)-Tensoroutput
    output:     segmentation masks, overlays, contours
    flags:
    SHOW:       show plt inline (for jupyter notebook)
    WRITE:      write masks, images to file
    HEADLESS:   hair does not follow HSC, default excluded in segmentation step
    kernel:     cv2 opening/closing kernel
    ITERS:       cv2 param for erosion / opening masks"""

    def __init__(self, verbosity=False, show=False, headless=True, kernel=None, write=False, processsteps=False,
                 iters=0, outputpath="out"):
        self.VERBOSE = verbosity
        # fine-grained adjustment what to show / save can be done later
        if show:
            self.SHOW_IMG = True
            self.SHOW_MASK = True
            self.SHOW_OVERLAY = True
        else:
            self.SHOW_IMG = False
            self.SHOW_MASK = False
            self.SHOW_OVERLAY = False
        if write:
            self.WRITE_IMG = True
            self.WRITE_MASK = True
            self.WRITE_OVERLAY = True
        else:
            self.WRITE_IMG = False
            self.WRITE_MASK = False
            self.WRITE_OVERLAY = False

        self.HEADLESS = headless
        self.kernel = kernel
        self.SHOW_PROCESS = processsteps
        self.ITERS = iters
        self.outputpath = outputpath
        self.maskdir_path = os.path.join(self.outputpath, "masks")
        self.overlaydir_path = os.path.join(self.outputpath, "overlay")
        self.image_dir = os.path.join(self.outputpath, "image")

    def mask_person(self, dp_pickle_path: str):
        """input: path to I-tensor pickle file (pre-processed for faster batch inference)"""

        with open(dp_pickle_path, 'rb') as pickled:
            detection_results = pickle.load(pickled)

        for i in range(0, len(detection_results)):
            result = detection_results[i]
            outname_prefix = result["file_name"]
            outname = outname_prefix.split('/')[-1]
            outname_prefix = outname_prefix.split('.')[0]

            original = cv2.imread(result["file_name"])
            original_height, original_width, channels = original.shape

            if self.VERBOSE:
                print("Original shape was", original.shape, "Entering padding mode for", result["file_name"])

            person_count = 0

            for k in range(0, len(result["scores"])):
                if result["scores"][k] > 0.95:
                    person_count += 1

                    if self.VERBOSE:
                        print("Now masking person", person_count)

                    # tensor unpickled as CUDA tensor
                    person_tensor = result["pred_densepose"][k].labels.cpu().numpy()

                    # increase mask intensity to 255 (WHITE)
                    intensify = np.where(person_tensor > 0, 255, 0).astype(np.uint8)

                    # padding section (only if padding=True)
                    # Syntax: cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
                    boundingbox = result["pred_boxes_XYXY"][k].numpy().astype(np.int64)
                    intensify = cv2.copyMakeBorder(intensify, boundingbox[1], original_height - boundingbox[3],
                                                   boundingbox[0], original_width - boundingbox[2], cv2.BORDER_CONSTANT,
                                                   value=0)

                    if intensify.shape != original.shape[:2]:
                        if self.VERBOSE:
                            print("I need to adjust this mask, sized", intensify.shape)
                        intensify = cv2.resize(intensify, (original_width, original_height))

                    person_mask = cv2.cvtColor(intensify, cv2.COLOR_GRAY2BGR)

                    img_person_masked = cv2.subtract(person_mask, original)
                    img_person_masked = cv2.subtract(person_mask, img_person_masked)

                    if self.SHOW_IMG:
                        img_person_masked = cv2.cvtColor(img_person_masked, cv2.COLOR_BGR2RGB)
                        plt.imshow(img_person_masked)
                        plt.show()

                    # headless mode only operates on torso and creates additional mask
                    if self.HEADLESS:
                        headless_tensor = np.array(person_tensor, copy=True)

                        # HEADLESS LINE + FILTERING OF HANDS 3+4, FEET 5+6, LOWER ARMS/LEGS 11-14, 19-22
                        headless_tensor = np.where(headless_tensor > 18, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 3, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 4, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 5, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 6, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 11, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 12, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 13, 0, headless_tensor)
                        headless_tensor = np.where(headless_tensor == 14, 0, headless_tensor)
                        # increase mask intensity to 255 (WHITE)
                        headless_tensor = np.where(headless_tensor > 0, 255, 0).astype(np.uint8)
                        # padding section (only if padding=True)
                        headless_tensor = cv2.copyMakeBorder(headless_tensor, boundingbox[1],
                                                             original_height - boundingbox[3],
                                                             boundingbox[0], original_width - boundingbox[2],
                                                             cv2.BORDER_CONSTANT, value=0)

                        if headless_tensor.shape != original.shape[:2]:
                            if self.VERBOSE:
                                print("Adjusting this mask, sized", headless_tensor.shape)
                            headless_tensor = cv2.resize(headless_tensor, (original_width, original_height))
                        headless_mask = cv2.cvtColor(headless_tensor, cv2.COLOR_GRAY2BGR)

                        # erode the mask for extraction of clothing (reflective skin portions cause noise in the masks)
                        headless_mask = cv2.erode(headless_mask, self.kernel, iterations=self.iters)

                        img_headless_masked = cv2.subtract(headless_mask, original)
                        img_headless_masked = cv2.subtract(headless_mask, img_headless_masked)

                        # Using HSC characteristic R>G>B to extract skin
                        hsc_detect_rg = np.where(img_headless_masked[:, :, 2] > img_headless_masked[:, :, 1], True,
                                                 False)
                        hsc_detect_gb = np.where(img_headless_masked[:, :, 1] > img_headless_masked[:, :, 0], True,
                                                 False)
                        hsc_detection = hsc_detect_gb & hsc_detect_rg
                        hsc_detection = np.where(hsc_detection == True, 0, 255).astype(np.uint8)
                        hsc_detection = cv2.cvtColor(hsc_detection, cv2.COLOR_GRAY2BGR)
                        hsc_out = cv2.subtract(hsc_detection, img_headless_masked)
                        hsc_out = cv2.subtract(hsc_detection, hsc_out)

                        if self.SHOW_IMG:
                            hsc_out = cv2.cvtColor(hsc_out, cv2.COLOR_BGR2RGB)
                            plt.imshow(hsc_out)
                            plt.show()

                        headless_mask = np.zeros_like(original)
                        headless_mask = np.where(hsc_out[:, :, 0] > 0, 255, 0)
                        headless_mask = np.where(hsc_out[:, :, 1] > 0, 255, 0)
                        headless_mask = np.where(hsc_out[:, :, 2] > 0, 255, 0)

                        headless_mask = cv2.cvtColor(headless_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        if self.SHOW_PROCESS:
                            plt.imshow(headless_mask)
                            plt.show()

                        # filter by contour area to avoid patchy masks
                        mask = cv2.cvtColor(headless_mask, cv2.COLOR_BGR2GRAY)
                        img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:3]
                        drawing_pad = np.zeros_like(original)

                        for index, cont in enumerate(contours):
                            area = cv2.contourArea(cont)
                            if index > 1:
                                cv2.drawContours(drawing_pad, [cont], 0, (0, 0, 255), 10)
                            else:
                                cv2.drawContours(drawing_pad, [cont], 0, (0, 255, 0), 10)

                        if self.SHOW_CONTOUR:
                            plt.imshow(cv2.cvtColor(drawing_pad, cv2.COLOR_BGR2RGB))
                            plt.show()

                        if self.WRITE_IMG:
                            cv2.imwrite(outname_prefix + str(person_count) + "_masked.jpg",
                                        cv2.cvtColor(img_person_masked, cv2.COLOR_BGR2RGB))
                        if self.WRITE_MASK:
                            mask_filename = outname + "_PERSON_" + str(person_count) + "_MASK.png"
                            cv2.imwrite(os.path.join(self.maskdir_path, mask_filename), mask)
                        if self.WRITE_CONTOUR:
                            cv2.imwrite(outname_prefix + str(person_count) + "contour.png", drawing_pad)

                        overlay = cv2.addWeighted(original, 1.0, drawing_pad, 0.7, 0)

                        if self.SHOW_OVERLAY:
                            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                            plt.show()
                        if self.WRITE_OVERLAY:
                            overlay_filename = outname + "_PERSON_" + str(person_count) + "_OVERLAY.jpg"
                            cv2.imwrite(os.path.join(self.overlay_path, overlay_filename), overlay)
        return

    def process_directory(self, inputpath_pickle):
        """process all dp-tensors in directory with masking method above
        throws: Exception (caught within method, creates print)"""
        start_time = time.time()
        filelist = os.listdir(inputpath_pickle)
        for file in filelist:
            if os.path.isfile(os.path.join(inputpath_pickle, file)):
                try:
                    self.mask_person(os.path.join(inputpath_pickle, file))
                except Exception as err:
                    print("Error processing file", os.path.join(inputpath_pickle, file))
                    print(err)

            if os.path.isdir(os.path.join(inputpath_pickle, file)):
                continue
        end_time = time.time()
        if self.VERBOSE:
            print("Processing", len(filelist), "elements took", (end_time - start_time) * 1000, "ms")
        return

    def print_area_labels(self):
        # call the docstring to see the area labels
        """
        0      = Background
        1, 2   = Torso    1 BACKSIDE 2 FRONTSIDE
        3      = Right Hand MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
        4      = Left Hand MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
        5      = Right Foot MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
        6      = Left Foot MOSTLY USELESS AS ITS FRONT/BACK INSENSITIVE
        7, 9   = Upper Leg Right   7 BACKSIDE  9 FRONTSIDE
        8, 10  = Upper Leg Left    8 BACKSIDE  10 FRONTSIDE
        11, 13 = Lower Leg Right   11 BACKSIDE 13 FRONTSIDE
        12, 14 = Lower Leg Left    12 BACKSIDE 14 FRONTSIDE
        15, 17 = Upper Arm Left
        16, 18 = Upper Arm Right
        19, 21 = Lower Arm Left
        20, 22 = Lower Arm Right
        23, 24 = Head   23 RIGHT SIDE  24 LEFT SIDE ATTENTION ITS INSENSITIVE TO HAIR COVERING
        """
    pass

