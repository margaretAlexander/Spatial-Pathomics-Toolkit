import argparse
from PIL import Image, ImageFilter
import skimage
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as scind
import scipy.sparse
import centrosome.cpmorphology
import pandas as pd
import feret
import centrosome.zernike
import centrosome.threshold

ZERNIKE_N = 9
SIZE_RANGE = [10, 200]
TH_SMOOTH = 0.5
TH_CORRECT = 1.0
TH_MIN = 0.0
TH_MAX = 1.0
SIZE_ADAPT_WINDOW = 50

# Identify foreground objects in image, must be grayscale
# Returns labeled array and number of objects
def label_objects(grayscale, threshold):
    binary_image = grayscale.point(lambda x: 255 if x > threshold else 0)
    binary_image = binary_image.filter(ImageFilter.MinFilter)  # remove noise around main objects
    binary_image = binary_image.filter(ImageFilter.MaxFilter)  # replace any area that was removed in previous step

    s = scind.generate_binary_structure(2, 2)
    labeled_image, object_count = scind.label(binary_image, structure=s)

    labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)  # fill any holes in labeled objects

    return labeled_image, object_count

def get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
    if image_resize_factor < 1.0:
        shape = np.array(image.shape) * image_resize_factor
        i_j = (
                np.mgrid[0: shape[0], 0: shape[1]].astype(float)
                / image_resize_factor
        )
        resized_image = scind.map_coordinates(image, i_j)
        resized_labels = scind.map_coordinates(
            labeled_image, i_j, order=0
        ).astype(labeled_image.dtype)

    else:
        resized_image = image
        resized_labels = labeled_image
    #
    # find local maxima
    #
    if maxima_mask is not None:
        binary_maxima_image = centrosome.cpmorphology.is_local_maximum(
            resized_image, resized_labels, maxima_mask
        )
        binary_maxima_image[resized_image <= 0] = 0
    else:
        binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
    if image_resize_factor < 1.0:
        inverse_resize_factor = float(image.shape[0]) / float(
            binary_maxima_image.shape[0]
        )
        i_j = (
                np.mgrid[0: image.shape[0], 0: image.shape[1]].astype(float)
                / inverse_resize_factor
        )
        binary_maxima_image = (
                scind.map_coordinates(binary_maxima_image.astype(float), i_j)
                > 0.5
        )
        assert binary_maxima_image.shape[0] == image.shape[0]
        assert binary_maxima_image.shape[1] == image.shape[1]

    # Erode blobs of touching maxima to a single point

    shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
    return shrunk_image

def separate_neighboring_objects(image, labeled_image):
    """Separate objects based on local maxima or distance transform

            labeled_image - image labeled by scind.label

            object_count  - # of objects in image

            returns revised labeled_image, object count, maxima_suppression_size,
            LoG threshold and filter diameter
            """

    if SIZE_RANGE[0] > 10:
        image_resize_factor = 10.0 / float(SIZE_RANGE[0])
        maxima_suppression_size = 7

        reported_maxima_suppression_size = (
                maxima_suppression_size / image_resize_factor
        )
    else:
        image_resize_factor = 1.0
        maxima_suppression_size = SIZE_RANGE[0] / 1.5

        reported_maxima_suppression_size = maxima_suppression_size
    maxima_mask = centrosome.cpmorphology.strel_disk(
        max(1, maxima_suppression_size - 0.5)
    )

    foreground = labeled_image > 0
    distance_transformed_image = scind.distance_transform_edt(
        foreground
    )
    # randomize the distance slightly to get unique maxima
    np.random.seed(0)
    distance_transformed_image += np.random.uniform(
        0, 0.001, distance_transformed_image.shape
    )
    maxima_image = self.get_maxima(
        distance_transformed_image,
        labeled_image,
        maxima_mask,
        image_resize_factor,
    )

    # Create the image for watershed
    watershed_image = 1 - image

    #
    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    labeled_maxima, object_count = scind.label(
        maxima_image, np.ones((3, 3), bool)
    )

    markers_dtype = (
        np.int16
        if object_count < np.iinfo(np.int16).max
        else np.int32
    )
    markers = np.zeros(watershed_image.shape, markers_dtype)
    markers[labeled_maxima > 0] = -labeled_maxima[
        labeled_maxima > 0
        ]

    #
    # Some labels have only one maker in them, some have multiple and
    # will be split up.
    #

    watershed_boundaries = skimage.segmentation.watershed(
        connectivity=np.ones((3, 3), bool),
        image=watershed_image,
        markers=markers,
        mask=labeled_image != 0,
    )

    watershed_boundaries = -watershed_boundaries
    
    return watershed_boundaries, object_count, reported_maxima_suppression_size

def filter_on_size(labeled_image, object_count):
    """ Filter the labeled image based on the size range
    labeled_image - pixel image labels
    object_count - # of objects in the labeled image
    returns the labeled image, and the labeled image with the
    small objects removed
    """
    if object_count > 0:
        areas = scind.measurements.sum(
            np.ones(labeled_image.shape),
            labeled_image,
            np.array(list(range(0, object_count + 1)), dtype=np.int32),
        )
        areas = np.array(areas, dtype=int)
        min_allowed_area = (
                np.pi * (SIZE_RANGE[0] * SIZE_RANGE[0]) / 4
        )
        max_allowed_area = (
                np.pi * (SIZE_RANGE[1] * SIZE_RANGE[1]) / 4
        )
        # area_image has the area of the object at every pixel within the object
        area_image = areas[labeled_image]
        labeled_image[area_image < min_allowed_area] = 0
        small_removed_labels = labeled_image.copy()
        labeled_image[area_image > max_allowed_area] = 0
    else:
        small_removed_labels = labeled_image.copy()
        
    return labeled_image, small_removed_labels

# for implementing zernike features
def zernike(labels, object_count):
    """The Zernike numbers measured by this module"""
    zernike_numbers = centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)
    inds = range(1, object_count + 1)

    zf = {}
    for n, m in zernike_numbers:
        zf[(n, m)] = np.zeros(object_count)

    zf_l = centrosome.zernike.zernike(
        zernike_numbers, labels, inds
    )
    for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
        zf[(n, m)] = z


# measure objects' shape and size
# Add in Zernike features
def measure_objects_sizeshape(labeled_image, object_count, desired_props):
    props = skimage.measure.regionprops_table(labeled_image, properties=desired_props)

    formfactor = 4.0 * np.pi * props["area"] / props["perimeter"] ** 2
    denom = [max(x, 1) for x in 4.0 * np.pi * props["area"]]
    compactness = props["perimeter"] ** 2 / denom

    max_radius = np.zeros(object_count)
    median_radius = np.zeros(object_count)
    mean_radius = np.zeros(object_count)
    min_feret_diameter = np.zeros(object_count)
    max_feret_diameter = np.zeros(object_count)

    for index, mini_image in enumerate(props["image"]):
        # Pad image to assist distance tranform
        mini_image = np.pad(mini_image, 1)
        distances = scind.distance_transform_edt(mini_image)
        max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scind.maximum(distances, mini_image)
        )
        mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scind.mean(distances, mini_image)
        )
        median_radius[index] = centrosome.cpmorphology.median_of_labels(
            distances, mini_image.astype("int"), [1]
        )

    temp_arrays = []
    for i in range(1, object_count + 1):
        temp_arrays.append(np.zeros(labeled_image.shape))
        for j in range(labeled_image.shape[0]):
            for k in range(labeled_image.shape[1]):
                if labeled_image[j][k] == i:
                    temp_arrays[i - 1][j][k] = 1
        max_feret_diameter[i - 1] = feret.max(temp_arrays[i - 1])
        min_feret_diameter[i - 1] = feret.min(temp_arrays[i - 1])

    features_to_record = {
        "Area": props["area"],
        "Perimeter": props["perimeter"],
        "MajorAxisLength": props["major_axis_length"],
        "MinorAxisLength": props["minor_axis_length"],
        "Eccentricity": props["eccentricity"],
        "Orientation": props["orientation"] * (180 / np.pi),
        "Center_X": props["centroid-1"],
        "Center_Y": props["centroid-0"],
        "BoundingBoxArea": props["bbox_area"],
        "BoundingBoxMinimumX": props["bbox-1"],
        "BoundingBoxMaximumX": props["bbox-3"],
        "BoundingBoxMinimumY": props["bbox-0"],
        "BoundingBoxMaximumY": props["bbox-2"],
        "FormFactor": formfactor,
        "Extent": props["extent"],
        "Soliditiy": props["solidity"],
        "Compactness": compactness,
        "EulerNumber": props["euler_number"],
        "MaximumRadius": max_radius,
        "MeanRadius": mean_radius,
        "MedianRadius": median_radius,
        "ConvexArea": props["convex_area"],
        "MinFeretDiameter": min_feret_diameter,
        "MaxFeretDiameter": max_feret_diameter,
        "EquivalentDiameter": props["equivalent_diameter"],
        "SpatialMoment_0_0": props["moments-0-0"],
        "SpatialMoment_0_1": props["moments-0-1"],
        "SpatialMoment_0_2": props["moments-0-2"],
        "SpatialMoment_0_3": props["moments-0-3"],
        "SpatialMoment_1_0": props["moments-1-0"],
        "SpatialMoment_1_1": props["moments-1-1"],
        "SpatialMoment_1_2": props["moments-1-2"],
        "SpatialMoment_1_3": props["moments-1-3"],
        "SpatialMoment_2_0": props["moments-2-0"],
        "SpatialMoment_2_1": props["moments-2-1"],
        "SpatialMoment_2_2": props["moments-2-2"],
        "SpatialMoment_2_3": props["moments-2-3"],
        "CentralMoment_0_0": props["moments_central-0-0"],
        "CentralMoment_0_1": props["moments_central-0-1"],
        "CentralMoment_0_2": props["moments_central-0-2"],
        "CentralMoment_0_3": props["moments_central-0-3"],
        "CentralMoment_1_0": props["moments_central-1-0"],
        "CentralMoment_1_1": props["moments_central-1-1"],
        "CentralMoment_1_2": props["moments_central-1-2"],
        "CentralMoment_1_3": props["moments_central-1-3"],
        "CentralMoment_2_0": props["moments_central-2-0"],
        "CentralMoment_2_1": props["moments_central-2-1"],
        "CentralMoment_2_2": props["moments_central-2-2"],
        "CentralMoment_2_3": props["moments_central-2-3"],
        "NormalizedMoment_0_0": props["moments_normalized-0-0"],
        "NormalizedMoment_0_1": props["moments_normalized-0-1"],
        "NormalizedMoment_0_2": props["moments_normalized-0-2"],
        "NormalizedMoment_0_3": props["moments_normalized-0-3"],
        "NormalizedMoment_1_0": props["moments_normalized-1-0"],
        "NormalizedMoment_1_1": props["moments_normalized-1-1"],
        "NormalizedMoment_1_2": props["moments_normalized-1-2"],
        "NormalizedMoment_1_3": props["moments_normalized-1-3"],
        "NormalizedMoment_2_0": props["moments_normalized-2-0"],
        "NormalizedMoment_2_1": props["moments_normalized-2-1"],
        "NormalizedMoment_2_2": props["moments_normalized-2-2"],
        "NormalizedMoment_2_3": props["moments_normalized-2-3"],
        "NormalizedMoment_3_0": props["moments_normalized-3-0"],
        "NormalizedMoment_3_1": props["moments_normalized-3-1"],
        "NormalizedMoment_3_2": props["moments_normalized-3-2"],
        "NormalizedMoment_3_3": props["moments_normalized-3-3"],
        "HuMoment_0": props["moments_hu-0"],
        "HuMoment_1": props["moments_hu-1"],
        "HuMoment_2": props["moments_hu-2"],
        "HuMoment_3": props["moments_hu-3"],
        "HuMoment_4": props["moments_hu-4"],
        "HuMoment_5": props["moments_hu-5"],
        "HuMoment_6": props["moments_hu-6"],
        "InertiaTensor_0_0": props["inertia_tensor-0-0"],
        "InertiaTensor_0_1": props["inertia_tensor-0-1"],
        "InertiaTensor_1_0": props["inertia_tensor-1-0"],
        "InertiaTensor_1_1": props["inertia_tensor-1-1"],
        "InertiaTensorEigenvalues_0": props["inertia_tensor_eigvals-0"],
        "InertiaTensorEigenvalues_1": props["inertia_tensor_eigvals-1"],
    }

    return features_to_record


# Measure intensity of objects
def measure_object_intensity(image, labeled_image, object_count):
    pixel_vals = np.asarray(image)
    inds = range(1, object_count + 1)

    integrated_intensity = np.zeros((object_count,))  # The sum of the pixel intensities within an object
    mean_intensity = np.zeros((object_count,))  # The average pixel intensity within an object.

    # needs to be implemented
    std_intensity = np.zeros((object_count,))  # The standard deviation of the pixel intensities within an object

    for i in inds:
        integrated_intensity[
            i - 1
            ] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scind.sum(pixel_vals, labeled_image, i)
        )

        lcount = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scind.sum(np.ones(len(pixel_vals)), labeled_image, i)
        )

        mean_intensity[i - 1] = (
                integrated_intensity[i - 1] / lcount
        )

    features_to_record = {
        "IntegratedIntesity": integrated_intensity,
        "MeanIntensity": mean_intensity
    }

    return features_to_record

"""
To implement:

def measure_texture():

def measure_object_intensity_distribution():
"""


parser = argparse.ArgumentParser(description='Process mask image')
parser.add_argument('mask_path', type=str, help='Path to the mask image file')
args = parser.parse_args()

mask_image = Image.open(args.mask_path)
grayscale = mask_image.convert('L')

local_threshold, global_threshold = get_threshold(TM_OTSU,
                                                  TS_ADAPTIVE,
                                                  grayscale,
                                                  threshold_range_min=TH_MIN,
                                                  threshold_range_max=TH_MAX,
                                                  threshold_correction_factor=TH_CORRECT,
                                                  adaptive_window_size=SIZE_ADAPT_WINDOW
                                                  )

labeled_image, object_count = label_objects(grayscale, global_threshold)

(
            labeled_image,
            object_count,
            maxima_suppression_size,
        ) = separate_neighboring_objects(grayscale, labeled_image)

unedited_labels = labeled_image.copy()

# Filter out small and large objects
size_excluded_labeled_image = labeled_image.copy()
labeled_image, small_removed_labels = filter_on_size(
    labeled_image, object_count
)
size_excluded_labeled_image[labeled_image > 0] = 0

#
# Fill holes again after watershed
#
labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)

# Relabel the image
labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)

# Make an outline image
outline_image = centrosome.outline.outline(labeled_image)
outline_size_excluded_image = centrosome.outline.outline(
    size_excluded_labeled_image
)


desired_props = [
    "label",
    "image",
    "area",
    "perimeter",
    "bbox",
    "bbox_area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "centroid",
    "equivalent_diameter",
    "extent",
    "eccentricity",
    "convex_area",
    "solidity",
    "euler_number",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "moments",
    "moments_central",
    "moments_hu",
    "moments_normalized",
]

sizeshape = measure_objects_sizeshape(labeled_image, object_count, desired_props)
intensity = measure_object_intensity(grayscale, labeled_image, object_count)
measurements = sizeshape.update(intensity)
measurements_df = pd.DataFrame(measurements)
measurements_df.to_csv('../output/Measurements.csv')
