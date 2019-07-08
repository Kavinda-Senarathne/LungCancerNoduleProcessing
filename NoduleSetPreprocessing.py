metadatapath = "./LIDC/LIDC-IDRI_MetaData.csv"
list32path = "./LIDC/list3.2.csv"
DOIfolderpath = './LIDC/LIDC-IDRI/'
datafolder = './processeddata'


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_fill_holes
from skimage import measure, morphology
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.draw import circle

# Load metadata
meta = pd.read_csv(metadatapath)
assert isinstance(meta.drop, object)
meta = meta.drop(meta[meta['Modality'] != 'CT'].index)
meta = meta.reset_index()

# Get folder names of CT data for each patient
patients = [DOIfolderpath + meta['Patient Id'][i] for i in range(len(meta))]
datfolder = []
# for i in range(0, len(meta) - 1):
#     for path in os.listdir(patients[i]):
#         if os.path.exists(patients[i] + meta['Series UID'][i]):
#             datfolder.append(patients[i] + meta['Series UID'][i])
datfolder = patients

# Load nodules locations
nodulelocations = pd.read_csv(list32path)


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s, force=True) for s in os.listdir(path) if s.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


# convert to ndarray
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def read_ct_scan(folder_name):
    # Read the slices from the dicom file
    slices = [pydicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]

    # Sort the dicom slices in their respective order
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # Get the pixel values for all the slices
    slices = np.stack([s.pixel_array for s in slices])
    slices[slices == -2000] = 0
    return slices


def segment_lung_mask(image, fill_lung_structures=True, dilate=False):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image = binary_image - 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    if dilate == 1:
        for i in range(binary_image.shape[0]):
            binary_image[i] = morphology.dilation(binary_image[i], np.ones([10, 10]))
    return binary_image


def sample_stack(stack, rows=7, cols=4):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 16])
    for i in range(rows * cols):
        ax[int(i / cols), int(i % cols)].set_title('slice %d' % i)
        ax[int(i / cols), int(i % cols)].imshow(stack[i], cmap='gray')
        ax[int(i / cols), int(i % cols)].axis('off')
    plt.show()


# Let's look at one of the patients

first_patient = load_scan(patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

sample_stack(first_patient_pixels)
