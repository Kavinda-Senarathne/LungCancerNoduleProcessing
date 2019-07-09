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
patients = [DOIfolderpath + meta['Patient Id'][i] for i in range(2)]
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


def coord_polar_to_cart(r, theta, center):
    '''Converts polar coordinates around center to Cartesian'''
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y


def coord_cart_to_polar(x, y, center):
    '''Converts Cartesian coordinates to polar'''
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    theta = np.arctan2((y - center[1]), (x - center[0]))
    return r, theta


def image_cart_to_polar(image, center, min_radius, max_radius, phase_width, zoom_factor=1):
    '''Converts an image from cartesian to polar coordinates around center'''

    # Upsample image
    if zoom_factor != 1:
        image = zoom(image, (zoom_factor, zoom_factor), order=4)
        center = (center[0] * zoom_factor + zoom_factor / 2, center[1] * zoom_factor + zoom_factor / 2)
        min_radius = min_radius * zoom_factor
        max_radius = max_radius * zoom_factor

    # pad if necessary
    max_x, max_y = image.shape[0], image.shape[1]
    pad_dist_x = np.max([(center[0] + max_radius) - max_x, -(center[0] - max_radius)])
    pad_dist_y = np.max([(center[1] + max_radius) - max_y, -(center[1] - max_radius)])
    pad_dist = int(np.max([0, pad_dist_x, pad_dist_y]))
    if pad_dist != 0:
        image = np.pad(image, pad_dist, 'constant')

    # coordinate conversion
    theta, r = np.meshgrid(np.linspace(0, 2 * np.pi, phase_width),
                           np.arange(min_radius, max_radius))
    x, y = coord_polar_to_cart(r, theta, center)
    x, y = np.round(x), np.round(y)
    x, y = x.astype(int), y.astype(int)
    x = np.maximum(x, 0)
    y = np.maximum(y, 0)
    x = np.minimum(x, max_x - 1)
    y = np.minimum(y, max_y - 1)

    polar = image[x, y]
    polar.reshape((max_radius - min_radius, phase_width))

    return polar


def mask_polar_to_cart(mask, center, min_radius, max_radius, output_shape, zoom_factor=1):
    '''Converts a polar binary mask to Cartesian and places in an image of zeros'''

    # Account for upsampling
    if zoom_factor != 1:
        center = (center[0] * zoom_factor + zoom_factor / 2, center[1] * zoom_factor + zoom_factor / 2)
        min_radius = min_radius * zoom_factor
        max_radius = max_radius * zoom_factor
        output_shape = map(lambda a: a * zoom_factor, output_shape)

    # new image
    image = np.zeros(output_shape)

    # coordinate conversion
    theta, r = np.meshgrid(np.linspace(0, 2 * np.pi, mask.shape[1]),
                           np.arange(0, max_radius))
    x, y = coord_polar_to_cart(r, theta, center)
    x, y = np.round(x), np.round(y)
    x, y = x.astype(int), y.astype(int)

    x = np.clip(x, 0, image.shape[0] - 1)
    y = np.clip(y, 0, image.shape[1] - 1)
    ix, iy = np.meshgrid(np.arange(0, mask.shape[1]), np.arange(0, mask.shape[0]))
    image[x, y] = mask

    # downsample image
    if zoom_factor != 1:
        zf = 1 / float(zoom_factor)
        image = zoom(image, (zf, zf), order=4)

    # ensure image remains a filled binary mask
    image = (image > 0.5).astype(int)
    image = binary_fill_holes(image)
    return image


def find_edge_2d(polar, min_radius):
    '''Dynamic programming algorithm to find edge given polar image'''
    if len(polar.shape) != 2:
        raise ValueError("argument to find_edge_2d must be 2D")

    # Dynamic programming phase
    values_right_shift = np.pad(polar, ((0, 0), (0, 1)), 'constant', constant_values=0)[:, 1:]
    values_closeright_shift = np.pad(polar, ((1, 0), (0, 1)), 'constant', constant_values=0)[:-1, 1:]
    values_awayright_shift = np.pad(polar, ((0, 1), (0, 1)), 'constant', constant_values=0)[1:, 1:]

    values_move = np.zeros((polar.shape[0], polar.shape[1], 3))
    values_move[:, :, 2] = np.add(polar, values_closeright_shift)  # closeright
    values_move[:, :, 1] = np.add(polar, values_right_shift)  # right
    values_move[:, :, 0] = np.add(polar, values_awayright_shift)  # awayright
    values = np.amax(values_move, axis=2)

    directions = np.argmax(values_move, axis=2)
    directions = np.subtract(directions, 1)
    directions = np.negative(directions)

    # Edge following phase
    edge = []
    mask = np.zeros(polar.shape)
    r_max, r = 0, 0
    for i, v in enumerate(values[:, 0]):
        if v >= r_max:
            r, r_max = i, v
    edge.append((r + min_radius, 0))
    mask[0:r + 1, 0] = 1
    for t in range(1, polar.shape[1]):
        r += directions[r, t - 1]
        if r >= directions.shape[0]: r = directions.shape[0] - 1
        if r < 0: r = 0
        edge.append((r + min_radius, t))
        mask[0:r + 1, t] = 1

    # add to inside of mask accounting for min_radius
    new_mask = np.ones((min_radius + mask.shape[0], mask.shape[1]))
    new_mask[min_radius:, :] = mask

    return np.array(edge), new_mask


def edge_polar_to_cart(edge, center):
    '''Converts a list of polar edge points to a list of cartesian edge points'''
    cart_edge = []
    for (r, t) in edge:
        x, y = coord_polar_to_cart(r, t, center)
        cart_edge.append((round(x), round(y)))
    return cart_edge


def cell_magic_wand_single_point(image, center, min_radius, max_radius,
                                 roughness=2, zoom_factor=1):
    '''Draws a border within a specified radius around a specified center "seed" point
    using a polar transform and a dynamic programming edge-following algorithm.
    Returns a binary mask with 1s inside the detected edge and
    a list of points along the detected edge.'''
    if roughness < 1:
        roughness = 1
        print("roughness must be >= 1, setting roughness to 1")
    if min_radius < 0:
        min_radius = 0
        print("min_radius must be >=0, setting min_radius to 0")
    if max_radius <= min_radius:
        max_radius = min_radius + 1
        print("max_radius must be larger than min_radius, setting max_radius to " + str(max_radius))
    if zoom_factor <= 0:
        zoom_factor = 1
        print("negative zoom_factor not allowed, setting zoom_factor to 1")
    phase_width = int(2 * np.pi * max_radius * roughness)
    polar_image = image_cart_to_polar(image, center, min_radius, max_radius,
                                      phase_width=phase_width, zoom_factor=zoom_factor)
    polar_edge, polar_mask = find_edge_2d(polar_image, min_radius)
    cart_edge = edge_polar_to_cart(polar_edge, center)
    cart_mask = mask_polar_to_cart(polar_mask, center, min_radius, max_radius,
                                   image.shape, zoom_factor=zoom_factor)
    return cart_mask, cart_edge


def cell_magic_wand(image, center, min_radius, max_radius,
                    roughness=2, zoom_factor=1, center_range=2):
    '''Runs the cell magic wand tool on multiple points near the supplied center and
    combines the results for a more robust edge detection then provided by the vanilla wand tool.
    Returns a binary mask with 1s inside detected edge'''

    centers = []
    for i in [-center_range, 0, center_range]:
        for j in [-center_range, 0, center_range]:
            centers.append((center[0] + i, center[1] + j))
    masks = np.zeros((image.shape[0], image.shape[1], len(centers)))
    for i, c in enumerate(centers):
        mask, edge = cell_magic_wand_single_point(image, c, min_radius, max_radius,
                                                  roughness=roughness, zoom_factor=zoom_factor)
        masks[:, :, i] = mask
    mean_mask = np.mean(masks, axis=2)
    final_mask = (mean_mask > 0.5).astype(int)
    return final_mask


def cell_magic_wand_3d(image_3d, center, min_radius, max_radius,
                       roughness=2, zoom_factor=1, center_range=2, z_step=1):
    '''Robust cell magic wand tool for 3D images with dimensions (z, x, y) - default for tifffile.load.
    This functions runs the robust wand tool on each z slice in the image and returns the mean mask
    thresholded to 0.5'''
    masks = np.zeros((int(image_3d.shape[0] / z_step), image_3d.shape[1], image_3d.shape[2]))
    for s in range(int(image_3d.shape[0] / z_step)):
        mask = cell_magic_wand(image_3d[s * z_step, :, :], center, min_radius, max_radius,
                               roughness=roughness, zoom_factor=zoom_factor,
                               center_range=center_range)
        masks[s, :, :] = mask
    mean_mask = np.mean(masks, axis=0)
    final_mask = (mean_mask > 0.5).astype(int)
    return final_mask


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


# Let's look at one of the patients

first_patient = load_scan(patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# import scipy

# Show some slice in the middle
# data=scipy.ndimage.interpolation.zoom(first_patient_pixels[41],[200,200])
plt.figure()
plt.imshow(first_patient_pixels[42])
plt.annotate('', xy=(317, 367), xycoords='data',
             xytext=(0.5, 0.5), textcoords='figure fraction',
             arrowprops=dict(arrowstyle="->"))
# plt.savefig("images/test.png",dpi=300)
plt.show()


def processimage(img):
    # function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    # Standardize the pixel values
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # plt.hist(img.flatten(),bins=200)
    # plt.show()
    # print(thresh_img[366][280:450])
    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # move the underflow bins
    img[img == max] = mean
    img[img == min] = mean
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    # plt.imshow(labels)
    # plt.show()
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
    return mask * img


def nodule_coordinates(nodulelocations, meta):
    slices = nodulelocations["slice no."][
        nodulelocations.index[nodulelocations["case"] == int(meta["Patient Id"][-4:])]]
    xlocs = nodulelocations["x loc."][nodulelocations.index[nodulelocations["case"] == int(meta["Patient Id"][-4:])]]
    ylocs = nodulelocations["y loc."][nodulelocations.index[nodulelocations["case"] == int(meta["Patient Id"][-4:])]]
    nodulecoord = []
    for i in range(len(slices)):
        nodulecoord.append([slices.values[i] - 1, xlocs.values[i] - 1, ylocs.values[i] - 1])
    return nodulecoord


noduleimages = np.ndarray([len(nodulelocations) * 3, 512, 512], dtype=np.float32)
nodulemasks = np.ndarray([len(nodulelocations) * 3, 512, 512], dtype=np.bool)
nodulemaskscircle = np.ndarray([len(nodulelocations) * 3, 512, 512], dtype=np.bool)
index = 0
totaltime = 50000
start_time = time.time()
elapsed_time = 0
nodulemeanhu = []
nonnodulemeanhu = []
thresh = -500
for i in range(len(patients)):
    print("Processing patient#", i, "ETA:", (totaltime - elapsed_time) / 3600, "hrs")
    coord = nodule_coordinates(nodulelocations, meta.iloc[i])
    if len(coord) > 0:
        patient = load_scan(patients[i])
        patient_pix = get_pixels_hu(patient)
        radius = nodulelocations["eq. diam."][
            nodulelocations.index[nodulelocations["case"] == int(meta["Patient Id"][i][-4:])]]
        nodulemask = np.ndarray([len(coord), 512, 512], dtype=np.bool)
        for j, cord in enumerate(coord):
            segmented_mask_fill = segment_lung_mask(patient_pix, True, False)
            if radius.iloc[j] > 5:
                # slice nodulecenter-1
                noduleimages[index] = processimage(patient_pix[cord[0] - 1])
                nodulemasks[index] = cell_magic_wand(-patient_pix[int(cord[0]) - 1], [int(cord[2]), int(cord[1])],
                                                     2, int(radius.iloc[j]) + 2)
                rr, cc = circle(int(cord[2]), int(cord[1]), int(radius.iloc[j]))
                imgcircle = np.zeros((512, 512), dtype=np.int16)
                imgcircle[rr, cc] = 1
                nodulepixcircle = imgcircle * patient_pix[cord[0] - 1]
                nodulepixcircle[nodulepixcircle < thresh] = 0
                nodulepixcircle[nodulepixcircle != 0] = 1
                nodulemaskscircle[index] = nodulepixcircle.astype(np.bool)

                nodulepix = nodulemasks[index] * patient_pix[cord[0] - 1]
                nodulepix[nodulepix < thresh] = 0
                nodulepix[nodulepix != 0] = 1
                nodulemasks[index] = nodulepix.astype(np.bool)
                index += 1

                # slice nodulecenter
                noduleimages[index] = processimage(patient_pix[cord[0]])
                nodulemasks[index] = cell_magic_wand(-patient_pix[int(cord[0])], [int(cord[2]), int(cord[1])], 2,
                                                     int(radius.iloc[j]) + 2)
                nodulepix = nodulemasks[index] * patient_pix[cord[0]]
                nodulepix[nodulepix < thresh] = 0
                nodulepixcircle = imgcircle * patient_pix[cord[0]]
                nodulepixcircle[nodulepixcircle < thresh] = 0

                # get mean nodule HU value

                # get mean non-nodule HU value
                nonnodule = (nodulemasks[index].astype(np.int16) - 1) * -1 * segmented_mask_fill[cord[0]] * patient_pix[
                    cord[0]]
                nonnodule[nonnodule < thresh] = 0
                nonnodulemeanhu.append(np.mean(nonnodule[nonnodule != 0]))
                plt.figure()
                # plt.hist(nodulepix[nodulepix!=0].flatten(),bins=80, alpha=0.5, color='blue')
                plt.hist(nonnodule[nonnodule != 0].flatten(), bins=80, alpha=0.5, color='orange')
                plt.hist(nodulepixcircle[nodulepix != 0].flatten(), bins=80, alpha=0.5, color='green')
                # plt.savefig("histplots/" + meta['Patient Id'].loc[i] + "slice" + str(cord) + ".png", dpi=300)
                plt.close()
                nodulemeanhu.append(np.mean(nodulepix[nodulepix != 0]))
                nodulepix[nodulepix != 0] = 1
                nodulemasks[index] = nodulepix.astype(np.bool)
                nodulepixcircle[nodulepixcircle != 0] = 1
                nodulemaskscircle[index] = nodulepixcircle.astype(np.bool)
                index += 1

                # slice nodulecenter+1
                noduleimages[index] = processimage(patient_pix[cord[0] + 1])

                nodulepix = nodulemasks[index] * patient_pix[cord[0] + 1]
                nodulepix[nodulepix < thresh] = 0
                nodulepix[nodulepix != 0] = 1
                nodulemasks[index] = nodulepix.astype(np.bool)
                nodulepixcircle = imgcircle * patient_pix[cord[0] + 1]
                nodulepixcircle[nodulepixcircle < thresh] = 0
                nodulepixcircle[nodulepixcircle != 0] = 1
                nodulemaskscircle[index] = nodulepixcircle.astype(np.bool)
                index += 1
    elapsed_time = time.time() - start_time
    totaltime = elapsed_time / (i + 1) * len(patients)
np.save(datafolder + '/noduleimages.npy', noduleimages)
np.save(datafolder + '/nodulemasks.npy', nodulemasks)
np.save(datafolder + '/nodulemaskscircle.npy', nodulemaskscircle)

plt.hist(nodulemeanhu, bins=20)
plt.hist(nonnodulemeanhu)
plt.show()
