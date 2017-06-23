import os
import numpy as np
import pandas as pd
import cv2

# image display
import IPython.display
import PIL.Image
from io import BytesIO

# specific tools
from sklearn.utils import shuffle
from scipy.signal import butter, lfilter


# DATA PROCESSING TOOLS

def get_filename(file_with_path):
    return file_with_path.split('/')[-1]


def assemble_data(suffixes, data_dir, df=None):
    '''
    1) assembles from a list of suffixes, the corresponding data_log_suffix.csv
    2) compares to see if any files are missing, and if so, excludes that line of the log
    3) returns a list of remaining rows
    '''
    for suffix in suffixes:

        directory = data_dir + 'IMG_' + suffix + '/'
        files = [directory + filename for filename in os.listdir(data_dir + 'IMG_' + suffix)]

        data = pd.read_csv(data_dir + 'driving_log_' + suffix + '.csv',
                           header = None,
                           names = ['center', 'left', 'right',
                                    'steering', 'throttle', 'brake', 'speed']
                           )

        data['keep'] = True

        for var in ['center', 'left', 'right']:
            data[var] = directory + data[var].map(get_filename)

        data['file_present'] = (data['center'].isin(files)) & \
            (data['left'].isin(files)) & \
            (data['right'].isin(files))

        data = data[data['file_present']]

        if df is None:
            df = data
        else:
            df = df.append(data, ignore_index=False)

    return df


def butter_lowpass(cutoff, fs, order=5):
    '''
    low pass filter using scipy
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
    applies a low pass filter to the data, in a forward pass w/ specified params
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def random_image_list(df_full, ang_factor, n):
    '''
    Building on the work of `assemble_data` this creates a random selection
    of augmented data of size n that can be used in a data generator.
    Images are not altered but instead are processed according to the columns
    in this list when fetched from the generator.
    '''
    df = df_full.sample(n)
    image_list = (list(df['center']) +
                  list(df['left']) +
                  list(df['right']))
    steer_list = list(df['steering']) + list(df['steering'] + ang_factor) + \
        list(df['steering'] - ang_factor)

    flip_image = len(image_list) * [True] + len(image_list) * [False]
    image_list *= 2
    steer_list *= 2
    df = pd.DataFrame(np.array([image_list, steer_list, flip_image]).T,
                      columns=['filename', 'steer', 'flip'])
    df['steer'] = df['steer'].astype(np.float32)

    return df


def load_image(filename):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return image


def pixel_normalization(p):
    return (p / 255.0) - 0.5


def region_of_interest(img, vertices, invert_mask=False):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        zeros = (0,) * channel_count
    else:
        ignore_mask_color = 255
        zeros = 0

    # defining a blank mask to start with
    if invert_mask:
        mask = np.full(img.shape, np.int32(255))
    else:
        mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    if invert_mask:
        mask = cv2.fillConvexPoly(mask, vertices, zeros)
    else:
        mask = cv2.fillConvexPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img.astype(np.int32), mask.astype(np.int32))
    return masked_image


def process_image(image, l_threshold=142, h_threshold=233, roi=False,
                  roi_vertices=np.array([[90, 80], [90, 30], [230, 30], [230, 80]],
                                        dtype=np.int32)):
    '''
    This is the image processing pipeline applied before submitting
    images for training as well as before submitting to the model for
    prediction.

    1) Creates a canny edge version of the image
    2) Optionally applies a region of interest mask,
    which may be inverted and must be convex
    2a) Change cv2.fillConvexPoly to cv2.fillPoly to use non convex shapes
    but expect a performance hit
    3) Adds this canny image as a 4th channel
    4) Normalized pixel values
    '''
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blurred, l_threshold, h_threshold)

    if roi:
        masked_image = region_of_interest(canny, roi_vertices, invert_mask=True).astype(np.uint8)
        new_image = np.dstack((image, masked_image))

    else:
        new_image = np.dstack((image, canny))

    new_image = pixel_normalization(new_image)

    return new_image


def generator(df, batch_size=32, x_or_y='both'):
    '''
    Generator for on the fly image generation, only slightly
    modified from Udacity presented version.
    '''
    num_samples = len(df)
    while 1:  # Loop forever so the generator never terminates
        df = shuffle(df)

        for offset in range(0, num_samples, batch_size):
            start = offset
            stop = min(num_samples, offset + batch_size)

            images = []
            angles = []

            for index in range(start, stop):
                angle = df.iloc[index]['steer']
                image = process_image(load_image(df.iloc[index]['filename']))

                if df.iloc[index]['flip']:
                    angle *= -1
                    image = np.fliplr(image)

                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            if x_or_y == 'both':
                yield X_train, y_train

            if x_or_y == 'x':
                yield X_train


# DATA INVESTIGATION TOOLS

def showarray(a, fmt='png', width=None, height=None):
    '''
    Displays an image without the ugliness of matplotlib
    '''
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue(), width=width, height=height))
