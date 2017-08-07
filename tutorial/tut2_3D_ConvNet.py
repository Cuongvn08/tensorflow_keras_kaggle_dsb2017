'''
Reference: https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet
'''

import dicom
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math


# load data
data_dir = '../data/stage1/'
patients = os.listdir(data_dir)

# load csv (data's labels)
labels_df = pd.read_csv('../data/stage1_labels.csv', index_col=0)
print(labels_df.head())
print()


# print dicom information of the 1st patient
for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    print('patient: {0}'.format(patient))
    print('volume size (x,y,z): ({0}, {1}, {2})'.format(slices[0].Columns,
                                                        slices[0].Rows,
                                                        len(slices)))
    print('label: {0}'.format(label))
    print('The 1st slice information: \n{0}'.format(slices[0]))
print()


# print the volume size of 10 patients
print('the number of patients: {0}'.format(len(patients)))

for patient in patients[:5]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    print('patient: {0}; volume size (x,y,z): ({1}, {2}, {3})'.format(patient,
                                                                      slices[0].Columns,
                                                                      slices[0].Rows,
                                                                      len(slices)))
print()

# display slice
for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    # display the first slice
    plt.imshow(slices[0].pixel_array, cmap='gray')
    plt.pause(0.1)

# resize slices
IMG_PX_SIZE = 150
for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    fig = plt.figure()
    for num,each_slice in enumerate(slices[:12]):
        y = fig.add_subplot(3,4,num+1)
        new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
        y.imshow(new_img, cmap='gray')
    plt.pause(0.1)


def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
    return sum(l) / len(l)

IMG_PX_SIZE = 150
HM_SLICES = 20

for patient in patients[:10]:
    try:
        label = labels_df.get_value(patient, 'cancer')
        path = data_dir + patient

        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

        new_slices = []
        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]
        chunk_sizes = math.ceil(len(slices) / HM_SLICES)
        for slice_chunk in chunks(slices, chunk_sizes):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)

        print(len(slices), len(new_slices))
    except:
        # some patients don't have labels, so we'll just pass on this for now
        pass

for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE))
            for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / HM_SLICES)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == HM_SLICES - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES + 2:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES], ])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val

    if len(new_slices) == HM_SLICES + 1:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES - 1], new_slices[HM_SLICES], ])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES - 1] = new_val

    fig = plt.figure()
    for num, each_slice in enumerate(new_slices):
        y = fig.add_subplot(4, 5, num + 1)
        y.imshow(each_slice, cmap='gray')
    plt.pause(0.1)

plt.show()
