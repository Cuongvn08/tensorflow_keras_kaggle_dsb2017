import numpy as np
from matplotlib import pyplot as plt

output_path = "../output/"

if __name__ == '__main__':
    # debug: display
    images = np.load(output_path + 'images_0002_0236.npy')
    masks = np.load(output_path + 'masks_0002_0236.npy')
    print('Number of images: %d' % len(images))

    for i in range(len(images)):
        print('Image %d' % i)
        fig, ax = plt.subplots(2, 2)

        ax[0, 0].imshow(images[i], cmap='gray',)
        ax[0, 0].set_title('image')

        ax[0, 1].imshow(masks[i], cmap='gray')
        ax[0, 1].set_title( 'mask')

        ax[1, 0].imshow(images[i]*masks[i], cmap='gray')
        ax[1, 0].set_title('image * mask')

        plt.show()
        raw_input('Please enter to continue ...')
