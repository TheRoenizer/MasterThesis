import cv2 as cv
import numpy as np
import os


def load_data(data_path, dtype=np.float32):
    N = 99            # Number of images
    M = 5             # Number of labels
    DIM = (480, 640)  # Image dimensions

    images = np.empty((N, *DIM, 3), dtype=dtype)
    labels = np.empty((N, *DIM, 1), dtype=dtype)

    for i in range(N):
        image_path = os.path.join(data_path, 'Images/Suturing ({}).png'.format(i + 1))
        images[i] = cv.imread(image_path).astype(dtype)
        images[i] = cv.normalize(images[i], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

        for j in range(M):
            label_path = os.path.join(data_path, 'Annotated/Suturing ({})/data/00{}.png'.format(i + 1, j))
            im = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
            mask = cv.threshold(im, dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY)[1]
            k = np.where(mask == 255)
            labels[i][k] = (j + 1) * 30  # set pixel value here

    return images, labels


# A little test:
if __name__ == '__main__':
    images, labels = load_data('C:/Users/chris/Google Drive/Jigsaw annotations')

    print(images.shape)
    print(labels.shape)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(cv.cvtColor(images[0], cv.COLOR_BGR2RGB))
    axes[1].imshow(np.squeeze(labels[0]), cmap='gray', vmin=0, vmax=255)

    plt.show()