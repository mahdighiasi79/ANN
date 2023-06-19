import numpy as np
import matplotlib.pyplot as plt


# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def fetch_data():
    # Reading The Train Set
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []
    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))

    # Reading The Test Set
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))

    # Plotting an image
    # show_image(train_set[1][0])
    # plt.show()

    # train_set[image_index][0:inputs, 1:label][pixel_index][0]

    return [train_set, test_set]


def smooth_data():
    data = fetch_data()
    train_set = data[0]
    test_set = data[1]
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for i in range(len(train_set)):
        train_image = []
        train_label = []
        for j in range(len(train_set[i][0])):
            train_image.append(train_set[i][0][j][0])
        for j in range(len(train_set[i][1])):
            train_label.append(train_set[i][1][j][0])
        train_images.append(train_image)
        train_labels.append(train_label)

    for i in range(len(test_set)):
        test_image = []
        test_label = []
        for j in range(len(test_set[i][0])):
            test_image.append(test_set[i][0][j][0])
        for j in range(len(test_set[i][1])):
            test_label.append(test_set[i][1][j][0])
        test_images.append(test_image)
        test_labels.append(test_label)

    return [train_images, train_labels, test_images, test_labels]
