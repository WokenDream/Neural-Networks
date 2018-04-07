import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_images(images, cols=1, titles=None):

    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)

        if image.ndim == 2:
            plt.gray()
            plt.axis('off')
        plt.imshow(image, interpolation='nearest')
        plt.axis('off')
    #fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    #plt.tight_layout()
    plt.show()


def main():

    d=[]
    for i in range(0, 100):
        fname = 'C:/Users/Elton Wong/PycharmProjects/a3/dropout_visualization/25/' + str(i + 1) + '.png'
        image = Image.open(fname).convert("L")
        d.append(np.asarray(image))

    show_images(d,cols=10, titles=None)

if __name__ == "__main__":
    main()