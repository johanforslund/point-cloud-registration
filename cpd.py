from functools import partial
import matplotlib.pyplot as plt
from pycpd import AffineRegistration
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import cv2

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def vis_pts(X=None, Y=None):
    if X is None:
        X = np.load('bu/pts1.npy')
    if Y is None:
        Y = np.load('bu/pts2.npy')

        Y = Y[~np.all(cdist(Y, X) >= 20, axis=1)]
    
    X = X[X[:, 0] < 6*105]
    X = X[X[:, 1] < 6*68]
    Y = Y[Y[:, 0] < 6*105]
    Y = Y[Y[:, 1] < 6*68]

    img = np.zeros((6*68, 6*105, 3))

    img[X[:, 1], X[:, 0], 1] = 1
    img[Y[:, 1], Y[:, 0], 0] = 1
    
    plt.imshow(img)
    plt.show()

def apa():
    A = np.random.rand(2, 2)
    B = np.random.rand(2)

    print(A)
    print(B)
    
    C = np.append(A, B[:, None], axis=1)
    print(C)
    print(C.shape)

def main():
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    all_pts1 = np.load('all_pts1.npy')
    all_pts2 = np.load('all_pts2.npy')
    np.load = np_load_old

    for ts, X in all_pts1.item().items():
        Y = all_pts2.item()[ts]

        Y = Y[~np.all(cdist(Y, X) >= 20, axis=1)]

        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0])

        reg = AffineRegistration(**{'X': Y, 'Y': X})
        TY, (B_reg, t_reg) = reg.register(callback)
        plt.show()

        vis_pts(X, Y)

        Xp = np.dot(X, B_reg) + np.tile(t_reg, (X.shape[0], 1))
        Xp = Xp.astype(int)

        vis_pts(Xp, Y)

        M = np.append(B_reg, t_reg[:, None], axis=1)

        frame = np.array(Image.open('frames/frame_' + str(ts) + '.png'))
        plt.imshow(frame)
        plt.show()
        frame_warped = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        plt.imshow(frame_warped)
        plt.show()

if __name__ == '__main__':
    main()