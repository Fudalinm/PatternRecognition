import numpy as np
from scipy.fftpack import dct, idct
from scipy.stats import truncnorm
import cv2
from math import sqrt
from random import sample
import cvxpy as cvx
# TODO: http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/

import matplotlib.pyplot as plt

# S: GLOBALS
resources = 'resources'
prepared = 'prepared'
dct_dir = 'dct'
reconstruction = 'inverse'
upscale_dir = 'upscaled'

detail_source_image_name = "detail.jpg"
face_source_image_name = "face.jpg"
landscape_source_image_name = "landscape.jpg"

images_names = [detail_source_image_name, face_source_image_name, landscape_source_image_name]
images_source_paths = [resources + "/" + detail_source_image_name, resources + "/" + face_source_image_name,
                       resources + "/" + landscape_source_image_name]
images_prepared_paths = [prepared + "/" + detail_source_image_name, prepared + "/" + face_source_image_name,
                         prepared + "/" + landscape_source_image_name]

resize_edge = 60

prepared = 'prepared'


# E: GLOBALS

# S: FIRST

def prepare_images():
    for name, source, destiny in zip(images_names, images_source_paths, images_prepared_paths):
        im = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (resize_edge, resize_edge))
        cv2.imwrite(destiny, im)


def load_images():
    images = []
    for read in images_prepared_paths:
        images.append(cv2.imread(read, cv2.IMREAD_GRAYSCALE))
    return images


def load_images2():
    images = []
    for read, name in zip(images_prepared_paths, images_names):
        images.append([cv2.imread(read, cv2.IMREAD_GRAYSCALE), name])
    return images


def calculate_dct(im, filename, N):
    # same as dct(dct(a.T).T)
    after_dct = dct(dct(im, type=2, axis=0, n=N, norm='ortho'), type=2, axis=1, n=N, norm='ortho')
    cv2.imwrite(dct_dir + '/' + filename, after_dct)
    return after_dct


def remove_square(im, f1, t1, f2, t2):
    to_ret = np.zeros([resize_edge, resize_edge])
    to_ret += im
    for i in range(f1, t1):
        for j in range(f2, t2):
            to_ret[i][j] = 0
    return to_ret


def leave_square(im, f1, t1, f2, t2):
    to_ret = np.zeros([resize_edge, resize_edge])
    # to_ret += im
    for i in range(f1, t1):
        for j in range(f2, t2):
            to_ret[i][j] += im[i][j]
    return to_ret


def remove_circle(im, x, y, r):
    to_ret = np.zeros([resize_edge, resize_edge])
    to_ret += im
    for i in range(resize_edge):
        for j in range(resize_edge):
            if (x - i) * (x - i) + (y - j) * (y - j) < r * r:
                to_ret[i][j] = 0
    return to_ret


def remove_circle2(im, x, y, r):
    nx, ny = im.shape
    to_ret = np.zeros(im.shape)
    to_ret += im
    for i in range(nx):
        for j in range(ny):
            if (x - i) * (x - i) + (y - j) * (y - j) < r * r:
                to_ret[i][j] = 0
    return to_ret


def div_components(im):
    to_ret = np.zeros([resize_edge, resize_edge])
    for i in range(resize_edge):
        for j in range(resize_edge):
            to_ret[i][j] = im[i][j] / int(sqrt((i + j + 2) / 2))
    return to_ret


def merge_images_grid(im1, im2):
    to_ret = np.zeros([resize_edge, resize_edge])
    for i in range(resize_edge):
        for j in range(resize_edge):
            if j % 2 == 0:
                to_ret[i][j] = im2[i][j]
            else:
                to_ret[i][j] = im1[i][j]
    return to_ret


def calculate_dct_all(images, size_dtc=None):
    to_ret = []
    for i, n in zip(images, images_names):
        to_ret.append(calculate_dct(i, str(size_dtc) + '__' + n, size_dtc))
    return to_ret


def run1():
    prepare_images()
    images = load_images()
    dct_images = calculate_dct_all(images)
    for dct_im, name in zip(dct_images, images_names):
        transformed = list()
        transformed.append(['removed_square', remove_square(dct_im, 0, 100, 0, 100)])
        transformed.append(['left_square', leave_square(dct_im, 0, 100, 0, 100)])
        transformed.append(['div_im', div_components(dct_im)])
        transformed.append(['circle_im', remove_circle(dct_im, 400, 400, 350)])
        for dct_im2, name2 in zip(dct_images, images_names):
            transformed.append(['merge' + str(name2), merge_images_grid(dct_im, dct_im2)])

        for name_to_save, im in transformed:
            inverse = idct(idct(im, axis=0, norm='ortho'), axis=1, norm='ortho')
            cv2.imwrite(reconstruction + '/' + name_to_save + "___" + name, inverse)


# E: FIRST

# S: SECOND

heat_map_dir = 'heatMaps'
compressive_sensing_recreation = 'compressiveReconstruction'


def load_original_images():
    to_ret = []
    for name, source, destiny in zip(images_names, images_source_paths, images_prepared_paths):
        to_ret.append([cv2.imread(source, cv2.IMREAD_GRAYSCALE), name])
    return to_ret


def draw_k_percentage_pixels(im, k):
    x, y = im.shape

    pixels = sample(range(1, x * y), int((x * y * k) / 100) - 1)
    to_ret = np.zeros([x, y])
    print(im.shape)
    print(to_ret.shape)
    print(pixels)
    to_ret2 = []
    for p in range(len(pixels)):
        nx = int(pixels[p] / y)
        ny = int(pixels[p] % y)
        to_ret[nx][ny] = im[nx][ny]
        pixels[p] = [im[nx][ny], nx, ny]
        to_ret2.append(nx * ny)

    to_ret_normal = np.zeros([x, y])
    to_ret_normal2 = []
    pixels_normal = []
    generatorX = truncnorm((1 - x / 2) / 100, (x / 200), loc=x / 2, scale=50)
    generatorY = truncnorm((1 - y / 2) / 100, (y / 200), loc=y / 2, scale=50)
    for x, y in zip(generatorX.rvs(int((x * y * k) / 100)), generatorY.rvs(int((x * y * k) / 100))):
        nx = int(x)
        ny = int(y)
        pixels_normal.append([im[nx][ny], nx, ny])
        to_ret_normal[nx][ny] = im[nx][ny]
        to_ret_normal2.append(nx * ny)

    # pixels[i] = [value,x,y]
    # to_ret[x][y] = value
    # to_ret*2 = nx*ny
    return [['random', to_ret, pixels, to_ret2], ['normal', to_ret_normal, pixels_normal, to_ret_normal2]]


def prepare_upscaling(im):
    x, y = im.shape
    to_ret = np.zeros([2 * x - 1, 2 * y - 1])
    for i in range(x):
        for j in range(y):
            to_ret[2 * i - 1][2 * j - 1] = im[i][j]
    return to_ret


def upscaling(im):
    return recreate_image(im, im, im)


def dct2(x):
    return dct(dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    return idct(idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


# pixels[i] = [value,x,y]
# to_ret[x][y] = value
# TODO: xD
# it is better to use im_sample_always same
def recreate_image(image, im_sample, pixels):
    ri = []

    ny, nx = im_sample.shape
    for x in range(nx):
        for y in range(ny):
            if im_sample[x][y] != 0:
                ri.append(x * ny + y)

    print(ri)

    b = image.T.flat[ri]
    b = np.expand_dims(b, axis=1)

    print(b.shape)
    print(b)

    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0)
    )
    A = A[ri, :]  # same as phi times kron

    vx = cvx.Variable((nx * ny, 1))
    objective = cvx.Minimize(cvx.norm(vx, 1))
    print(A.shape)
    print(vx.shape)
    print(b.shape)

    constraints = [A * vx == b]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    Xat2 = np.array(vx.value).squeeze()

    Xat = Xat2.reshape(nx, ny).T  # stack columns
    Xa = idct2(Xat)

    return Xa


#
def loss_function(original_im, recreted_im):
    x, y = original_im.shape
    recreated_resize = cv2.resize(recreted_im, (y, x))
    loss_map = np.zeros([x, y])
    for i in range(x):
        for j in range(y):
            loss_map[i][j] = abs(original_im[i][j] - recreated_resize[i][j])
    return 0, 0, loss_map


def prepare_image(im):
    # prepare type: None, lower_resolution, remove shape
    x, y = im.shape
    im_normal = np.zeros([x, y])
    im_lower_resolution = cv2.resize(im, (int(x / 2), int(y / 2)))
    im_remove_shape = remove_circle2(im, int(x / 2), int(y / 2), 50)

    im_normal += im
    im_remove_shape += im

    to_ret = list()
    to_ret.append(['normal', im_normal])
    to_ret.append(['lower_resolution', im_lower_resolution])
    to_ret.append(['remove_shape', im_remove_shape])

    return to_ret


def run2():
    prepare_images()
    images = load_images2()
    # images = load_original_images()
    to_fig = []
    for current_image, name in [images[0], images[1], images[2]]:
        for prepare_type, prepared_image in prepare_image(current_image):
            if prepare_type == 'normal' or prepare_type == 'lower_resolution':
                # we need to draw several samples and colect them
                for random_type_loop in ['random', 'normal']:

                    for k in [50, 70, 90]:
                        repeat = 3
                        l1_total = 0
                        l2_total = 0
                        loss_map_total = np.zeros(current_image.shape)
                        recreated_image_total = np.zeros(prepared_image.shape)
                        for i in range(repeat):
                            index_random = 0
                            if random_type_loop == 'normal':
                                index_random = 1
                            random_type, im_sample, samples, nxny = draw_k_percentage_pixels(prepared_image, k)[
                                index_random]

                            recreated_image = recreate_image(prepared_image, im_sample, samples)

                            l1, l2, loss_map = loss_function(current_image, recreated_image)
                            l1_total += l1
                            l2_total += l2
                            loss_map_total += loss_map
                            recreated_image_total += recreated_image
                        to_fig.append([k, np.mean(loss_map_total)])

                        plt.clf()
                        plt.imshow(loss_map_total, cmap='hot', interpolation='nearest')
                        plt.colorbar()
                        plt.savefig(heat_map_dir + '/' + str(name) + "_" + str(prepare_type) + '_' +
                                    str(random_type_loop) + '_' + str(k) + '_' + 'heatMap.png')
                        cv2.imwrite(compressive_sensing_recreation + '/' + str(name) + "_" + str(prepare_type) +
                                    str(random_type_loop) + '_' + str(k) + '_' + 'recreation.png',
                                    recreated_image_total)


            elif prepare_type == 'remove_shape':
                recreated_image = recreate_image(prepared_image, prepared_image, prepared_image)
                plt.clf()

                cv2.imwrite(compressive_sensing_recreation + '/' + str(name) + "_" + str(prepare_type) +
                            '_' + 'recreation.png', recreated_image)

        to_upscale = prepare_upscaling(current_image)
        upscaled = upscaling(to_upscale)
        cv2.imwrite(upscale_dir + '/' + str(name) + ".png", upscaled)

        print(to_fig)
        print_fig(to_fig)

# [[50, 94.85952782037512], [70, 58.21856192294773], [90, 19.922502322585004], [50, 152.04764718146316], [70, 148.4737379774926], [90, 146.95690765907548], [50, 150.21736775621247], [70, 145.27452087630203], [90, 139.89586320778463], [50, 163.11934221854867], [70, 165.09846905624843], [90, 161.78084235766647], [50, 28.67416705222562], [70, 15.89825033797689], [90, 5.069485195064802], [50, 242.88650127003552], [70, 243.35442164893033], [90, 240.1718069397603], [50, 56.42912600342044], [70, 41.44843396864137], [90, 32.82565783132139], [50, 270.4735968836821], [70, 256.90198818528927], [90, 256.2350387332263], [50, 47.86740994610104], [70, 28.19784530392849], [90, 9.529193450819935], [50, 101.86093326663472], [70, 100.80731758490705], [90, 100.1138591512], [50, 72.84369040325832], [70, 66.03817658930454], [90, 60.09246926023768], [50, 105.51040598093714], [70, 104.59321377461015], [90, 105.12840147938616]]
def print_fig(data):
    plt.clf()
    fifty = []
    seventy = []
    ninty = []

    for k, precision in data:
        if k == 50:
            fifty.append(precision)
        elif k == 70:
            seventy.append(precision)
        else:
            ninty.append(precision)
    fifty = sum(fifty) / len(fifty)
    seventy = sum(seventy) / len(seventy)
    ninty = sum(ninty) / len(ninty)

    print(fifty)
    print(seventy)
    print(ninty)

    plt.bar(50, fifty)
    plt.bar(70, seventy)
    plt.bar(90, ninty)

    plt.savefig('lastfig.png')


# E: SECOND


if __name__ == "__main__":
    # run1()
    # run2()
    print_fig([[50, 94.85952782037512], [70, 58.21856192294773], [90, 19.922502322585004], [50, 152.04764718146316], [70, 148.4737379774926], [90, 146.95690765907548], [50, 150.21736775621247], [70, 145.27452087630203], [90, 139.89586320778463], [50, 163.11934221854867], [70, 165.09846905624843], [90, 161.78084235766647], [50, 28.67416705222562], [70, 15.89825033797689], [90, 5.069485195064802], [50, 242.88650127003552], [70, 243.35442164893033], [90, 240.1718069397603], [50, 56.42912600342044], [70, 41.44843396864137], [90, 32.82565783132139], [50, 270.4735968836821], [70, 256.90198818528927], [90, 256.2350387332263], [50, 47.86740994610104], [70, 28.19784530392849], [90, 9.529193450819935], [50, 101.86093326663472], [70, 100.80731758490705], [90, 100.1138591512], [50, 72.84369040325832], [70, 66.03817658930454], [90, 60.09246926023768], [50, 105.51040598093714], [70, 104.59321377461015], [90, 105.12840147938616]]
)
