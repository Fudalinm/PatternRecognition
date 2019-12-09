import cv2
import math
import random

from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
from build_reference_table import *
from match_table import *
from find_maxima import *
import numpy as np

normal = 'images/normal/normal'
dense = 'images/normalDense/normalDense'
oneOnAnother = 'images/oneOnAnother/oneOnAnother'
tilted1 = 'images/tilted/tilted'
tilted2 = 'images/tilted2/tilted2'
tilted3 = 'images/tilted3/tilted3'

template = 'images/templatesmaller'

type = '.jpg'

allJPG = [normal, dense, oneOnAnother, tilted1, tilted2, tilted3]

for core in allJPG:

    input_image = core + "smaller" + type
    for noise in ['None', 'Blurr', 'S&P', 'RandomElements']:
        print("xDDD1")
        negative_file = core + 'Negativ' + type
        noised_file = core + noise + 'Noised' + type
        result_file = core + noise + 'Result' + type
        voting_file = core + noise + 'Voting' + type
        edges_file = core + noise + 'Edges' + type
        refim_file = core + noise + 'CannyEdge' + type

        refim = cv2.imread(template + type)
        refim = cv2.cvtColor(refim, cv2.COLOR_BGR2GRAY)
        loaded = cv2.imread(input_image)
        loadedGray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)

        # negative
        # loadedGray = cv2.bitwise_not(cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY))
        # cv2.imwrite(negative_file, loadedGray)

        if noise == 'Blurr':
            noised = cv2.GaussianBlur(src=loadedGray,
                                      ksize=(19, 19), sigmaX=0)
        elif noise == 'S&P':
            noised = loadedGray
            height, width = noised.shape
            for x in range(height):
                for y in range(width):
                    if random.random() < 0.01:
                        noised[x][y] = random.randrange(0, 255)
        elif noise == 'RandomElements':
            howMany = 5
            noised = loadedGray
            height, width = noised.shape
            thickness = 2

            # trojkaty
            for i in range(howMany):
                t = np.array([(random.randrange(0, width), random.randrange(0, height)),
                              (random.randrange(0, width), random.randrange(0, height)),
                              (random.randrange(0, width), random.randrange(0, height))])
                color = random.randrange(20, 255)
                cv2.drawContours(noised, [t], 0, (color, color, color), thickness=thickness)

            # elipses
            for i in range(howMany):
                color = random.randrange(20, 255)
                center1 = random.randrange(100, width - 100)
                dev1 = random.randrange(40, 60)
                center2 = random.randrange(100, height - 100)
                dev2 = random.randrange(40, 60)

                cv2.ellipse(noised, (center1, center2), (dev1, dev2), 0, 0, 360,
                            color=(color, color, color), thickness=thickness)
        else:
            noised = loadedGray
        cv2.imwrite(noised_file, noised)

        print("xDDD2")
        # we need to make canny on both of them
        edges_image = cv2.Canny(noised, 250, 500, apertureSize=3)
        cv2.imwrite(edges_file, edges_image)
        refim = cv2.Canny(refim, 120, 500, apertureSize=3)
        cv2.imwrite(refim_file, refim)

        table = buildRefTable(refim)
        print(table)
        acc = matchTable(edges_image, table)
        print("xDDD4")
        val, ridx, cidx = findMaxima(acc)
        print("xDDD5")

        #####
        # code for drawing bounding-box in accumulator array...

        acc[ridx - 5:ridx + 5, cidx - 5] = val
        acc[ridx - 5:ridx + 5, cidx + 5] = val

        acc[ridx - 5, cidx - 5:cidx + 5] = val
        acc[ridx + 5, cidx - 5:cidx + 5] = val

        plt.figure(1)
        imshow(acc)
        imsave(voting_file, acc)

        plt.show()

        # code for drawing bounding-box in original image at the found location...

        # find the half-width and height of template
        hheight = np.floor(refim.shape[0] / 2) + 1
        hwidth = np.floor(refim.shape[1] / 2) + 1

        # find coordinates of the box
        rstart = int(max(ridx - hheight, 1))
        rend = int(min(ridx + hheight, noised.shape[0] - 1))
        cstart = int(max(cidx - hwidth, 1))
        cend = int(min(cidx + hwidth, noised.shape[1] - 1))

        # draw the box
        noised[rstart:rend, cstart] = 255
        noised[rstart:rend, cend] = 255

        noised[rstart, cstart:cend] = 255
        noised[rend, cstart:cend] = 255

        # show the image
        plt.figure(2), imshow(refim)
        plt.figure(3), imshow(noised)
        plt.show()

        ####
        cv2.imwrite(result_file, noised)

"""
# for core in allJPG:
#     im = Image.open(core + type)
#     height = 672
#     width = 504
#     tmp = im.resize((width, height), Image.NEAREST)
#     tmp.save(core + "smaller" + type)
"""
