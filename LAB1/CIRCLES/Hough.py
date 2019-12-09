# whole concept is based on
# codingame.com/playgrounds/38470/how-to-detect-circles-in-images?fbclid=IwAR2jYAJqeZzOyI0DFRlv7LHX5ffq-n06H3eX4G_aJRj-xxbIGgAF7FwIiOc

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin, atan2
from collections import defaultdict
from skimage.util import random_noise

# global variables
width = 800
height = 800
thickness = 2
title = 'circles'
circleNumber = 6
radius = 50
rmin = 48
rmax = 52
threshold = 0.4
original_file = 'original.jpeg'
edge_file = 'edges.jpeg'
canny_file = 'canny.jpeg'
result_file = 'origin_detect.jpeg'
vote_result_file = 'votingResult.jpeg'
steps = 100


############################################## S: CANNY ##############################################

def canny_edge_detector(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)

    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep


def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale


def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx ** 2 + magy ** 2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x + a, y + b) not in keep:
                    newkeep.add((x + a, y + b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)


# if __name__ == "__main__":
#     from PIL import Image, ImageDraw
#     input_image = Image.open("input.png")
#     output_image = Image.new("RGB", input_image.size)
#     draw = ImageDraw.Draw(output_image)
#     for x, y in canny_edge_detector(input_image):
#         draw.point((x, y), (255, 255, 255))
#     output_image.save("canny.png")


############################################## E: CANNY ##############################################


############################################## random image creation based on global variables
origin_img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
for i in range(circleNumber):
    color = random.randrange(20, 255)
    cv2.circle(origin_img,
               (random.randrange(radius + 1, width - radius - 1), random.randrange(radius + 1, height - radius - 1)),
               radius, (color, color, color), thickness)

# # trojkaty
#
# for i in range(5):
#     t = np.array([(random.randrange(0, width), random.randrange(0, height)),
#                   (random.randrange(0, width), random.randrange(0, height)),
#                   (random.randrange(0, width), random.randrange(0, height))])
#     color = random.randrange(20, 255)
#     cv2.drawContours(origin_img, [t], 0, (color, color, color), thickness=thickness)

# elipses
# for i in range(circleNumber):
#     color = random.randrange(20, 255)
#     center1 = random.randrange(100, width - 100)
#     dev1 = random.randrange(40, 60)
#     center2 = random.randrange(100, height - 100)
#     dev2 = random.randrange(40, 60)
#
#     cv2.ellipse(origin_img, (center1, center2), (dev1, dev2), 0, 0, 360,
#                 color=(color, color, color), thickness=thickness)


# blur image
# apply Gaussian blur, creating a new image
# origin_img = cv2.GaussianBlur(src=origin_img,
#                               ksize=(19, 19), sigmaX=0)

# peper and salt
for x in range(width):
    for y in range(height):
        if random.random() < 0.05:
            origin_img[x][y] = random.randrange(0, 255)

cv2.imwrite(original_file, origin_img)

##############################################

############################################## S: CIRCLE ##############################################
# Load image:
input_image = Image.open(original_file)

# Output image:
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

# glosowanie w step kierunkach na najbardziej odpowiadajace srodki okregow
acc = defaultdict(int)
for x, y in canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

# saving vote results
voteImage = Image.new("RGB", input_image.size)
for k, v in acc.items():
    x, y, r = k
    if (x < width and y < height):
        voteImage.putpixel((x, y), (v * 30))
voteImage.save(vote_result_file)

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))

for x, y, r in circles:
    draw_result.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))

# Save output image
output_image.save(result_file)
############################################## E: CIRCLE ##############################################
