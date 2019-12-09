import cv2
from math import copysign, log10
import matplotlib.pyplot as plt

# maybe it should be one image P, one image R, one image W

resources_directory = 'images/'
figure_directory = 'figures/'
# resources_subdir = ['grouped/', 'M/', 'P/', 'R/']
resources_subdir = ['M/', 'P/', 'R/']
type = '.jpg'

normal = 'normal'  # black
rotation = 'rotation'  # red
font_image = 'font'  # green
scale_down_image = 'scaleDown'  # blue
scale_up_image = 'scaleUp'  # purple
translation_image = 'translation'  # yellow

color = ['k', 'r', 'g', 'b', 'm', 'y']

images_type = [normal, rotation, font_image, scale_down_image, scale_up_image, translation_image]
images = []
for j in resources_subdir:
    for i in range(len(images_type)):
        images.append(resources_directory + j + images_type[i] + type)

result_raw = []
result_log = []

for current_image_path in images:
    # read image
    image_read = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)

    # convert to on/off pixels
    # image_read_binary = cv2.threshold(image_read, 128, 255, cv2.THRESH_BINARY)

    # calculate basic moments
    moments_basic = cv2.moments(image_read)
    # calculate Hu moments
    moments_hu = cv2.HuMoments(moments_basic)
    moments_hu_log = []
    # convert it to log scale
    for i in range(7):
        moments_hu_log.append(-1 * copysign(1.0, moments_hu[i]) * log10(abs(moments_hu[i])))
    # add to results
    result_raw.append((current_image_path, moments_hu))
    result_log.append((current_image_path, moments_hu_log))
    #result_log.append((current_image_path, moments_hu))

color_counter = -1
sub_dir_counter = -1
x = [1, 2, 3, 4, 5, 6, 7]
for i in range(len(images)):
    color_counter += 1
    if i % len(images_type) == 0:
        if i != 0:
            print(i)
            plt.legend()
            plt.savefig(figure_directory + resources_subdir[sub_dir_counter][:-1], dpi=1500)
            plt.clf()

        sub_dir_counter += 1
        plt.title(resources_subdir[sub_dir_counter])
        color_counter = 0
    # podpis linii to sciezka
    # plt.plot(x, result_log[i][1], 'r', color=color[color_counter])
    plt.scatter(x, result_log[i][1], s=0.01, color=color[color_counter], label=result_log[i][0])

    # print("###############")
    # print(result_log[i][0])
    # print(x)
    # print(result_log[i][1])
    # print(color)
    # print(color_counter)
    # print("###############")

plt.legend()
plt.savefig(figure_directory + resources_subdir[sub_dir_counter][:-1], dpi=1500)
