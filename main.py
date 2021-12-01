import cv2 as cv
from create_output import *
import itertools

images = []
for i in range(2, 3):
    print(i)
    img1 = cv.imread('data/' + str(i - 1) + '.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('data/' + str(i) + '.jpg', cv.IMREAD_GRAYSCALE)
    scale_percent = 60
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)

    dim = (width, height)
    img1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)
    img2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)
    images.append((img1, img2))

win_size = 5
min_disp = -1
max_disp = 63
num_disp = max_disp - min_disp
stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                              numDisparities=num_disp,
                              blockSize=5,
                              uniquenessRatio=5,
                              speckleWindowSize=5,
                              speckleRange=5,
                              disp12MaxDiff=1,
                              P1=8 * 3 * win_size ** 2,
                              P2=32 * 3 * win_size ** 2)
focal_length = 1
Q2 = np.float32([[1, 0, 0, 0],
                 [0, -1, 0, 0],
                 [0, 0, focal_length * 0.05, 0],
                 [0, 0, 0, 1]])

result_points = []
result_colors = []

for image_pair in images:
    print("\nComputing the disparity  map...")
    disparity_map = stereo.compute(image_pair[0], image_pair[1])

    print("\nGenerating the 3D map...")
    h, w = image_pair[0].shape[:2]

    points_3D = cv.reprojectImageTo3D(disparity_map, Q2)
    colors = cv.cvtColor(image_pair[0], cv.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min()
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    result_points.append(output_points)
    result_colors.append(output_colors)


result_points = list(itertools.chain(*result_points))
result_colors = list(itertools.chain(*result_colors))


output_file = 'reconstructed.ply'
print("\n Creating the output file... \n")
create_output(np.array(result_points), np.array(result_colors), output_file)

