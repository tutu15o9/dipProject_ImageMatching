# Library imports
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
from tqdm import tqdm

sift = cv.SIFT_create()


def calculate_SIFT(img):
    # Find the keypoints and descriptors using SIFT features
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def knn_match(des1, des2, nn_ratio=0.75):
    # FLANN parameters
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Match features from each image
    matches = flann.knnMatch(des1, des2, k=2)

    # store only the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < nn_ratio * n.distance:
            good.append(m)

    return good


def angle_horizontal(v):
    return -np.arctan2(v[1], v[0])


def knn_clasif(good_matches):
    best_template, highest_logprob = None, 0.0

    sum_good_matches = sum([len(gm) for gm in good_matches])
    for i, gm in enumerate(good_matches):
        logprob = len(gm) / sum_good_matches
        # save highest
        if logprob > highest_logprob:
            highest_logprob = logprob
            best_template = i
        logger.info('p(t_{} | x) = {:.4f}'.format(i, logprob))
    return best_template


template_name = "./images/nemo_template.jpg"
query_name = "./images/nemo.jpg"

template_img = cv.imread(template_name, cv.IMREAD_GRAYSCALE)
template_kp, template_des = calculate_SIFT(template_img)
query_img = cv.imread(query_name, cv.IMREAD_GRAYSCALE)
query_kp, query_des = calculate_SIFT(query_img)

gm = knn_match(template_des, query_des)
src_pts = np.float32(
    [template_kp[m.queryIdx].pt for m in gm]).reshape(-1, 1, 2)
dst_pts = np.float32(
    [query_kp[m.trainIdx].pt for m in gm]).reshape(-1, 1, 2)

# find the matrix transformation M
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# Make it affine
M[2, 2] = 1.0
M[2, 0] = 0.0
M[2, 1] = 0.0
# Calculate the rectangle enclosing the query image
h, w = template_img.shape
# Define the rectangle in the coordinates of the template image
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                  [w - 1, 0]]).reshape(-1, 1, 2)
# transform the rectangle from the template "coordinates" to the query "coordinates"
dst = cv.perspectiveTransform(pts, M)

# calculate template "world" reference vectors
w_v = np.array([w - 1, 0])
h_v = np.array([h - 1, 0])
# calculate query "world" reference vectors
w_vp = (dst[3] - dst[0])[0]
h_vp = (dst[1] - dst[0])[0]

angle = angle_horizontal(w_vp)
# estimate the scale using the top-horizontal line and left-vertical line
scale_x = np.linalg.norm(w_vp) / np.linalg.norm(w_v)
scale_y = np.linalg.norm(h_vp) / np.linalg.norm(h_v)
# retrieve translation from original matrix M
M = np.array([[scale_x * np.cos(angle), np.sin(angle), M[0, 2]],
               [-np.sin(angle), scale_y * np.cos(angle), M[1, 2]],
               [0, 0, 1.]])
# retransform the rectangle with the new matrix
dst = cv.perspectiveTransform(pts, M)

# draw the rectangle in the image
out = cv.polylines(query_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
plt.imshow(out, 'gray')
plt.savefig('outline.png')
plt.show()
# show the matching features
params = dict(matchColor=(0, 255, 0),  # draw matches in green color
              singlePointColor=None,
              matchesMask=matchesMask,  # draw only inliers
              flags=2)
# draw the matches image
out = cv.drawMatches(template_img, template_kp,
                     query_img, query_kp,
                     gm,
                     None, **params)

# show result
plt.imshow(out, 'gray')
plt.savefig('linematches.png')
plt.show()
