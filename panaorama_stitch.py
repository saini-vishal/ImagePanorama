import sys, os
import time
import cv2 as cv2
import numpy as np
import collections
from scipy.spatial.distance import cdist

DEFAULT_FILENAME = 'panorama.jpg'
IMAGE_EXTENSION = '.jpg'
UBID = 50325387
np.random.seed(UBID)


#
# Computes a homography from 4-corresponding points
#
def calculate_homography(correspondences):
    # Create equations
    eqns_list = np.empty((0, 9), dtype='float64')
    for corr in correspondences:
        point1 = np.asarray([corr.item(0), corr.item(1), 1])

        point2 = np.asarray([corr.item(2), corr.item(3), 1])

        eqn1 = np.asarray(
            [-point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2), 0, 0,
             0,
             point2.item(0) * point1.item(0), point2.item(0) * point1.item(1), point2.item(0) * point1.item(2)])
        eqn2 = np.asarray(
            [0, 0, 0, -point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1),
             -point2.item(2) * point1.item(2),
             point2.item(1) * point1.item(0), point2.item(1) * point1.item(1), point2.item(1) * point1.item(2)])
        eqns_list = np.append(eqns_list, [eqn1], axis=0)
        eqns_list = np.append(eqns_list, [eqn2], axis=0)

    svd_matrix = np.asarray(eqns_list)
    # decompose into SVD
    u, s, v = np.linalg.svd(svd_matrix)

    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    # normalize H
    h = np.divide(h, h.item(8))
    return h


#
# Calculate Reprojection errors
#
def calculate_error(correspondence, h):
    p1 = np.transpose(np.asarray([correspondence[0], correspondence[1], 1]))
    estimated_points = np.dot(h, p1)
    estimated_points = np.divide(estimated_points, estimated_points.item(2))
    p2 = np.transpose(np.asarray([correspondence[2], correspondence[3], 1]))
    error = np.subtract(p2, estimated_points)
    return np.linalg.norm(error)


#
# Estimate homography matrix using RANSAC
#
def ransac(corr, thresh):
    max_inliers = []
    final_homography = None
    num_of_iterations = np.round(np.log(1 - 0.99) / np.log(1 - (1 - 0.6) ** 5))
    for i in range(int(num_of_iterations)):
        # find 4 random points to calculate a homography
        rand_gen = np.random.default_rng()
        indexes = rand_gen.choice(corr.shape[0], size=4, replace=False)
        random_four = corr[indexes, :]

        # call the homography function on those points
        temp_homography = calculate_homography(random_four)

        if np.linalg.matrix_rank(temp_homography) < 3:
            continue
        temp = []

        for j in range(len(corr)):
            d = calculate_error(corr[j], temp_homography)
            if d < 5:
                temp.append(corr[j])

        if len(temp) > len(max_inliers):
            max_inliers = temp
            final_homography = temp_homography

        if len(max_inliers) > (len(corr) * thresh):
            break
    return final_homography, max_inliers


def get_homography_inliers(keypoints1, keypoints2, matches):
    correspondence_points = []
    for match in matches:
        (x1, y1) = keypoints1[match[0]].pt
        (x2, y2) = keypoints2[match[1]].pt
        correspondence_points.append([x1, y1, x2, y2])
    correspondence_list = np.asarray(correspondence_points)

    # run ransac algorithm
    threshold = .4
    H, inliers = ransac(correspondence_list, threshold)
    return H, inliers


def keypoint_matching(img1, img1_keypoints_des, img2, img2_keypoints_des):
    descriptors1 = img1_keypoints_des[1]
    descriptors2 = img2_keypoints_des[1]

    # using brute force
    d = cdist(descriptors1, descriptors2)

    # check if the images are same
    indexes_img1 = np.asarray([row.argmin() for row in d]).reshape(-1, 1)
    if np.diag(d).sum() == 0:
        matches = np.append(indexes_img1, indexes_img1, 1)
        return matches

    good_matches = {}

    # cross-check matches to remove false matches
    for i in range(len(indexes_img1)):
        if d[:, indexes_img1[i]].argmin() == i:
            good_matches[d[i, indexes_img1[i].item(0)]] = (i, indexes_img1[i])

    good_matches = collections.OrderedDict(sorted(good_matches.items()))
    matches = np.array(list(good_matches.values()), dtype=int)
    return matches


# Calculate SIFT keypoints and descriptors
def sift_keypoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img, None)
    return keypoints1, descriptors1


# Calculate panorama size and offset
def get_pano_size_offest(H, grayimg1, grayimg2):
    height1, width1 = grayimg1.shape[:2]
    height2, width2 = grayimg2.shape[:2]
    top_left = np.dot(H, np.asarray([0, 0, 1]))
    top_right = np.dot(H, np.asarray([width2, 0, 1]))
    bottom_left = np.dot(H, np.asarray([0, height2, 1]))
    bottom_right = np.dot(H, np.asarray([width2, height2, 1]))

    # normalize the corners
    top_left = top_left / top_left[2]
    top_right = top_right / top_right[2]
    bottom_left = bottom_left / bottom_left[2]
    bottom_right = bottom_right / bottom_right[2]

    left = int(min(top_left[0], bottom_left[0], 0))
    right = int(max(top_right[0], bottom_right[0], width1))
    # Use the corners to determine Width
    width = right - left

    top = int(min(top_left[1], top_right[1], 0))
    bottom = int(max(bottom_left[1], bottom_right[1], height1))

    # Use the corners to determine Height
    height = bottom - top
    offset_x = int(min(top_left[0], bottom_left[0], 0))
    offset_y = int(min(top_left[1], top_right[1], 0))
    return height, offset_x, offset_y, width


def warp_image(img1, img2, H):
    grayimg1 = get_gray_image(img1)
    grayimg2 = get_gray_image(img2)

    height1, width1 = img1.shape[:2]

    # Inverse the Homography matrix
    H = np.linalg.inv(H)

    height, ox, oy, width = get_pano_size_offest(H, grayimg1, grayimg2)
    ox = -ox
    oy = -oy
    panorama = np.zeros((height, width, 3), np.uint8)

    translation = np.asarray([
        [1.0, 0.0, ox],
        [0, 1.0, oy],
        [0.0, 0.0, 1.0]
    ])

    H = np.dot(translation, H)
    cv2.warpPerspective(img2, H, (width, height), panorama)
    panorama[oy:height1 + oy, ox:ox + width1] = img1
    panorama = panorama[oy:height1 + oy, :]
    return panorama


# Stitch edges to panorama with the remaining indexes in matches_dict
def stitch_edges(matches_dict, middle_img, images, gray_images_list, keypoints_descriptors_list):
    pano = middle_img
    if len(matches_dict) == 2:
        for k, v in matches_dict.items():
            pano = merge_images_with_pano(pano, images, gray_images_list, k, keypoints_descriptors_list)
    return pano


def merge_images_with_pano(img1, images, gray_images_list, i, keypoints_descriptors_list):
    img2 = images[i]
    grayimg1 = get_gray_image(img1.astype('uint8'))
    grayimg2 = gray_images_list[i]
    keypoints1, descriptors1 = sift_keypoints(grayimg1)
    keypoints2, descriptors2 = keypoints_descriptors_list[i]
    matches = get_matches(grayimg1, grayimg2, (keypoints1, descriptors1),
                          (keypoints2, descriptors2))
    H, inliers = get_homography_inliers(keypoints1, keypoints2, matches[:500])
    return warp_image(img1, img2, H)


def merge_images(images, gray_images_list, i, k, keypoints_descriptors_list):
    img1 = images[i]
    img2 = images[k]
    keypoints1, descriptors1 = keypoints_descriptors_list[i]
    keypoints2, descriptors2 = keypoints_descriptors_list[k]
    grayimg1 = gray_images_list[i]
    grayimg2 = gray_images_list[k]
    matches = get_matches(grayimg1, grayimg2, (keypoints1, descriptors1),
                          (keypoints2, descriptors2))
    H, inliers = get_homography_inliers(keypoints1, keypoints2, matches[:500])
    return warp_image(img1, img2, H)


# Calculate matches between images using brute-force method
def get_matches(grayimg1, grayimg2, img1_features, img2_features):
    (keypoints1, descriptors1) = img1_features
    (keypoints2, descriptors2) = img2_features
    matches = keypoint_matching(grayimg1, (keypoints1, descriptors1), grayimg2, (keypoints2, descriptors2))
    return matches


def get_gray_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def image_stitching(data_dir, images):
    keypoints_descriptors_list = []
    gray_images_list = []

    panorama = None
    for image in images:
        keypoints, descriptors = sift_keypoints(image)
        keypoints_descriptors_list.append((keypoints, descriptors))
        gray_images_list.append(get_gray_image(image))

    matches_dict = {}
    middle_images_list = []

    # Check if only two images
    if len(images) == 2:
        panorama = merge_images(images, gray_images_list, 0, 1, keypoints_descriptors_list)
    else:
        for i in range(len(images)):
            grayimg1 = gray_images_list[i]
            keypoints1, descriptors1 = keypoints_descriptors_list[i]
            for j in range(len(images)):
                if i != j:
                    if matches_dict and j in matches_dict.keys() and matches_dict[j] == i:
                        matches_dict[i] = j
                    else:
                        keypoints2, descriptors2 = keypoints_descriptors_list[j]
                        grayimg2 = gray_images_list[j]
                        matches = get_matches(grayimg1, grayimg2, (keypoints1, descriptors1),
                                              (keypoints2, descriptors2))
                        print('Matches between image:', i, ' and ', j, ':', len(matches))

                        H, inliers = get_homography_inliers(keypoints1, keypoints2, matches)
                        print('Inliers found for image :', i, ' in ', j, ':', len(inliers), '\n')

                        if len(inliers) / len(matches) > 0.25:
                            if i not in matches_dict:
                                matches_dict[i] = j
                            else:
                                if not middle_images_list:
                                    middle_images_list.append([i])
                                    del matches_dict[i]
                                    break
                                else:
                                    new_matches_list = []
                                    index = -1
                                    for l in middle_images_list:
                                        new_matches_list = list(l).copy()
                                        if list(l).__contains__(matches_dict[i]):
                                            new_matches_list.insert(list(l).index(matches_dict[i]), i)
                                        elif list(l).__contains__(j):
                                            new_matches_list.insert(list(l).index(j), i)
                                        else:
                                            middle_images_list.append([i])
                                            break
                                        index += 1
                                    if not index == -1:
                                        middle_images_list[index] = new_matches_list
                                    del matches_dict[i]
                                    break

        # Check if there are images in middle list
        if not middle_images_list:
            if len(matches_dict) > 0:
                first_key = next(iter(matches_dict))
                i = first_key
                k = matches_dict[first_key]
                panorama = merge_images(images, gray_images_list, i, k, keypoints_descriptors_list)
        else:
            list_sizes = [len(x) for x in middle_images_list]
            middle_images_list = list(middle_images_list[list_sizes.index(max(list_sizes))])

            if len(middle_images_list) == 1:
                panorama = stitch_edges(matches_dict, images[middle_images_list[0]], images, gray_images_list,
                                        keypoints_descriptors_list)
            else:
                for i in range(len(middle_images_list) - 1):
                    if i == 0:
                        panorama = merge_images(images, gray_images_list, middle_images_list[i],
                                                middle_images_list[i + 1], keypoints_descriptors_list)
                    else:
                        panorama = merge_images_with_pano(panorama, images, gray_images_list, middle_images_list[i + 1],
                                                          keypoints_descriptors_list)
                panorama = stitch_edges(matches_dict, panorama, images, gray_images_list, keypoints_descriptors_list)
    if panorama is not None:
        cv2.imwrite(os.path.join(data_dir, DEFAULT_FILENAME), panorama)


# Read images with jpg
def read_img():
    start_time = time.time()
    data_dir = (sys.argv[1:])[0]
    data_dir = os.path.join(os.getcwd(), os.path.abspath(data_dir)).replace("\\", "/")
    images = []
    for file in os.listdir(data_dir):
        if file.endswith(IMAGE_EXTENSION):
            images.append(cv2.imread(os.path.join(data_dir, file)))

    # Stitch Images
    image_stitching(data_dir, images)
    print("Total Runtime--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    read_img()
