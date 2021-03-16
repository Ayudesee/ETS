import cv2
import numpy as np
from test import proc_screen, find_traffic_light, transform2, sobel, lap


def process_images(original_images):
    processed_images = []
    for original_image in original_images:
        processed_images.append(process_image_v2(original_image))

    processed_images = np.array(processed_images)
    return processed_images


# def process_image(original_image):
#     vertices_minimap = np.array([[160, 88], [214, 88], [214, 135], [160, 135]])  # 216x135
#     vertices_road = np.array([[0, 80], [40, 80], [70, 72], [150, 72], [180, 100], [216, 100], [216, 0], [0, 0]])
# 
#     mask = np.full_like(original_image, fill_value=0)
#     cv2.fillPoly(mask, [vertices_minimap], (255, 255, 255))
# 
#     masked = cv2.bitwise_and(original_image, mask)
#     canny = cv2.Canny(image=original_image, threshold1=100, threshold2=170)
#     canny = cv2.GaussianBlur(canny, (3, 3), 0)
# 
#     minLineLength = 20
#     maxLineGap = 5
# 
#     try:
#         lines = cv2.HoughLinesP(image=canny, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=10,
#                                 minLineLength=minLineLength, maxLineGap=maxLineGap)
#         canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
# 
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 if y2 - y1 == 0:
#                     continue
#                 slope = (x2 - x1) / (y2 - y1)
#                 angle = np.rad2deg(np.arctan(slope))
# 
#                 if abs(angle) <= 70:
#                     cv2.line(canny, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     except:
#         print('no lines')
# 
#     cv2.fillPoly(canny, [vertices_road], (0, 0, 0))
#     cv2.fillPoly(canny, [vertices_minimap], (0, 0, 0))
# 
#     processed_image = cv2.add(canny, masked)
#     return processed_image

def process_image_v1(original_image):
    Y = original_image * 1
    minimap = Y[130:180, 224:295, :]
    vertices_roi_minimap_warped = np.array([[300, 190], [228, 190], [300, 140], [300, 190]])

    pts1_mm = np.float32([[0, 0], [71, 0], [0, 50], [71, 50]])  # minimap
    pts2_mm = np.float32([[229, 150], [300, 150], [229, 200], [300, 200]])

    pts1 = np.float32([[140, 120], [160, 120], [90, 200], [210, 200]])  # road
    pts2 = np.float32([[100, 0], [200, 0], [100, 200], [200, 200]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    M_m = cv2.getPerspectiveTransform(pts1_mm, pts2_mm)
    warped_mm = cv2.warpPerspective(minimap, M_m, (300, 200))

    warped = cv2.warpPerspective(original_image, M, (300, 200))
    mask = np.full_like(warped, 255)
    cv2.fillPoly(mask, [vertices_roi_minimap_warped], (0, 0, 0))
    masked = cv2.bitwise_and(warped, mask)
    # CANNY
    canny_warped = cv2.GaussianBlur(masked, (3, 3), 0)
    canny_warped = cv2.cvtColor(canny_warped, cv2.COLOR_RGB2GRAY)
    canny_warped = cv2.Canny(canny_warped, threshold1=80, threshold2=100, apertureSize=3)

    # #FINDING LINES
    minLineLength = 20
    maxLineGap = 20
    left_coordinate = []
    right_coordinate = []
    try:
        lines = cv2.HoughLinesP(image=canny_warped, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=25,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        canny_warped = cv2.cvtColor(canny_warped, cv2.COLOR_GRAY2RGB)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y2 - y1 == 0:
                    continue
                slope = (x2 - x1) / (y2 - y1)
                angle = np.rad2deg(np.arctan(slope))
                if abs(angle) <= 60 and slope < 0:
                    left_coordinate.append([x1, y1, x2, y2])
                    cv2.line(canny_warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif abs(angle) <= 60 and slope > 0:
                    right_coordinate.append([x1, y1, x2, y2])
                    cv2.line(canny_warped, (x1, y1), (x2, y2), (255, 0, 0), 2)
    except:
        pass

    canny_warped = cv2.fillPoly(canny_warped, np.array([[[229, 150], [300, 150], [300, 200], [229, 200]]]), 0) # закрашиваем место под цветную миникарту
    # processed_image1 = cv2.add(canny_warped, warped_mm)

    processed_image2 = find_traffic_light(original_image)
    processed_image = cv2.add(canny_warped, processed_image2)
    return processed_image


def process_image_v2(original_image):
    processed_image1 = proc_screen(original_image)
    processed_image2 = find_traffic_light(original_image)
    processed_image = cv2.add(processed_image1, processed_image2)

    return processed_image


def process_image_v3(original_image):
    processed_image1 = transform2(original_image)
    processed_image2 = find_traffic_light(original_image)
    processed_image = cv2.add(processed_image1, processed_image2)

    return processed_image


def process_image_v4(original_image):
    processed_image1 = sobel(original_image)
    processed_image2 = find_traffic_light(original_image)
    processed_image = cv2.add(processed_image1, processed_image2)

    return processed_image


def process_image_v5(original_image):
    processed_image1 = lap(original_image)
    processed_image2 = find_traffic_light(original_image)
    processed_image = cv2.add(processed_image1, processed_image2)

    return processed_image


# file = np.load(f"D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-1.npy", allow_pickle=True)
# screen = process_image_v2(file[0][0])
# cv2.imshow('screen', screen)
# cv2.waitKey(0)
