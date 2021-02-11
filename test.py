import cv2
import numpy as np
import os
from process_image import process_image


def main(start_file_number):
    FILE_I_END = 1
    while True:
        file_name = f'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{FILE_I_END}.npy'

        if os.path.isfile(file_name):
            print(FILE_I_END)
            FILE_I_END += 1
        else:
            FILE_I_END -= 1
            print('Last file:', FILE_I_END)
            break

    for i in range(FILE_I_END):
        if start_file_number + i > FILE_I_END:
            break
        screen = np.load(f"D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{start_file_number + i}.npy", allow_pickle=True)

        n = 0
        while n < 500:
            print(f"n:{n}, {screen[n][1]}, file:{start_file_number+i}")

            processed_image = hsv_screen(screen[n][0])

            cv2.imshow('raw', screen[n][0])
            cv2.imshow('processed', processed_image)
            cv2.waitKey(0)
            n += 1
        i += 1


def hsv_screen(original_image):
        vertices_minimap = np.array([[160, 88], [214, 88], [214, 122], [160, 122]])  # 216x135
        vertices_road = np.array([[0, 80], [40, 80], [70, 72], [150, 72], [180, 100], [216, 100], [216, 0], [0, 0]])

        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        cv2.imshow('raw', original_image)
        mask = np.full_like(original_image, fill_value=255)
        cv2.fillPoly(mask, [vertices_minimap], (0, 0, 0))
        cv2.fillPoly(mask, [vertices_road], (0, 0, 0))

        masked = original_image
        masked = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)


        minLineLength = 20
        maxLineGap = 2

        canny = cv2.Canny(image=gray, threshold1=150, threshold2=170)
        canny = cv2.GaussianBlur(canny, (3, 3), 0)

        cv2.fillPoly(canny, [vertices_road], (0, 0, 0))
        cv2.fillPoly(canny, [vertices_minimap], (0, 0, 0))

        lines = cv2.HoughLinesP(image=canny, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=10,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(masked, (x1, y1), (x2, y2), (0, 255, 0), 2)

        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        processed_image = cv2.add(masked, canny)

        return processed_image


def transform2(original_image):
    pts1_mm = np.float32([[0, 0], [71, 0], [0, 50], [71, 50]])  # minimap
    pts2_mm = np.float32([[229, 150], [300, 150], [229, 200], [300, 200]])

    pts1 = np.float32([[140, 120], [160, 120], [90, 200], [210, 200]])  # road
    pts2 = np.float32([[100, 0], [200, 0], [100, 200], [200, 200]])

    Y = original_image * 1
    minimap = Y[130:180, 224:295, :]
    vertices_roi_minimap_warped = np.array([[300, 190], [228, 190], [300, 140], [300, 190]])

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
        print('no lines')

    canny_warped = cv2.fillPoly(canny_warped, np.array([[[229, 150], [300, 150], [300, 200], [229, 200]]]), 0)
    processed_image = cv2.add(canny_warped, warped_mm)

    return processed_image


main(start_file_number=500)
