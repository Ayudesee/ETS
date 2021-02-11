import cv2
import numpy as np
import math
from process_image import process_image

screen = np.load("D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-220.npy", allow_pickle=True)  # 182 ДТП


# screen = np.load("D:/Ayudesee/Other/Data/ets-data-shuffled/training_data-1.npy", allow_pickle=True)


def main():
    n = 0
    while n < 500:
        screen[n][0] = cv2.resize(screen[n][0], (216, 135))
        # vertices_minimap = np.array([[80, 44], [107, 44], [107, 61], [80, 61]])  #  108x68
        # vertices_road = np.array([[0, 29], [75, 29], [85, 38], [108, 38], [108, 0], [0, 0]])  #  108x68
        vertices_minimap = np.array([[160, 88], [214, 88], [214, 122], [160, 122]])  # 216x135
        vertices_road = np.array([[0, 58], [150, 58], [170, 79], [216, 79], [216, 0], [0, 0]])  # 216x135
        mask = np.full_like(screen[n][0], fill_value=0)
        cv2.fillPoly(mask, [vertices_minimap], (255, 255, 255))
        cv2.fillPoly(mask, [vertices_road], (255, 255, 255))

        masked = cv2.bitwise_and(screen[n][0], mask)
        canny = cv2.Canny(image=screen[n][0], threshold1=100, threshold2=111)

        cv2.fillPoly(canny, [vertices_road], (0, 0, 0))
        cv2.fillPoly(canny, [vertices_minimap], (0, 0, 0))
        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

        final_image = cv2.addWeighted(masked, 1, canny, 1, 0)
        # final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
        cv2.imshow('final_image', final_image)
        cv2.imshow('masked', masked)

        cv2.imshow('raw', screen[n][0])
        cv2.imshow('canny', canny)

        print(f"n:{n}")
        cv2.waitKey(0)
        n += 1


def test_find_color():
    n = 0
    while n < 500:
        screen[n][0] = cv2.resize(screen[n][0], (216, 135))
        # vertices_minimap = np.array([[80, 44], [107, 44], [107, 61], [80, 61]])  #  108x68
        # vertices_road = np.array([[0, 29], [75, 29], [85, 38], [108, 38], [108, 0], [0, 0]])  #  108x68
        vertices_minimap = np.array([[160, 88], [214, 88], [214, 122], [160, 122]])  # 216x135
        vertices_road = np.array(
            [[0, 100], [40, 90], [70, 72], [90, 72], [150, 72], [180, 90], [216, 100], [216, 0], [0, 0]])  # 216x135
        hsv_image = cv2.cvtColor(screen[n][0], cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(screen[n][0], cv2.COLOR_RGB2GRAY)

        # good roll - 50, 0, 100 - 250 150 200 (50, 0, 100 - 200 250 250 - with radar path)
        # good roll - 0 , 0, 100 - 200, 100, 250

        lower_white = np.array([80, 0, 100])
        upper_white = np.array([120, 250, 155])
        hsv_in_range = cv2.inRange(hsv_image, lower_white, upper_white)
        cv2.imshow('my_hsv', hsv_in_range)

        hsv_in_range = cv2.inRange(hsv_image, np.array([50, 0, 100]),
                                   np.array([250, 150, 200]))  # 50, 0, 100 - 250 150 200
        cv2.imshow('50, 0, 100 - 250 150 200', hsv_in_range)

        hsv_in_range = cv2.inRange(hsv_image, np.array([0, 100, 0]),
                                   np.array([100, 250, 150]))  # 0, 100, 0 - 100 250 150
        cv2.imshow('0, 100, 0 - 100 250 150', hsv_in_range)

        canny = cv2.Canny(gray, 160, 170)
        cv2.fillPoly(canny, [vertices_road], (0, 0, 0))
        cv2.fillPoly(canny, [vertices_minimap], (0, 0, 0))
        cv2.imshow('canny', canny)
        cv2.imshow('raw', screen[n][0])
        cv2.imshow('hsv_image', hsv_image)
        print(f"n:{n}, {screen[n][1]}")
        cv2.waitKey(0)
        n += 1


def hlp():
    n = 0
    while n < 500:
        screen[n][0] = cv2.resize(screen[n][0], (216, 135))
        # vertices_minimap = np.array([[80, 44], [107, 44], [107, 61], [80, 61]])  #  108x68
        # vertices_road = np.array([[0, 29], [75, 29], [85, 38], [108, 38], [108, 0], [0, 0]])  #  108x68
        vertices_minimap = np.array([[158, 86], [216, 86], [216, 135], [158, 135]])  # 216x135
        vertices_road = np.array(
            [[0, 100], [40, 100], [70, 72], [90, 72], [150, 72], [160, 100], [216, 100], [216, 0], [0, 0]])  # 216x135
        canny = cv2.Canny(screen[n][0], 100, 170)
        canny = cv2.GaussianBlur(canny, (5, 5), 0)
        hlp_screen = screen[n][0] * 1

        cv2.fillPoly(canny, [vertices_road], (0, 0, 0))
        cv2.fillPoly(canny, [vertices_minimap], (0, 0, 0))

        minLineLength = 20
        maxLineGap = 2
        try:
            lines = cv2.HoughLinesP(image=canny, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=10,
                                    minLineLength=minLineLength, maxLineGap=maxLineGap)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(hlp_screen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except:
            print('no lines')

        cv2.imshow('hlp_screen', hlp_screen)
        cv2.imshow('raw', screen[n][0])
        cv2.imshow('canny', canny)
        print(f"n:{n}, {screen[n][1]}")
        cv2.waitKey(0)
        n += 1


def proc():
    n = 0
    while n < 500:
        vertices_minimap = np.array([[160, 88], [214, 88], [214, 122], [160, 122]])  # 216x135
        vertices_road = np.array([[0, 80], [40, 80], [70, 72], [150, 72], [180, 100], [216, 100], [216, 0], [0, 0]])
        gray = cv2.cvtColor(screen[n][0], cv2.COLOR_RGB2GRAY)
        cv2.imshow('raw', screen[n][0])
        mask = np.full_like(screen[n][0], fill_value=255)
        cv2.fillPoly(mask, [vertices_minimap], (0, 0, 0))
        cv2.fillPoly(mask, [vertices_road], (0, 0, 0))

        masked = screen[n][0] * 1
        masked = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)
        # masked = cv2.inRange(masked, np.array([100, 50, 50]), np.array([179, 255, 255]))
        # masked = cv2.bitwise_and(masked, mask)

        minLineLength = 20
        maxLineGap = 2

        canny = cv2.Canny(image=gray, threshold1=150, threshold2=170)
        canny = cv2.GaussianBlur(canny, (3, 3), 0)
        cv2.imshow('canny-gray', canny)

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

        cv2.imshow('masked', masked)
        cv2.imshow('mask', mask)
        cv2.imshow('processed_image', processed_image)
        cv2.waitKey(0)
        n += 1


def finding_lines():
    n = 0
    while n < 500:
        # gray = cv2.cvtColor(screen[n][0], cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(screen[n][0], 120, 170)
        canny = cv2.GaussianBlur(canny, (3, 3), 1)

        vertices_minimap = np.array([[160, 88], [214, 88], [214, 135], [160, 135]])  # 216x135
        vertices_road = np.array([[0, 80], [40, 80], [70, 72], [150, 72], [180, 100], [216, 100], [216, 0], [0, 0]])

        minLineLength = 20
        maxLineGap = 20
        left_coordinate = []
        right_coordinate = []
        try:
            lines = cv2.HoughLinesP(image=canny, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=10,
                                    minLineLength=minLineLength, maxLineGap=maxLineGap)
            canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if y2 - y1 == 0:
                        continue
                    slope = (x2 - x1) / (y2 - y1)
                    angle = np.rad2deg(np.arctan(slope))
                    if abs(angle) <= 70 and slope < 0:
                        left_coordinate.append([x1, y1, x2, y2])
                        cv2.line(canny, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif abs(angle) < 30 and slope > 0:
                        right_coordinate.append([x1, y1, x2, y2])

                l_avg = np.average(left_coordinate, axis=0)
                r_avg = np.average(right_coordinate, axis=0)
                l = l_avg.tolist()
                r = r_avg.tolist()

        except:
            print('no lines')

        cv2.fillPoly(canny, [vertices_road], (0, 0, 0))
        cv2.fillPoly(canny, [vertices_minimap], (0, 0, 0))
        cv2.imshow('canny', canny)
        cv2.imshow('raw', screen[n][0])
        cv2.waitKey(0)
        n += 1


def transform():
    n = 0
    while n < 500:
        print(f"n:{n}, {screen[n][1]}")
        Y = screen[n][0] * 1
        minimap = Y[89:122, 160:214, :]
        vertices_roi_minimap_warped = np.array([[260, 200], [247, 158], [300, 100], [300, 200]])

        pts1_mm = np.float32([[0, 0], [53, 0], [0, 30], [53, 30]])
        pts2_mm = np.float32([[236, 164], [300, 164], [236, 200], [300, 200]])

        pts1 = np.float32([[98, 88], [128, 88], [63, 135], [158, 135]])
        pts2 = np.float32([[100, 100], [200, 100], [100, 200], [200, 200]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        M_m = cv2.getPerspectiveTransform(pts1_mm, pts2_mm)
        warped_mm = cv2.warpPerspective(minimap, M_m, (300, 200))

        warped = cv2.warpPerspective(screen[n][0], M, (300, 200))
        mask = np.full_like(warped, 255)
        cv2.fillPoly(mask, [vertices_roi_minimap_warped], (0, 0, 0))
        masked = cv2.bitwise_and(warped, mask)
        # CANNY
        canny_warped = cv2.GaussianBlur(masked, (3, 3), 0)
        canny_warped = cv2.Canny(canny_warped, threshold1=80, threshold2=100, apertureSize=3)

        cv2.imshow('blur+canny', canny_warped)
        # #FINDING LINES
        minLineLength = 20
        maxLineGap = 20
        left_coordinate = []
        right_coordinate = []
        try:
            lines = cv2.HoughLinesP(image=canny_warped, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=10,
                                    minLineLength=minLineLength, maxLineGap=maxLineGap)
            canny_warped = cv2.cvtColor(canny_warped, cv2.COLOR_GRAY2RGB)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if y2 - y1 == 0:
                        continue
                    slope = (x2 - x1) / (y2 - y1)
                    angle = np.rad2deg(np.arctan(slope))
                    if abs(angle) <= 35 and slope < 0:
                        left_coordinate.append([x1, y1, x2, y2])
                        cv2.line(canny_warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif abs(angle) <= 35 and slope > 0:
                        right_coordinate.append([x1, y1, x2, y2])
                        cv2.line(canny_warped, (x1, y1), (x2, y2), (255, 0, 0), 2)
        except:
            print('no lines')

        canny_warped = cv2.fillPoly(canny_warped, np.array([[[236, 164], [300, 164], [300, 200], [236, 200]]]), 0)
        final_image = cv2.add(canny_warped, warped_mm)

        cv2.imshow('f', final_image)
        cv2.imshow('canny_warped', canny_warped)
        cv2.imshow('raw', screen[n][0])
        cv2.waitKey(0)
        n += 1


def transform2():
    n = 0
    while n < 500:
        # cv2.line(screen[n][0], (224, 130), (295, 130), (0, 255, 0), 1)  # minimap
        # cv2.line(screen[n][0], (224, 180), (295, 180), (0, 255, 0), 1)
        #
        # cv2.line(screen[n][0], (90, 200), (140, 120), (0, 255, 0), 2)  # road
        # cv2.line(screen[n][0], (210, 200), (160, 120), (0, 255, 0), 2)

        pts1_mm = np.float32([[0, 0], [71, 0], [0, 50], [71, 50]])  # minimap
        pts2_mm = np.float32([[229, 150], [300, 150], [229, 200], [300, 200]])

        pts1 = np.float32([[140, 120], [160, 120], [90, 200], [210, 200]])  # road
        pts2 = np.float32([[100, 0], [200, 0], [100, 200], [200, 200]])

        cv2.imshow('raw', screen[n][0])
        print(f"n:{n}, {screen[n][1]}")
        Y = screen[n][0] * 1
        minimap = Y[130:180, 224:295, :]
        vertices_roi_minimap_warped = np.array([[300, 190], [228, 190], [300, 140], [300, 190]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        M_m = cv2.getPerspectiveTransform(pts1_mm, pts2_mm)
        warped_mm = cv2.warpPerspective(minimap, M_m, (300, 200))

        cv2.imshow('warped_mm', warped_mm)

        warped = cv2.warpPerspective(screen[n][0], M, (300, 200))
        cv2.imshow('warped_screen', warped)

        mask = np.full_like(warped, 255)
        cv2.fillPoly(mask, [vertices_roi_minimap_warped], (0, 0, 0))
        masked = cv2.bitwise_and(warped, mask)
        # CANNY
        canny_warped = cv2.GaussianBlur(masked, (3, 3), 0)
        canny_warped = cv2.cvtColor(canny_warped, cv2.COLOR_RGB2GRAY)
        canny_warped = cv2.Canny(canny_warped, threshold1=80, threshold2=100, apertureSize=3)

        cv2.imshow('blur+canny', canny_warped)
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
        final_image = cv2.add(canny_warped, warped_mm)

        cv2.imshow('f', final_image)
        cv2.imshow('canny_warped', canny_warped)
        cv2.imshow('raw', screen[n][0])
        cv2.waitKey(0)
        n += 1


# finding_lines()
# transform2()
# hlp()
# test_find_color()
proc()

# img = process_image(screen[0][0])
# cv2.imshow('1', img)
# cv2.waitKey(0)
