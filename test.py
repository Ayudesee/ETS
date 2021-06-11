import cv2
import numpy as np
import os


# 49, 54город, 62-повороты+светофоры, 700 - туман, 300 - ночь+объезд, 406 - солнце, ничего не видно

def main_func(start_file_number):
    FILE_I_END = 1
    while True:
        file_name = f'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{FILE_I_END}.npy'

        if os.path.isfile(file_name):
            FILE_I_END += 1
        else:
            FILE_I_END -= 1
            print('Last file:', FILE_I_END)
            break

    for i in range(FILE_I_END):
        if start_file_number + i > FILE_I_END:
            break
        screen = np.load(f"D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{start_file_number + i}.npy",
                         allow_pickle=True)

        n = 0
        while n < 500:
            print(f"n:{n}, {screen[n][1]}, file:{start_file_number + i}")

            # cv2.line(screen[n][0], (105, 190), (145, 110), (255, 255, 0), 1)
            # cv2.line(screen[n][0], (155, 110), (195, 190), (255, 255, 0), 1)

            # processed_image1 = transform2(screen[n][0])
            # processed_image1 = proc_screen(screen[n][0])
            # processed_image1 = sobel(screen[n][0])

            # processed_image1 = lap(screen[n][0])
            # processed_image1 = find_lines_inrange(screen[n][0], 0, 0, 77, 255, 87, 255)
            processed_image2 = find_traffic_light(screen[n][0])
            # processed_image3 = find_lines_inrange(screen[n][0], 98, 90, 123, 176, 255, 255)
            processed_image1 = lanes(screen[n][0])
            processed_image = cv2.add(processed_image1, processed_image2)
            # processed_image = cv2.add(processed_image, processed_image3)
            # processed_image = find_lines_inrange(screen[n][0])
            # processed_image = lanes(screen[n][0])

            cv2.imshow('raw', screen[n][0])
            cv2.imshow('processed', processed_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):  # next file
                cv2.destroyAllWindows()
                break
            n += 1
        i += 1


def proc_screen(original_image):
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    cv2.imshow('hsv', hsv_image)
    cv2.fillPoly(hsv_image, np.array([[[0, 100], [300, 100], [300, 0], [0, 0]]]), 0)
    vertices_road = np.array([[90, 200], [130, 150], [170, 150], [210, 200]])
    left_coordinate = []
    right_coordinate = []
    mask = np.full_like(original_image, fill_value=0)
    cv2.fillPoly(mask, [vertices_road], (255, 255, 255))

    line_image = original_image * 0
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    pts1 = np.float32([[130, 120], [170, 120], [90, 200], [210, 200]])  # road
    pts2 = np.float32([[90, 20], [210, 20], [90, 200], [210, 200]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(gray, M, (300, 200))

    minLineLength = 10
    maxLineGap = 8

    canny = cv2.Canny(image=warped, threshold1=80, threshold2=150, apertureSize=3)  # 80 150 good
    canny = cv2.GaussianBlur(canny, (3, 3), 0)
    cv2.imshow('canny', canny)
    lines = cv2.HoughLinesP(image=canny, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=22,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y2 - y1 == 0:
                continue
            slope = (x2 - x1) / (y2 - y1)
            angle = np.rad2deg(np.arctan(slope))
            if abs(angle) <= 65 and slope < 0:
                left_coordinate.append([x1, y1, x2, y2])
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            elif abs(angle) <= 65 and slope > 0:
                right_coordinate.append([x1, y1, x2, y2])
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    M = cv2.getPerspectiveTransform(pts2, pts1)
    line_image = cv2.warpPerspective(line_image, M, (300, 200))
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    canny = cv2.warpPerspective(canny, M, (300, 200))
    processed_image = cv2.addWeighted(canny, 0.4, line_image, 0.6, 1)
    # processed_image = line_image
    cv2.fillPoly(processed_image, np.array([[[225, 130], [290, 130], [290, 180], [225, 180]]]), 0)
    return processed_image


def transform2(original_image):
    # pts1_mm = np.float32([[0, 0], [71, 0], [0, 50], [71, 50]])  # minimap
    # pts2_mm = np.float32([[229, 150], [300, 150], [229, 200], [300, 200]])

    pts1 = np.float32([[145, 110], [155, 110], [105, 190], [195, 190]])  # road
    pts2 = np.float32([[130, -50], [170, -50], [100, 200], [200, 200]])

    # Y = original_image * 1
    # minimap = Y[130:180, 224:295, :]

    M = cv2.getPerspectiveTransform(pts1, pts2)
    # M_m = cv2.getPerspectiveTransform(pts1_mm, pts2_mm)
    # warped_mm = cv2.warpPerspective(minimap, M_m, (300, 200))

    warped = cv2.warpPerspective(original_image, M, (300, 200))

    mask = np.full_like(warped, 255)
    # cv2.fillPoly(mask, [vertices_roi_minimap_warped], (0, 0, 0))
    masked = cv2.bitwise_and(warped, mask)

    # CANNY
    canny_warped = cv2.GaussianBlur(masked, (5, 5), 0)
    canny_warped = cv2.cvtColor(canny_warped, cv2.COLOR_RGB2GRAY)

    # cv2.imshow('canny_warped', canny_warped)
    canny_warped = cv2.Canny(canny_warped, threshold1=80, threshold2=100, apertureSize=3)

    # #FINDING LINES
    minLineLength = 2
    maxLineGap = 30
    left_coordinate = []
    right_coordinate = []
    try:
        lines = cv2.HoughLinesP(image=canny_warped, rho=1, theta=np.pi / 180, threshold=20,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        canny_warped = cv2.cvtColor(canny_warped, cv2.COLOR_GRAY2RGB)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y2 - y1 == 0:
                    # cv2.line(canny_warped, (x1, y1), (x2, y2), (100, 100, 100), 3)
                    continue
                slope = (x2 - x1) / (y2 - y1)
                angle = np.rad2deg(np.arctan(slope))
                if abs(angle) <= 60 and slope < 0:
                    left_coordinate.append([x1, y1, x2, y2])
                    # cv2.line(canny_warped, (x1, y1), (x2, y2), (0, 255, 0), 3)
                elif abs(angle) <= 60 and slope > 0:
                    right_coordinate.append([x1, y1, x2, y2])
                    # cv2.line(canny_warped, (x1, y1), (x2, y2), (255, 0, 0), 3)
    except:
        print('no lines')

    canny_warped = cv2.fillPoly(canny_warped, np.array([[[225, 130], [290, 130], [290, 200], [225, 200]]]), 0)

    return canny_warped


def find_traffic_light(original_image):
    # 116, 86, 63 - 168, 255, 255 RED
    # 88, 83, 175 - 96, 255, 255 - YELLOW
    # 32, 78, 58 - 73, 255, 255 - GREEN
    hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    lower_red = np.array([116, 86, 63])
    upper_red = np.array([168, 255, 255])
    lower_yellow = np.array([88, 83, 175])
    upper_yellow = np.array([96, 255, 255])
    lower_green = np.array([32, 78, 58])
    upper_green = np.array([73, 255, 255])
    mask_r = cv2.inRange(hsv, lower_red, upper_red)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_g = cv2.inRange(hsv, lower_green, upper_green)

    red_screen = cv2.bitwise_and(original_image, original_image, mask=mask_r)
    yellow_screen = cv2.bitwise_and(original_image, original_image, mask=mask_y)
    green_screen = cv2.bitwise_and(original_image, original_image, mask=mask_g)
    processed_image = cv2.add(red_screen, yellow_screen)
    processed_image = cv2.add(processed_image, green_screen)

    return processed_image


def find_lines_inrange(original_image, hue_lower, sat_lower, value_lower, hue_upper, sat_upper, value_upper): #35, 34, 96, 102, 48, 169 - white color(file 75); 0, 0, 113, 101, 25, 185(file 461)
    #middle - 0, 0, 96, 101, 101, 185;   29, 0, 56, 107, 32, 255 for 321(night)
    # 90 0 77 114 87 255
    hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    lower = np.array([hue_lower, sat_lower, value_lower])
    upper = np.array([hue_upper, sat_upper, value_upper])
    mask = cv2.inRange(hsv, lower, upper)

    screen = cv2.bitwise_and(original_image, original_image, mask=mask)
    return screen


def find_color_with_taskbars():  # 116, 86, 63 - 168, 255, 255
    file = np.load(f"D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-507.npy", allow_pickle=True)
    original_image = file[0][0]
    cv2.namedWindow('Trackbars')
    hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    cv2.createTrackbar("L - H", "Trackbars", 0, 180, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 77, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 255, 180, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 87, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    while True:
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_red = np.array([l_h, l_s, l_v])
        upper_red = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        processed_image = cv2.bitwise_and(hsv, original_image, mask=mask)
        cv2.imshow('hsv', mask)
        cv2.imshow('raw', original_image)
        cv2.imshow('processed_image', processed_image)
        if cv2.waitKey(1) == 27:
            break

    # processed_image = original_image
    return processed_image


def find_edges_with_sobel():
    file = np.load(f"D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-569.npy", allow_pickle=True)
    original_image = file[0][0]
    cv2.namedWindow('Trackbars')
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    cv2.createTrackbar("dx", "Trackbars", 1, 9, nothing)
    cv2.createTrackbar("dy", "Trackbars", 1, 9, nothing)
    cv2.createTrackbar("ksize", "Trackbars", 3, 17, nothing)
    cv2.createTrackbar("kernel", "Trackbars", 1, 17, nothing)

    while True:
        dx = cv2.getTrackbarPos("dx", "Trackbars")
        dy = cv2.getTrackbarPos("dy", "Trackbars")
        ksize = cv2.getTrackbarPos("ksize", "Trackbars")
        kernel = cv2.getTrackbarPos("kernel", "Trackbars")
        if ksize % 2 == 0:
            ksize += 1
        if kernel % 2 == 0:
            kernel += 1
        gray_median_blur = cv2.medianBlur(gray, kernel, 0)
        processed_image_sobel = cv2.Sobel(gray, ddepth=cv2.CV_8U, dx=dx, dy=dy, ksize=ksize)
        processed_image_lap = cv2.Laplacian(gray, ddepth=cv2.CV_8U, ksize=ksize)
        processed_image_blur = cv2.Laplacian(gray_median_blur, ddepth=cv2.CV_8U, ksize=ksize)

        cv2.imshow('raw', original_image)
        cv2.imshow('proc_sobel', processed_image_sobel)
        cv2.imshow('proc_blur_lap', processed_image_blur)
        cv2.imshow('proc_lap', processed_image_lap)
        if cv2.waitKey(1) == 27:
            break

    return processed_image_blur


def nothing(x):
    pass


def sobel(original_image):
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    processed_image = cv2.Sobel(gray, ddepth=cv2.CV_8U, dx=2, dy=1, ksize=5)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    processed_image = cv2.fillPoly(processed_image, np.array([[[225, 130], [290, 130], [290, 180], [225, 180]]]), 0)

    return processed_image


def lap(original_image):
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # cv2.fillPoly(gray,
                 # np.array([[[0, 200], [70, 200], [90, 120], [210, 120], [230, 200], [300, 200], [300, 0], [0, 0]]]), 0)

    processed_image = cv2.Laplacian(gray, ddepth=cv2.CV_8U, ksize=3)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    return processed_image


def lanes(original_image):
    canny_image = cv2.Canny(original_image, 90, 110, apertureSize=3)
    mask = np.full_like(original_image, fill_value=0)
    vertices_road = np.array([[90, 200], [100, 120], [200, 120], [210, 200]])
    vertices_speed = np.array([[0, 0], [76, 0], [76, 19], [0, 19]])
    cv2.fillPoly(mask, [vertices_road], 255)
    # cv2.fillPoly(mask, [vertices_speed], 255)
    canny_image = cv2.bitwise_and(cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB), mask)
    processed_image = canny_image
    processed_image[:18, :76, :] = cv2.add(processed_image[:18, :76, :], original_image[:18, :76, :])
    # cv2.imshow('canny', canny_image)

    return processed_image


# main_func(start_file_number=507)


# find_color_with_taskbars()
# find_edges_with_sobel()
