import cv2
import numpy as np
import os



# 49, 54город,983 - светофор,  700 - туман, 300 - ночь+объезд, 406 - солнце, ничего не видно


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
        screen = np.load(f"D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{start_file_number + i}.npy",
                         allow_pickle=True)

        n = 0
        while n < 500:
            print(f"n:{n}, {screen[n][1]}, file:{start_file_number + i}")

            # cv2.line(screen[n][0], (105, 190), (145, 110), (255, 255, 0), 1)
            # cv2.line(screen[n][0], (155, 110), (195, 190), (255, 255, 0), 1)

            processed_image1 = transform2(screen[n][0])
            # processed_image1 = proc_screen(screen[n][0])
            processed_image2 = find_traffic_light(screen[n][0])
            processed_image = cv2.add(processed_image1, processed_image2)



            cv2.imshow('raw', screen[n][0])
            cv2.imshow('processed', processed_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):  # next file
                cv2.destroyAllWindows()
                break
            n += 1
        i += 1


def proc_screen(original_image):
    vertices_road = np.array([[90, 200], [130, 150], [170, 150], [210, 200]])
    left_coordinate = []
    right_coordinate = []
    mask = np.full_like(original_image, fill_value=0)
    cv2.fillPoly(mask, [vertices_road], (255, 255, 255))

    line_image = original_image * 0
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    pts1 = np.float32([[130, 120], [170, 120], [90, 200], [210, 200]])  # road
    pts2 = np.float32([[90, 20], [210, 20], [20, 200], [280, 200]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(gray, M, (300, 200))

    minLineLength = 10
    maxLineGap = 8

    canny = cv2.Canny(image=warped, threshold1=80, threshold2=150, apertureSize=3)  # 80 150 good
    canny = cv2.GaussianBlur(canny, (3, 3), 0)
    lines = cv2.HoughLinesP(image=canny, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=20,
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
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif abs(angle) <= 65 and slope > 0:
                right_coordinate.append([x1, y1, x2, y2])
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # M = cv2.getPerspectiveTransform(pts2, pts1)
    # line_image = cv2.warpPerspective(line_image, M, (300, 200))
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    processed_image = cv2.add(canny, line_image)
    return processed_image


def transform2(original_image):
    # pts1_mm = np.float32([[0, 0], [71, 0], [0, 50], [71, 50]])  # minimap
    # pts2_mm = np.float32([[229, 150], [300, 150], [229, 200], [300, 200]])

    pts1 = np.float32([[145, 110], [155, 110], [105, 190], [195, 190]])  # road
    pts2 = np.float32([[100, 0], [200, 0], [100, 200], [200, 200]])

    Y = original_image * 1
    # minimap = Y[130:180, 224:295, :]
    vertices_roi_minimap_warped = np.array([[300, 190], [228, 190], [300, 140], [300, 190]])

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

    cv2.imshow('canny_warped', canny_warped)
    canny_warped = cv2.Canny(canny_warped, threshold1=80, threshold2=100, apertureSize=3)

    # #FINDING LINES
    minLineLength = 5
    maxLineGap = 30
    left_coordinate = []
    right_coordinate = []
    try:
        lines = cv2.HoughLinesP(image=canny_warped, rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=15,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        canny_warped = cv2.cvtColor(canny_warped, cv2.COLOR_GRAY2RGB)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y2 - y1 == 0:
                    continue
                slope = (x2 - x1) / (y2 - y1)
                angle = np.rad2deg(np.arctan(slope))
                if abs(angle) <= 65 and slope < 0:
                    left_coordinate.append([x1, y1, x2, y2])
                    cv2.line(canny_warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif abs(angle) <= 65 and slope > 0:
                    right_coordinate.append([x1, y1, x2, y2])
                    cv2.line(canny_warped, (x1, y1), (x2, y2), (255, 0, 0), 2)
    except:
        print('no lines')

    canny_warped = cv2.fillPoly(canny_warped, np.array([[[229, 120], [300, 120], [300, 200], [229, 200]]]), 0)
    # processed_image = cv2.add(canny_warped, warped_mm)

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


def find_color_with_taskbars():  # 116, 86, 63 - 168, 255, 255
    file = np.load(f"D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-58.npy", allow_pickle=True)
    original_image = file[184][0]
    cv2.namedWindow('Trackbars')
    hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    cv2.createTrackbar("L - H", "Trackbars", 116, 180, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 86, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 63, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 168, 180, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
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


def nothing(x):
    pass


main(start_file_number=143)

# find_color_with_taskbars()


