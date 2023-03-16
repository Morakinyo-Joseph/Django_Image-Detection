import cv2
import numpy as np
import os



def blue_detection(image_obj):
    image = str(image_obj.item)

    # Get the current working directory
    cwd = os.getcwd()

    # Define the path to the image file relative to the current working directory
    image_path = os.path.join(cwd, "media", f"{image}")

    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the blue color
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the position and percentage of the blue object
    x_pos = 0
    y_pos = 0
    area = 0
    image_area = img.shape[0] * img.shape[1]

    # Loop over all contours to find the largest blue object
    for cnt in contours:
        # Get the area of the contour
        cnt_area = cv2.contourArea(cnt)
        # If the contour area is smaller than 1% of the image area, skip it
        if cnt_area < image_area * 0.01:
            continue
        # Get the position of the contour
        x,y,w,h = cv2.boundingRect(cnt)
        # If the contour is larger than the previous one, update the position and area
        if cnt_area > area:
            x_pos = x + w // 2
            y_pos = y + h // 2
            area = cnt_area

    # Calculate the percentage of the blue object in the image
    percentage = area / image_area * 100

    # Print out the position and percentage of the blue object
    print("Position of the blue object: ({}, {})".format(x_pos, y_pos))
    print("Percentage of the blue object in the image: {:.2f}%".format(percentage))


    # Draw contours on the original image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # # Resize the output window  
    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Output", 600, 800)

    # # Display the output image
    # cv2.imshow("Output", img)
    # cv2.waitKey(0)

    # to delete the file afterwards
    image_obj.delete()

    # to get the image file and remove it
    cwd = os.getcwd()
    file_path = os.path.join(cwd, "media", f"{image}")

    if os.path.exists(file_path):
        os.remove(file_path)
        print("file was deleted")

    position = "Position of the blue object: ({}, {})".format(x_pos, y_pos)
    percentages = "Percentage of the blue object in the image: {:.2f}%".format(percentage)

    results = position, percentages

    return results
