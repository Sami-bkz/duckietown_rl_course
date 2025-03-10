import cv2 
import numpy as np



def process_image(image):
    """
    Process the input image by masking the top two-thirds, detecting blue and white lines,
    finding the centroids of both lines, and calculating the distance from a reference point.

    Parameters:
    - image: The input image (RGB format).

    Returns:
    - The image with the centroids of blue and white lines and the reference point marked, and the distance from the reference point to the centroids.
    If no blue or white line is detected, returns -np.inf.
    """
    # image = self.boost_contrast_and_saturation(image)

    # Step 1: Mask the top two-thirds of the image (make them black)
    height, width, _ = image.shape
    # mask_height = height // 2  # Top third of the image is blacked out
    # image[mask_height:, :] = 0  # Set the top third to black (0)
    # cv2.imshow("Clipped Image", image)
    # cv2.waitKey(1)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    else:
        image = np.uint8(image)
    # Step 2: Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # # Define the range for yellow color in HSV
    # lower_yellow = np.array([15, 150, 150])  # Lower bound for yellow
    # upper_yellow = np.array([45, 255, 255])  # Upper bound for yellow
    lower_blue = np.array([20, 50, 100])  # Lower bound for blue
    upper_blue = np.array([50, 255, 255])  # Upper bound for blue

    # Define the range for white color in HSV
    lower_white = np.array([0, 0, 100])  # Lower bound for white
    upper_white = np.array([180, 25, 255])  # Upper bound for white

    # Create masks for blue and white regions
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Step 3: Find contours in the blue mask

    contours_blue, _ = cv2.findContours(
        mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 4: Find contours in the white mask
    contours_white, _ = cv2.findContours(
        mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    

    # Find the centroid of the blue region
    centroid_x_blue = None
    centroid_y_blue = None
    for contour in contours_blue:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Check if the contour is non-empty
            centroid_x_blue = int(M["m10"] / M["m00"])
            centroid_y_blue = int(M["m01"] / M["m00"])
            break  # Take the first detected blue contour

    # Find the centroid of the white region
    centroid_x_white, centroid_y_white = None, None
    for contour in contours_white:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Check if the contour is non-empty
            centroid_x_white = int(M["m10"] / M["m00"])
            centroid_y_white = int(M["m01"] / M["m00"])
            break  # Take the first detected white contour

    # Step 5: Define the reference point (x=center of image, y=10% of image height)
    ref_x = width
    ref_y = int(
        height / 2
    )  # int(height * 0.1)  # 10% of the image height (near the top)

    # Step 6: Calculate the distance from the reference point to the centroids
    distance_from_blue = -np.inf
    distance_from_white = -np.inf
    x_blue_center = None
    y_blue_center = None
    if centroid_x_blue is not None and centroid_y_blue is not None:
        # print(f"centroid_x_blue: {centroid_x_blue} centroid_y_blue: {centroid_y_blue}")
        x_blue_center = (centroid_x_blue/width - 0.5) * 2
        y_blue_center = -(centroid_y_blue/height - 0.5) * 2
        # distance from center
        distance_from_blue = np.sqrt(
            (centroid_x_blue - width) ** 2 + (centroid_y_blue - height / 2) ** 2
        )
    x_white_center = None
    y_white_center = None
    if centroid_x_white is not None and centroid_y_white is not None:
        x_white_center = (centroid_x_white/width - 0.5) * 2 
        y_white_center = -(centroid_y_white/height - 0.5) * 2
        # print(f"x_white_center: {x_white_center} y_white_center: {y_white_center}")
        distance_from_white = np.sqrt(
            (centroid_x_white - 0) ** 2 + (centroid_y_white - height / 2) ** 2
        )

    # Step 7: Create a new black image for the result
    result_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw the centroid (green for blue, red for white) and reference point (blue)
    if centroid_x_blue is not None and centroid_y_blue is not None:

        cv2.circle(
            result_image, (centroid_x_blue, centroid_y_blue), 2, (0, 255, 0), -1
        )  # Green dot for blue
    if centroid_x_white is not None and centroid_y_white is not None:
        cv2.circle(
            result_image,
            (centroid_x_white, centroid_y_white),
            2,
            (255, 255, 255),
            -1,
        )  # White dot for white

    # Draw reference point in blue
    cv2.circle(result_image, (ref_x, ref_y), 2, (255, 0, 0), -1)

    # Show the processed image
    # cv2.imshow("Processed Image", result_image)
    # cv2.waitKey(1)

    # Step 9: Determine the action based on the white centroid's position
    action_according_white = None
    if centroid_x_white is not None and centroid_y_white is not None:
        # Check if white centroid is in the left or right area
        if centroid_x_white < width / 2:  # In the left area

            action_according_white = "Go Right"
        else:  # In the right area
            action_according_white = "Go Left"

    action_according_blue = None
    if centroid_x_blue is not None and centroid_y_blue is not None:
        # Check if blue centroid is in the left or right area
        if centroid_x_blue < width / 2:  # In the left area
            action_according_blue = "Go Left"
        else:  # In the right area
            action_according_blue = "Go Right"

    # Return the result image and distances (for both blue and white lines)

    return (
        x_blue_center, y_blue_center,
        x_white_center, y_white_center,
        distance_from_blue,
        distance_from_white,
        action_according_white,
        action_according_blue,
    )

def compute_custom_reward(obs, a): 
        """ action : (vel_abs, vel_angle) """
        
        (   x_blue_center, y_blue_center,
            x_white_center, y_white_center,
            distance_from_blue,
            distance_from_white,
            action_fased_on_white,
            action_based_on_blue,
        ) = process_image(obs)
        if x_blue_center is None or y_blue_center is None:
            reward = -100
        else:
            speed_action = (a[0]*10. if a[0]>0.1 else -1.)
            reward = np.exp(-np.sqrt(x_blue_center**2+y_blue_center**2)**2) + speed_action #TODO
        return reward/10