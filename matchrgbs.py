#importing all the necessary libraries
from scipy.spatial import distance as dist
from imutils import perspective, contours
import numpy as np
import imutils
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10, 6)
import cv2 as cv
import builtins
zip = builtins.zip
import matplotlib.pyplot as plt
from tabulate import tabulate
from PIL import Image
import base64
import os
import joblib
# import sklearn
from imutils.perspective import order_points
from xgboost import XGBRegressor
import xgboost
import cv2
import os
import glob
import pandas as pd
# from google.colab.patches import cv2_imshow
import sys

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def extract_rgb_values(image):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    #cv2_imshow(thresh)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_area = 10000
    valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]
    orig=image.copy()
    l=[]
    ls = 0
    for contour in valid_contours:
      if cv.contourArea(contour) < 200:
        continue
      else:
          # Get the minimum area rectangle around the contour
          box = cv.minAreaRect(contour)
          box = cv.boxPoints(box) if not imutils.is_cv2() else cv.cv.BoxPoints(box)
          box = np.array(box, dtype="int")
          box = order_points(box)
          cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
          # Compute the midpoints between box points
          (tl, tr, br, bl) = box
          (tltrX, tltrY) = midpoint(tl, tr)
          (blbrX, blbrY) = midpoint(bl, br)
          (tlblX, tlblY) = midpoint(tl, bl)
          (trbrX, trbrY) = midpoint(tr, br)

          # Compute the distances between the midpoints
          dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
          dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
          pixels_per_metric = dB / float(10.08)
          dimA = dA / pixels_per_metric
          dimB = dB / pixels_per_metric
          length = max(dimA, dimB)
          breadth = min(dimA, dimB)
          l.append(length)
          cv.putText(orig, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
          cv.putText(orig, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    for i in l:
      if i > 140:
        ls = i

    mask = np.zeros_like(image, dtype=np.uint8)

    for cnt in valid_contours:
        cv.drawContours(mask, [cnt], -1, (255, 255, 255), thickness=cv.FILLED)

    contoured_image = image.copy()
    cv.drawContours(contoured_image, valid_contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2
    #cv2_imshow(contoured_image)
    # Erode the mask to shrink it slightly
    erosion_kernel = np.ones((2, 2), np.uint8)
    mask = cv.erode(mask, erosion_kernel, iterations=1)

    # Apply the mask to the original image
    result = cv.bitwise_and(image, mask)
    #cv2_imshow(result)

    # Get valid pixels for RGB extraction
    valid_mask = mask[:, :, 0] > 0
    rgb_values = image[valid_mask]


    if len(rgb_values) == 0:
        return [0, 0, 0]  # Handle case with no valid RGB values

    average_rgb = np.mean(rgb_values, axis=0)
    print(average_rgb)
    return average_rgb, thresh, contoured_image, result,ls

def whiteness_analysis(im):
    if im is None:
        raise ValueError("Image could not be loaded.")
    #image = cv.cvtColor(np.array(image_cropped), cv.COLOR_RGB2BGR)
    im_pil = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    # Cropping the main region (original cropped image)
    image_cropped = im_pil.crop(box=(640, 50, 3950, 3350))
    # Create a copy of the cropped image for splitting
    image_copy = image_cropped.copy()
    # Get dimensions of the cropped image
    width, height = image_copy.size
    # Calculate split points (40% for strip, 60% for bowl)
    split_point = int(width * 0.35)
    # Split the image into two parts: strip (40%) and bowl (60%)
    strip_region = image_copy.crop((0, 0, split_point, height))  #
    bowl_region = image_copy.crop((split_point, 0, width, height))  #
    strip_rgb,thresh_strip, contoured_image_strip, result_strip,ls = extract_rgb_values(cv.cvtColor(np.array(strip_region), cv.COLOR_RGB2BGR))
    bowl_rgb, thresh_bowl, contoured_image_bowl, result_bowl,bls = extract_rgb_values(cv.cvtColor(np.array(bowl_region), cv.COLOR_RGB2BGR))
    # WI = 100 - np.sqrt((255 - bowl_rgb[0])**2 + (255 - bowl_rgb[1])**2 + (255 - bowl_rgb[2])**2)
    #cv2_imshow(result_strip)
    # scale_r = (255/strip_rgb[0])
    # scale_g = (255/strip_rgb[1])
    # scale_b = (255/strip_rgb[2])
    # new_r = bowl_rgb[0]*scale_r
    # new_g = bowl_rgb[1]*scale_g
    # new_b = bowl_rgb[2]*scale_b2
    # WI_correct = 100 - np.sqrt((255 - new_r)**2 + (255 - new_g)**2 + (255 - new_b)**2)
    # wi_ls_correct = WI_correct * (150/ls)

    # return bowl_rgb, strip_rgb, WI, WI_correct, thresh_bowl, contoured_image_bowl, result_bowl, scale_r, scale_g, scale_b, new_r, new_g, new_b,ls,wi_ls_correct
    return bowl_rgb, strip_rgb, thresh_bowl, contoured_image_bowl, result_bowl,ls

import os
import cv2 as cv
import pandas as pd
import json
from datetime import datetime
import pytz

# Specify the image file path
image_path = 'C:/Agsure/AI-ML/Whiteness_Index/content/S_1036_25_217_1_20250210_112956.jpg'  # Replace with your image path
image_dir = '/content'
output_dir = 'C:/Agsure/AI-ML/Whiteness_Index/content'
results_json_folder = os.path.join(output_dir, 'results_json')  # Folder to store JSON results

os.makedirs(results_json_folder, exist_ok=True)  # Create folder for JSON results

columns = ['Image', 'REF_WI', 'ls', 'strip_r', 'strip_g', 'strip_b', 'bowl_r', 'bowl_g', 'bowl_b', 'Exposure']

def process_single_image(image_path):
    result_dict = {}  # Dictionary to store the results
    try:
        # Extract the image name without the extension
        image_name = os.path.basename(image_path).replace('.jpg', '')
        a = image_name.split('_')  # Split after the first space

        im1 = cv.imread(image_path)
        # Assuming whiteness_analysis is defined somewhere
        bowl_rgb, strip_rgb, thresh_bowl, contoured_image_bowl, result_bowl, ls = whiteness_analysis(im1)

        india_timezone = pytz.timezone('Asia/Kolkata')  # IST timezone
        current_time = datetime.now(india_timezone).strftime("%Y-%m-%d %H:%M:%S")

        # Determine the response based on ls value
        if ls > 146:
            response = 'Yes'
        else:
            response = 'No'

        # Construct the result dictionary (this part is what gets returned)
        result_dict = {
            'Image': image_name,
            'strip_r': strip_rgb[0],
            'strip_g': strip_rgb[1],
            'strip_b': strip_rgb[2],
            'bowl_r': bowl_rgb[0],
            'bowl_g': bowl_rgb[1],
            'bowl_b': bowl_rgb[2],
            'ls': ls,
            'Response': response,  # Add the response to the result dictionary
            'DateTime': current_time  # Add the current date and time to the result dictionary
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    return result_dict  # Return the results as a dictionary

def save_result_to_json(result, json_file_path):
    # Check if the JSON file already exists
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            try:
                existing_data = json.load(json_file)  # Load existing data
                if not isinstance(existing_data, list):
                    existing_data = []  # If the data isn't a list, reset to an empty list
            except json.JSONDecodeError:
                existing_data = []  # If the file is empty or corrupt, initialize an empty list
    else:
        existing_data = []  # If no file exists, initialize an empty list

    # Append the new result to the existing data
    existing_data.append(result)

    # Save the updated data back to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)
        print(f"Result saved to {json_file_path}")


# Process a single image and get results as a dictionary
result = process_single_image(image_path)

# Show the result on the console
print("Processed Result:")
print(json.dumps(result, indent=4))  # Print the dictionary in a pretty JSON format

# Define the JSON output path
json_output_path = os.path.join(results_json_folder, 'result.json')

# Save the result to a JSON file
save_result_to_json(result, json_output_path)

