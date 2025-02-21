import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os

from numpy.ma.extras import average

input_dir = "./captured_images"
output_csv = "./result_csv/RGB.csv"

result = []

#Loop through all the image files in directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpeg",".jpg", ".png")):
        #load and process the image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue #Skip files that cannot be read as image

        #Noise Removal
        kernel = np.ones((3,3),np.uint8)

        #Strip mask
        crp_strip = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        cropped_strip = crp_strip.crop(box=(640,50,1000,3350))
        abc_strip = cv2.cvtColor(np.array(cropped_strip), cv2.COLOR_GRAY2BGR)
        imgray = cv2.cvtColor(abc_strip, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

        #noise removal
        thresh_strip = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        #Find Contour
        contour_strip, _ = cv2.findContours(thresh_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minarea = 10000
        valid_cnt = [cnt for cnt in contour_strip if cv2.contourArea(cnt)>minarea]

        #mask
        mask = np.zeros(thresh.shape, dtype="uint8")

        #Draw Contour
        for cnt in valid_cnt:
            cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

        #Erode the mask to represent the shrunked contour such that only necessary pixels are captured
        erosion = np.ones((1,1),np.uint8)
        shrunk = cv2.erode(mask, erosion, iterations=1)

        results = cv2.bitwise_and(abc_strip, abc_strip, mask=shrunk)

        # valid_strip = (mask[:, :, 0] != 0 | mask[:, :, 1] != 0 | mask[:, :, 2] != 0)
        valid_strip = mask != 0

        #Extract RGB
        rgb = results[valid_strip]
        average = np.mean(rgb, axis=0)

        result.append(
            {
                "Average R": average[0],
                "Average G": average[1],
                "Average B": average[2],
            }
        )
df = pd.DataFrame(result)
a = df.describe()
analyze = [{"Mean_R" : np.mean(df["Average R"]),
"Mean_G ": np.mean(df["Average G"]),
"Mean_B" : np.mean(df["Average B"]),
"Std_R" : np.std(df["Average R"]),
"Std_G" : np.std(df["Average G"]),
"Std_B" : np.std(df["Average B"]),
"Min_R" : np.min(df["Average R"]),
"Min_G" : np.min(df["Average G"]),
"Min_B" : np.min(df["Average B"]),
"Max_R" : np.max(df["Average R"]),
"Max_G" : np.max(df["Average G"]),
"Max_B" : np.max(df["Average B"])
}]
ana = pd.DataFrame(analyze)
df.join(ana)
box_id = input("ENTER BOX ID: ")
df.to_csv(f"{output_csv}_{box_id}", index=False)