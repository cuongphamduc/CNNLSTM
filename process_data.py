import os
import cv2

outpath = r"data"

for i in os.listdir("video"):
    if "avi" in i or "mp4" in i:
        imgs = []
        count = 0

        cap = cv2.VideoCapture(os.path.join("video", i))
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if count % 5 == 0:
                imgs.append(frame)

            count += 1

        while len(imgs) % 15 != 0:
            imgs.pop(-1)

        for (j, k) in enumerate(imgs):
            jj = "{:=06d}".format(j)

            file_name = os.path.join(outpath, i.split(".")[0] + "_" + jj + ".png")
            cv2.imwrite(file_name, k)
