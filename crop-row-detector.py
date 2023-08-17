import cv2
import numpy as np
from icecream import ic
import math

def main():
  filename = "data/2023-03-17_Alm_rajgraes_cleopatra_jsj.tif"
  img = cv2.imread(filename)
  cv2.imwrite("output/00_input_image.png", img)

  # Calculate excess green image
  (B, G, R) = cv2.split(img)
  exg = 2 * G - R - B
  temp = 2.0*G - 1.0*R - 1.0*B
  threshold, bw = cv2.threshold(temp, 10, 255, cv2.THRESH_BINARY)
  
  cv2.imwrite("output/10_excess_green.png", temp)
  cv2.imwrite("output/12_segmented.png", bw)

  bw = ic(bw.astype(np.uint8))

  lines = cv2.HoughLines(bw, 1, np.pi / 360, 300, None, 0, 0)

  img_hough = np.copy(img)
  if lines is not None:
    for i in range(0, len(lines)):
      rho = lines[i][0][0]
      theta = lines[i][0][1]
      a = math.cos(theta)
      b = math.sin(theta)
      x0 = a * rho
      y0 = b * rho
      pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
      pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
      cv2.line(img_hough, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
  cv2.imwrite("output/20_hough_lines.png", img_hough)


  linesP = cv2.HoughLinesP(bw, 1, np.pi / 360, 300, 2, 50, 10)

  img_houghP = np.copy(img)
  if linesP is not None:
    for i in range(0, len(linesP)):
      l = linesP[i][0]
      cv2.line(img_houghP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

  cv2.imwrite("output/25_houghP_lines.png", img_houghP)

main()
