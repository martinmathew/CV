import cv2
import numpy as np


# For Your Eyes Only
frizzy = cv2.imread('img/frizzy.png')
froomer = cv2.imread('img/froomer.png')

frizzy_edge = cv2.Canny(frizzy, 100, 200)
froomer_edge = cv2.Canny(froomer,100, 200)

common = frizzy_edge & froomer_edge

cv2.imshow('Frizzy', common)
cv2.imshow('Froomer', froomer_edge)
cv2.waitKey(0)

# TODO: Find edges in frizzy and froomer images

# TODO: Display common edge pixels
