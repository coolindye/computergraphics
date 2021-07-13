import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img1 = cv.imread('waldo-distancing-temp.jpg',
                 cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('waldo-social-distancing.jpg',
                 cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp1 = star.detect(img1, None)
kp2 = star.detect(img2, None)
# compute the descriptors with BRIEF
kp1, des1 = brief.compute(img1, kp2)
kp2, des2 = brief.compute(img2, kp1)

print(brief.descriptorSize())
#print(des.shape)

# create BFMatcher object
#bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
#matches = bf.match(des1, des2)
# Sort them in the order of their distance.
#matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
#img3 = cv.drawMatches(img1, kp1, img2, kp2,
#                      matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#plt.imshow(img3), plt.show()
