import cv2
import numpy as np


#img = cv2.imread('test.png')
#img = cv2.imread('tmt.jpeg')
img = cv2.imread('tmt2.jpg')

gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
red_lower_hsv = np.array([0,0,0])
red_upper_hsv = np.array([10,255,255])
hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Masking
initial_mask = cv2.inRange(hsv, red_lower_hsv, red_upper_hsv)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
closing = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40))
erosion = cv2.erode(closing,kernel,iterations = 2)
dilation = cv2.dilate(erosion,kernel,iterations = 2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)




mask = opening

initial_segment = cv2.bitwise_and(img, img, mask=initial_mask)

segment = cv2.bitwise_and(img, img, mask=mask)


#Show Image List
show_imagename = ['Original Image',
                  'GrayScale',
                  'initial mask',
                  'erosion',
                  'dilation',
                  'closing',
                  'opening',
                  'final mask',
                  'hsv',
                  'initial_segment',
                  'segment'
                  ]
show_image = [img,
              gs,
              initial_mask,
              erosion,
              dilation,
              closing,
              opening,
              mask,
              hsv,
              initial_segment,
              segment
              ]

n_showimg = len(show_image)


resize = True
rwidth = 550
rheight = 600
#Image Showing Sequencing
for k in range (0,n_showimg):
    if resize == True:
        cv2.namedWindow(show_imagename[k],cv2.WINDOW_NORMAL)    
        cv2.resizeWindow(show_imagename[k],rwidth,rheight)    
    cv2.imshow(show_imagename[k],show_image[k])

# SHOWING IMAGE ALGORITHMS ---------------------------------------------------------------------- END





cv2.waitKey(0)
cv2.destroyAllWindows()
