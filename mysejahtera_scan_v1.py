import cv2 
import numpy as np
import imutils

PATH_TEMPLATE_MSLOGO = r"C:\Users\ASUS\Desktop\Working\PROJECT\Mybotic Product\mySejahtera Scan\images\template images\mslogo.png"
PATH_TEMPLATE_TICKMARK = r"C:\Users\ASUS\Desktop\Working\PROJECT\Mybotic Product\mySejahtera Scan\images\template images\tickmark.png"
SIMILARITY_TH = 0.85
IMAGE_SCANNED = 'imagescanned.png'
DEBUG = True

def ScanMysejahtera(PATH_TEMPLATE):
    template = cv2.imread(PATH_TEMPLATE,0)
    #template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[::]
    #cv2.imshow("Template", template) 

    # load the image, convert it to grayscale, and initialize the bookkeeping variable to keep track of the matched region
    image = cv2.imread(IMAGE_SCANNED)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:   
        # resize the image according to the scale, and keep track of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))  
        r = gray.shape[1] / float(resized.shape[1])

        #if the resized image is smaller than the template, then break from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW: 
            break
        
        #detect edges in the resized, grayscale image and apply template matching to find the template in the image
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        #print(maxVal)

        if(maxVal >= SIMILARITY_TH):
            if(DEBUG is True):
                #if we have found a new maximum correlation value, then update the bookkeeping variable
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

                #unpack the bookkeeping variable and compute the (x, y) coordinates of the bounding box based on the resized ratio
                (_, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

                #draw a bounding box around the detected result and display the image
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.imshow("Image", image)
                cv2.waitKey()

            return True

def CaptureMysejahteraScan():
    cam = cv2.VideoCapture(0)
    retval, frame = cam.read()

    cv2.imwrite(IMAGE_SCANNED, frame)
    #cv2.imshow("IMAGE_SCANNED", frame)

    return retval

retval = CaptureMysejahteraScan()
if(retval is True):
    Tickmark_Scan = ScanMysejahtera(PATH_TEMPLATE_TICKMARK)
    MSLogo_Scan = ScanMysejahtera(PATH_TEMPLATE_MSLOGO)

    if(Tickmark_Scan is True and MSLogo_Scan is True):
        print("Tickmark:", Tickmark_Scan, "; MSLogo:", MSLogo_Scan)
    else:
        print("Tickmark:", Tickmark_Scan, "; MSLogo: ", MSLogo_Scan)

else: 
    print("Cannot read frame")