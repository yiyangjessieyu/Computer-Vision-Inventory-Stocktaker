import cv2 as cv

LOCAL_PATH = "/csse/users/yyu69/Desktop/COSC428/Project-april21/Computer-Vision-Inventory-Stocktaker/resources/"
INPUT_IMAGE_PATH = 'side_elastic_10.png'


img = cv.imread(LOCAL_PATH + INPUT_IMAGE_PATH,0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(50000)

# Again compute keypoints and check its number.
kp, des = surf.detectAndCompute(img,None)
print( len(kp) )


img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()
