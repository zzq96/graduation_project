import cv2
import  os
from ImageUtil import  *
import  numpy as np
class XiHua(object):
    @staticmethod
    def VThin(image,array):
        h = image.shape[0]
        w = image.shape[1]
        NEXT = 1
        for i in range(h):

            for j in range(w):
                if NEXT == 0:
                    NEXT = 1
                else:
                    M = image[i,j-1]+image[i,j]+image[i,j+1] if 0<j<w-1 else 1
                    if image[i,j] == 0  and M != 0:
                        a = [0]*9
                        for k in range(3):
                            for l in range(3):
                                if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                    a[k*3+l] = 1
                        sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                        image[i,j] = array[sum]*255
                        if array[sum] == 1:
                            NEXT = 0
        return image

    @staticmethod
    def HThin(image,array):
        h = image.shape[0]
        w = image.shape[1]

        NEXT = 1
        for j in range(w):
            for i in range(h):
                if NEXT == 0:
                    NEXT = 1
                else:
                    M = image[i-1,j]+image[i,j]+image[i+1,j] if 0<i<h-1 else 1
                    if image[i,j] == 0 and M != 0:
                        a = [0]*9
                        for k in range(3):
                            for l in range(3):
                                if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                    a[k*3+l] = 1
                        sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                        image[i,j] = array[sum]*255
                        if array[sum] == 1:
                            NEXT = 0
        return image

    @staticmethod
    def Xihua(image,num=10):
        iXihua =image.copy()# cv.CreateImage(cv.GetSize(image),8,1)
        # cv.Copy(image,iXihua)
        for i in range(num):
            XiHua.VThin(iXihua,XiHua.array)
            XiHua.HThin(iXihua,XiHua.array)
        return iXihua

    @staticmethod
    def Two(image):
        w = image.width
        h = image.height
        size = (w,h)
        iTwo = np.zeros(size, dtype= np.uint8)#cv.CreateImage(size,8,1)
        for i in range(h):
            for j in range(w):
                iTwo[i,j] = 0 if image[i,j] < 200 else 255
        return iTwo

    array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
             1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

if __name__ == "__main__":
    dir = r'Data\word'
    print(dir)
    for img_name in os.listdir(dir):

        img_path = os.path.join(dir, img_name)

        img, _ = ImageProcessing.get_img(img_path)
        img = cv2.resize(img, (50, 50))
        th,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # iTwo = Two(img)
        iThin = XiHua.Xihua(img)
        print(img)
        plt.imshow(iThin,"gray")
        plt.show()
