import os
import  random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from queue import  Queue
import pickle
#%%
class ImageProcessing(object):

    @staticmethod
    def reverse(img):
        return  255 - img

    @staticmethod
    def closing(img,kernel_size = 3):
        """
        开运算
        :param img:
        :param kernel_size:
        :return:
        """
        img = img.copy()
        img = ImageProcessing.reverse(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        img = cv2.dilate(img,kernel)
        img = cv2.erode(img,kernel)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
        img = ImageProcessing.reverse(img)
        return  img

    @staticmethod
    def opening(img,kernel_size = 3):
        img = img.copy()
        img = ImageProcessing.reverse(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
        img = cv2.erode(img,kernel)
        img = cv2.dilate(img,kernel)
        img = ImageProcessing.reverse(img)
        return  img

    @staticmethod
    def erode(img,kernel_size = 3):
        img = img.copy()
        img = ImageProcessing.reverse(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        img = cv2.erode(img, kernel)
        img = ImageProcessing.reverse(img)
        return  img

    @staticmethod
    def dilate(img,kernel_size = 3):
        img = img.copy()
        img = ImageProcessing.reverse(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        img = cv2.dilate(img, kernel)
        img = ImageProcessing.reverse(img)
        return  img

    @staticmethod
    def delete_spot(img, max_len = 10):
        """
        将与边框粘连的长宽小于max_len的块去掉,本问题中只需要考虑左右边框即可
        :param img:需要处理的图像
        :param max_len:
        :return:
        """
        img = img.copy()
        flag = np.zeros(img.shape)
        y = 0 #先处理左边框
        for x in range(img.shape[0]):
            if img[x][y] == 0 and flag[x][y] == 0:
                img, flag = ImageProcessing.__bfs(img, flag,(x,y),max_len)

        y = img.shape[1] - 1  #先处理左边框
        for x in range(img.shape[0]):
            if img[x][y] == 0 and flag[x][y] == 0:
                img, flag = ImageProcessing.__bfs(img, flag,(x,y),max_len)
        return  img

    moves = [(1,0),(-1,0), (0, 1), (0, -1), (1,1), (1, -1),(-1, 1), (-1, -1)]
    @staticmethod
    def __bfs(raw_img, flag,s_point,max_len ):
        # print(s_point)
        img = raw_img.copy()
        Up, Down, Left , Right = 0, img.shape[0] -1 ,0, img.shape[1] -1
        up, down, left, right = s_point[0], s_point[0],s_point[1], s_point[1]
        Q = Queue()
        Q.put(s_point)
        flag[s_point[0],s_point[1]] =1
        while not Q.empty():
            x,y = Q.get()
            for _x,_y in ImageProcessing.moves:
                new_x = x + _x
                new_y = y + _y
                if new_x < Up or new_x >Down or \
                        new_y < Left or new_y > Right or \
                        img[new_x,new_y] != 0:
                    continue
                flag[new_x,new_y] =1
                img[new_x, new_y] = 255
                Q.put((new_x, new_y))
                up = min(up, new_x)
                down = max(down, new_x)
                left = min(left, new_y)
                right = max(right, new_y)
        if down - up + 1 >= max_len or right - left + 1 >= max_len:
            return  raw_img,flag
        else :
            return  img,flag

    @staticmethod
    def crop_margin(img,crop =0):
        img = img.copy()
        flag = np.zeros(img.shape)
        height, width = img.shape[0],img.shape[1]
        # print(height, width)
        col_sum = np.sum(img==0, axis = 0)
        row_sum = np.sum(img==0, axis = 1)
        left = 0
        right = 0
        up = 0
        down = 0
        #寻找字体上边界
        for i in range(height):
            if row_sum[i] > 0:
                up = i
                break
        #寻找字体下边界
        for i in range(height - 1, -1, -1):
            if row_sum[i] > 0:
                down = i
                break
        #寻找字体左边界
        for i in range(width):
            if col_sum[i] > 0:
                left = i
                break
        #寻找字体右边界
        for i in range(width - 1, -1, -1):
            if col_sum[i] > 0:
                right = i
                break
        # print(up, down, left, right)
        left += random.randint(0,crop)
        right -= random.randint(0,crop)
        img = img[up:down, left:right]
        return  img

    @staticmethod
    def get_img(img_path):
        """

        :param img_path:
        :return:img.shape = (height,width)
        """
        PIL_img = Image.open(img_path)
        img =np.asarray(PIL_img.convert("L"))#转化为灰度图像
        # img.setflags(write=1)
        # plt.imshow(img,cmap="gray")
        # plt.show()
        th,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return img, PIL_img.convert('RGB')

    @staticmethod
    def plot_line(img, line,axis):
        img = img.copy()
        if axis ==0:
            img[:,line-2:line] = 150
        else:
            img[line-2:line,:] = 100
        return  img

    @staticmethod
    def get_outline(raw_img):
        img = raw_img.copy()
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x,y] == 0:
                    cnt = ImageProcessing.__white_plot_num(raw_img,(x,y))
                    if cnt == 0:
                        img[x,y] = 255
        return img

    @staticmethod
    def __white_plot_num(img,plot):
        x,y = plot
        Up, Down, Left , Right = 0, img.shape[0] -1 ,0, img.shape[1] -1
        cnt = 0
        for _x,_y in ImageProcessing.moves:
            new_x, new_y = x + _x, y + _y
            if new_x < Up or new_x >Down or \
                    new_y < Left or new_y > Right or \
                    img[new_x,new_y] == 0:
                continue
            cnt =cnt + 1
        # print(cnt)

        return  cnt

    @staticmethod
    def test(raw_img,th=4):
        img = raw_img.copy()
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if raw_img[x,y] == 0:
                    Up, Down, Left , Right = 0, img.shape[0] -1 ,0, img.shape[1] -1
                    cnt = 0
                    for _x,_y in ImageProcessing.moves:
                        new_x, new_y = x + _x, y + _y
                        if new_x < Up or new_x >Down or \
                                new_y < Left or new_y > Right or \
                                raw_img[new_x,new_y] != raw_img[x,y]:
                            continue
                        cnt =cnt + 1
                    if cnt <=th:
                        img[x,y] = 255-img[x,y]
        return  img

class Captha2Words(object):

    def __init__(self, k,weight, height):
        self.weight = weight
        self.height = height
        self.k = k

    def run(self, img_path):
        """

        :param img_path:
        :return: narray shape = (k,height, weight)
        """
        img, PIL_img = ImageProcessing.get_img(img_path)
        centers = self.Kmeans(img)
        self.plot_centers(PIL_img,centers)
        words = self.test(img, centers)
        return  words

    def test(self, raw_img, centers):
        # plt.imshow(img,cmap='gray')
        # plt.show()
        img = raw_img.copy()
        count = np.sum(img == 0,axis= 0)
        left_boundary,right_boundary = 0,0
        for i in range(img.shape[1]):
            if count[i] != 0 :
                left_boundary = i
                break
        for i in range(img.shape[1]-1,-1,-1):
            if count[i] != 0 :
                right_boundary = i
                break
        # print(left, right)
        min_quarter = (right_boundary - left_boundary)//self.k//4
        min_weight = (right_boundary - left_boundary)//self.k *7 //8

        left = left_boundary + min_weight
        split_lines = [left_boundary]#字符之间的分割线位置
        for i in range(self.k - 1):
            right = centers[i+1][1] - min_quarter
            if left > right :
                minx = 0
            else :
                minx = np.argmin(count[left:right])
            line = left + minx
            img = ImageProcessing.plot_line(img,line,axis = 0)
            left = line + min_weight
            split_lines.append(line)
        split_lines.append(right_boundary)
        img = ImageProcessing.plot_line(img,left_boundary,axis = 0)
        img = ImageProcessing.plot_line(img,right_boundary,axis = 0)
        # plt.imshow(img,cmap='gray')
        # plt.show()
        words = self.get_words(raw_img,split_lines)
        return  words

    def get_words(self,img, split_lines):
        words = np.zeros((self.k,self.height, self.weight))
        for i in range(self.k):
            words[i,:,:] = self.get_word(img[:,split_lines[i]:split_lines[i+1]])
        return  words

    def get_word(self,img):
        # plt.imshow(img,cmap="gray")
        # plt.show()
        count = np.sum(img==0, axis= 1)
        # print(count)
        up ,down = 0,0
        for i in range(img.shape[0]):
            if count[i] != 0:
                up = i
                break
        for i in range(img.shape[0]-1,-1,-1):
            if count[i] != 0:
                down = i
                break
        word1 = img[up:down,:]
        word2 = ImageProcessing.delete_spot(word1)
        word2 = cv2.resize(word2, (self.height, self.weight))
        th,word3= cv2.threshold(word2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        word4 = ImageProcessing.closing(word3)
        # fig,ax = plt.subplots(1,4)
        # ax[0].imshow(word1, cmap="gray")
        # ax[1].imshow(word2, cmap="gray")
        # ax[2].imshow(word3, cmap="gray")
        # ax[3].imshow(word4, cmap="gray")
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        return  word4


    def Kmeans(self,img):
        X = self.get_black_point(img)
        # plt.figure(figsize=(10,2))
        km = KMeans(n_clusters=self.k)
        y_pred = km.fit_predict(X)
        centers = km.cluster_centers_
        centers = np.asarray(centers,dtype = np.int16)
        centers = centers[np.argsort(centers[:,1])]
        # print(centers)
        # plt.scatter(X[:,1], 88-X[:,0], c = y_pred, linewidths=0.01)
        # plt.scatter(centers[:,1],88-centers[:,0],c='r', linewidths= 10)
        # count = np.sum(img == 0,axis= 0)
        # plt.plot(count,'b')
        # plt.show()
        return  centers

    def plot_centers(self, PIL_img, centers):
        im = PIL_img.copy()
        bgdr = ImageDraw.Draw(im)
        for y, x in centers:
            bgdr.ellipse((x-3, y-3, x+3, y+3), fill ="red", outline ='red')
        # im.show()
        # print(im)


    def get_black_point(self, img):
        len = np.sum(img == 0)
        X = np.zeros((len,2))
        cnt = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 0:
                    X[cnt,0],X[cnt,1] = i,j
                    cnt =cnt +1
        return X

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

    array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1, \
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1, \
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1, \
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1, \
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0, \
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1, \
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1, \
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1, \
             0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1, \
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0, \
             1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0, \
             1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0, \
             1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0, \
             1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

class Word2Feature(object):
    def __init__(self,rownum=4, colnum =9):
        self.rownum = 4
        self.colnum =9

    def run(self,img,show_img = 0):
        sub_img = self.dissolve(img)

        if show_img:
            fig, ax =plt.subplots(1,5,figsize = (32,8))
            ax[0].imshow(img,cmap='gray')
            ax[1].imshow(sub_img[0,...],cmap='gray')
            ax[2].imshow(sub_img[1,...],cmap='gray')
            ax[3].imshow(sub_img[2,...],cmap='gray')
            ax[4].imshow(sub_img[3,...],cmap='gray')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[2].set_xticks([])
            ax[2].set_yticks([])
            ax[3].set_xticks([])
            ax[3].set_yticks([])
            ax[4].set_xticks([])
            ax[4].set_yticks([])
            plt.axis("off")
            plt.show()

        features = self.get_features(img, sub_img, show_img)
        tot = np.sum(sub_img==0,axis=1)
        tot = np.sum(tot,axis=1)
        Features = np.zeros(4*9*4)
        # print(tot)
        for i in range(len(features)):
            features[i] = features[i]/ tot
            Features[4*i:4*i+4] = features[i]
        return Features

    #局部弹性网格划分
    def get_features(self,img, sub_img,show_img = 0):
        img = img.copy()
        sub_img = sub_img.copy()
        features = []
        self.__dfs(img,sub_img,2,features)

        if show_img:
            fig, ax =plt.subplots(1,5,figsize = (32,8))
            ax[0].imshow(img,cmap='gray')
            ax[1].imshow(sub_img[0,...],cmap='gray')
            ax[2].imshow(sub_img[1,...],cmap='gray')
            ax[3].imshow(sub_img[2,...],cmap='gray')
            ax[4].imshow(sub_img[3,...],cmap='gray')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[2].set_xticks([])
            ax[2].set_yticks([])
            ax[3].set_xticks([])
            ax[3].set_yticks([])
            ax[4].set_xticks([])
            ax[4].set_yticks([])
            plt.axis("off")
            plt.show()

        return  features

    def __dfs(self,img,sub_img,depth,features):
        img01 = (img ==0)
        sub_img01 = (sub_img ==0)
        if depth ==0:
            if img.shape[0] == 0 or img.shape[1] ==0:
                features.append(np.zeros(4))
                return
                # print(img.shape)
            sum = np.sum(sub_img01,axis=1)
            sum = np.sum(sum,axis=1)
            features.append(sum)
            img[-1,:]=0
            img[:,-1]=0

            sub_img[:,-1,:]=0
            sub_img[:,:,-1]=0
            return
        if img.shape[0] == 0 or img.shape[1] ==0:
            for i in range(1,4):
                for j in range(1,3):
                    self.__dfs(img[0:0,0:0],img[0:0,0:0],depth-1,features)
        else:
            row_cnt = np.sum(img01,axis=1)
            col_cnt = np.sum(img01,axis=0)
            split_row =self.__split(row_cnt,3)#[0,...,img01.shape[0]}
            split_col =self.__split(col_cnt,2)
            # print(split_row,split_col,depth)
            # print(img01.shape,sub_img01.shape)
            # print(split_row,split_col)
            for i in range(1,4):
                for j in range(1,3):
                    # print(sub_img01.shape)
                    self.__dfs(img[split_row[i-1]:split_row[i],split_col[j-1]:split_col[j]], \
                               sub_img[:,split_row[i-1]:split_row[i],split_col[j-1]:split_col[j]], \
                               depth-1,features)

    def __split(self,cnt,num):
        split =[]
        split.append(0)
        tot = np.sum(cnt)
        average = tot//num
        sum = 0
        c = 0
        for i in range(cnt.size):
            sum = sum + cnt[i]
            if sum >=average:
                split.append(i+1)
                sum = 0
                c = c +1
                if c == num-1:
                    break
        split.append(cnt.size)
        for i in range(len(split)):
            if split[i] >=cnt.size:
                split[i] = cnt.size
        for i in range(num+1-len(split)):
            split.append(cnt.size)

        return  split

    def dissolve(self,img):
        img = img.copy()
        sub_img = np.zeros((4,img.shape[0],img.shape[1]))
        sub_img[...] = 255
        #AND 在横竖方向上分解较好，OR 在撇捺分解较好
        for x in range(1,img.shape[0]-1):
            for y in range(1,img.shape[1]-1):
                if img[x,y] == 255:
                    continue
                if img[x,y-1]==0 and  img[x,y+1]==0:
                    sub_img[0,x,y] = 0
                if img[x-1,y]==0 and  img[x+1,y]==0:
                    sub_img[1,x,y] = 0
                if img[x+1,y-1]==0 or  img[x-1,y+1]==0:
                    sub_img[2,x,y] = 0
                if img[x-1,y-1]==0 or  img[x+1,y+1]==0:
                    sub_img[3,x,y] = 0


        return sub_img

#图片x轴的投影，如果有数据（黑色像素点0）值为1否则为0
class RotatingCalipers(object):
    @staticmethod
    def get_projection_x(image,invert=False):
        p_x = [0 for x in range(image.size[0])]
        for w in range(image.size[1]):
            for h in range(image.size[0]):
                # print(image.getpixel((h,w)))
                if invert:
                    if image.getpixel((h,w)) == 255:
                        p_x[h] = 1
                        continue
                else:
                    if image.getpixel((h,w)) == 0:
                        p_x[h] = 1
                        continue
        return p_x

    @staticmethod
    def get_img_width(projection_x):
        start_pos = 0
        stop_pos = 0
        pro_len = len(projection_x) - 1
        for idx in range(pro_len):
            if projection_x[idx] > 0:
                start_pos = idx
                break
        for idx in range(pro_len):
            if projection_x[pro_len - idx] > 0:
                stop_pos = pro_len - idx
                break
        return stop_pos - start_pos

    @staticmethod
    def rotating_calipers(raw_img):
        raw_img = Image.fromarray(raw_img.astype('uint8')).convert('L')
        img = raw_img.copy()
        min_width = 100
        min_angle =  100
        for angle in range(-20,20,2):
            temp_img = img.rotate(angle, expand = True,fillcolor ="white")
            jection = RotatingCalipers.get_projection_x(temp_img, False)
            cur_width = RotatingCalipers.get_img_width(jection)
            if cur_width < min_width:
                min_width = cur_width
                min_angle = angle
        img = img.rotate(min_angle,expand=True, fillcolor = "white")
        img =np.asarray(img.convert("L"))#转化为灰度图像
        # img.setflags(write=1)
        img = ImageProcessing.crop_margin(img)
        img = cv2.resize(img,raw_img.size)
        th,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return img
        # plt.show()

class KNNRecognition(object):
    def __init__(self,data_dir):
        self.train_X = np.load(os.path.join(data_dir,'train_X.npy'))
        self.train_Y = np.load(os.path.join(data_dir,'train_Y.npy'))
        self.test_X = np.load(os.path.join(data_dir,'test_X.npy'))
        self.test_Y = np.load(os.path.join(data_dir,'test_Y.npy'))
        print(self.train_Y,self.test_Y)

        self.id2word = {}
        self.word2id = {}
        with open(r"Data/id2word.pkl","rb") as f:
            self.id2word = pickle.load(f)
        with open(r"Data/word2id.pkl","rb") as f:
            self.word2id = pickle.load(f)

    def run(self,img_dir,k = 1):
        kk = 7
        weight =100
        height = 100
        wordsfeature = Word2Feature()
        knn = KNN(k)
        get_words = Captha2Words(kk,weight, height)

        kind_right = 0
        reverse_right = 0
        reverse_TP = 0
        reverse_FP = 0
        reverse_FN = 0
        reverse_TN = 0
        all_right = 0
        test_num = kk * len(os.listdir(img_dir))
        for img_name in (os.listdir(img_dir)):
            print(img_name)
            split_name = img_name.split(" ")
            labels = np.zeros((kk,2),dtype="int16")
            for i in range(0,kk):
                if split_name[2*i] in self.word2id.keys():
                    labels[i,0] = self.word2id[split_name[2*i]]
                else :
                    labels[i,0] = -1
                labels[i,1] = int(split_name[2*i+1])

            # print(labels)
            img_path = os.path.join(img_dir, img_name)
            words = get_words.run(img_path)
            # rename =''
            for i in range(kk):
                label = labels[i,:]
                word = words[i,...]
                # word = Word2Feature.rotating_calipers(word)
                word = XiHua.Xihua(word)
                feature = wordsfeature.run(word)
                distence = np.linalg.norm(self.train_X - feature,axis=1)
                predict_label = knn.run(distence,self.train_Y)
                if predict_label[0] == labels[i,0]:
                    kind_right = kind_right + 1
                if predict_label[1] == labels[i,1]:
                    reverse_right = reverse_right + 1
                if predict_label[0] == labels[i,0] and predict_label[1] == labels[i,1]:
                    all_right = all_right + 1

                if label[1] ==1 and predict_label[1] ==1:
                    reverse_TP = reverse_TP + 1
                if label[1] ==0 and predict_label[1] ==1:
                    reverse_FP = reverse_FP + 1
                if label[1] ==1 and predict_label[1] ==0:
                    reverse_FN = reverse_FN + 1
                if label[1] ==0 and predict_label[1] ==0:
                    reverse_TN = reverse_TN + 1
                # print(reverse_TP, reverse_FP, reverse_FN,reverse_TN)
                # rename = rename + id2word[predict_label[0]] +' '+str(predict_label[1]) +(' ' if i != kk else '')
                # rename = rename +'.gif'
                # os.rename(img_path,os.path.join(img_dir,rename))
                # print(rename)
        reverse_precision = reverse_TP/(reverse_TP+reverse_FP)
        reverse_recall = reverse_TP/(reverse_TP + reverse_FN)
        reverse_F1score = 2* reverse_precision*reverse_recall/(reverse_precision+reverse_recall)
        print(k,"准确率：汉字识别成功率{0:3f},反转识别成功率{1:3f}，F1score{2:3f},同时识别成功率{3:3f}".format(kind_right/test_num,reverse_right/test_num,reverse_F1score,all_right/test_num))

class KNN(object):

    def __init__(self,k):
        self.k =k

    def run(self,distence,label):
        arg = np.argsort(distence)
        knn_label = label[arg[:self.k],:]
        kind = knn_label[:,0]
        reverse = knn_label[:,1]
        unq_kind,counts_kind = np.unique(kind,return_counts=True)
        unq_reverse,counts_reverse = np.unique(reverse,return_counts=True)
        predict_kind = unq_kind[np.argmax(counts_kind)]
        predict_reverse = unq_reverse[np.argmax(counts_reverse)]
        return np.array([predict_kind,predict_reverse])

if __name__ == '__main__':
    img_dir = r'.\CaptchaData'
    k = 7#字符个数
    weight =100
    height = 100
    get_words = Captha2Words(k,weight, height)
    for img_name in (os.listdir(img_dir))[:5]:
        img_path = os.path.join(img_dir, img_name)
        words = get_words.run(img_path)
        print("words:",words.shape)

#%%
