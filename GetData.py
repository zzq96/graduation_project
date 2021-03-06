
#%%
from Until import *
import time
import random
import tensorflow
import matplotlib.pyplot as plt
import pickle
import  xlrd
import os
import numpy as np
from PIL import Image,ImageFont,ImageDraw
import xlwt
import  cv2
import  shutil

#得到初始数据集
class Font2Img(object):
    """

    """
    def __init__(self,
                 width,
                 height,
                 ):
        self.width = width
        self.height = height

    def get_image(self,font_path, word, rotate, crop,is_reversed ):
        """
        :param font_path: str,字体文件绝对路径
        :param word: str,需要转化为图像的一个汉字
        :return: narray,size=(height,width),字体图像
        """
        #白色背景
        y=1
        img = Image.new("RGB", (self.width+100, self.height+100), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width+50))
        draw.fontmode = "1"
        #黑色字体
        draw.text((0, 0), word,(0,0,0),font = font)
        img = img.rotate(rotate,  expand= 1, fillcolor="white")
        # print(img.size, type(img))
        img = np.asarray(img, dtype='uint8')[:,:,0]
        # plt.imshow(img, "gray")
        # plt.show()
        img = ImageProcessing.crop_margin(img,crop)
        img = cv2.resize(img,(self.height,self.width),Image.ANTIALIAS)
        # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        th,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if is_reversed:
            img = img[::-1,::-1]
        return img


def get_data1(data_dir, width, height, Rotate, Rotate_step, test_ratio, crop,reversed_ratio,is_aug=0):
    id2word = {}
    word2id = {}
    with open(r"Data/id2word.pkl","rb") as f:
        id2word = pickle.load(f)
    with open(r"Data/word2id.pkl","rb") as f:
        word2id = pickle.load(f)

    # for item in
    train_dir =os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    for word, id in  word2id.items():
        print(word, id)
        image_list = []
        font2img = Font2Img(width, height)
        for font_name in os.listdir(r"./Fonts")[:]:
            # print(font_name)
            font_path = os.path.join(r"./Fonts",font_name)
            for rotate in range(-Rotate, Rotate+1, Rotate_step):
                is_reversed = random.random()<reversed_ratio
                img = font2img.get_image(font_path, word, rotate,crop,is_reversed)
                image_list.append((img,is_reversed))
                aug_img = None
                flag = None
                if is_aug :
                    rand = random.random()
                    if rand < 0.25:
                        aug_img = ImageProcessing.dilate(img,3)
                        flag =1
                    elif rand < 0.5:
                        aug_img = ImageProcessing.erode(img,3)
                        flag =2
                    elif rand < 0.75:
                        aug_img = ImageProcessing.closing(img,3)
                        flag =3
                    else:
                        aug_img = ImageProcessing.opening(img,3)
                        flag =4
                    image_list.append((aug_img,is_reversed))

        test_num = len(image_list) * test_ratio
        np.random.shuffle(image_list)
        count = 0
        for i in range(len(image_list)):
           img = image_list[i][0]
           is_reversed = image_list[i][1]
           if count < test_num:
               word_dir =os.path.join(test_dir,'%0.5d'%id)
           else:
               word_dir =os.path.join(train_dir,'%0.5d'%id)

           if not os.path.isdir(word_dir):
              os.makedirs(word_dir)
           path_image = os.path.join(word_dir, '%d %d.png'% (count,is_reversed))
           cv2.imwrite(path_image, img)
           count +=1

def get_data_knn(data_dir, width, height, Rotate, Rotate_step, test_ratio, crop,reversed_ratio):
    id2word = {}
    word2id = {}
    with open(r"Data/id2word.pkl","rb") as f:
        id2word = pickle.load(f)
    with open(r"Data/word2id.pkl","rb") as f:
        word2id = pickle.load(f)
    num = len(word2id)
    X = []
    Y = []
    word2feature = Word2Feature()
    time1 = time.clock()
    for word, id in  list(word2id.items())[:]:
        print(word, id)
        image_list = []
        font2img = Font2Img(width, height)
        for font_name in os.listdir(r"./KNNFonts")[:]:
            font_path = os.path.join(r"./KNNFonts",font_name)
            for rotate in range(-Rotate, Rotate+1, Rotate_step):
                is_reversed = random.random()<reversed_ratio
                img = font2img.get_image(font_path, word, rotate,crop,is_reversed)
                feature = word2feature.run(img)
                X.append(feature)
                Y.append([id,is_reversed])
        time2 = time.clock()
        print(time2-time1)
    #
    X = np.array(X)
    Y = np.array(Y)
    shuffle = [i for i in range(len(X))]
    random.shuffle(shuffle)
    test_num = int(len(X)*test_ratio)
    test_X = X[shuffle[:test_num]]
    train_X = X[shuffle[test_num:]]
    test_Y = Y[shuffle[:test_num]]
    train_Y = Y[shuffle[test_num:]]
    print(test_num,len(X)-test_num)
    print(test_X.shape,train_X.shape)
    np.save(os.path.join(data_dir,'test_X'),test_X)
    np.save(os.path.join(data_dir,'train_X'),train_X)
    np.save(os.path.join(data_dir,'test_Y'),test_Y)
    np.save(os.path.join(data_dir,'train_Y'),train_Y)

def get_data_cnn(data_dir, width, height, Rotate, Rotate_step, test_ratio, crop,reversed_ratio):
    id2word = {}
    word2id = {}
    with open(r"Data/id2word.pkl","rb") as f:
        id2word = pickle.load(f)
    with open(r"Data/word2id.pkl","rb") as f:
        word2id = pickle.load(f)

    num = len(word2id)
    X = []
    Y = []
    for word, id in  list(word2id.items())[:]:
        print(word, id)
        image_list = []
        font2img = Font2Img(width, height)
        for font_name in os.listdir(r"./CNNFonts")[:]:
            font_path = os.path.join(r"./CNNFonts",font_name)
            for rotate in range(-Rotate, Rotate+1, Rotate_step):
                is_reversed = random.random()<reversed_ratio
                img = font2img.get_image(font_path, word, rotate,crop,is_reversed)
                # if abs(rotate) >5:
                #     img = Word2Feature.rotating_calipers(img)
                X.append(img)
                Y.append([id,is_reversed])
    #
    X = np.array(X)
    Y = np.array(Y)
    shuffle = [i for i in range(len(X))]
    random.shuffle(shuffle)
    test_num = int(len(X)*test_ratio)
    test_X = X[shuffle[:test_num]]
    train_X = X[shuffle[test_num:]]#.astype('uint8')
    test_Y = Y[shuffle[:test_num]]
    train_Y = Y[shuffle[test_num:]]

    print(test_num,len(X)-test_num)
    print(test_X.shape,train_X.shape)
    # np.save(os.path.join(data_dir,'test_X'),test_X)
    # np.save(os.path.join(data_dir,'train_X'),train_X)
    # np.save(os.path.join(data_dir,'test_Y'),test_Y)
    # np.save(os.path.join(data_dir,'train_Y'),train_Y)
    print(test_X.shape, test_X.dtype)
    print(test_X)

if __name__ =="__main__":
    width = 100
    height = 100
    Rotate = 20
    Rotate_step = 5
    test_ratio = 0.02
    crop = 15# 字左右随机裁剪0-15个像素
    reversed_ratio = 0.4
    data_dir = r'kNNData1'
    get_data_knn(data_dir, width, height, Rotate, Rotate_step, test_ratio, crop,reversed_ratio)
    # width = 40
    # height = 40
    # Rotate = 20
    # Rotate_step = 5
    # test_ratio = 0.02
    # crop = 15# 字左右随机裁剪0-15个像素
    # reversed_ratio = 0.5
    # data_dir = r'CNNData'
    # get_data_cnn(data_dir, width, height, Rotate, Rotate_step, test_ratio, crop,reversed_ratio)

