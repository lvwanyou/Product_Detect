import cv2  # [1]导入OpenCv开源库
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class PicCutting(object):

    parser = argparse.ArgumentParser("""Image classifical!""")
    parser.add_argument('--test_single_path', type=str, default='./data/Products/perfection_imperfection/test_single',
                        help="""Save model path""")
    parser.add_argument('--train_single_path', type=str, default='./data/Products/perfection_imperfection/train_single',
                        help="""Save model path""")
    args = parser.parse_args()


    def get_files_name(file_dir):
        L = []
        for root, dirs, files in os.walk(file_dir):
            # print(dirs) #当前路径下所有子目录
            # print(files) #当前路径下所有非目录子文件
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    str_root = str(root)
                    str_root = str_root.replace('\\', '/')+'/'
                    L.append(str_root+file)
        print (L)
        return L


    ## 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(filePath = ""):
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img


    def cut_pict(file_name):
        str_file_name =str(file_name)
        srcImg = PicCutting.cv_imread(str_file_name)
        # './data/Products/perfection_imperfection/test_single/imperfection/a.jpg'
        # cv2.namedWindow("[srcImg]", cv2.WINDOW_AUTOSIZE)  # [3]创建显示窗口
        # cv2.imshow("[srcImg]", srcImg)  # [4]在刚才创建的显示窗口中显示刚在加载的图片

        # image1 = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)
        # plt.subplot(131)
        # plt.imshow(image1)
        # plt.show()
        # cv2.waitKey(0)

        # ========================================================================================================
        # 模块说明:
        #       由于OpenCv中,imread()函数读进来的图片,其本质上就是一个三维的数组,这个NumPy中的三维数组是一致的,所以设置图片的
        #   ROI区域的问题,就转换成数组的切片问题,在Python中,数组就是一个列表序列,所以使用列表的切片就可以完成ROI区域的设置
        # ========================================================================================================

        image_save_path_head = str_file_name[0:len(str_file_name)-4]+"_"
        image_save_path_head= image_save_path_head.replace('val', 'val_single')
        image_save_path_tail = ".jpg"
        seq = 1
        for j in range(3):  # [1]column-----------height  [2]column-----------width
            if j == 0:
                img_roi = srcImg[0:2048, 480:980]
            elif j == 1:
                img_roi = srcImg[0:2048, 980:1480]
            else:
                img_roi = srcImg[0:2048, 1580:2080]
            image_save_path = "%s%d%s" % (image_save_path_head, seq, image_save_path_tail)  # 将整数和字符串连接在一起
            # cv2.imwrite(image_save_path, img_roi)
            if not os.path.exists(image_save_path):
                cv2.imencode('.jpg', img_roi)[1].tofile(image_save_path)
                seq = seq + 1
        # os.remove(file_name)  # 删除原有图片


    def execution(L):
        # L =[]
        # L = get_files_name(args.test_single_path)
        for item in L:
            PicCutting.cut_pict(item)
        L = PicCutting.get_files_name(PicCutting.args.train_single_path)
        for item in L:
            PicCutting.cut_pict(item)