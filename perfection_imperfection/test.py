import os
import cv2
def file_name(file_dir):
    L = []

    for root, dirs, files in os.walk(file_dir):

        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                str_root = str(root)
                str_root = str_root.replace('\\', '/')+'/'
                L.append(str_root+file)
    return L
def main():
    # file_name('./data/Products/perfection_imperfection/train_single')
    # a= ' a/b/c'
    # a=a.replace('b', 'aaaa')
    # print(a)


    fname = './data/Products/perfection_imperfection/val/perfection/6.jpg'
    img = cv2.imread(fname)
    # 画矩形框
    cv2.rectangle(img, (450, 400), (985, 1900), (0, 0, 255), 4)
    # 标注文本
    # font = cv2.FONT_HERSHEY_SUPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '001'
    cv2.putText(img, text,(450, 400), font, 2, (0, 0, 255),5)

    # 画矩形框
    cv2.rectangle(img, (1050, 400), (1537, 1900), (0, 0, 255), 4)
    # 标注文本
    # font = cv2.FONT_HERSHEY_SUPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '001'
    cv2.putText(img, text, (1050, 400), font, 2, (0, 0, 255), 5)


    # 画矩形框
    cv2.rectangle(img, (1600, 400), (2100, 1900), (0, 0, 255), 6)
    # 标注文本
    # font = cv2.FONT_HERSHEY_SUPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '001'
    cv2.putText(img, text,(1600, 400), font, 2, (0, 0, 255), 5)
    cv2.imwrite('./data/Products/perfection_imperfection/val/perfection/6_new.jpg', img)

if __name__ == '__main__':
    main()