"""
# author: lvwanyou
"""

import argparse
import os
import cv2
import torch
import torchvision
from torchvision import transforms
from pic_cutting import PicCutting
from prediction_single import PredictionSingle


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='./data/Products/perfection_imperfection/',
                    help="""image dir path default: './data/Products/perfection_imperfection/'.""")
parser.add_argument('--batch_size', type=int, default=256,
                    help="""Batch_size default:154.""")
parser.add_argument('--num_classes', type=int, default=2,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='./models/pytorch/Products/perfection_imperfection/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='PerfectionImperfection.pth',
                    help="""Model name.""")
parser.add_argument('--single_model_name', type=str, default='SinglePerfectionImperfection.pth',
                    help="""Single Model name.""")
parser.add_argument('--result_dir_path', type=str, default='./data/Products/perfection_imperfection/result',
                    help="""image dir path default: './data/Products/perfection_imperfection/result'.""")
args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.Resize(128),  # 将图像转化为128 * 128
    transforms.RandomCrop(114),  # 从图像中裁剪一个114 * 114的
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
])

# Load data
test_datasets = torchvision.datasets.ImageFolder(root=args.path + 'val/',
                                                 transform=transform)


test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=args.batch_size,
                                          shuffle=False)

# train_datasets zip
item = test_datasets.class_to_idx

class Product:
    name = str()
    left = bool(True)       # 0: right ; 1: error
    middle = bool(True)
    right = bool(True)

def main():
    print("#################PICT PREDICTION BEGIN#####################")
    print(f"total test numbers: {len(test_datasets)}.")
    # Load model
    if torch.cuda.is_available():
        model = torch.load(args.model_path + args.model_name).to(device)
    else:
        model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model.eval()

    correct = 0.
    total = 0
    conclusion = []
    imperfection_coll=[]
    for images, labels in test_loader:
        # to GPU
        images = images.to(device)
        labels = labels.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)

        # print(f"label kind:{labels}")             # 0:imperfections ;  1:perfection

        print(f"predicted kind:{predicted}")
        for i in range(len(predicted)):
            file = str(test_datasets.imgs[i])[2:-5]
            if int(predicted[i]) == 1:
                conclusion.append(file + ": 合格")
                print(f"{i+1}.({file}) is perfection!")
            else:
                conclusion.append(file + ": 不合格")
                print(f"{i+1}.({file}) is imperfection!")
                imperfection_coll.append(file)

        # val_loader total
        total += labels.size(0)
        # add correct
        correct += (predicted == labels).sum().item()
    # print(f"Acc: {100 * correct / total:.4f}")
    print("#################PICT PREDICTION END#####################")
    print('\n')

    # 将result下的所有的文件进行清空
    path = './data/Products/perfection_imperfection/val_single'
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

    # ========================================================================================================
    # 模块说明:
    #   针对imperfection的照片,先按照固定比例进行切割；
    #   然后利用事先训练好的模型判断哪根有问题
    #   最后再将结果进行汇总，进行输出。
    # ========================================================================================================
    for item in imperfection_coll:
        item= item.replace('\\\\', '/')
        PicCutting.cut_pict(item)

    # 将不合格根数输出
    # 在图片上打点
    imperfection_detail = []
    imperfection_single_coll= PredictionSingle.predict(0)
    for item in imperfection_single_coll:
        item = item.replace('\\\\', '/')
        file_names = str(item).split('/')
        assembled_file_name = file_names[len(file_names)-1]
        file_name = assembled_file_name[0: len(assembled_file_name)-6]+".jpg"
        assembled_index = item[len(item)-5:len(item)-4]

        temp = Product()
        count = 0
        exist_flag = False
        for index in range(len(imperfection_detail)):
            if imperfection_detail[index].name == file_name:
                temp = imperfection_detail[index]
                exist_flag = True
                break
            count += 1
        if count >= len(imperfection_detail):
            temp.name = file_name

        if int(assembled_index) == 1:
            temp.left = False
        elif int(assembled_index) == 2:
            temp.middle = False
        else:
            temp.right = False
        if not exist_flag:
            imperfection_detail.append(temp)

    # 将结果输出到txt中去
    for item in imperfection_detail:
        for index in range(len(conclusion)):
            if str(conclusion[index]).__contains__(item.name):
                sub_str = str()
                if not item.left:
                    sub_str += "   1"
                if not item.middle:
                    sub_str += "   2"
                if not item.right:
                    sub_str += "   3"
                conclusion[index] = conclusion[index] + " || " + sub_str

    for index in range(len(conclusion)):
        conclusion[index] = str(conclusion[index]).replace('\\\\', '/')

    print(conclusion)
    # 将框选结果后的图片输出到result文件夹下
    if not os.path.exists(args.result_dir_path):
        os.makedirs(args.result_dir_path)

    # 将result下的所有的文件进行清空
    path = './data/Products/perfection_imperfection/result'
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)


    for item in imperfection_detail:
        fname = './data/Products/perfection_imperfection/val/perfection/' + item.name
        img = cv2.imread(fname)
        if not item.left:
            # 画矩形框
            cv2.rectangle(img, (450, 400), (985, 1900), (0, 0, 255), 4)
            # 标注文本
            # font = cv2.FONT_HERSHEY_SUPLEX
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'offset'
            cv2.putText(img, text, (450, 400), font, 2, (0, 0, 255), 5)
        if not item.middle:
            # 画矩形框
            cv2.rectangle(img, (1050, 400), (1537, 1900), (0, 0, 255), 4)
            # 标注文本
            # font = cv2.FONT_HERSHEY_SUPLEX
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'offset'
            cv2.putText(img, text, (1050, 400), font, 2, (0, 0, 255), 5)
        if not item.right:
            # 画矩形框
            cv2.rectangle(img, (1600, 400), (2100, 1900), (0, 0, 255), 6)
            # 标注文本
            # font = cv2.FONT_HERSHEY_SUPLEX
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'offset'
            cv2.putText(img, text, (1600, 400), font, 2, (0, 0, 255), 5)
        cv2.imwrite('./data/Products/perfection_imperfection/result/' + item.name[0:len(item.name)-4]+'_new.jpg', img)

    with open("./data/Products/perfection_imperfection/result.txt", "w") as f:
        for str_item in conclusion:
            f.write(str_item + '\n')    # 这句话自带文件关闭功能，不需要再写f.close()


if __name__ == '__main__':
    main()
