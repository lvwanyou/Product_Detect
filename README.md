#  根据图片判断产品是否合格

<br>

用到的数据集来自 人工智能课程，这份数据集包含了100份 带有标签的图片。

<br>

## 总数据集包含两个文件夹（imperfection product & perfection product）：

- 训练文件夹：它包含了 60 张图片（合格和不合格图片各三十张），每张图片都含有标签，这个标签是作为文件名的一部分。我们将用这个文件夹来训练和评估我们的模型。

- 测试文件夹：它包含了 40 张图片（合格和不合格图片各二十张），每张图片都以数字来命名。对于这份数据集中的每幅图片来说，我们的模型都要预测这张图片上是合格还是不合格（1= 合格，0= 不合格）。

文件目录结构如下:

	  └── train
        └── perfection
            └── p_product.1.jpg
            ...
            └── p_product.n.jpg
        └── imperfection
            └── imp_product.1.jpg
            ...
            └── imp_product.n.jpg
     └── test
        └── perfection
            └── p_product.1.jpg
            ...
            └── p_product.n.jpg
        └── imperfection
            └── imp_product.1.jpg
            ...
            └── imp_product.n.jpg
    └── val
        └── unknown
            └── p_product.1.jpg
            ...
            └── imp_product.n.jpg

[项目及数据下载点这里](https://github.com/lvwanyou/scilearn2)

### 使用的是Pytorch框架

#### 使用说明

- train:

`python train.py`

- test:

`python test.py`

- val:

`python prediction.py`
#** model 之前已经经过 train & test ,存放在 Path:'./models/pytorch/Products/perfection_imperfection/'下的'PerfectionImperfection.pth'中  **#

#### 使用步骤：

1.将待预测的数据放入到'./data/Products/perfection_imperfection/val' 相对路径下的 'perfection' dir下；
2.Run 'prediction.py'；
3. prediction 过程分为两步：
               a. predict 数据集中不合格的图片为哪几张
               b. 在不合格的图片中 predict 哪几根 不合格
 因此在 步骤2 后，'./data/Products/perfection_imperfection/val_single' 中产生了针对不合格图片切割后产生的图片      ///切割规则：将一张图片切割为 三张（h,w）:(2048:500)的图片
 4. 查看结果：
                a. 在 Path '/data/Products/perfection_imperfection/'下 的 ' result.txt '中查看输出的结论
                b. 在 Path '/data/Products/perfection_imperfection/result' 下查看图像化展示缺陷位置的图片


环境：
        python 3.6 + pytorch + Anaconda3

亮点：
        1. nn  、optim   （include 'torch'）
        2. resnet18  、 resnet50  (training model )
        3. 数据增强 ( 具体查看代码 'train.py' 中的 transforms)
        4. 整个预测过程分为两个子预测过程，两个子过程之间是 independence。由于数据集较少， 因此单个预测结果准确度上限有限。因此采用多个独立预测过程
        5. opencv 实现 缺陷位置框选、 缺陷类别提示
        6. lbp 实现区分 缺陷为 偏移|| 焊黑 的功能（尚未完成， 详情见 : 'lbp.py'）


## LINCENSE: MIT
