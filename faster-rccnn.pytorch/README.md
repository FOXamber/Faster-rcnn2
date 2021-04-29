# 1. 准备工作

## 1.1 数据下载

1. Download the training, validation, test data and VOCdevkit

   ```
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. Create symlinks for the PASCAL VOC dataset

   ```
   cd $FRCN_ROOT/data
   ln -s $VOCdevkit VOCdevkit2007
   ```

   Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.

## 1.2 预训练模型下载

ResNet 101 [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)
下载完成后把他们放到data/pretrained_model目录下。

## 1.3 工程编译

进入faster-rcnn.pytorch`目录下，在进入到lib文件夹内：

```
cd lib
python3 setup.py build develop
```

[参考你提供的csdn博客](https://blog.csdn.net/weixin_43869778/article/details/96837042?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161968646216780366521612%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161968646216780366521612&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-6-96837042.pc_search_result_cache&utm_term=pytorch1.0)

# 2. 训练

```
CUDA_VISIBLE_DEVICES=0 python3 trainval_net.py --dataset pascal_voc --net res101 --epochs 20 --bs 1 --num_workers 4 --lr 1e-2 --lr_decay_step 8 --mGPUs --cuda
```

# 3. 测试

If you want to evlauate the detection performance of a pre-trained res101 model on pascal_voc test set, simply run

```
python3 test_net.py --dataset pascal_voc --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```

Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

# 4. 改动记录

原始的faster rcnn在Backbone的最末端进行目标预测：

![image-20210429224034739](/Users/nickccnie/Library/Application Support/typora-user-images/image-20210429224034739.png)

FPN结构：在多个尺度进行目标分配和预测：

![image-20210429224054843](/Users/nickccnie/Library/Application Support/typora-user-images/image-20210429224054843.png)

PANet-FPN结构：

![image-20210429224242434](/Users/nickccnie/Library/Application Support/typora-user-images/image-20210429224242434.png)

```
lib/faster_rcnn --> lib/fpn
```

