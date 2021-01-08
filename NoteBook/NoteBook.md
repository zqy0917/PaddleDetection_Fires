1: 处理数据集，转化成 voc 或者 coco 格式，用于训练和验证  
2: 模型选择，选择和修改模型配置文件，分类个数等  
3: 开始训练，设置好模型保存路径  
4: 对模型进行评估  
5: 开始预测  
PPYolo 链接：https://yeyupiaoling.blog.csdn.net/article/details/108069066?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1.control  
github： https://github.com/yeyupiaoling/PP-YOLO  

paddleDetection: https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-beta/docs/tutorials/DetectionPipeline.md

### 1: 生成 Annotation xml 文件


```python
# 安装 lxml 库
# pip3 install lxml
```


```python
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
from PIL import Image
# 参考文章：https://blog.csdn.net/w113691/article/details/80393487

workPath = "/home/aistudio/work"
save_xml_path = workPath + "/VOC_Fires/Annotations/"
annotationsDirPath = workPath + "/Fires/train/annotations/"
imagesDirPath = workPath + "/Fires/train/images/"
pasclavocTxt = "/home/aistudio/work/pascla_voc.txt"

# 将所有的 Annotations txt 文件合并成一个文件
def mergeAllAnnotation():
    if os.path.exists(pasclavocTxt) == 0:
        !touch {pasclavocTxt}
    data = []
    for root, dir, files in os.walk(annotationsDirPath):
        # files 表示该文件夹下的文件list
        for f in files:
            fullname = os.path.join(root, f)
            fileF = open(fullname) 
            for line in fileF.readlines():
                data.append(line)
    f = open(pasclavocTxt, 'w+', encoding='utf-8')
    for i, p in enumerate(data):
        strP = p.replace("非金属打火机", "NonmetallicFire")
        f.write(strP)
    f.close()

# 生成 xml 文件
def make_xml(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, clname, image_name):
 
    node_root = Element('annotation')
 
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'
 
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name
 
    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(xmin_tuple))
 
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    im = Image.open(imagesDirPath + image_name)
    width, height = im.size
    node_width.text = str(width)
 
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
 
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    
    for i in range(len(xmin_tuple)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(clname[i])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
 
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin_tuple[i])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin_tuple[i])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax_tuple[i])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax_tuple[i])
 
 
    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    return dom

# 将 xml 写入文件
def writeXmlToFile(xmlFileName, dom):
    if os.path.exists(xmlFileName) == 0:
        with open(xmlFileName, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

# 制作所有的 xml 文件
def makeXMLAnnotations():
    if os.path.exists(save_xml_path) == 0:
        os.makedirs(save_xml_path)
    f = open(pasclavocTxt)   
    result = []
    n = 0
    for line in f.readlines():
        result.append(line.split(' '))
        n = n + 1
    img_nameinit = result[0][0]
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    clname = []
    for i in range(0, n):
        if img_nameinit == result[i][0]:
            clname.append(result[i][1])
            xmin.append(result[i][2])
            ymin.append(result[i][3])
            xmax.append(result[i][4])
            ymax.append(result[i][5])
        else:
            dom = make_xml(xmin, ymin, xmax, ymax, clname, img_nameinit)
            por = os.path.splitext(img_nameinit)
            xml_name = os.path.join(save_xml_path, por[0] + '.xml')
            writeXmlToFile(xml_name, dom)
            img_nameinit = result[i][0]
            clname = [result[i][1]]
            xmin = [result[i][2]]
            ymin = [result[i][3]]
            xmax = [result[i][4]]
            ymax = [result[i][5]]
    dom = make_xml(xmin, ymin, xmax, ymax, clname, img_nameinit)
    por = os.path.splitext(img_nameinit)
    xml_name = os.path.join(save_xml_path, por[0] + '.xml')
    writeXmlToFile(xml_name, dom)
    
mergeAllAnnotation()
makeXMLAnnotations()
!rm {pasclavocTxt}

```

### 生成标准 VOC 文件夹


```python
VOC_image_path = workPath + "/VOC_Fires/JPEGImages/"
save_xml_path = workPath + "/VOC_Fires/Annotations/"
ImageSetsDirPath = workPath + "/Fires/train/ImageSets/"
trainvalTxt = workPath + "/VOC_Fires/trainval.txt"


# 移动图片到另一个文件夹
def moveImage():
    if os.path.exists(VOC_image_path) == 0:
        os.makedirs(VOC_image_path)
        !cp -r {imagesDirPath}* {VOC_image_path}

def makeImageSets():
    if os.path.exists(ImageSetsDirPath) == 0:
        os.makedirs(ImageSetsDirPath)


def makeTrainTxt():
    data = []
    # 图片路径 xml路径
    for root, dir, files in os.walk(annotationsDirPath):
        # files 表示该文件夹下的文件list
        for f in files:
            fullname = os.path.join(root, f)
            por = os.path.splitext(fullname)
            prefix = por[0].split('/')[-1]
            line = VOC_image_path+prefix + '.jpg' ' ' + save_xml_path+prefix + '.xml' + '\n'
            data.append(line)
    f = open(trainvalTxt, 'w+', encoding='utf-8')
    for i, p in enumerate(data):
        f.write(p)
    f.close()


moveImage()
makeImageSets()
makeTrainTxt()
```

    1212


### 生成测试文件txt


```python
testImagesDirPath = workPath + "/Fires/test/images/"
testTxt = workPath + "/VOC_Fires/test.txt"

def makeTestTxt():
    data = []
    # 图片路径 xml路径
    for root, dir, files in os.walk(testImagesDirPath):
        # files 表示该文件夹下的文件list
        for f in files:
            fullname = os.path.join(root, f)
            por = os.path.splitext(fullname)
            prefix = por[0].split('/')[-1]
            line = VOC_image_path+prefix + '.jpg' + '\n'
            data.append(line)
            print(line)
    f = open(testTxt, 'w+', encoding='utf-8')
    for i, p in enumerate(data):
        f.write(p)
    f.close()
    
makeTestTxt()

```