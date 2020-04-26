# tensorflow_Neural_Style_Transfer

Original NST accomplished in Tensorflow2.0  
NST风格迁移原论文的tensorflow2.0实现，按照原论文使用IMAGENET预训练过的VGG19作为feature extractor

## 示例
![examples from original paper](https://github.com/miknyko/tensorflow_Neural_Style_Transfer/blob/master/output/1580470789(1).png)

## 使用环境
python = 3.6  
tensorflow = 2.0 (可使用GPU)  

## 使用方法
**（可以直接运行main.py进行测试）**  
1.将内容照片（如一张人像，一张风景）放入input/content下，将风格照片(如梵高的印象派画作)放入input/style下  
2.修改main.py中的content image和style image路径 Modify the image path in main.py  
3.运行main.py  

P.S. 在CPU上可能耗时较长，建议在能够使用gpu的机器上运行

## 更多详情请参阅原论文
Original Paper:[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)


