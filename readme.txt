论文使用的数据集在：链接：https://pan.baidu.com/s/1qVAXj82gkw82hbyb3kDoCg 提取码：8hgs 
（解压时候可能存在错误，请记录下对应的错误图片，将相邻的图片进行复制重命名替代即可，大约有4张错误图像）
代码中使用到的预训练权重：链接：https://pan.baidu.com/s/1u-u2SKDi43NSzTR1fyk0Xw 提取码：qgda 
请将权重下载之后添加到premodels文件夹中

环境配置在requirement.txt中，可能需要再补充个别包，使用python3.8

当准备工作完成之后，需要修改main中的root_path，将DAD数据集所在目录地址替换即可

修改view可以训练4种不同视图

batch_size可根据自身设备进行设置，默认的bs是效果最好的参数配置

device可以根据自身配置设置，支持多卡训练

其他参数可以不需要调整

最后使用 python main.py 即可训练
