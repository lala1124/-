# -
基于light——yolov5s的火灾检测系统

所用系统环境为Ubuntu 18.04.2，Jetpack 4.6.4，CUDA 10.2.300，cuDNN 8.2.1.32，TensorRT 8.2.1.8，Opencv 4.1.1。

jetpack必须是4.6.4   之前用4.5.1版本无法正常启动TensorRT

将以上所有文件放入Ubuntu文档目录下
打开终端
启动fire文件，指令为 ./fire
将1.txt的TensorRT指令输入回车
选择启动方式运行即可
想加入其他功能，修改fire.cpp即可
onnx模型改名为best.onnx
第一次生成engine文件会很慢。需要等待10分钟左右，请耐心等待，生成后则无需再次生成
每改变一次onnx模型，需要重新生成一次engine文件
