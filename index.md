# 超分模型部署

| 版本 | 修改时间  | 修改者 | 备注             |
| ---- | --------- | ------ | ---------------- |
| V1.0 | 2023-5-26 | 张森华 | 内部( ) 公开(√ ) |

| 训练配置 | RTX3090(24GB) TensorFlow 2.5.0 Python 3.8(ubuntu18.04) Cuda 11.2 请注意numpy版本请保持为1.19.2 否则将出现错误 |
| -------- | ------------------------------------------------------------ |
| 部署配置 | 全志A64, 64位，四核，Mali400MP2 GPU, buildroot2018, QT4.9，openblas |

## 1.开源代码（论文）分析

这个项目是一个基于面向边缘的卷积块（ECB）的超分辨率（SR）网络，名为ECBSR。它的目标是在移动设备上实现实时的图像超分辨率。它的主要特点是：

- 在训练阶段，ECB在多个路径上提取特征，包括一个普通的3x3卷积，一个通道扩展和压缩卷积，以及从中间特征中提取的一阶和二阶空间导数。
- 在推理阶段，多个操作可以合并为一个单一的3x3卷积，从而减少计算量和内存消耗。
- ECB可以作为一个通用的模块，替换普通的3x3卷积，提高模型性能，而不增加推理阶段的开销。
- ECBSR在五个基准数据集上实现了与最先进的轻量级SR模型相当的PSNR/SSIM性能，同时可以在普通的移动设备上实时地将270p/540p的图像超分辨率到1080p。

项目的主要代码结构如下：

- **configs：包含了不同模型和数据集的配置文件，用于训练和测试。**

- **datas：包含了数据加载和预处理的相关代码。**

- **deploy：纯c++卷积，填充，激活等算法。**

- **experiments：包含了训练和测试结果的保存路径。**

- **models：包含了模型定义和损失函数的相关代码。**

- **convert.py：用于将训练好的PyTorch模型转换为不同前端的格式。**

- **requirements.txt：用于安装项目所需的依赖库。**

- **train.py：用于训练和测试模型。**

  

## 2.onnx移植

由于本次目的主要在于探寻ECBSR在边缘设备上的前向执行时间，因此模型仅在服务器训练10个epoch, 得到pytorch模型，然后执行conver.py将模型转换为onnx模型。然后基于qt框架调用onnx的api接口实现前向推理。

主要接口qt代码：

```c++
int MainWindow::InitModel(const char *onnxPath)
{
    // Check if model file exists
    QFile model_file(onnxPath);
    if (!model_file.exists())
    {
        qDebug("Model file not found.");
        return EXIT_FAILURE;
    }
    // --- init onnxruntime env
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default");
    // set options
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(5); // extend the number to do parallel
    session_option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session = new Ort::Session(*env, onnxPath, session_option); // delete the session object when ~
    if (session != nullptr)
        return EXIT_FAILURE;
    else
        return EXIT_SUCCESS;
}
void MainWindow::Inference()
{

    std::vector<int64_t> input_shape = {1, 1, 270, 480};
    std::vector<float> input_data(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 21.123);
    std::vector<const char *> input_names = {"input"};
    std::vector<Ort::Value> input_tensors;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    input_tensors.push_back(std::move(input_tensor));
   // session->GetInputCount(input_names[0], input_tensors);
    std::vector<const char *> output_names = {"output"};
    std::vector<Ort::Value> output_tensors;
    // inference start
    for (int i = 0; i < 10; i++) {
        qDebug("%d",i);
        output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());
    }
    QTime inference_start = QTime::currentTime();
    output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());
    // inference end
    QTime inference_end = QTime::currentTime();
    qDebug() << "inference duration:" << (float)inference_start.msecsTo(inference_end) / 100;
    // get the output data
    const float *output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output shape: ";
    for (int64_t dim : output_shape)
    {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "First output element: " << output_data[0] << std::endl;
}

```

**编译准备：A64源码编译出buildroot文件系统，基于qmake产生makefile，注意将onnx的库文件提前拷贝至开发板，并在qt工程加入头文件**。make产生可执行文件，拷贝开发板执行。

输入：与训练时一致，1×1×270×480

![image-20230526215417310](2023-5-26_%E8%B6%85%E5%88%86%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2.assets/image-20230526215417310.png)

输出：

![image-20230526215501615](2023-5-26_%E8%B6%85%E5%88%86%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2.assets/image-20230526215501615.png)

10次测试时间：

| 线程数 | 平均时间（s） |
| ------ | ------------- |
| 2      | 0.825064      |
| 3      | 0.79972       |
| 5      | 1.630877      |
| 10     | 1.906421      |

**此次实验的架构为完整架构，既包含6个卷积层，一个Pixelshuffle**

此次全部由CPU承担运算任务，A64gpu暂时未能调用起来。此次并行计算操作所使用的线程数设置为3时执行时间最优。**A64单纯靠CPU执行，所以执行时间比带有NPU的RK3588应该是要低很多的**。<u>由于官方的文档没有给出A64的浮点运算能力，但是经过测试，其运算能力大概在5GFLOPS左右，而RK3588的算力可达到6TFLOPS，差了3个量级。因此按理论来讲，RK3588使用NPU来跑onnx，在同个任务至少会比A64的时间少2-3个量级。</u>

## 3.算法测试（deploy源码）

为了方便Makefile文件的生成，使用qt作为框架移植算法。

**头文件做如下更改，使用交叉编译出来的文件系统带的头文件，另外一个 #include <emmintrin.h>注释。**

```c++
#include "/home/zsh/A64/cqa64_linuxqt5.8_bv3s/brandy/toolchain/gcc-linaro-4.9-2015.02-3-x86_64_aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/4.9.3/include/omp.h"
// #include <emmintrin.h>
```

分别对以下运算步骤进行运算时间评估：

| 步骤                                    | 耗时（s） |
| --------------------------------------- | --------- |
| elmt durations                          | 0.000003  |
| padding durations                       | 0.024179  |
| **conv3x3** durations                   | 4.949411  |
| padding **conv3x3** durations           | 5.505700  |
| padding **conv3x3** relu durations      | 5.565609  |
| padding **conv3x3** bias relu durations | 5.723547  |
| pixelshuffle duration                   | 0.054799  |

由于A64的gpu暂时无法调用，导致运算时间达到秒级别，因此在A64上想实现实时超分基本不可能。但这并不影响优化的进行，依旧可以在此平台研究算法优化有多少提升。

从结果来看，耗时最大的是卷积3*3的操作，从开源的代码来看，还有较大提升空间。函数如下

```c++
int32_t float_conv3x3s1p1(
    FloatTensor *out, 
    FloatTensor *inp, 
    FloatTensor *ker, 
    FloatTensor *bias, 
    OperationConfig)
{
    //...
    .....
        
        ///
            ker0 += 9;
        }
    }
    return 0;
}
```

这段代码中的循环结构有待改进并且内存访问低效。

1. 嵌套循环过多：该代码存在多层嵌套循环，导致每次迭代需要执行大量的乘法和加法操作。这会导致计算时间的增加。

   2.内存访问模式不佳：代码中频繁地进行内存访问，通过逐个元素地访问输入和卷积核的数据，这会导致较多的内存访问开销。

可以做以下优化：

1. **使用矩阵运算库如im2col、openblas优化卷积运算过程。**
2. **优化数据结构，避免频繁访问内存。**
3. **尝试调用A64的gpu进行加速运算。**

