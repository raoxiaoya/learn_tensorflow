MNIST机器学习手写数字识别



https://www.bilibili.com/video/BV1y7411v7zd/?spm_id_from=333.337.search-card.all.click&vd_source=0c75dc193ee55511d0515b3a8c375bd0



两个示例

https://mp.weixin.qq.com/s/FwgabUUSkcCd-gWHnBnL9w

https://mp.weixin.qq.com/s/zuOVHVVjfEYCY39jUTLpvg



在深度学习中，“Hello World”通常指的是一个简单的示例，展示了如何使用深度学习框架来解决一个基本的问题，最经典的“Hello World”示例是使用神经网络来识别手写数字，这个问题通常被称为MNIST手写数字识别问题，在这个问题中，目标是训练一个卷积神经网络CNN，使其能够准确地识别手写数字图像中的数字，当然接下来的代码是通过调用相关库进行实现。

这个问题之所以成为“Hello World”，是因为它是深度学习中最简单、最基础的问题之一，很多人在学习深度学习时会从这个问题开始，通过解决这个问题，可以学会如何搭建神经网络、准备数据、进行训练和评估模型的性能等基本技能。

```bash
conda create -n tensorflow2.19.0 python==3.10.16
conda activate tensorflow2.19.0

pip install tensorflow==2.19.0
```

官网教程：https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh-cn

```python
mnist = tf.keras.datasets.mnist
model = tf.keras.models.Sequential
```

对于我这个初学者而言，想要找到 keras 在哪里都有点困难，它使用的是动态加载模块，所以编辑器也无法提示，包含 keras 的目录有下面几个，到底是哪一个呢？

```bash
d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\keras\src\models
D:\ProgramData\Anaconda3\envs\tensorflow2.19.0\Lib\site-packages\keras\models
D:\ProgramData\Anaconda3\envs\tensorflow2.19.0\Lib\site-packages\keras\_tf_keras\keras\models
D:\ProgramData\Anaconda3\envs\tensorflow2.19.0\Lib\site-packages\tensorflow\python\keras
```

我是比较喜欢看源码的，找不到某个方法的实现，强迫症都要犯了。这是在网上找的一张图片，展示了 tensorflow 的架构。

![image-20250716085146354](D:\dev\php\magook\trunk\server\md\img\image-20250716085146354.png)



[tensorflow](https://github.com/tensorflow/tensorflow) 的源码是用C++写的，它提供了其他语言的绑定，比如 Python, Java, Go, Javascript；我们使用 pip 安装 tensorflow 的时候，就是下载了 tensorflow 以及 python 的绑定。

然后就要阅读 tensorflow 的 `__init__.py`文件。

```bash
d:/ProgramData/Anaconda3/envs/tensorflow2.19.0/lib/site-packages/tensorflow/__init__.py
```

重点是要找到 keras 是从哪里加载的。

```bash
globals()
是一个内建函数，用于返回当前全局作用域中的所有变量、函数、类等符号的字典（dictionary）。这个字典包含了当前模块中所有全局变量的命名空间。
```

```bash
在 d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\tensorflow\__init__.py 的最后面打印

print(globals())
```

打印测试

```python
import tensorflow as tf
print(tf.__version__)
```

可以看到

```bash
 'keras': <module 'keras._tf_keras.keras' from 'd:/ProgramData/Anaconda3/envs/tensorflow2.19.0/lib/site-packages/keras/_tf_keras/keras/__init__.py'>,
```

从源码知道，minist 数据集被下载到`~/.keras/datasets`，这是 c 盘的当前用户的数据目录。

这个 `Sequential` 类既可以写成 `tf.keras.Sequential`也可以写成`tf.keras.models.Sequential`，其代码在 `D:\ProgramData\Anaconda3\envs\tensorflow2.19.0\Lib\site-packages\keras\src\models\sequential.py`

下面，使用 tensorflow 官网的例子来说明，https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh-cn

```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist

# 加载并准备 MNIST 数据集。将样本数据从整数转换为浮点数
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 通过堆叠层来构建 tf.keras.Sequential 模型
model = tf.keras.models.Sequential([
  	# 将二维的图像展平为一维向量 
    tf.keras.layers.Flatten(input_shape=(28, 28)),   
    # 添加一个具有16个神经元的全连接层，激活函数为ReLU
    tf.keras.layers.Dense(16, activation='relu'),      
    tf.keras.layers.Dense(16, activation='relu'), 
    # 添加一个Dropout层，防止过拟合 正则化有20%的神经元被丢弃以减少过拟合
    tf.keras.layers.Dropout(0.2),  
    # 添加一个具有10个神经元的输出层，激活函数为softmax
    tf.keras.layers.Dense(10, activation='softmax')  
])
```

首先就是模型的实例化，`Sequential`类传入参数 `layer`数组，也可以使用 `model.add()`来添加`layer`。

```python
# 对于每个样本，模型都会返回一个包含 logits 或 log-odds 分数的向量，每个类一个。
# 取第一个样本传入神经网络进行计算。
# x_train[:1] 为 (1, 28, 28)
predictions = model(x_train[:1]).numpy()
print(predictions)
# [[-0.09807709 -0.7592189   0.32323447 -0.26165694  0.7865137   0.26053178   0.3715647   0.7433355   0.1019477  -0.21061346]]

# tf.nn.softmax 函数将这些 logits 转换为每个类的概率
res = tf.nn.softmax(predictions).numpy()
print(res)
# [[0.07258627 0.0374735  0.1106185  0.06163291 0.17580345 0.1038954  0.11609603 0.16837412 0.08865927 0.06486055]]
```

`model()`表示调用对象的`__call__`方法，传参为训练集的第一个样本，意思就是将这个样本输入神经网络进行计算，查看输出的结果。其实就是`predict`操作，只是模型的权重还没有经过训练。

在文件 `D:\ProgramData\Anaconda3\envs\tensorflow2.19.0\Lib\site-packages\keras\src\trainers\trainer.py`中给出了注释

```python
For small numbers of inputs that fit in one batch,
directly use `__call__()` for faster execution, e.g.,
`model(x)`, or `model(x, training=False)` if you have layers such as
`BatchNormalization` that behave differently during
inference.

Note: See [this FAQ entry](https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
for more details about the difference between `Model` methods `predict()` and `__call__()`.
```

现在问题是`Sequential`类里面没有定义`__call__`方法，它又继承了一些父类，需要梳理调用链。使用 python debuger 调试发现它无法进入依赖的包里面。然后凭借直觉，在`sequential.py`的`call`方法打印调用栈

```python
import traceback
traceback.print_stack()
```

输出如下

```bash
File "D:\dev\php\magook\trunk\server\learn_tensorflow\basic.py", line 36, in <module>
    predictions = model(x_train[:1]).numpy()
  File "d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\keras\src\utils\traceback_utils.py", line 117, in error_handler
    return fn(*args, **kwargs)
  File "d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\keras\src\layers\layer.py", line 936, in __call__
    outputs = super().__call__(*args, **kwargs)
  File "d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\keras\src\utils\traceback_utils.py", line 117, in error_handler
    return fn(*args, **kwargs)
  File "d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\keras\src\ops\operation.py", line 58, in __call__
    return call_fn(*args, **kwargs)
  File "d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\keras\src\utils\traceback_utils.py", line 156, in error_handler
    return fn(*args, **kwargs)
  File "d:\ProgramData\Anaconda3\envs\tensorflow2.19.0\lib\site-packages\keras\src\models\sequential.py", line 221, in call
    traceback.print_stack()
```

在`operation.py`第58行打印

```python
print(call_fn)
```

才看到，它调用的的确是`Sequential.call`

```python
def call(self, inputs, training=None, mask=None, **kwargs):
        import traceback
        traceback.print_stack()
        if self._functional:
            return self._functional.call(
                inputs, training=training, mask=mask, **kwargs
            )

        # Fallback: Just apply the layer sequence.
        # This typically happens if `inputs` is a nested struct.
        for layer in self.layers:
            # During each iteration, `inputs` are the inputs to `layer`, and
            # `outputs` are the outputs of `layer` applied to `inputs`. At the
            # end of each iteration `inputs` is set to `outputs` to prepare for
            # the next layer.
            layer_kwargs = {
                k: kwargs[k]
                # only inject if this layer’s signature actually has that arg
                for k in getattr(layer, "_call_has_context_arg", {})
                if k in kwargs
            }
            if layer._call_has_mask_arg:
                layer_kwargs["mask"] = mask
            if layer._call_has_training_arg and training is not None:
                layer_kwargs["training"] = training
            outputs = layer(inputs, **layer_kwargs)
            inputs = outputs

            mask = tree.map_structure(backend.get_keras_mask, outputs)
        return outputs
```

`layer(inputs, **layer_kwargs)`就是依次调用各个`layer`的`call`方法来计算，计算结果作为下一个`layer`的输入。

```bash
Once the model is created, you can config the model with losses and metrics
with `model.compile()`, train the model with `model.fit()`, or use the model
to do prediction with `model.predict()`.
```

```python
# 配置模型
model.compile(
    optimizer='adam', # 优化器   
    loss='sparse_categorical_crossentropy', # 损失函数               
    metrics=['accuracy'] # 评价指标
)

# 打印模型的结构信息
model.summary()

# 训练模型，记录训练过程中的损失值
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# 绘制损失曲线
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 在测试集上评估模型性能
test_loss, test_acc = model.evaluate(x_test, y_test)
print("测试集损失:", test_loss)
print("测试集准确率:", test_acc)

# 保存模型
model.save('handwritten_digit.keras')
```

打印信息

```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dense (Dense)                (None, 16)                12560
_________________________________________________________________
dense_1 (Dense)              (None, 16)                272
_________________________________________________________________
dropout (Dropout)            (None, 16)                0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                170
=================================================================
Total params: 13,002
Trainable params: 13,002
Non-trainable params: 0
_________________________________________________________________
```

**模型解释：**

- Flatten 层: 这是输入层，用于将输入的二维图像数据展平成一维向量，输入图像的尺寸为 28x28 像素，所以展平后的向量长度为 784

- Dense 层: 这是第一个隐藏层，包含 16 个神经元，该层是全连接层，每个神经元与上一层中的所有神经元相连，参数数量为 784 (输入向量长度) * 16 (神经元数量) + 16 (偏置项) = 12560

- Dense 层: 这是第二个隐藏层，同样包含 16 个神经元，参数数量为 16 (上一层神经元数量) * 16 (神经元数量) + 16 (偏置项) = 272

- Dropout 层: 这是一个 Dropout 层，用于在训练过程中随机将一部分神经元的输出置为零，以防止过拟合，在本模型中，Dropout 层的输出形状与上一层相同，即 (None, 16)

- Dense 层: 这是输出层，包含 10 个神经元，对应着 10 个类别（0 到 9 的数字），参数数量为 16 (上一层神经元数量) * 10 (神经元数量) + 10 (偏置项) = 170

总参数数量为13002，其中所有参数都是可训练的，模型的训练目标是最小化损失函数，使得模型能够准确地预测输入图像对应的数字类别。

关于损失函数`sparse_categorical_crossentropy`与`categorical_crossentropy`，虽然都是交叉熵，但是计算格式不一样，前者的targets是数字编码，比如`1, 2, 3`，后者是one-hot编码，比如`[0, 1, 0]`。他们都应用于多分类问题。在这个例子中，我们的样本中每一个图像的标签是0-9的数值，因此选择`sparse_categorical_crossentropy`。

数值转换成one-hot：对于一个数9，将其转换成10位的one-hot向量为`[0,0,0,0,0,0,0,0,0,1]`

```bash
Epoch 1/20
1875/1875 [==============================] - 5s 2ms/step - loss: 1.0119 - accuracy: 0.6633 - val_loss: 0.2844 - val_accuracy: 0.9179
Epoch 2/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3920 - accuracy: 0.8817 - val_loss: 0.2388 - val_accuracy: 0.9302
Epoch 3/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3255 - accuracy: 0.9032 - val_loss: 0.2211 - val_accuracy: 0.9358
Epoch 4/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3046 - accuracy: 0.9107 - val_loss: 0.2036 - val_accuracy: 0.9414
Epoch 5/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2756 - accuracy: 0.9177 - val_loss: 0.2029 - val_accuracy: 0.9415
Epoch 6/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2537 - accuracy: 0.9255 - val_loss: 0.1937 - val_accuracy: 0.9444
Epoch 7/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2382 - accuracy: 0.9294 - val_loss: 0.1925 - val_accuracy: 0.9465
Epoch 8/20
1875/1875 [==============================] - 3s 1ms/step - loss: 0.2375 - accuracy: 0.9292 - val_loss: 0.1877 - val_accuracy: 0.9461
Epoch 9/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2300 - accuracy: 0.9328 - val_loss: 0.1886 - val_accuracy: 0.9456
Epoch 10/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2212 - accuracy: 0.9340 - val_loss: 0.1883 - val_accuracy: 0.9471
Epoch 11/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2138 - accuracy: 0.9369 - val_loss: 0.1860 - val_accuracy: 0.9476
Epoch 12/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2013 - accuracy: 0.9416 - val_loss: 0.1855 - val_accuracy: 0.9471
Epoch 13/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1994 - accuracy: 0.9396 - val_loss: 0.1830 - val_accuracy: 0.9500
Epoch 14/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1976 - accuracy: 0.9413 - val_loss: 0.1812 - val_accuracy: 0.9506
Epoch 15/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1983 - accuracy: 0.9411 - val_loss: 0.1832 - val_accuracy: 0.9504
Epoch 16/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1892 - accuracy: 0.9443 - val_loss: 0.2024 - val_accuracy: 0.9471
Epoch 17/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1855 - accuracy: 0.9438 - val_loss: 0.2021 - val_accuracy: 0.9458
Epoch 18/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1858 - accuracy: 0.9453 - val_loss: 0.1853 - val_accuracy: 0.9516
Epoch 19/20
1875/1875 [==============================] - 3s 1ms/step - loss: 0.1853 - accuracy: 0.9446 - val_loss: 0.1849 - val_accuracy: 0.9497
Epoch 20/20
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1803 - accuracy: 0.9456 - val_loss: 0.1888 - val_accuracy: 0.9506
313/313 [==============================] - 0s 803us/step - loss: 0.1888 - accuracy: 0.9506
测试集损失: 0.18882906436920166
测试集准确率: 0.9506000280380249
```



![image-20240906115901155](D:\dev\php\magook\trunk\server\md\img\image-20240906115901155.png)



**模型调用识别数字**

![图片](https://mmbiz.qpic.cn/mmbiz_png/cCtGVD6h9medUfD2dmZdcBBJs4q8IKlCYDmnt47pHkhI8odN1hXxTQypDRldBbeMD1JRgP2jBQ3o9iaeHJ3QEyA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

加载一个已经训练好的神经网络模型，该模型用于识别手写数字，然后读取一张包含手写数字的图像（上图），并对图像进行预处理，使其与模型的输入格式相匹配，最后通过该模型对预处理后的图像进行预测，输出预测的手写数字

```bash
pip install opencv-python
```

```python
import tensorflow as tf
import cv2
import numpy as np
# 加载训练好的模型
model = tf.keras.models.load_model('handwritten_digit.keras')
# 读取现实中的手写数字图像
image = cv2.imread('handwritten_digit.png', cv2.IMREAD_GRAYSCALE)
# 调整图像尺寸为模型输入的大小（28x28）
image_resized = cv2.resize(image, (28, 28))
# 对图像进行归一化处理
image_normalized = image_resized / 255.0
# 将图像转换为模型所需的形状 (1, 28, 28)
image_input = np.expand_dims(image_normalized, axis=0)
prediction = model.predict(image_input)
predicted_digit = np.argmax(prediction)
print("预测结果:", predicted_digit)
```

```bash
预测结果: 6
```



**python代码实现2-使用卷积神经网络**

CNN的通用架构：`卷积-池化-卷积-池化-全连接-全连接-输出`

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 1、构建CNN模型
# 构建一个最基础的连续的模型，所谓连续，就是一层接着一层
model = tf.keras.models.Sequential()
# 第一层为一个卷积，卷积核大小为(3,3), 输出通道32，使用 relu 作为激活函数
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 第二层为一个最大池化层，池化核为（2,2)
# 最大池化的作用，是取出池化核（2,2）范围内最大的像素点代表该区域
# 可减少数据量，降低运算量。
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 又经过一个（3,3）的卷积，输出通道变为64，也就是提取了64个特征。
# 同样为 relu 激活函数
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# 上面通道数增大，运算量增大，此处再加一个最大池化，降低运算
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# dropout 随机设置一部分神经元的权值为零，在训练时用于防止过拟合
# 这里设置25%的神经元权值为零
model.add(tf.keras.layers.Dropout(0.25))
# 将结果展平成1维的向量
model.add(tf.keras.layers.Flatten())
# 增加一个全连接层，用来进一步特征融合
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 再设置一个dropout层，将50%的神经元权值为零，防止过拟合
# 由于一般的神经元处于关闭状态，这样也可以加速训练
model.add(tf.keras.layers.Dropout(0.5))
# 最后添加一个全连接softmax激活，输出10个分类，分别对应0-9 这10个数字
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 编译上述构建好的神经网络模型
# 指定优化器为 rmsprop
# 制定损失函数为交叉熵损失
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型的结构信息
model.summary()

# 2、处理数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 给标签增加维度,使其满足模型的需要 
# 原始标签，比如训练集标签的维度信息是[60000, 28, 28, 1]
X_train = x_train.reshape(60000, 28, 28, 1)
X_test = x_test.reshape(10000, 28, 28, 1)
# 特征转换为one-hot编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3、开始训练              
model.fit(
    X_train, y_train,  # 指定训练特征集和训练标签集
    validation_split=0.3,  # 部分训练集数据拆分成验证集
    epochs=5,  # 训练轮次为5轮
    batch_size=128)  # 以128为批量进行训练

# 在测试集上进行模型评估
score = model.evaluate(X_test, y_test)
print('测试集预测准确率:', score[1])  #  打印测试集上的预测准确率


#  预测验证集第一个数据
pred = model.predict(X_test[0].reshape(1, 28, 28, 1))
# 把one-hot码转换为数字
print(pred[0], "转换一下格式得到：", pred.argmax())
# 导入绘图工具包
# 输出这个图片
plt.imshow(X_test[0].reshape(28, 28), cmap='Greys')

```

模型信息

```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
dropout (Dropout)            (None, 5, 5, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
dense (Dense)                (None, 128)               204928
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 225,034
Trainable params: 225,034
Non-trainable params: 0
_________________________________________________________________
```



to_categorical 的作用是将样本标签转为 one-hot 编码，而 one-hot  编码的作用是可以对于类别更好的计算概率或得分。这个例子中，数字 0-9 转换为的独热编码为：

```bash
array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
```



**卷积核的通道数以及参数量的计算**

卷积神经网络的原理：https://www.bilibili.com/video/BV1f54y1f7rs/?spm_id_from=333.788&vd_source=dd488f2825c3a352e192887d5d63e429

卷积核可以是多层立体的，就像魔方那样，层数就是深度也叫通道数，比如RGB图像有三个通道，我们也应该用三层的卷积核去扫它，每一层负责扫一个通道。

所以卷积核一般不用设置输入的通道数，因为它与输入图像的通道数一样。

但是需要设置输出结果的通道数，比如你可以将RGB三通道图像经过卷积核变成32通道的图像。通道越多相当于升维了，便于提取特征。

![image-20240906151300918](D:\dev\php\magook\trunk\server\md\img\image-20240906151300918.png)

卷积层的参数量跟卷积核有关。我们先来看看多层卷积核是怎么计算的，此处以三通道为例。

![image-20240906152359129](D:\dev\php\magook\trunk\server\md\img\image-20240906152359129.png)

![image-20240906154046640](D:\dev\php\magook\trunk\server\md\img\image-20240906154046640.png)

从图中可以看出，一个卷积核的输出结果为一个通道，其参数量：`N = 输入的通道数 * 核宽 * 核高 + 1`，这个`1`表示偏置量。如果输出的通道数为P，那么就需要P个卷积核，参数量就是`P*N`。总结下来就是

```bash
L = P * (D * U * V + 1)
P：输出通道数
D：输入通道数
U：核宽
V：核高
```



于是示例2中的`conv2D`层的参数计算如下：

```bash
32 * (1 * 3 * 3 + 1) = 320
64 * (32 * 3 * 3 + 1) = 18496
```



**可视化MNIST的运行过程**

https://tensorspace.org/index.html

![image-20240906161235863](D:\dev\php\magook\trunk\server\md\img\image-20240906161235863.png)
