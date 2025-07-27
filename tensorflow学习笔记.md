tensorflow学习笔记



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

首先就是模型的实例化，`Sequential`类传入参数 `layer`数组，也可以使用 `model.add()`来添加`layer`。接着往下。

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

加载模型进行预测

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

`predict`的实现在 `D:\ProgramData\Anaconda3\envs\tensorflow2.19.0\Lib\site-packages\keras\src\backend\tensorflow\trainer.py`













```bash
__new__ 与 __init__ 

```

