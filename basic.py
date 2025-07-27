import tensorflow as tf


'''

print(tf.__version__)

'''

'''

1、加载一个预构建的数据集。
2、构建对图像进行分类的神经网络机器学习模型。
3、训练此神经网络。
4、评估模型的准确率。

'''


mnist = tf.keras.datasets.mnist

# 加载并准备 MNIST 数据集。将样本数据从整数转换为浮点数
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x = x_train[:1]
print(x.shape) # (1, 28, 28)
print(x_train.shape) # (1, 28, 28)

'''

# 通过堆叠层来构建 tf.keras.Sequential 模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# 对于每个样本，模型都会返回一个包含 logits 或 log-odds 分数的向量，每个类一个。
# 取第一个样本传入神经网络进行计算。
predictions = model(x_train[:1]).numpy()
# [[-0.09807709 -0.7592189   0.32323447 -0.26165694  0.7865137   0.26053178   0.3715647   0.7433355   0.1019477  -0.21061346]]


# tf.nn.softmax 函数将这些 logits 转换为每个类的概率
res = tf.nn.softmax(predictions).numpy()
# [[0.07258627 0.0374735  0.1106185  0.06163291 0.17580345 0.1038954  0.11609603 0.16837412 0.08865927 0.06486055]]

# 损失函数，交叉熵
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 配置模型
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 训练模型，使用 Model.fit 方法调整您的模型参数并最小化损失
model.fit(x_train, y_train, epochs=5)

# Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上检查模型性能
model.evaluate(x_test,  y_test, verbose=2)

'''