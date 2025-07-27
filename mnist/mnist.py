import tensorflow as tf

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist

# 加载训练数据和测试数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 打印数据集的维度信息
print("训练数据集维度:", x_train.shape)
print("训练标签集维度:", y_train.shape)
print("测试数据集维度:", x_test.shape)
print("测试标签集维度:", y_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建神经网络模型
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

# 对于每个样本，模型都会返回一个包含 logits 或 log-odds 分数的向量，每个类一个。
# 取第一个样本传入神经网络进行计算。
predictions = model(x_train[:1]).numpy()
# [[-0.09807709 -0.7592189   0.32323447 -0.26165694  0.7865137   0.26053178   0.3715647   0.7433355   0.1019477  -0.21061346]]


# tf.nn.softmax 函数将这些 logits 转换为每个类的概率
res = tf.nn.softmax(predictions).numpy()
# [[0.07258627 0.0374735  0.1106185  0.06163291 0.17580345 0.1038954  0.11609603 0.16837412 0.08865927 0.06486055]]

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