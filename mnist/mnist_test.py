import tensorflow as tf
import cv2
import numpy as np

# 加载训练好的模型
model = tf.keras.models.load_model('handwritten_digit.keras')

# 读取现实中的手写数字图像
image = cv2.imread('handwritten_digit.png', cv2.IMREAD_GRAYSCALE)  # (28, 28)

# 调整图像尺寸为模型输入的大小（28x28）
image_resized = cv2.resize(image, (28, 28))

# 对图像进行归一化处理
image_normalized = image_resized / 255.0

# 将图像转换为模型所需的形状 (1, 28, 28)
image_input = np.expand_dims(image_normalized, axis=0)
prediction = model.predict(image_input)
predicted_digit = np.argmax(prediction)
print("预测结果:", predicted_digit)
