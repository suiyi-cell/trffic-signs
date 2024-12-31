import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime
from tf2onnx import convert


# 改进数据增强策略，添加更多操作
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,  # 随机旋转角度范围
    width_shift_range=0.1,  # 水平平移范围
    height_shift_range=0.1,  # 垂直平移范围
    shear_range=0.1,  # 错切变换范围
    zoom_range=0.1,  # 缩放范围
    horizontal_flip=True  # 水平翻转
)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 指向训练集文件夹路径
train_generator = train_datagen.flow_from_directory(
    'C:/Users/25507/Desktop/train-sign/train',  # 替换成实际的训练集文件夹路径
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# 指向测试集文件夹路径
test_generator = test_datagen.flow_from_directory(
    'C:/Users/25507/Desktop/train-sign/test',  # 替换成实际的测试集文件夹路径
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# 定义 Inception 模块
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    return output


# 构建 GoogLeNet 模型结构（调整后的模型结构，增加了更多的 Inception 模块）
input_layer = Input(shape=(224, 224, 3))

# 第一部分：开始的卷积和池化层
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# 第二部分：多个 Inception 模块堆叠（增加了更多的 Inception 模块）
x = inception_module(x, 64, 96, 128, 16, 32, 32)
x = inception_module(x, 128, 128, 192, 32, 96, 64)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, 192, 96, 208, 16, 48, 64)
x = inception_module(x, 160, 112, 224, 24, 64, 64)
x = inception_module(x, 128, 128, 256, 24, 64, 64)
x = inception_module(x, 112, 144, 288, 32, 64, 64)
x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = inception_module(x, 384, 192, 384, 48, 128, 128)
x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = inception_module(x, 384, 192, 384, 48, 128, 128)
x = inception_module(x, 256, 160, 320, 32, 128, 128)
x = inception_module(x, 384, 192, 384, 48, 128, 128)

# 第三部分：平均池化、全连接层和输出层
x = AveragePooling2D((7, 7), strides=(1, 1))(x)
x = Flatten()(x)
# 适当增加 Dropout 比率
x = Dropout(0.5)(x)
output_layer = Dense(49, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# 调整学习率，可尝试不同值
adam = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# 设置早停回调，监控验证集的损失，当连续 5 轮损失不再下降时停止训练
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 添加 TensorBoard 回调
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',  # 存储 TensorBoard 日志的目录
    histogram_freq=1,  # 每一个 epoch 后记录激活直方图
    write_graph=True,  # 写入模型图
    write_images=True  # 写入模型权重的可视化表示
)


# 使用生成器来训练模型，添加早停回调和 TensorBoard 回调
model.fit(
    train_generator,
    epochs=500,  # 可根据实际情况调整
    validation_data=test_generator,
    callbacks=[early_stopping_callback, tensorboard_callback]
)

# 导出为 ONNX 模型
onnx_model, _ = convert.from_keras(model)
onnx.save(onnx_model, "googlenet.onnx")

# 在测试集上进行预测，获取预测结果和真实标签
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# 计算精准率、召回率、F1-score
# precision = precision_score(y_true, y_pred, average='weighted')
# recall = recall_score(y_true, y_pred, average='weighted')
# f1 = f1_score(y_true, y_pred, average='weighted')

# 计算混淆矩阵
confusion_mtx = confusion_matrix(y_true, y_pred)


# 可视化混淆矩阵的函数
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(10, 10))  # 调整图像大小
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 显示颜色条
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")  # 旋转 x 轴标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()  # 调整布局


# 在测试集上评估模型
score = model.evaluate(test_generator)
print(model.summary())
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1-score:', f1)

# 调用函数可视化混淆矩阵
class_names = [str(i) for i in range(49)]
plot_confusion_matrix(confusion_mtx, class_names)
plt.show()