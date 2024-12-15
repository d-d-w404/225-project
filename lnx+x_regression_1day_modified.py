import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 读取数据
file_path = './assignment2/bitcoin_2019-11-29_2024-11-27.csv'
data = pd.read_csv(file_path)

# 转换日期列为日期类型，并按时间顺序排序
data['Start'] = pd.to_datetime(data['Start'])
data = data.sort_values(by='Start').reset_index(drop=True)
data['High-Low'] = data['High'] - data['Low']

# 使用前一天的数据作为特征，预测后一天的 Close
data['Prev_Volume'] = data['Volume'].shift(1)
data['Prev_Market Cap'] = data['Market Cap'].shift(1)
data['Prev_High-Low'] = data['High-Low'].shift(1)
#data['Prev_Close'] = data['Close'].shift(1)

# 删除第一行（因 shift 导致的缺失值）
data = data.dropna().reset_index(drop=True)

# 提取输入特征和目标值
X = data[['Prev_Volume', 'Prev_Market Cap', 'Prev_High-Low']]
y = data['Close']

# 根据日期划分训练集（前4年）和验证集（最后1年）
train_data = data[data['Start'] < '2024-6-27']
val_data = data[data['Start'] >= '2024-6-27']

X_train = train_data[['Prev_Volume', 'Prev_Market Cap', 'Prev_High-Low']]
y_train = train_data['Close']
X_val = val_data[['Prev_Volume', 'Prev_Market Cap', 'Prev_High-Low']]
y_val = val_data['Close']

# 对输入特征应用 ln(x) + x 变换
def transform_features(X):
    X_transformed = np.log(X) + X
    return X_transformed

# 转换训练集和验证集特征
X_train_transformed = transform_features(X_train)
X_val_transformed = transform_features(X_val)

# 多项式回归模型（使用二次项为例）
degree = 1
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X_train_transformed, y_train)

# 预测验证集
y_poly_pred = poly_model.predict(X_val_transformed)

# 计算模型评估指标
mse_poly = mean_squared_error(y_val, y_poly_pred)
r2_poly = r2_score(y_val, y_poly_pred)

print(mse_poly, r2_poly)

# 创建一个可视化函数
def visualize_predictions(dates, actual, predicted):
    plt.figure(figsize=(12, 6))
    
    # 绘制实际值
    plt.plot(dates, actual, label='Actual Values', color='blue', linewidth=2)
    
    # 绘制预测值
    plt.plot(dates, predicted, label='Predicted Values', color='orange', linestyle='--', linewidth=2)
    
    # 添加标题和标签
    plt.title('Comparison of Actual and Predicted Values', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Close Price', fontsize=14)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 旋转日期标签
    plt.xticks(rotation=45)
    
    # 显示图形
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 准备日期、实际值和预测值
dates_val = val_data['Start']
visualize_predictions(dates_val, y_val, y_poly_pred)
