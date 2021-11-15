import numpy as np
from alipy import ToolBox
from sklearn.tree import DecisionTreeClassifier
import random as rd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from alipy.experiment.al_experiment import AlExperiment
from sklearn.ensemble import RandomForestClassifier
def get_train_label(dir):
    array = [[]]
    file = dir
    f = open(file)
    lines = f.readlines()
    for i in range(len(lines)):
        list = lines[i].split()
        #del (list[0])
        array.append(list)
    del (array[0])
    array=np.array(array)

    return array
def get_train_data(dir):
    array = [[]]
    t_file = dir
    x_t = np.load(t_file, allow_pickle=True)['arr_0']
    x = np.array(x_t)
    return x
def get_val_data(dir):
    array = [[]]
    t_file = dir
    x_t = np.load(t_file,allow_pickle=True)['arr_0']
    x=np.array(x_t)
    return x
def get_val_label(dir):
    array = [[]]
    file = dir
    f = open(file)
    lines = f.readlines()
    for i in range(len(lines)):
        list = lines[i].split()
        # del (list[0])
        array.append(list)
    del (array[0])
    array = np.array(array)

    return array
def loadDataSet():  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    train_data_path = r'.\dataset\train_set\train_fea.npz'
    train_label_path = r'.\dataset\train_set\train_label.txt'
    test_data_path = r'.\dataset\test_set\test_fea.npz'
    test_label_path = r'.\dataset\test_set\test_label.txt'
    # test_data_path = r'.\dataset\test_set\dataset.npz'
    # test_label_path = r'.\dataset\test_set\ss.txt'
    X_train = get_train_data(train_data_path)
    Y_train = get_train_label(train_label_path)
    Y_train_s = np.array(Y_train).reshape(-1)
    x_test = get_val_data(test_data_path)
    y_test = get_val_label(test_label_path)
    # X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=1, shuffle=True)
    y_test_s = np.array(y_test).reshape(-1)
    return X_train,Y_train_s,x_test,y_test_s
X_train,Y_train_s,x_test,y_test_s = loadDataSet()
x_train=np.array(X_train)
x_test=np.array(x_test)
alibox = ToolBox(X=x_train, y=Y_train_s, query_type='AllLabels', saving_path='.')    #一般使用‘AllLabels’
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.05, split_count=10)
de_model = alibox.get_default_model()    #获取默认模型
de_model.fit(x_train, Y_train_s)    #传入数据及标签
pred = de_model.predict(x_test)    #标签预测
print(pred)

#pred = de_model.predict_proba(x)
acc = alibox.calc_performance_metric(y_true=y_test_s, y_pred=pred, performance_metric='accuracy_score')
print("accuracy score:", acc)
