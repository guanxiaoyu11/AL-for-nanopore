import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from alipy.index.index_collections import IndexCollection
from alipy.query_strategy import (QueryInstanceQBC, QueryInstanceGraphDensity,
                                  QueryInstanceUncertainty, QueryRandom)
from alipy import ToolBox
import warnings
from sklearn.metrics import confusion_matrix
import json
from tqdm import tqdm
import os
import time
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
warnings.filterwarnings("ignore")
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
class Normalize(object):

    def normalize(self, X):
        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X)
        return X

    def inverse(self, X):
        X= self.scaler.inverse_transform(X)

        return X
train_data_path = r'.\dataset\train_set\train_fea.npz'
train_label_path = r'.\dataset\train_set\train_label.txt'
test_data_path = r'.\dataset\test_set\test_fea.npz'
test_label_path = r'.\dataset\test_set\test_label.txt'
# test_data_path = r'.\dataset\test_set\dataset.npz'
# test_label_path = r'.\dataset\test_set\ss.txt'
X_raw = get_train_data(train_data_path)
y_raw = get_train_label(train_label_path)
y_raw = np.array(y_raw).reshape(-1)
x_test = get_val_data(test_data_path)
y_test = get_val_label(test_label_path)
# X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=1, shuffle=True)
y_test_s = np.array(y_test).reshape(-1)
# X, y = load_iris(return_X_y=True)
X=np.append(X_raw,x_test,axis=0)
y=np.append(y_raw,y_test_s).astype(int)
normalizer = Normalize()
X=normalizer.normalize(X)
alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')
alibox.split_AL1( split_count=10)

# Split data
# alibox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = RandomForestClassifier(n_estimators=500)
model.fit(X=X[0:1020], y=y[0:1020])
pred = model.predict(X[1020:])
feature_name = ['data_mean', 'data_std', 'data_med', 'data_max', 'data_min', 'data_len', 'data_skew',
                            'data_kurt', 'step_2', 'step_1',
                            'noise']
labels = ['0', '1', '2', '3', '4', '5', '6']
###10.17 gxy add the tree generate
# for idx, estimator in enumerate(model.estimators_):
#     # 导出dot文件
#     tree.export_graphviz(estimator,
#                         out_file='output/tree1/tree{}.dot'.format(idx),
#                         feature_names=feature_name,
#                         class_names=labels,
#                         rounded=True,
#                         proportion=False,
#                         precision=2,
#                         filled=True)
#     # 转换为png文件
#     os.system('dot -Tpng output/tree1/tree{}.dot -o output/tree1/tree{}.png'.format(idx, idx))
###
accuracy = alibox.calc_performance_metric(y_true=y[1020 :],
                                              y_pred=pred,
                                              performance_metric='accuracy_score')
print("Random Forest best performance:"+str(accuracy))

# The cost budget is 50 times querying
stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 100)

def selcet_test(label_index, unlabel_index, batch_size,models):
    unlabel_x = X[unlabel_index, :]
    unlabel_x_mean=np.mean(unlabel_x)
    unlabel_x_std=np.std(unlabel_x)
    # unlabel_x=abs(unlabel_x-unlabel_x_mean)
    # pv= models.predict(unlabel_x)
    pv=models.predict_proba(unlabel_x)
    pv = np.asarray(pv)  # predict value
    spv = np.shape(pv)
    pat = np.partition(pv, (spv[1] - 2, spv[1] - 1), axis=1)
    pat = pat[:, spv[1] - 2] - pat[:, spv[1] - 1]
    argret = np.argsort(pat)
    return argret[argret.size - batch_size:]
def main_loop(alibox, strategy, round):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    # Set initial performance point
    model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                              y_pred=pred,
                                              performance_metric='accuracy_score')
    init_label=label_ind._innercontainer
    print("init label is "+','.join('%s' %id for id in label_ind._innercontainer))
    print(accuracy)
    saver.set_initial_point(accuracy)
    selected_label=[]
    # If the stopping criterion is simple, such as query 50 times. Use `for i in range(50):` is ok.
    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = strategy.select(label_index=label_ind, unlabel_index=unlab_ind, batch_size=10,model=model)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)
        for i in range(len(select_ind)):
            selected_label.append(select_ind[i])
        # print("selected label is "+','.join('%s' %id for id in select_ind.tolist()))
        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')
        # if accuracy>0.934:
        #     # feature_name = ['data_mean', 'data_std', 'data_med', 'data_max', 'data_min', 'data_len', 'data_skew',
        #     #                 'data_kurt', 'step_2', 'step_1',
        #     #                 'noise']
        #     # labels = ['0', '1', '2', '3', '4', '5', '6']
        #     # ###10.17 gxy add the tree generate
        #     # for idx, estimator in enumerate(model.estimators_):
        #     #     # 导出dot文件
        #     #     tree.export_graphviz(estimator,
        #     #                     out_file='output/tree/tree{}.dot'.format(idx),
        #     #                     feature_names=feature_name,
        #     #                     class_names=labels,
        #     #                     rounded=True,
        #     #                     proportion=False,
        #     #                     precision=2,
        #     #                     filled=True)
        #     #     # 转换为png文件
        #     #     os.system('dot -Tpng output/tree/tree{}.dot -o output/tree/tree{}.png'.format(idx, idx))
        #     # ###
        #     #
        #     # # 特征重要性
        #     # y_importances = model.feature_importances_
        #     # x_importances = feature_name
        #     # y_pos = np.arange(len(x_importances))
        #     # # 横向柱状图
        #     # total_heigh, n = 0.55, 2
        #     # heigh = total_heigh / n
        #     # plt.figure(3, dpi=120)
        #     # plt.barh(y_pos, y_importances, align='center', height=heigh, tick_label=feature_name)
        #     # plt.yticks(y_pos, x_importances, fontsize='10', fontproperties='arial')
        #     # plt.xlabel('Importances', fontsize='10', fontproperties='arial')
        #     # plt.xlim(0, 0.7)
        #     # plt.title('Features Importances', fontsize='10', fontproperties='arial')
        #     # plt.show()
        #     # obj1 = confusion_matrix(y_test.reshape(559).astype(int), pred)
        #     # print('confusion_matrix\n', obj1)
        #     # classes = list(set(pred))
        #     # classes.sort()
        #     # plt.figure(figsize=(12, 8), dpi=120)
        #     # plt.imshow(obj1, cmap=plt.cm.Blues)
        #     #
        #     # indices = range(len(obj1))
        #     # plt.xticks(indices, classes)
        #     # plt.yticks(indices, classes)
        #     # plt.colorbar()
        #     # plt.xlabel('label2 of test set sample')
        #     # plt.ylabel('label2 of predict')
        #     # plt.title('Confusion Matrix', fontsize='10', fontproperties='arial')
        #     # for first_index in range(len(obj1)):
        #     #     for second_index in range(len(obj1[first_index])):
        #     #         plt.text(first_index, second_index, obj1[first_index][second_index], fontsize=10, va='center',
        #     #                  ha='center')
        #     # labels = ['0', '1', '2', '3', '4', '5', '6']
        #     # cm = confusion_matrix(y_test.reshape(559).astype(int), pred)
        #     # tick_marks = np.array(range(len(labels))) + 0.5
        #     #
        #     # def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
        #     #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #     #     plt.title(title)
        #     #     plt.colorbar()
        #     #     xlocations = np.array(range(len(labels)))
        #     #     plt.xticks(xlocations, labels, rotation=90)
        #     #     plt.yticks(xlocations, labels)
        #     #     plt.ylim(len(labels) - 0.5, -0.5)
        #     #     plt.ylabel('True label')
        #     #     plt.xlabel('Predicted label')
        #     #     cm = confusion_matrix(y_test.reshape(559).astype(int), pred)
        #     #     np.set_printoptions(precision=2)
        #     #
        #     # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #     # plt.figure(figsize=(12, 8), dpi=120)
        #     #
        #     # ind_array = np.arange(len(labels))
        #     # x1, y1 = np.meshgrid(ind_array, ind_array)
        #     # thresh = cm.max() / 2
        #     # for x_vall, y_vall in zip(x1.flatten(), y1.flatten()):
        #     #     c = cm_normalized[y_vall][x_vall]
        #     #     if c > 0.0001:
        #     #         plt.text(x_vall, y_vall, "%0.4f" % (c,), color="white" if cm[x_vall, y_vall] > thresh else "black",
        #     #                  fontsize=10, va='center', ha='center')
        #     # # offset the tick
        #     # plt.gca().set_xticks(tick_marks, minor=True)
        #     # plt.gca().set_yticks(tick_marks, minor=True)
        #     # plt.gca().xaxis.set_ticks_position('none')
        #     # plt.gca().yaxis.set_ticks_position('none')
        #     # plt.grid(True, which='minor', linestyle='-')
        #     # plt.gcf().subplots_adjust(bottom=0.15)
        #     #
        #     # plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        #     # # show confusion matrix
        #     #
        #     # plt.show()
        #     break
        # print(accuracy)
        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()

    # return saver,init_label,selected_label
    return saver


unc_result = []
qbc_result = []
eer_result = []
quire_result = []
density_result = []
bmdr_result = []
spal_result = []
lal_result = []
rnd_result = []

_I_have_installed_the_cvxpy = False

for round in range(1):
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # n_labeled_examples = X_raw.shape[0]
    # train_idx = np.arange(0, n_labeled_examples)
    # label_ind = np.random.randint(low=0, high=n_labeled_examples + 1, size=10)
    # test_idx = np.arange(n_labeled_examples, n_labeled_examples + x_test.shape[0])
    # unlab_ind = np.delete(train_idx,label_ind, axis=0)
    # train_idx=IndexCollection(train_idx)
    # label_ind=IndexCollection(label_ind)
    # test_idx=IndexCollection(test_idx)
    # unlab_ind=IndexCollection(unlab_ind)
    # Use pre-defined strategy
    # unc = alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty")
    unc = QueryInstanceUncertainty(X, y,measure='margin')
    qbc = alibox.get_query_strategy(strategy_name="QueryInstanceQBC")
    # eer = alibox.get_query_strategy(strategy_name="QueryExpectedErrorReduction")
    rnd = alibox.get_query_strategy(strategy_name="QueryInstanceRandom")
    quire = alibox.get_query_strategy(strategy_name="QueryInstanceQUIRE", train_idx=train_idx)
    density = alibox.get_query_strategy(strategy_name="QueryInstanceGraphDensity", train_idx=train_idx)
    #lal = alibox.get_query_strategy(strategy_name="QueryInstanceLAL", cls_est=10, train_slt=False)
    # lal.download_data()
    # lal.train_selector_from_file(reg_est=30, reg_depth=5)
    # unc_main_loop,init_label,selected_label=main_loop(alibox, unc, round)
    # save_data = r"output\init.npz"
    # np.savez(save_data, np.array(init_label[:10]))
    # save_data = r"output\selected.npz"
    # np.savez(save_data, np.array(selected_label))
    # rnd_main_loop, init_label, selected_label = main_loop(alibox, rnd, round)
    time_1 = time.time()
    unc_result.append(copy.deepcopy(main_loop(alibox, unc, round)))
    time_2 = time.time()
    print('unc_result cost %f seconds' % (time_2 - time_1))
    # unc_result.append(copy.deepcopy(unc_main_loop))
    qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))
    time_3 = time.time()
    print('qbc_result cost %f seconds' % (time_3 - time_2))
    # eer_result.append(copy.deepcopy(main_loop(alibox, eer, round)))
    rnd_result.append(copy.deepcopy(main_loop(alibox, rnd, round)))
    time_4 = time.time()
    print('rnd_result cost %f seconds' % (time_4 - time_3))
    quire_result.append(copy.deepcopy(main_loop(alibox, quire, round)))
    time_5 = time.time()
    print('quire_result cost %f seconds' % (time_5 - time_4))
    density_result.append(copy.deepcopy(main_loop(alibox, density, round)))
    time_6 = time.time()
    print('density_result cost %f seconds' % (time_6 - time_5))
    # lal_result.append(copy.deepcopy(main_loop(alibox, lal, round)))

    # if _I_have_installed_the_cvxpy:
    #     bmdr = alibox.get_query_strategy(strategy_name="QueryInstanceBMDR", kernel='rbf')
    #     spal = alibox.get_query_strategy(strategy_name="QueryInstanceSPAL", kernel='rbf')
    #
    #     bmdr_result.append(copy.deepcopy(main_loop(alibox, bmdr, round)))
    #     spal_result.append(copy.deepcopy(main_loop(alibox, spal, round)))

analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='QBC', method_results=qbc_result)
qbc_result_nd=[]
# for i in range(len(qbc_result[0]._StateIO__state_list)):
#     qbc_result_nd.append(qbc_result[0]._StateIO__state_list[i]._save_seq['performance'])
# np.savez(r"result\qbc.npz",np.array(qbc_result_nd))
analyser.add_method(method_name='Unc', method_results=unc_result)
unc_result_nd=[]
# for i in range(len(unc_result[0]._StateIO__state_list)):
#     unc_result_nd.append(unc_result[0]._StateIO__state_list[i]._save_seq['performance'])
# np.savez(r"result\unc.npz",np.array(unc_result_nd))
#analyser.add_method(method_name='EER', method_results=eer_result)
analyser.add_method(method_name='Random', method_results=rnd_result)
rnd_result_nd=[]
# for i in range(len(rnd_result[0]._StateIO__state_list)):
#     rnd_result_nd.append(rnd_result[0]._StateIO__state_list[i]._save_seq['performance'])
# np.savez(r"result\rnd.npz",np.array(rnd_result_nd))
analyser.add_method(method_name='QUIRE', method_results=quire_result)
quire_result_nd=[]
# for i in range(len(quire_result[0]._StateIO__state_list)):
#     quire_result_nd.append(quire_result[0]._StateIO__state_list[i]._save_seq['performance'])
# np.savez(r"result\quire.npz",np.array(quire_result_nd))
analyser.add_method(method_name='Density', method_results=density_result)
density_result_nd=[]
# for i in range(len(density_result[0]._StateIO__state_list)):
#     density_result_nd.append(density_result[0]._StateIO__state_list[i]._save_seq['performance'])
# np.savez(r"result\density.npz",np.array(density_result_nd))
# analyser.add_method(method_name='LAL', method_results=lal_result)
# if _I_have_installed_the_cvxpy:
#     analyser.add_method(method_name='BMDR', method_results=bmdr_result)
#     analyser.add_method(method_name='SPAL', method_results=spal_result)
print(analyser)
x = get_train_data(train_data_path)
y_raw = get_train_label(train_label_path)
y = np.array(y_raw).reshape(-1)
x_test = get_val_data(test_data_path)
y_test = get_val_label(test_label_path)
    # X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=1, shuffle=True)
y_test_s = np.array(y_test).reshape(-1)
    # Random values in a given shape
rng = np.random.RandomState(37)
y_rand = rng.rand(y.shape[0])
    # 根据随机数小于0.2的值，设置未标记样本
smi_result=[]
time_1 = time.time()
for i in range(0,1010,10):
    y_20 = np.copy(y)
    y_rand.sort()
    y_20[10+i:] = -1
        # 标签传播，图半监督学习
    model = RandomForestClassifier(n_estimators=500)
    model.fit(x, y_20)
    smi_result.append(model.score(x_test, y_test))
time_2 = time.time()
print('smi_result cost %f seconds' % (time_2 - time_1))
# np.savez(r"result\smi.npz",np.array(smi_result))
plt.figure(figsize=(8, 8))
plt.plot(range(len(smi_result)),smi_result,label='semi')
plt.legend()
plt.ylim(0.4, 1)
plt.xlabel('Number of queries')
plt.ylabel('Performance')
plt.title('Example of alipy')

analyser.plot_learning_curves(title='Example of alipy', std_area=False)