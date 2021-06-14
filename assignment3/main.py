from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
import seaborn as sns
import math

def DecisionTree(x_train, y_train, x_test, y_test, feature, target):
    # Decision Tree
    treelist = []
    # 1.parameter list
    # 1.1 max depth
    max_depth = 2
    while max_depth<=15:
        dt = DecisionTreeRegressor(max_depth=max_depth)
        dt.fit(x_train, y_train)
        treelist.append(dt)
        max_depth+=1
    min_leaf = 2
    while min_leaf<=30:
        dt = DecisionTreeRegressor(min_samples_leaf=min_leaf)
        dt.fit(x_train, y_train)
        treelist.append(dt)
        min_leaf+=1

    test_err_list = []
    train_err_list = []
    for tree in treelist:

    # 3. predict
        train_err = 1/(x_train.shape[0]) * sum((tree.predict(x_train) - y_train) ** 2)
        test_err = 1/(x_test.shape[0]) * sum((tree.predict(x_test) - y_test) ** 2)
        train_err_list.append(train_err)
        test_err_list.append(test_err)

    # 4. train,test error
    error_plot("Tree",train_err_list[:14], test_err_list[:14], "Tree depth", range(2,16),14)
    error_plot("Tree",train_err_list[14:], test_err_list[14:], "Minimum leaf sample", range(2,31),29)



    # 5. best model
    t = DecisionTreeRegressor(min_samples_leaf=4,max_depth=5)
    t.fit(x_train,y_train)
    train_err = 1/(x_train.shape[0]) * sum((t.predict(x_train) - y_train) ** 2)
    test_err = 1/(x_test.shape[0]) * sum((t.predict(x_test) - y_test) ** 2)
    print("Best Regression model:","train error SSE:",train_err, "test error SSE: ",test_err)
    # 5.1 visualize deicision tree
    fig = plt.figure(figsize=(25, 20))
    _ = plot_tree(t,
                      feature_names=feature,
                      class_names=target,
                      filled=True)
    plt.title("Decision Tree Regression")
    plt.show()
    # 5.2 Feature importance
    feature_importances("Tree", t, feature, x_train)
    return train_err,test_err




def Bagging(x_train, y_train, x_test, y_test, feature, target):
    # 1. parameter list
    test_err_list = []
    train_err_list = []
    oob_score_list=[]
    variance_list = []
    bias_list = []


    # 2. train
    n_learner = 5
    base_learner = DecisionTreeRegressor(max_depth=5)
    while n_learner<=200:
        bg = BaggingRegressor(base_estimator=base_learner, n_estimators=n_learner, oob_score=True, max_features=10)
        bg.fit(x_train, y_train)
        train_err = 1/(x_train.shape[0]) * sum((bg.predict(x_train) - y_train) ** 2)
        # variance = 1/(x_train.shape[0])*sum(bg.predict(x_train))
        # bias = 1/(x_test.shape[0])*sum(bg.predict(x_test))
        test_err = 1/(x_test.shape[0]) * sum((bg.predict(x_test) - y_test) ** 2)

        train_err_list.append(train_err)
        test_err_list.append(test_err)
        oob_score_list.append(bg.oob_score_)
        n_learner += 1

    depth = 2
    while depth <= 20:
        base_learner = DecisionTreeRegressor(max_depth=depth)
        bg = BaggingRegressor(base_estimator=base_learner, n_estimators=20, oob_score=True, max_features=10)
        bg.fit(x_train, y_train)
        train_err =  1/(x_train.shape[0]) *sum((bg.predict(x_train) - y_train) ** 2)
        test_err = 1/(x_test.shape[0]) * sum((bg.predict(x_test) - y_test) ** 2)
        train_err_list.append(train_err)
        test_err_list.append(test_err)
        oob_score_list.append(bg.oob_score_)
        depth +=1


    # 3. train test error
    error_plot("Bagging",train_err_list[:196], test_err_list[:196], "number of learner", range(5,201), 196)
    error_plot("Bagging",train_err_list[196:], test_err_list[196:], "base learner depth", range(2,21), 19)
    ## oob score
    oob_plot("Bagging: number of learner", [list(range(5,201)),], [oob_score_list[:196],],["number of learner"])
    oob_plot("Bagging: base learner depth", [list(range(2,21)),], [oob_score_list[196:],],["base learner depth"])

    # bias and variance

    # 4. best model
    base_learner = DecisionTreeRegressor(max_depth=15)
    best_bg = BaggingRegressor(base_estimator=base_learner, n_estimators=161, oob_score=True, max_features=10)
    best_bg.fit(x_train, y_train)
    train_err = 1/(x_train.shape[0]) * sum((best_bg.predict(x_train) - y_train) ** 2)
    test_err = 1/(x_test.shape[0]) * sum((best_bg.predict(x_test) - y_test) ** 2)
    print("Best bagging:", "train error SSE: ",train_err, "test error SSE: ",test_err)

    # feature importance
    feature_importances("Bagging", best_bg, feature, x_train)

    return train_err,test_err



def RandomForest(x_train, y_train, x_test, y_test, feature, target):
    test_err_list = []
    train_err_list = []
    oob_score_list = []
    oob_score_lists = []
    # 1. parameter list
    max_features_list = ['auto', 'sqrt', 'log2']
    m = {'auto': len(feature), 'sqrt': math.sqrt(len(feature)), 'log2': math.log2(len((feature)))}
    # 2. train
    for mtry in max_features_list:
        for n in range(5,201):
            rf = RandomForestRegressor(warm_start=True,n_estimators=n, max_depth=10, oob_score=True, max_features=mtry)
            rf.fit(x_train, y_train)
            train_err = 1/(x_train.shape[0]) * sum((rf.predict(x_train) - y_train) ** 2)
            test_err = 1/(x_test.shape[0]) *sum((rf.predict(x_test) - y_test) ** 2)
            train_err_list.append(train_err)
            test_err_list.append(test_err)
            oob_score_list.append(rf.oob_score_)
        # error SSE plot
        error_plot("Random Forest",train_err_list, test_err_list, "number of leaner (mtry="+str(m[mtry])+")", range(5, 201), 196)
        oob_score_lists.append(oob_score_list)
        # oob score
        test_err_list = []
        train_err_list = []
        oob_score_list = []
    oob_plot("Random Forest: number of learner", [list(range(5, 201))]*3, oob_score_lists, max_features_list)

    # 3. best model
    best_rf = RandomForestRegressor(warm_start=True,n_estimators=193, max_depth=10, oob_score=True, max_features='auto')
    best_rf.fit(x_train, y_train)
    train_err = 1/(x_train.shape[0]) * sum((best_rf.predict(x_train) - y_train) ** 2)
    test_err = 1/(x_test.shape[0]) *sum((best_rf.predict(x_test) - y_test) ** 2)
    print("Best random forest:", "train error SSE: ", train_err, "test error SSE: ",test_err)
    # feature importance
    feature_importances("Random forest", best_rf, feature, x_train)

    return train_err, test_err

def Boosting(x_train, y_train, x_test, y_test, feature, target):
    # AdaBoost 1. parameter list
    learning_rate_list = [0.1, 0.5, 1]

    test_err_list = []
    train_err_list = []
    oob_score_list = []
    oob_score_lists = []

    # 2. train
    for learning_rate in learning_rate_list:
        for n in range(5,201):
            ada = AdaBoostRegressor(learning_rate=learning_rate, n_estimators=n,
                                        loss='square')
            ada.fit(x_train[90:], y_train[90:])
            train_err = 1/(x_train.shape[0]) * sum((ada.predict(x_train) - y_train) ** 2)
            test_err = 1/(x_test.shape[0]) * sum((ada.predict(x_test) - y_test) ** 2)
            train_err_list.append(train_err)
            test_err_list.append(test_err)
            oob_score = ada.score(x_train[:90], y_train[:90])
            oob_score_list.append(oob_score)
        error_plot("Boosting",train_err_list,test_err_list,"number of learner (learning rate="+str(learning_rate)+")",range(5,201),196)
        oob_score_lists.append(oob_score_list)

        test_err_list = []
        train_err_list = []
        oob_score_list = []

    oob_plot("Adaboost: number of learner", [list(range(5, 201))] * 3, oob_score_lists, learning_rate_list)

    # 3. best model
    best_ada = AdaBoostRegressor(learning_rate=1, n_estimators=88,
                                        loss='square')
    best_ada.fit(x_train,y_train)
    train_err = 1/(x_train.shape[0]) * sum((best_ada.predict(x_train) - y_train) ** 2)
    test_err = 1/(x_test.shape[0]) *sum((best_ada.predict(x_test) - y_test) ** 2)
    best_ada.fit(x_train, y_train)

    print("Best adaboost","train error SSE: ", train_err, "test error SSE: ",test_err)
    # feature importance
    feature_importances("Adaboost", best_ada, feature, x_train)
    return train_err,test_err


def dataPreprocessing(data):
    # Explore data

    # distribution
    print(data["Sales"].describe())
    plt.figure(figsize=(9, 8))
    sns.distplot(data['Sales'], color='g', bins=100, hist_kws={'alpha': 0.4})

    # encode categorical data
    data["ShelveLoc"] = data["ShelveLoc"].astype('category')
    data["ShelveLoc"] = data["ShelveLoc"].cat.codes
    data.head()

    data["Urban"] = data["Urban"].astype('category')
    data["Urban"] = data["Urban"].cat.codes
    data.head()

    data["US"] = data["US"].astype('category')
    data["US"] = data["US"].cat.codes
    data.head()

    # split x, y
    d_x = data.iloc[:, 1:]
    d_y = data.iloc[:, 0]

    # feature histogram
    d_x.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

    # data correlation
    data_corr = data.corr()['Sales'][1:]  # -1 because the latest row is SalePrice
    golden_features_list = data_corr[abs(data_corr) > 0.1].sort_values(ascending=False)
    print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list),
                                                                              golden_features_list))

    for i in range(1, len(data.columns), 5):
        sns.pairplot(data=data,
                     x_vars=data.columns[i:i + 5],
                     y_vars=['Sales'])
    return data, d_x, d_y

def dataSplit(d_x,d_y):
    x_train = d_x[:300]
    y_train = d_y[:300]

    x_test = d_x[300:]
    y_test = d_y[300:]
    return x_train, y_train, x_test, y_test

def error_plot(model_name, train_err_list, test_err_list, title, parameter_list, length):
    plt.figure()
    plt.title(model_name+ title+" - train_test error")
    for i in range(length):
        x = parameter_list[i]
        y = train_err_list[i]
        plt.plot(x, y, 'bo',c="blue")
        # plt.text(x * (1 + 0.01), y * (1 + 0.01), int(y), fontsize=12)
        y2 = test_err_list[i]
        plt.plot(x, y2, 'bo', c="orange")
        # plt.text(x * (1 + 0.01), y2 * (1 + 0.01), int(y2), fontsize=12)
    plt.xlabel(title)
    plt.ylabel("SSE")
    plt.show()
    return

def oob_plot(title, parameter_lists, oob_score_lists, labels):
    plt.figure()
    plt.title(title + " - oob score")
    color = ['lightblue','orange','green']
    for i in range(len(parameter_lists)):
        maxindex = np.argmax(oob_score_lists[i])
        x = parameter_lists[i][maxindex]
        y = oob_score_lists[i][maxindex]
        plt.text(x+0.01,y+0.01,x)
        plt.axvline(x,ls="--",c=color[i])
        plt.plot(parameter_lists[i], oob_score_lists[i], label=labels[i])
        plt.legend()
    plt.xlabel(title)
    plt.ylabel("OOB")
    plt.show()

    return


def feature_importances(model_name, model, feature, x_train):
    if (model_name == "Bagging"):
        importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    else:
        importances = model.feature_importances_
    indices = np.argsort(importances)
    std = np.std([importances],axis=0)


    plt.figure()
    plt.title("Feature importances: "+model_name)
    plt.barh(range(len(feature)), importances[indices],
             color="r", xerr=std[indices], align="center")
    plt.yticks(range(x_train.shape[1]), feature[indices])
    plt.ylim([-1, len(feature)])
    plt.show()


def main():
    data = pd.read_csv("Carseats.csv")
    data.info()

    # 1. data preprocessing
    data, d_x, d_y = dataPreprocessing(data)
    feature = np.array(d_x.columns)
    target = d_y.name
    # 2. data split
    x_train, y_train, x_test, y_test = dataSplit(d_x,d_y)


    # 3. model training and fitting
    final_train_err = []
    final_test_err = []

    print("============ Decision Tree ============")
    train_err, test_err = DecisionTree(x_train, y_train, x_test, y_test, feature, target)
    final_train_err.append(train_err)
    final_test_err.append(test_err)
    print("============ Bagging ============")
    train_err, test_err = Bagging(x_train, y_train, x_test, y_test, feature, target)
    final_train_err.append(train_err)
    final_test_err.append(test_err)
    print("============ Random Forest ============")
    train_err, test_err = RandomForest(x_train, y_train, x_test, y_test, feature, target)
    final_train_err.append(train_err)
    final_test_err.append(test_err)
    print("============ Boosting ============")
    train_err, test_err = Boosting(x_train, y_train, x_test, y_test, feature, target)
    final_train_err.append(train_err)
    final_test_err.append(test_err)

    # 4. model comparison
    plt.figure()
    plt.title("Ensemble Learning model")
    model = ["Tree", "Bagging", "Random Forest", "Adaboost"]
    plt.xlabel = "model"
    plt.ylabel = "SSE"
    plt.plot(model, final_train_err, 'bo', c="lightblue")
    plt.plot(model, final_test_err, 'bo', c='orange')
    plt.show()


if __name__ == "__main__":
    main()
