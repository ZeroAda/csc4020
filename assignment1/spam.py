import numpy as np
import matplotlib.pyplot as plt

# initInput
def initInput(filename):
    with open(filename,"rb") as file:
        lines = file.readlines()
        input = [list(map(eval,line.split())) for line in lines]
        file.close()
    input = np.mat(input)
    m, n = input.shape

    # # build up 01 array
    # ones = np.ones(3065,dtype=bool)
    # zeros = np.zeros(1536,dtype=bool)
    # # build up index to split train and test data
    # train_index = np.concatenate((ones, zeros))
    # np.random.shuffle(train_index)
    # test_index = ~train_index
    # with open("index.txt","w") as indexfile:
    #     for i in train_index:
    #         indexfile.write(str(i+0)+'\n')

    # open 01 file to separate data
    with open("index.txt", "r") as handle:
        lines = handle.readlines()
        train_index = []
        for line in lines:
            train_index.append(int(line))
        handle.close()
    train_index0 = []
    for i in train_index:
        if i == 1: train_index0.append(True)
        else: train_index0.append(False)
    train_index0 = np.array(train_index0)
    test_index0 = ~train_index0
    # print(train_index0)

    #slice data with 01
    train_data = input[train_index0]
    test_data = input[test_index0]

    # target array
    train_target = train_data[:,-1:]
    test_target = test_data[:,-1:]

    # reorganize train and test data
    train_mat = np.zeros((3065,n))
    train_mat[:,0] = 1
    train_mat[:,1:] = train_data[:,:-1]

    test_mat = np.zeros((1536,n))
    test_mat[:,0] = 1
    test_mat[:,1:] = test_data[:,:-1]

    # train_mat: 3065*58 matrix, test_mat: 1536*58 matrix
    # train_target: 3065*1 matrix, test_target: 1536* 1 matrix
    return train_mat, test_mat, train_target, test_target


def standardize(data):
    #for each colum, x = (x-mean) / max-min
    for i in range(1,data.shape[1]):
        mean = np.sum(data[:, i])/data.shape[0]
        max = np.max(data[:,i])
        min = np.min(data[:,i])
        data[:,i] = (data[:,i]-mean)/(max-min)
    return data

def logistic(u):
    return 1/(1+np.exp(-u))


def loss(w, h, target, m, lambdax=None):
    # print(target)
    # print(np.log(h))
    target = np.array(target)
    h = np.array(h)
    # print("w*w is ",np.multiply(w,w))
    if lambdax:
        temp = np.sum(target*np.log(h)+(1-target)*np.log(1-h)) - lambdax/m * np.sum(np.multiply(w,w))
    else:
        temp = np.sum(target*np.log(h)+(1-target)*np.log(1-h))
    return (-1/m)*(temp)

def trainLogistic(train_data, train_target, test_data, test_target, m, n, iterations, alpha):
    w = np.ones((58, 1))
    # loss list
    train_losslist = []
    test_losslist = []

    # training
    for k in range(iterations):
        u = train_data * np.mat(w)
        h = logistic(u)
        train_ls = loss(w, h, train_target, m)
        train_losslist.append(train_ls)

        h_test = logistic(test_data * np.mat(w))
        test_ls = loss(w, h_test, test_target, n)
        test_losslist.append(test_ls)

        # gradient is 58*1
        error = h - train_target
        gradient = 1 / m * np.sum(np.multiply(error, train_data), axis=0)
        gradient.reshape(58, 1)

        w = w - np.multiply(alpha, gradient).reshape(58, 1)

    return w, train_losslist, test_losslist

def trainLogisticL2(train_data, train_target, test_data, test_target, m, n, iterations, alpha, lambdax):
    w = np.ones((58, 1))
    train_losslist = []
    test_losslist = []

    for k in range(iterations):
            # print("episode",k)
            # u = w*x
        u = train_data * np.mat(w)
            # print (u)
            # hypothesis function
        h = logistic(u)
            # error is 3065*1
            # error
        error = h - train_target
            # print(h)
            # print(train_target)
            # gradient is 58*1

        train_ls = loss(w, h, train_target, m)
        train_losslist.append(train_ls)

        h_test = logistic(test_data * np.mat(w))
        test_ls = loss(w, h_test, test_target, n)
        test_losslist.append(test_ls)

        w0 = w[0]
        # print(w)
        # print(lambdax * w)
        gradient = 1 / m * (np.sum(np.multiply(error, train_data), axis=0).reshape(58,1) + lambdax * w)
        # print(error)
        # print(train_data[:,0])
        # print(np.sum(np.multiply(error.reshape(1,3065), train_data[:,0])))
        gradient0 = 1 / m * (np.sum(np.multiply(error.reshape(1,m), train_data[:,0])))
        # gradient0.reshape(1, 1)
        # print(gradient0)
        w = w - np.multiply(alpha, gradient).reshape(58, 1)
        # print(w)
        # print(gradient0)
        w[0] = w0 - np.multiply(alpha, gradient0)
        # print(w)


    return w, train_losslist, test_losslist


def classifier(test_data, weight):
    u =test_data * np.mat(weight)
    prob = logistic(u)
    h = np.where(prob>0.5, 1, 0)
    return h

def fit(data, target, weight, m):
    result = classifier(data, weight)
    error_rate = sum(abs(result - target))/m
    return error_rate

def crossValidation(data, target, m, k, iteration, alpha, lambdax):
    weightlist = []
    errorlist = []
    train_lossList = []
    test_lossList = []
    j = m//k
    for i in range(k):
        start = i*j
        end = i*j+j
        test = data[start:end]
        test_target = target[start:end]
        train = np.concatenate((data[:start],data[end:]),axis=0)
        train_target = np.concatenate((target[:start],target[end:]),axis=0)
        weight, train_losslist, test_losslist= trainLogisticL2(train, train_target, test, test_target,
                                                               2452, 613, iteration, alpha, lambdax)
        error = fit(test, test_target, weight, 613)
        errorlist.append(error)
        weightlist.append(weight)
        train_lossList.append(train_losslist)
        test_lossList.append(test_losslist)
    minError = min(errorlist)
    minIndex = errorlist.index(minError)
    minTrainLosslist = train_lossList[minIndex]
    minTestLosslist = test_lossList[minIndex]

    return weightlist[minIndex], minTrainLosslist, minTestLosslist


def weightImportance(weight):
    # importance is weight
    for i in range(len(weight)):
        if weight[i,:] > 1:
            print("the weight ",i,"is ", weight[i])


def lossCurve(iterations,train_losslist, test_losslist):
    plt.scatter(range(iterations), train_losslist, c="blue")
    plt.scatter(range(iterations), test_losslist, c="orange")
    plt.legend(["train loss","test loss"])
    plt.show()


def main():
    # initialize data
    train_mat, test_mat, train_target, test_target = initInput("spam.data")

    # standardize data
    train_data = standardize(train_mat)
    test_data = standardize(test_mat)

    # parameter list
    alphaList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # model1: standard logistic regression model
    print("model1: standard logistic regression model")
    errorList = []
    weightList = []
    train_lossList = []
    test_lossList = []
    alpha_list = []


    for alpha in alphaList:
        weight, train_losslist, test_losslist = trainLogistic(train_data, train_target, test_data, test_target, 3065,
                                                                 1536, 600, alpha)
        error_rate = fit(test_data, test_target, weight, 1536)

        weightList.append(weight)
        errorList.append(error_rate)
        train_lossList.append(train_losslist)
        test_lossList.append(test_losslist)
        alpha_list.append(alpha)

    minError = min(errorList)
    minIndex = errorList.index(minError)
    minTrainLosslist = train_lossList[minIndex]
    minTestLosslist = test_lossList[minIndex]
    minWeight = weightList[minIndex]
    minAlpha = alpha_list[minIndex]


    print("The best standard logistic regression model parameter: ", minWeight.reshape(58,1))
    print("The test error rate is:",minError)
    print("The train error rate is:", fit(train_data, train_target, minWeight, 3065))

    # evaluate model
    lossCurve(600, minTrainLosslist, minTestLosslist)
    weightImportance(minWeight)

    # model2: logistic regression with L2 regularization, cross validation
    print("model2: logistic regression model with L2 regularization")

    errorList = []
    weightList = []
    train_lossList = []
    test_lossList = []
    lambda_list= []

    lambdalist = [0.01, 0.1, 1, 10, 100]
    alphaList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for alpha in alphaList:
        for lambdax in lambdalist:
            weight, trainLoss, testLoss = crossValidation(train_data, train_target, 3065, 5, 600, alpha, lambdax)
            error_rate = fit(test_data, test_target, weight, 1536)
            errorList.append(error_rate)
            weightList.append(weight)
            train_lossList.append(trainLoss)
            test_lossList.append(testLoss)
            lambda_list.append(lambdax)
    minError = min(errorList)
    minIndex = errorList.index(minError)
    minTrainLosslist = train_lossList[minIndex]
    minTestLosslist = test_lossList[minIndex]
    minWeight = weightList[minIndex]
    minLambda = lambda_list[minIndex]


    print("The best logistic regression model with L2 regularization parameter: ", minWeight.reshape(58,1))
    print("The lambda parameter is:", minLambda)
    print("The min test error rate is:", minError)
    print("The min train error rate is: ", fit(train_data, train_target, minWeight,3065))
    lossCurve(600, minTrainLosslist, minTestLosslist)
    weightImportance(minWeight)


if __name__ == "__main__":
    main()

