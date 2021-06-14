import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time


def initialData(filename):
    with open(filename,"rb") as file:
        lines = file.readlines()
        input = [list(map(eval,line.split())) for line in lines]
        file.close()
    input = np.mat(input)
    m, n = input.shape
    feature = input[:,:-1]
    classes = input[:,-1]
    # dimension
    d = n - 1
    classes = np.reshape(classes,(1,m))
    reshapeclasses = np.zeros(m)
    for i in range(m):
        reshapeclasses[i] = classes[:,i]
    return feature, reshapeclasses, d

def initialization(features, k):
    """
    initialize cluster centers
    input: features(Data, n*d), k
    output: a list of k points with d dimensions; each point is randomly assigned with the value between min and max
    for each dimension
    """
    centers = []
    dimensions = features.shape[1]
    # print(dimensions)
    for i in range(k):
        rand_point = []
        for j in range(dimensions):
            # min_val = min_max['min_%d' % i]
            # max_val = min_max['max_%d' % i]
            #
            # rand_point.append(random.uniform(min_val, max_val))
            min_val = np.min(features[:,j])
            max_val = np.max(features[:,j])
            rand_point.append(random.uniform(min_val,max_val))
        centers.append(rand_point)
    # print(centers)
    return centers
def EMinitialization(features, k, d):
    # randomize mean list
    mean_list = np.mat(initialization(features, k))
    # covariance matrix should be positive semidefinite
    cov_list = []
    for i in range(k):
        m = np.random.rand(d,d)
        cov_list.append(np.mat(np.dot(m, m.T)))
    cov_list = cov_list
    # randomize class list
    mix_coefficient = np.repeat([1/k],k)

    return mean_list, cov_list, mix_coefficient

def Gaussian(x, mean, cov, d):
    inv = np.linalg.inv(cov)
    # print(inv)
    det = np.linalg.det(cov)
    # print(det)
    media0 = x - mean
    media1 = 1 / pow(math.pi, d / 2) * pow(det, -1 / 2)
    exponent = -1 / 2 * np.matmul(np.matmul(media0, inv), media0.T)
    media2 = math.exp(exponent)
    p = media1 * media2
    return p

def E_step(features, mean_list, cov_list, mix_coef, k, n, d):
    response = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            # print(j)
            response[i,j] = mix_coef[j] * Gaussian(features[i], mean_list[j], cov_list[j], d)
    sum_list = np.sum(response,axis=1)
    for i in range(n):
        response[i] = response[i]/sum_list[i]

    return response

def M_step(features, response, k, n, d):
    n_k = np.sum(response,axis=0)
    mix_coef = n_k/n
    # print(mix_coef)
    mean_list = list()
    for j in range(k):
        mean = np.zeros((1,d))
        for i in range(n):
            mean += response[i,j] * features[i]
        mean_list.append(mean/n_k[j])
    cov_list = list()
    for j in range(k):
        cov = np.zeros((d,d))
        for i in range(n):
            mul = features[i] - mean_list[j]
            # print(mul)
            mult = np.zeros((d,d))
            for q in range(d):
                for p in range(d):
                    mult[q,p] = mul[0,q] * mul[0,p]
            cov += response[i,j] * mult
        cov_list.append(cov/n_k[j])
    # return mean_list, cov_list, mix_coef
    return mean_list, cov_list, mix_coef

def EM(features, classes, n, d):
    k = 3
    # initialize
    # cov_list a list of covairiance matrix (k) (d*d)
    # mean_list a list of mean (k) (d dimension)
    # mix_coef a list of probability (k) sum is 1
    mean_list, cov_list, mix_coef = EMinitialization(features, k, d)
    # # print(mean_list)
    # print(cov_list)
    # print(mix_coef)
    pur_list = list()
    rand_list = list()
    mutual_info_list = list()
    time_start = time.time()
    for epoch in range(9):
        response = E_step(features, mean_list, cov_list, mix_coef, k, n, d)
        mean_list, cov_list, mix_coef = M_step(features, response, k, n, d)

        assignments = np.argmax(response, axis=1)
        pur = purity(assignments, classes, k, 3, n)
        pur_list.append(pur)
        rand_in = rand_index(assignments, classes, k, 3, n)
        rand_list.append(rand_in)
        mutual_info = normalized_mutual_information(assignments, classes, k, 3, n)
        mutual_info_list.append(mutual_info)

    plt.plot(range(0, epoch + 1), pur_list)
    plt.title("purity list")
    plt.show()

    plt.plot(range(0, epoch + 1), rand_list)
    plt.title("rand index list")
    plt.show()

    plt.plot(range(0, epoch + 1), mutual_info_list)
    plt.title("mutual information list")
    plt.show()

    time_end = time.time()
    print('EM ', time_end - time_start, 's')

    return assignments

# Metrics
def purity(assignments, classes, k, c, n):
    """
    purity metric
    input: assignments (n*1) , classes (n*1), k clusters, c classes, n data points
    output: purity number
    """
    # find maximum part of the classes in each clusters
    cluster_class = {}
    correctness = 0

    for i in range(k):
        cluster_class[i] = []
    for j in range(n):
        cluster_class[assignments[j]].append(classes[j])

    print("Each cluster has the following classes:\n")
    for p in range(k):
        print("Cluster",p)
        print(cluster_class[p])

        class_number_dict = {}
        for class_index in cluster_class[p]:
            class_number_dict[class_index] = class_number_dict.get(class_index, 0)+1
        max_amount = 0
        max_class = 0
        for (class_index, class_number) in class_number_dict.items():
            if class_number > max_amount:
                max_class = class_index
                max_amount = class_number
        print("For cluster {}, the maximum class {} has {} data points\n".format(p, max_class, max_amount))
        correctness += max_amount
    pur = (1/n) * correctness
    print("Purity: ", pur)
    return pur


def bionomial(a,b):
    if (b>a):
        return 0
    if b == 0 or a == b:
        return 1
    return bionomial(a-1, b-1) + bionomial(a-1, b)

def rand_index(assignments, classes, k, c, n):
    """
    rand_index = (TP + TN) / (TP + TN + FP + FN)
    input: assignments, classes, k, n
    output: randindex
    """
    index_matrix = np.zeros((k,c))
    # print(assignments[1],classes[1])
    for j in range(n):
        classin = int(classes[j]-1)
        # print(assignments[j])
        # print(classin)
        # print(index_matrix[1,1])
        index_matrix[assignments[j],classin] += 1
    # TP + FP
    TP = 0
    for m in range(k):
        for mm in range(c):
            TP += bionomial(index_matrix[m][mm],2)
    TP_FP = 0
    for p in np.sum(index_matrix,axis=1):
        TP_FP += bionomial(p,2)
    # TP + FN
    TP_FN = 0
    for q in np.sum(index_matrix, axis=0):
        TP_FN += bionomial(q, 2)
    # whole
    whole = bionomial(n,2)
    # TN + FN
    TN_FN = whole - TP_FP
    TN = TP + TN_FN - TP_FN
    rand_in = (TP+TN)/whole
    print("Rand index: ",rand_in)
    return rand_in


def mutual_information(assignments, classes, k, c, n):
    I_a_b = 0
    cluster_class = {}
    class_number = {}
    cluster_number = {}
    for i in range(k):
        cluster_class[i] = {}
    # for (a,b) in cluster_class.items():
    #     print(a,b)
    # I_a_b = 1
    for j in range(n):
        classin = int(classes[j])
        # print(cluster_class[assignments[j]].get(1,0))
        cluster_class[assignments[j]][classin] = cluster_class[assignments[j]].get(classes[j],0) + 1
        class_number[classin] = class_number.get(classin, 0) + 1
        cluster_number[assignments[j]] = cluster_number.get(assignments[j],0) + 1
    # print(class_number)
    # print("---")
    # print(cluster_number)
    # print("---")
    # print(cluster_class)
    for (cluster, class_set) in cluster_class.items():
        for (class_index, class_number_cluster) in class_set.items():
            I_a_b += class_number_cluster * np.log((n*class_number_cluster)/(cluster_number[cluster]*class_number[class_index]))
    I_a_b = 1/n * I_a_b
    return I_a_b

def entropy(a, N):
    H_a = 0
    a_number_dict = {}
    for i in range(N):
        a_number_dict[a[i]] = a_number_dict.get(a[i],0)+1
    for (a_index, a_number) in a_number_dict.items():
        H_a += -1 * (a_number/N) * np.log(a_number/N)
    return H_a

def normalized_mutual_information(assignments, classes, k, c, n):
    """
    normalized mutual information
    """
    I_k_c = mutual_information(assignments, classes, k, c, n)
    H_k = entropy(assignments, n)
    H_c = entropy(classes, n)
    NMI = I_k_c / ((H_k + H_c)/2)
    print("NMI: ",NMI)
    return NMI



def main():
    # initialize data
    features, classes, d = initialData("seeds_dataset.txt")
    (n, d) = features.shape


    # EM algorithm
    assignments = EM(features, classes, n, d)


    return

if __name__ == "__main__":
    main()