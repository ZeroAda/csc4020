# algorithm
#k means
# soft kmeans
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np


# data
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

# algorithm clustering
# input: feature data, parameters
# output: clustering for each data
# parameter: k (cluster number)
# choose parameter: objective function see
# objective function distances from every point to each centroid
# k clusters, n data point
# centers: centers for each cluster
# assignments: assignment of each point to a cluster index

############################## Algorithm

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


def assign_step(features, centers):
    """
    assign points step
    input: features (n*d), centers (k*d)
    output: assignments (n*1)
    assign each points with its cluster index. the cluster index is the index of the centers closest to each point.
    """
    assignments = []
    for point in features:
        shortest = float("inf") # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            dis = distance(point, centers[i])
            if dis < shortest:
                shortest = dis
                shortest_index = i
        assignments.append(shortest_index)
    # print("Assignment step: ",assignments)
    return assignments

def distance(a,b):
    return np.sum(np.square(a-b))

def cluster_mean(cluster_points):
    """
    compute the cluster mean of given cluster points
    """
    mean = np.mean(np.array(cluster_points),axis=0)
    return mean

def update_step(features, assignments, n, k):
    """
    update_step
    input: features (n*d), assignments (n*1), k (clusters), n (data point)
    output: centers (k*d)
    update centers according to given features (data) and assignments
    """
    centers = []
    # cluster dict collect all the data with same corresponding cluster index
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = []
    for j in range(n):
        assignment = assignments[j]
        data = features[j]
        cluster_dict[assignment].append(data)

    distances = 0
    # new centers is the list of data with mean value in each cluster
    for i in range(k):
        print("for cluster",i,"there is: ",len(cluster_dict[i]))

        mean = cluster_mean(cluster_dict[i])
        for data in (cluster_dict[i]):
            distances += distance(data, mean)

        centers.append(mean)
    return centers, distances

def k_means(features, classes):
    '''
    k-means algorithm
    input: features (Data n*d)
    output: centers (k*d) and assignments (n*d) for each points
    '''
    # parameter_list = range(1,5)
    parameter_list = [3]
    # n data points
    n = features.shape[0]

    for k in parameter_list:
        objective_list = []
        print("=========K = {} ==========".format(k))
        time_start = time.time()
    # initialization: random initialization of cluster center
    # centers: k*d
        centers = initialization(features, k)
        assignments = assign_step(features, centers)
        origin_distance = 0
        for nn in range(n):
            origin_distance += distance(features[nn],centers[assignments[nn]])
        objective_list.append(origin_distance)

        old_assignments = None
        epoch = 0
        while assignments != old_assignments:
            epoch +=1
            print("====epoch ",epoch," ====")
            # if (epoch == 10):
            #     break
            # update step
            new_centers, distances = update_step(features, assignments, n, k)
            objective_list.append(distances)
            # print(new_centers)
            # assign step
            old_assignments = assignments
            assignments = assign_step(features, new_centers)
        plt.plot(range(0,epoch+1),objective_list)
        plt.title("k means obj function")
        plt.show()
        # print(len(objective_list))
        pur = purity(assignments, classes, k, 3, n)
        rand_in = rand_index(assignments, classes, k, 3, n)
        mutual_info = normalized_mutual_information(assignments, classes, k, 3, n)


    time_end = time.time()
    print('k-means time cost: ', time_end - time_start, 's')

    return assignments

##############################################################################

# def accelerated_k_means(features, classes):
#     """
#     accelerated k_means
#     """
#     parameter_list = [1, 2, 3, 4, 5]
#     # n data points
#     n = features.shape[0]
#     time_start = time.time()
#
#     for k in parameter_list:
#         objective_list = []
#         print("=========K = {} ==========".format(k))
#         # initialization: random initialization of cluster center
#         # centers: k*d
#         centers = initialization(features, k)
#         assignments = assign_step(features, centers)
#         origin_distance = 0
#         for nn in range(n):
#             origin_distance += distance(features[nn], centers[assignments[nn]])
#         objective_list.append(origin_distance)
#         epoch = 0
#         sc = []
#         for i in range(k):
#             j = 0
#
#
#         plt.plot(range(0, epoch + 1), objective_list)
#         plt.title(k)
#         plt.show()
#         # print(len(objective_list))
#         pur = purity(assignments, classes, k, 3, n)
#
#     time_end = time.time()
#     print('k-means time cost: ', time_end - time_start, 's')
#
#     return


def soft_assign_step(features, centers, k, n):
    """
    soft assig step
    input: feature, center, n, k
    parameter: beta
    output: responsibility matrix(n*k)
    """
    beta = -1
    responsibility = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            dis = distance(features[i], centers[j])
            responsibility[i,j] = np.exp(beta*dis)
    summation = np.sum(responsibility,axis=1)
    # print("sum",summation)
    for i in range(n):
        for j in range(k):
            responsibility[i,j] = responsibility[i,j]/summation[i]
            # print("Responsibility")
            # print(i,j,responsibility)
    return responsibility

def soft_update_step(features, responsibility, k, n, d):
    """
        soft update step
        input: feature, responsibility, k, n,d
        output: new center
        """
    new_centers = []
    for i in range(k):
        c = responsibility[:,i]
        # d = np.ones((210,7))
        # print(c.shape)
        dd = np.reshape(np.repeat(c, d),(n,d))
        # soft feature
        # print(features.shape,d.shape)
        soft_feature = np.ones((n,d))
        for j in range(n):
            soft_feature[j,:] = features[j] * c[j]
        # print("s",soft_feature.shape,soft_feature)
        # soft_feature = np.ones((210,7)) * np.ones((210,7))
        # soft_features = np.mat(1,(210,70)) *  np.mat(2,(210,70))
        # soft_Sum d dimension
        soft_sum = np.sum(soft_feature,axis=0)
        responsibility_sum = np.sum(c)
        new_center = soft_sum/responsibility_sum
        # print("new center: ",new_center)
        new_centers.append(new_center)
    return new_centers

def soft_k_means(features, classes):
    """
    soft k means:
    1. initialize centers
    2. assignment step: for each point, assign responsibility to it
    3. update step: for each center, update center
    4. repeat 2,3 until center not change
    """
    # parameter_list = range(1,5)
    parameter_list = [3]
    # n data points
    n = features.shape[0]
    # d dimension
    d = features.shape[1]

    time_start = time.time()

    for k in parameter_list:
        objective_list = []
        print("=========K = {} ==========".format(k))
        time_start = time.time()
        # initialization: random initialization of cluster center
        # centers: k*d
        centers = initialization(features, k)
        responsibility = soft_assign_step(features, centers,k,n)
        assignments = np.argmax(responsibility,axis=1)
        origin_distance = 0
        for nn in range(n):
            origin_distance += distance(features[nn], centers[assignments[nn]])
        objective_list.append(origin_distance)
        new_centers = None
        new_assignments = np.zeros(210)

        epoch = 0
        while epoch < 10:
            epoch += 1
            print("====epoch ", epoch, " ====")
            # if (epoch == 10):
            #     break
            # update step
            centers = new_centers
            new_centers = soft_update_step(features,responsibility,k,n,d)
            # assign step
            assignments = new_assignments
            responsibility = soft_assign_step(features, new_centers, k,n)

            new_assignments = np.argmax(responsibility, axis=1)
            # print(assignments,new_assignments)
        print(epoch)



        # assignment
        assignments = np.argmax(responsibility,axis=1)
        # print("assignment",assignments)
        # center
        #
        # plt.plot(range(0, epoch + 1), objective_list)
        # plt.title("soft k means obj function")
        # plt.show()
        # print(len(objective_list))
        pur = purity(assignments, classes, k, 3, n)
        rand_in = rand_index(assignments, classes, k, 3, n)
        mutual_info = normalized_mutual_information(assignments, classes, k, 3, n)

        time_end = time.time()
        print('soft-k-means time cost: ', time_end - time_start, 's')
        return assignments

########################################################

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

# ########################################################3Metrics
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

# def max_likelihood():
#
#     return mle


def main():
    # initialize data
    features, classes, d = initialData("seeds_dataset.txt")

    # clustering
    # k_means
    assignments = k_means(features, classes)
    #
    # # soft k_means
    assignments = soft_k_means(features, classes)

    # accelerated k_means

    # EM algorithm
    EM(features, classes, 210, d)

    return

if __name__ == "__main__":
    main()


# problems
# 1.initial -- sensiyivity
