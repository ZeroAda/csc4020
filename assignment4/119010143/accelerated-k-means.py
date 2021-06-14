import numpy as np
import matplotlib.pyplot as plt
import random

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
    return np.array(centers)


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

def distance(a,b):
    return np.sum(np.square(a-b))


def accelerated_k_means(features, classes, n, k, d):
    centers = initialization(features, k=3)
    # set initial lower bound
    l_mat = np.zeros((n,k))
    u_list = np.zeros(n)
    d_mat = np.zeros((n,k))

    # assignment
    assignments = []
    for j in range(n):
        point = features[j]
        shortestindex = 0
        dis = distance(point, centers[0])
        l_mat[j,0] = dis
        for i in range(1,len(centers)):
            dis2 = distance(centers[shortestindex], centers[i])
            if 2* dis > dis2:
                dis = distance(point, centers[i])
                l_mat[j,i] = dis
                shortestindex = i
        # assign upper bound
        u_list[j] = dis
        # print(u_list)
        assignments.append(shortestindex)

    r_list = [False]*n

    for epoch in range(10):
        s_list = np.zeros(k)


        for j in range(len(centers)):

            shortestdist = float("inf")
            for q in range(len(centers)):
                d = distance(centers[q], centers[j])
                if j != q and shortestdist < d:
                    shortestdist = d
            s_list[j] = 1/2 * shortestdist

        # all points smaller
        for i in range(n):
            # 3a
            if u_list[i] <= s_list[assignments[i]]:
                continue
            if r_list[i] == True:
                r_list[i] = False
                d_mat[i,assignments[i]] = distance(features[i], assignments[i])
            else:
                d_mat[i,assignments[i]] = u_list[i]
            # 3b
            for j in range(len(centers)):
                if assignments[i] == j or u_list[i] <= l_mat[i,j] or u_list[i] <= 1/2*distance(assignments[i], centers[j]):
                    continue
                if d_mat[i, assignments[i]] > l_mat[i,j] or d_mat[i, assignments[i]] > 1/2*distance(centers[j], assignments[i]):
                    dd = distance(features[i], centers[j])
                    if dd < d_mat[i, assignments[i]]:
                        assignments[i] = j
        # print(assignments)

    # 4
        cluster_dict = {}
        m_list = list()
        for i in range(k):
            cluster_dict[i] = []
        for j in range(n):
            assignment = assignments[j]
            data = features[j]
            cluster_dict[assignment].append(data)

        for i in range(k):
            mean = np.mean(np.array(cluster_dict[i]),axis=0)
            m_list.append(mean)
        m_list = np.array(m_list)
        for i in range(n):
            for j in range(k):
                l_mat[i,j] = np.max(l_mat[i,j]-distance(centers[j],m_list[j]),0)
            u_list[i] = u_list[i] + distance(m_list[assignments[i]],assignments[i])
            r_list[i] = True

        centers = m_list
        print(assignments)


    return assignments




def main():
    # initialize data
    features, classes, d = initialData("seeds_dataset.txt")
    (n, d) = features.shape


    # EM algorithm
    k = 3
    assignments = accelerated_k_means(features, classes, n, k, d)
    print(assignments)


    return

if __name__ == "__main__":
    main()