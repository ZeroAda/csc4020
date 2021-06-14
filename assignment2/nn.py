import numpy as np
import sys
from time import *
from math import sqrt,log
import random
import matplotlib.pyplot as plt




class FullyConnected:
    def __init__(self, input_num, output_num, learning_rate):
        self.output_num = output_num
        self.input_num = input_num
        self.lr = learning_rate
        # weight matrix output*input
        self.weight = 0.01*np.random.rand(self.output_num, input_num)
        # bias term: 1 * output_num
        # self.bias = np.random.rand(1, self.output_num)
        self.bias = np.zeros([1,self.output_num])


    def forward(self, inputs):
        # input: 1*m data
        # output: 1*n data
        self.inputs = inputs

        outputs = self.weight.dot(inputs.T).T + self.bias
        self.outputs = outputs
        return outputs

    # dy: 1*n, inputs: 1*m
    def backward(self, dy):
        # print("Fully backward")
        # dw : n*m matrix!
        dw = (dy.T).dot(self.inputs)
        # db: 1*n
        db = dy
        # dx: weight.T * dy
        dx = (self.weight).T.dot(dy.T)
        # weight: n*m matrix
        self.weight = self.weight - dw * self.lr
        # bias: 1*n
        self.bias = self.bias - db * self.lr
        return dx

    def extract(self):
        print("Layer Fully Connected \n intputs: ", self.inputs, "\noutputs: ", self.outputs)


class SoftMax:
    def __int__(self):
        pass

    # input: 1*m data
    # output: 1*n data
    def forward(self, inputs):
        self.inputs = inputs
        normal = np.sum(np.exp(inputs))
        outputs = (np.exp(inputs) / normal)
        self.outputs = outputs
        return outputs

    # input: label dy 1*3
    # output: dError/du
    def backward(self, dy):
        # print("Softmaxbackward")
        # return  1*3
        dx = self.outputs - dy
        return dx

    def extract(self):
        print("Layer Softmax \n intputs: ", self.inputs, "\noutputs: ", self.outputs)
        return


class Relu:
    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        outputs = inputs.copy()
        outputs[outputs < 0] = 0
        self.outputs = outputs
        return outputs

    # dy
    def backward(self, dy):
        # print("Relubackward")
        dx = dy.copy()
        # dx = relugradient * dy
        # relugradient = 0(if x<0), 1(if x >0)
        dx[self.inputs < 0] = 0
        return dx

    def extract(self):
        print("Layer Relu \n intputs: ", self.inputs, "\noutputs: ", self.outputs)
        return


# verify ok
class DataLoader:
    def __init__(self, filename):
        f = open(filename, 'r')
        f.readline()
        # 3
        num_label = len(f.readline()[2:].split())
        # 256
        num_attr = int(f.readline()[2:])
        # 16
        dim_size = int(sqrt(num_attr))
        # 369 train 105 test

        lines = f.readlines()
        self.data_size = len(lines)
        f.close()
        input = [list(map(eval, line.split())) for line in lines]
        oridata = np.mat(input)
        print("Data length: ", self.data_size)

        # data: (datasize * attribute + 1 ) matrix
        # label: (datasize * num_label) matrix
        data = np.zeros([self.data_size, num_attr])
        data[:, :] = oridata[:, :-num_label]

        print("Data is : ", data.shape[0], "*", data.shape[1], "matrix")

        label = oridata[:, num_attr:]
        print("label is length: ", label.shape[0], "*", label.shape[1], "matrix")

        self.data = data
        self.label = label

        # print("Data 1:",data[1,:]," Label: ",label[1])


# label: 1*n
# h: 1*n
# verify ok
def lossf(label, h):
    for i in range(3):
        if label[:, i] != 0:
            mul = h[:, i]
            res = -1 * label[:, i] * log(mul)
            return res


class NN:
    # initialize hyperparameter for nn
    def __init__(self, hidden_node, input_node, output_node):
        self.layers = []
        self.output_node = output_node
        self.input_node = input_node
        self.output_node = output_node
        self.layer_num = 0

    def appendLayer(self, layer):
        self.layers.append(layer)
        self.layer_num += 1

    def randomize_mini_batch(self, batch_size, train_data, train_label, seed):
        # 1. shuffle data
        np.random.seed(seed)
        m = train_data.shape[0]
        permutation = np.random.permutation(m)
        shuffle_data = train_data[permutation, :]
        shuffle_label = train_label[permutation, :]
        # 2. partition data
        mini_batches = []
        batch_num = train_data.shape[0] // batch_size
        for i in range(batch_num):
            data = shuffle_data[i * batch_size:(i + 1) * batch_size, :]
            label = shuffle_label[i * batch_size:(i + 1) * batch_size, :]
            mini_batches.append((data, label))
        if (train_data.shape[0] % batch_size != 0):
            data = shuffle_data[batch_num * batch_size:m, :]
            label = shuffle_label[batch_num * batch_size:m, :]
            mini_batches.append((data, label))
        return mini_batches

    def train(self, batch_size, epoch, o_train_data, o_train_label, test_data, test_label):
        # train data: 369*256
        # train label: 1*256
        total_train_acc = 0
        train_data = o_train_data[:296]
        train_label = o_train_label[:296]
        val_data = o_train_data[296:]
        val_label = o_train_label[286:]
        m = train_data.shape[0]
        n = test_data.shape[0]
        epoch_loss_list = []
        epoch_acc_list = []
        test_loss_list = []
        test_acc_list = []
        val_loss_list = []
        val_acc_list = []
        seed = 0
        for ep in range(epoch):
            ep_loss = 0
            ep_acc = 0
            ep_class_acc = [0,0,0]
            ep_class_total = [0,0,0]
            start_time = time()
            # Minibatch gradient descent
            seed += 1
            # if batch_size = 1, stochastic gradient descent
            # layers are designed for stochastic gradient descent
            mini_batches = self.randomize_mini_batch(batch_size, train_data, train_label, seed)

            for mini_batch in mini_batches:
                # for one batch
                batch_data = mini_batch[0]
                batch_label = mini_batch[1]
                # forward prop
                u = batch_data
                for j in range(0, self.layer_num):
                    l = self.layers[j]
                    h = l.forward(u)
                    # l.extract()
                    u = h
                ep_loss += lossf(batch_label, h)
                # for gradient descent
                ep_class_total[int(np.argmax(batch_label))] += 1
                if (np.argmax(batch_label) == np.argmax(h)):
                    ep_class_acc[int(np.argmax(batch_label))] +=1
                    ep_acc += 1


                # backward prop
                dy = batch_label
                for j in range(self.layer_num - 1, 0, -1):
                    l = self.layers[j]
                    dx = l.backward(dy)
                    # l.extract()
                    dy = dx

            # time
            end_time = time()
            ep_time = end_time - start_time
            # validate
            val_loss, val_acc, val_class_acc, val_class_total = self.test(val_data,val_label)
            # test
            test_loss, test_acc, test_class_acc, test_class_total = self.test(test_data, test_label)
            # result
            ep_loss /= train_data.shape[0]
            epoch_loss_list.append(ep_loss)
            epoch_acc_list.append(ep_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            class6_acc = ep_class_acc[0]/ep_class_total[0]
            class7_acc = ep_class_acc[1]/ep_class_total[1]
            class8_acc = ep_class_acc[2]/ep_class_total[2]

            test_class6_acc = test_class_acc[0]/test_class_total[0]
            test_class7_acc = test_class_acc[1]/test_class_total[1]
            test_class8_acc = test_class_acc[2]/test_class_total[2]

            print("=====================Epoch {}====================".format(ep))
            print("Train: Loss = {}, Accuracy = {} / {}".format(ep_loss, ep_acc, m))
            print("Train: class 6: {} class 7: {} class 8: {}".format(class6_acc, class7_acc, class8_acc))
            print("Val: Loss = {}, Accuracy = {} / {}".format(val_loss, val_acc, 369-m))
            print("Test: Loss = {}, Accuracy = {} / {}".format(test_loss, test_acc, n))
            print("Test: class 6: {} class 7: {} class 8: {}".format(test_class6_acc, test_class7_acc, test_class8_acc))

        plt.scatter(range(epoch), epoch_loss_list, c="lightblue")
        plt.scatter(range(epoch), val_loss_list, c="purple")
        plt.legend(["train loss","validation loss"])
        plt.show()

        plt.scatter(range(epoch), epoch_acc_list, c="orange")
        plt.axhline(y=m, ls=":", c="orange")
        plt.scatter(range(epoch), val_acc_list, c="goldenrod")
        plt.axhline(y=369-m, ls=":", c="goldenrod")
        plt.legend(["train acc","validation acc"])
        plt.show()

        plt.scatter(range(epoch), test_loss_list, c="purple")
        plt.legend(["test loss"])
        plt.show()

        plt.scatter(range(epoch), test_acc_list, c="goldenrod")
        plt.axhline(y=n, ls=":", c="goldenrod")
        plt.legend(["test acc"])
        plt.show()



    def test(self, test_data, test_label):
        m = test_data.shape[0]
        loss = 0
        acc = 0
        test_class_acc = [0, 0, 0]
        test_class_total = [0, 0, 0]

        for i in range(m):
            u = test_data[i]
            for j in range(0, self.layer_num):
                l = self.layers[j]
                h = l.forward(u)
                # l.extract()
                u = h
            loss += lossf(test_label[i], h)
                # for gradient descent
            test_class_total[int(np.argmax(test_label[i]))] += 1
            if (np.argmax(test_label[i]) == np.argmax(h)):
                test_class_acc[int(np.argmax(test_label[i]))] += 1
                acc += 1

            # result
        loss /= test_data.shape[0]
        return loss, acc, test_class_acc, test_class_total


def main():
    # load data
    train_data_loader = DataLoader("train1.txt")
    test_data_loader = DataLoader("test1.txt")

    # nn class
    # parameter list
    # batch_size_list = range(2, 256, 4)
    learn_rate_list = [x / 100 for x in range(1, 150, 20)]
    hidden_node_list = [173, 500]
    # experiment
    nn = NN(173, 256, 3)
    nn.appendLayer(FullyConnected(256, 173, 0.01))
    nn.appendLayer(Relu())
    nn.appendLayer(FullyConnected(173, 3, 0.01))
    nn.appendLayer(SoftMax())
    nn.train(1, 1000, train_data_loader.data, train_data_loader.label, test_data_loader.data, test_data_loader.label)

    return 0


if __name__ == "__main__":
    main()
