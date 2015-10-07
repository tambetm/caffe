import caffe
import numpy as np
import sklearn.preprocessing

class contrastive_accuracy(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need four inputs (2 vecs 2 labels) to compute contrastive distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Features must have the same dimension.")

        # check input dimensions match
        if bottom[2].count != bottom[3].count:
            raise Exception("Labels must have the same dimension.")

        # check input dimensions match
        if bottom[2].count == 1 :
            raise Exception("Labels must have one dimension.")

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #print "bottom[0].shape: ", bottom[0].data.shape
        #print "bottom[1].shape: ", bottom[1].data.shape
        #print "bottom[2].shape: ", bottom[2].data.shape, bottom[2].data[:10]
        #print "bottom[3].shape: ", bottom[3].data.shape, bottom[3].data[:10]

        data0 = sklearn.preprocessing.normalize(bottom[0].data)
        data1 = sklearn.preprocessing.normalize(bottom[1].data)

        distances = np.linalg.norm(data0 - data1, axis=1)
        #print "distances.shape: ", distances.shape, distances[:10]

        predictions = (distances < 1)
        #print "predictions.shape: ", predictions.shape, predictions[:10]

        labels = (bottom[2].data == bottom[3].data)
        #print "labels.shape: ", labels.shape, labels[:10]

        correct = (predictions == labels)
        #print "correct.shape: ", correct.shape, correct[:10]

        accuracy = np.mean(correct)
        #print "accuracy:", accuracy.shape, accuracy

        top[0].data[...] = accuracy
