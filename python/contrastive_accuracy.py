import caffe
import numpy as np

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


        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        f=open("jobu.txt","w")
        f.write(str(bottom[0].num)+" "+str(np.shape(bottom[0].data)))
        
	errors=0
        correct=0
	for i in range(bottom[0].num):
		distance = np.sqrt(np.sum(self.diff[i]**2))
                f.write(str(distance)+"\n")
		if distance < 0.8 and not bottom[2].data[i]==bottom[3].data[i]:
			errors+=1
		elif distance > 0.8 and bottom[2].data[i]==bottom[3].data[i]:
			errors+=1
		else:
			correct+=1
        f.close()
	top[0].data[...] = 1.0*errors/(errors+correct)

