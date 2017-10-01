from numpy import *

class mlp:
    
    def __init__(self,inputs,targets,nhidden):
        """ Constructor """
        # Set up network size
        self.nin = shape(inputs)[1]#colomn
        self.nout = shape(targets)[1]
        self.ndata = shape(inputs)[0]#row
        self.nhidden = nhidden

        # self.beta = beta
        # self.momentum = momentum
        # self.outtype = outtype
    
        # Initialise network
        self.wi = (random.rand(self.nin+1,self.nhidden)-0.5)*2/sqrt(self.nin)
        self.wo = (random.rand(self.nhidden+1,self.nout)-0.5)*2/sqrt(self.nhidden)

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = concatenate((valid,-ones((shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print count
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*sum((validtargets-validout)**2)
            
        print "Stopped", new_val_error,old_val_error1, old_val_error2
        return new_val_error
    	

            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = dot(inputs,self.wi);
        self.hidden = 1.0/(1.0+exp(-self.hidden))
        self.hidden = concatenate((self.hidden,-ones((shape(inputs)[0],1))),axis=1)

        outputs = dot(self.hidden,self.wo);

        return 1.0/(1.0+exp(-outputs))

        # Different types of output neurons
        # if self.outtype == 'linear':
        # 	return outputs
        # elif self.outtype == 'logistic':
        #     return 1.0/(1.0+exp(-outputs))
        # elif self.outtype == 'softmax':
        #     normalisers = sum(exp(outputs),axis=1)*ones((1,shape(outputs)[0]))
        #     return transpose(transpose(exp(outputs))/normalisers)
        # else:
        #     print "error"

    def sumErrors(self):
        error=0.0
        for p in

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = argmax(outputs,1)
            targets = argmax(targets,1)

        cm = zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = sum(where(outputs==i,1,0)*where(targets==j,1,0))

        print "Confusion matrix is:"
        print cm
        correctRate=trace(cm)/sum(cm)*100
        return  correctRate
        # print "Percentage Correct: ",

    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((self.ndata,1))),axis=1)
        change = range(self.ndata)

        updatew1 = zeros((shape(self.wi)))
        updatew2 = zeros((shape(self.wo)))

        for n in range(niterations):

            self.outputs = self.mlpfwd(inputs)

            error = 0.5*sum((targets-self.outputs)**2)
            if (mod(n,100)==0):
                print "Iteration: ",n, " Error: ",error

            deltao = (targets-self.outputs)/self.ndata
            # Different types of output neurons
            # if self.outtype == 'linear':
            # 	deltao = (targets-self.outputs)/self.ndata
            # elif self.outtype == 'logistic':
            # 	deltao = (targets-self.outputs)*self.outputs*(1.0-self.outputs)
            # elif self.outtype == 'softmax':
            # 	#deltao = (targets-self.outputs)*self.outputs/self.ndata
            #     deltao = (targets-self.outputs)/self.ndata
            # else:
            # 	print "error"

            deltah = self.hidden*(1.0-self.hidden)*(dot(deltao,transpose(self.wo)))

            updatew1 = eta*(dot(transpose(inputs),deltah[:,:-1])) + updatew1
            updatew2 = eta*(dot(transpose(self.hidden),deltao)) + updatew2
            self.wi += updatew1
            self.wo += updatew2

            # Randomise order of inputs
            random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]

