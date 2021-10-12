# library
import math

# Lidstone Smoothing
class Lidstone_smoothing:
    def __init__(self, dev_data, k):

        self.dev_data = dev_data

        # log10 probability of each 3 words phase
        self.hash = {}

        # para, k=1 is Laplacian Smoothing
        self.k = k

        # len of dev_data
        self.len = len(self.dev_data)

    def smoothing(self, hash2):
        for i in range(2, self.len):
            # the previous two words
            prefix = self.dev_data[i-2] + ' ' + self.dev_data[i-1]
            # 3 words phase
            tmp = self.dev_data[i-2] + ' ' + self.dev_data[i-1] + ' '  + self.dev_data[i]
            # prefix not in hash2
            if (prefix not in hash2):
                pass
            elif (tmp not in hash2[prefix]):
                hash2[prefix][tmp] = 0

        for key2 in hash2:
            # number of str start in prefix "key2"
            v = len(hash2[key2].keys())
            m = sum(hash2[key2].values())
            for key3 in hash2[key2]:
                # Lidstone Smoothing
                self.hash[key3] = math.log10((hash2[key2][key3] + self.k) / (m + v*self.k))

    def calculate_PPL(self):
        # calculate the PPL
        ppl = 0

        # number of 3 words phase
        cnt = 0

        for i in range(2, self.len):
            # 3 words phase
            tmp = self.dev_data[i-2] + ' ' + self.dev_data[i-1] + ' '  + self.dev_data[i]
            if (tmp not in self.hash):
                continue
            ppl += self.hash[tmp]
            cnt += 1
    
        ppl *= -1
        ppl /= cnt
        ppl = 10**ppl

        return ppl

# Jelinek-Mercer Smoothing
class Interpolation_smoothing:
    def __init__(self, dev_data, test_data, lambda_1, lambda_2):
        self.dev_data = dev_data
        self.test_data = test_data

        # para lambda, learn from dev_set
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # len of dev_data
        self.len = len(self.dev_data)

    # maxiter, stepSize and tol represent maxDepth of iteration, update step size and stop flag respectively
    def train_paras(self, hash1, hash2, hash3, dev_data, maxiter, stepSize, tol):
    
        # gradient decent for lambda_2
        def gd2(lambda_1, lambda_2):
            res = 0

            # number of derivative
            cnt = 0

            for i in range(2, len(dev_data)):
                word3 = dev_data[i-2] + ' ' + dev_data[i-1] + ' '  + dev_data[i]
                word2 = dev_data[i-1] + ' '  + dev_data[i]
                word1 = dev_data[i]

                # probability of different length phase
                p3 = hash3[word3][1] if word3 in hash3 else 0
                p2 = hash2[word2][1] if word2 in hash2 else 0
                p1 = hash1[word1] if word1 in hash1 else 1          

                # sum the derivative of each phase
                res += (p3-lambda_1*p2-(1-lambda_1)*p1)/(lambda_2*p3 + lambda_1*(1-lambda_2)*p2 + (1-lambda_1)*(1-lambda_2)*p1)
                cnt += 1

            # get the avg of PPL derivative
            res /= -cnt

            return res

        # gradient decent for lambda_1
        def gd1(lambda_1, lambda_2):
            res = 0

            # number of derivative
            cnt = 0

            for i in range(2, len(dev_data)):
                word3 = dev_data[i-2] + ' ' + dev_data[i-1] + ' '  + dev_data[i]
                word2 = dev_data[i-1] + ' '  + dev_data[i]
                word1 = dev_data[i]

                # probability of different length phase
                p3 = hash3[word3][1] if word3 in hash3 else 0
                p2 = hash2[word2][1] if word2 in hash2 else 0
                p1 = hash1[word1] if word1 in hash1 else 1           

                # sum the derivative of each phase
                res += ((1-lambda_2)*p2-(1-lambda_2)*p1)/(lambda_2*p3 + lambda_1*(1-lambda_2)*p2 + (1-lambda_1)*(1-lambda_2)*p1)
                cnt += 1

            # get the avg of PPL derivative
            res /= -cnt

            return res


        # fst train para lambda_1 and lambda2 on the dev_set

        # EM algorithm - fst fix one var and update another, then fix another var and update the previous one
        while (gd2(self.lambda_1, self.lambda_2)**2 >= tol or gd1(self.lambda_1, self.lambda_2)**2 >= tol):
            # update lambda_2
            cnt = 0
            gd = gd2(self.lambda_1, self.lambda_2)
            while (cnt <= maxiter and gd**2 >= tol):
                # update lambda_2 based on the gradient
                self.lambda_2 -= gd*stepSize

                gd = gd2(self.lambda_1, self.lambda_2)
                cnt += 1

            # update lambda_1
            cnt = 0
            gd = gd1(self.lambda_1, self.lambda_2)
            while (cnt <= maxiter and gd**2 >= tol):
                # update lambda_1 based on the gradient
                self.lambda_1 -= gd*stepSize

                gd = gd1(self.lambda_1, self.lambda_2)
                cnt += 1


        print("--------------------------------------")
        print("lambda_1: {}".format(self.lambda_1))
        print("lambda_2: {}".format(self.lambda_2))


    def calculate_PPL(self, hash1, hash2, hash3):
        # calculate the PPL
        ppl = 0

        for i in range(2,self.len):
            word3 = self.test_data[i-2] + ' ' + self.test_data[i-1] + ' '  + self.test_data[i]
            word2 = self.test_data[i-1] + ' '  + self.test_data[i]
            word1 = self.test_data[i]

            # probability of different length phase
            p3 = hash3[word3][1] if word3 in hash3 else 0
            p2 = hash2[word2][1] if word2 in hash2 else 0
            p1 = hash1[word1] if word1 in hash1 else 1         

            # interpolation
            # lambda_1 and lambda_2 are learned from dev_set
            ppl += math.log10(self.lambda_2*p3 + self.lambda_1*(1-self.lambda_2)*p2 + (1-self.lambda_1)*(1-self.lambda_2)*p1)

        ppl *= -1
        ppl /= (self.len-2)
        ppl = 10**ppl
        
        return ppl
    
# Good Turing Discounting
class discounting:
    def __init__(self, hash2, test2, count):
        # train_set
        self.hash2 = hash2

        # test_set
        self.test2 = test2

        # define the number of low frequency word
        self.numlow = count

        # record probability of 3 word phase
        self.hash3 = {}

    def discounting(self):
        for key2 in hash2:
            if (key2 not in self.test2):
                continue

            # find the maximum number of occurrences in train_set
            values = list(self.hash2[key2].values())
            m = max(values)

            # vocabulary size
            v = sum(self.hash2[key2].values())

            # probability array
            p = [0 for _ in range(m+1)]

            # record the number of words of the same occurrences
            cnt = [0 for _ in range(m+1)]

            # record the number of words of the same occurrences
            for word3 in self.test2[key2]:
                if (word3 not in self.hash2[key2]):
                    cnt[0] += 1
                else:
                    cnt[self.hash2[key2][word3]] += 1

            # Good Turing Discounting for lower frequency word
            n = min(m, self.numlow)
            for i in range(n):
                if (cnt[i] == 0):
                    p[i] = 0
                else:
                    j = i+1
                    while (j < n-1 and cnt[j]==0):
                        j += 1
                    p[i] = (j)*cnt[j]/cnt[i] / v

            # the higher item retains the original probability
            if (n < m+1): 
                for i in range(n, m+1):
                    p[i] = cnt[i] / v

            # normalize to ensure the sum of probability is 1
            ss = sum(p)
            if (ss == 0):
                continue
            for i in range(m+1):
                p[i] /= ss

            # record probability of 3 word phase
            for word3 in self.test2[key2]:
                if (word3 not in self.hash2[key2]):
                    self.hash3[word3] = p[0]
                else:
                    self.hash3[word3] = p[self.hash2[key2][word3]]

    def calculate_PPL(self):
        # calculate the PPL
        ppl = 0

        cnt = 0

        for key2 in self.test2:
            for word3 in self.test2[key2]:
                if (word3 not in self.hash3 or self.hash3[word3]==0):
                    continue
                ppl += math.log10(self.hash3[word3])
                cnt += 1

        ppl /= -cnt
        ppl = 10**ppl
        
        return ppl

# align
length = 30
if __name__ == '__main__':

    train_data = []

    # f = open("hw1_dataset\\train_set.txt", "r")
    with open(r"hw1_dataset\train_set.txt", "r") as f:
        # read by line
        for line in f:
            train_data.append(line)

    # split by whitespace
    train_data = train_data[0].split(" ")

    # begin and end symbol
    train_data.insert(0, '<s>')
    train_data.append('</s>')

    dev_data = []

    with open(r"hw1_dataset\dev_set.txt",'r') as f:
        # read by line
        for line in f:
            dev_data.append(line)

    # split by whitespace
    dev_data = dev_data[0].split(" ")
        
    # insert begin and end symbols
    dev_data.insert(0, '<s>')
    dev_data.append('</s>')

    test_data = []

    with open(r"hw1_dataset\test_set.txt",'r') as f:
        # read by line
        for line in f:
            test_data.append(line)

    # split by whitespace
    test_data = test_data[0].split(" ")
        
    # insert begin and end symbols
    test_data.insert(0, '<s>')
    test_data.append('</s>')

    # test the Lidstone Smoothing algorithm

    # record 3 words phase with its prefix
    hash2 = {}

    # 2-gram
    for i in range(1,len(train_data)):
        # concatenate two words 
        tmp = train_data[i-1] + ' ' + train_data[i]

        # record the previous word
        if (tmp not in hash2):
            hash2[train_data[i-1] + ' ' + train_data[i]] = {}

    # 3-gram
    for i in range(2,len(train_data)):
        # the previous two words
        prefix = train_data[i-2] + ' ' + train_data[i-1]
        tmp = train_data[i-2] + ' ' + train_data[i-1] + ' ' + train_data[i]
        if (tmp not in hash2[prefix]):
            hash2[prefix][tmp] = 1
        else:
            hash2[prefix][tmp] += 1
    
    # pass in the initial value of para k
    instance1 = Lidstone_smoothing(dev_data, k=0.5)

    # smoothing
    instance1.smoothing(hash2)

    # calculate the PPL of test_set
    res1 = instance1.calculate_PPL()
    print(res1)


    # fst train paras lambda_1 and lambda_2 on dev_set

    # 1-gram
    hash1 = {}

    # 2-gram
    hash2 = {}

    # 3-gram
    hash3 = {}

    # 1-gram
    for word in train_data:
        if (word not in hash1):
            hash1[word] = 1
        else:
            hash1[word] += 1

    # 2-gram
    for i in range(1,len(train_data)):
        # concatenate two words 
        tmp = train_data[i-1] + ' ' + train_data[i]

        # record the previous word
        if (tmp not in hash2):
            hash2[train_data[i-1] + ' ' + train_data[i]] = [train_data[i-1], 1]
        else:
            hash2[train_data[i-1] + ' ' + train_data[i]][1] += 1

    # 3-gram
    for i in range(2,len(train_data)):
        # concatenate three words
        tmp =  train_data[i-2] + ' ' + train_data[i-1] + ' ' + train_data[i]

        if (tmp not in hash3):
            hash3[tmp] = [train_data[i-2] + ' ' + train_data[i-1], 1]
        else:
            hash3[tmp][1] += 1
    
    # proportion
    for key in hash3:
        hash3[key][1] /= hash2[hash3[key][0]][1]

    for key in hash2:
        hash2[key][1] /= hash1[hash2[key][0]]

    for key in hash1:
        hash1[key] /= len(train_data)

    # snd test the model on the test_set with paras lambda_1 and lambda_2

    # pass in the initial values of lambda_1 and lambda_2 
    instance2 = Interpolation_smoothing(dev_data, test_data, lambda_1=0.5, lambda_2=0.5)

    # train lambda_1 and lambda_2 on the dev_set
    instance2.train_paras(hash1, hash2, hash3, dev_data, maxiter=1000, stepSize=0.05, tol=1e-5)

    # calculate PPL on the test_set
    res2 = instance2.calculate_PPL(hash1, hash2, hash3)
    print("PPL of test_set: {}".format(res2))


    # test the Good Turing Discounting algorithm

    # record 3 words phase with its prefix
    hash2 = {}

    # 2-gram
    for i in range(len(train_data)-1):
        # concatenate two words 
        tmp = train_data[i]

        # record the previous word
        if (tmp not in hash2):
            hash2[train_data[i]] = {}

    # 3-gram
    for i in range(1,len(train_data)):
        # the previous two words
        prefix = train_data[i-1]
        tmp = train_data[i-1] + ' ' + train_data[i]
        if (tmp not in hash2[prefix]):
            hash2[prefix][tmp] = 1
        else:
            hash2[prefix][tmp] += 1

    # record 3 words phase with its prefix
    test2 = {}

    # 2-gram
    for i in range(len(test_data)-1):
        # concatenate two words 
        tmp = test_data[i]

        # record the previous word
        if (tmp not in test2):
            test2[test_data[i]] = {}

    # 3-gram
    for i in range(1,len(test_data)):
        # the previous two words
        prefix = test_data[i-1]
        tmp = test_data[i-1] + ' ' + test_data[i]
        if (tmp not in test2[prefix]):
            test2[prefix][tmp] = 1
        else:
            test2[prefix][tmp] += 1

    # pass in the train_set, the test_set and the number of low frequency word
    instance3 = discounting(hash2, test2, count=10)

    # Good Turing Discounting
    instance3.discounting()

    # calculate the PPL of test_set
    res3 = instance3.calculate_PPL()
    print("ppl for discounting: {}".format(res3))