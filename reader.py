import pickle
import pandas
k = 10000
with open("./Qlearning/rewards"+str(k) +".pkl", 'rb') as fp:
            print(fp)
            qlearning = pickle.load(fp)

with open("./SARSA/rewards"+str(k) +".pkl", 'rb') as fp:
            print(fp)
            sarsa = pickle.load(fp)

with open("./tdlearning/rewards"+str(k) +".pkl", 'rb') as fp:
            print(fp)
            tdlearning = pickle.load(fp)

print(sum(qlearning))

print(sum(sarsa))
print(sum(tdlearning))

def summary(qlearning):
    ql = pandas.DataFrame(qlearning)
    ql = ql.describe()
    ql = ql.transpose()
    print(ql.head() )

summary(qlearning)
summary(sarsa)
summary(tdlearning)
