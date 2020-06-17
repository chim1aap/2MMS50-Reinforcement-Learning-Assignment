import pickle
import pandas
from plotnine import * # ggplot package for python.
import numpy


'''
Load results 
'''
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

'''
Print summary
'''
print(sum(qlearning))
print(sum(sarsa))
print(sum(tdlearning))

def summary(vector):
    ql = pandas.DataFrame(vector, columns = {"Reward" } )
    ql.insert(1, "Run", range(len(vector)))
    qls = ql.describe()
    qls = qls.transpose()
    print(qls.head() )

    return ql
summary(qlearning)
summary(sarsa)
summary(tdlearning)
'''
plot some results. 
'''
df = pandas.DataFrame(range(len(sarsa)) , columns = {"Run" })
df.insert(1, "Qlearning", numpy.cumsum(qlearning) )
df.insert(2, "Sarsa" , numpy.cumsum(sarsa) )
df.insert(3, "TD(0)" , numpy.cumsum(tdlearning ))
print(df.head())
g = (ggplot(df) +
     geom_line(aes( x  = "Run", y = "Run"), color = "black") +
     geom_line(aes( x  = "Run", y = "Qlearning"), color= "blue") +
     geom_line(aes( x  = "Run", y = "Sarsa"), color = "Red") +
     geom_line(aes( x  = "Run", y = "TD(0)"), color = "green") +
     ggtitle("Results")
     )
#print(g)

#save to file
g.save(filename="./images/results.pdf")
