import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.cm as cm

def em(data, classes, epoch):
    sample_mean = np.average(data, axis=0)

    component_variance = 1.0/len(data)*sum((data - sample_mean)**2)

    mu = [data[np.random.choice(range(data.shape[0]))] for i in range(classes)]

    sigma2 = [component_variance for i in range(classes)]

    psi = [1.0/classes for i in range(classes)]

    for i in range(epoch):
        gamma = e_step(data, classes, mu, sigma2, psi)
        mu,sigma2,psi = m_step(data,classes,gamma)

    return gamma

def e_step(data, classes, mu, sigma2, psi):
    gamma = np.zeros((len(data), classes))

    for i in range(len(data)):
        normals = [psi[k]*stats.multivariate_normal.pdf(data[i], mu[k], np.diag(sigma2[k]))\
                     for k in range(classes)]
        
        for k in range(classes):
            gamma[i][k] = normals[k]/sum(normals)
    return gamma


def m_step(data, classes, gamma):
    sumGamma = [sum(gamma[:,k]) for k in range(classes)]
    psi = [sumGamma[k]/data.shape[0] for k in range(classes)]
    mu = [scalarSum(gamma[:,k], data)/sumGamma[k] for k in range(classes)]
    sigma2 = [scalarSquareSum(gamma[:,k], data, mu[k])/sumGamma[k] for k in range(classes)]
    return mu, sigma2, psi

def scalarSum(gamma, x):
    total = np.zeros(x[0].shape)
    for i in range(x.shape[0]):
        total += gamma[i]*x[i]
    return total

def scalarSquareSum(gamma, x, m):
    total = np.zeros(x[0].shape)
    for i in range(x.shape[0]):
        total += gamma[i]*(x[i] - m)**2
    return total

def generateData(dimension, amount, classes):
    dimension = 2
    centers = []
    r = 10.0
    for i in range(classes):
        centers.append(np.array([r * np.cos(2 * np.pi * i / classes), r * np.sin(2 * np.pi * i / classes)]))
    data = []
    labels = []
    for sample in range(amount):
        center = np.random.choice(range(classes))
        labels.append(center)
        C = centers[center]
        data.append(np.array([np.random.normal(C[0],1.0), np.random.normal(C[1], 1.0)]))

    
    return np.array(data),labels

classes = 5

data,labels = generateData(dimension=2,amount=100,classes=classes)

probs = em(data, classes=classes, epoch=50)

prediction = np.array([np.argmax(probs[i]) for i in range(probs.shape[0])])
print(prediction)
print(labels)


colors = cm.rainbow(np.linspace(0, 1, classes))

for i in range(classes):
    indices = np.where(prediction == i)
    points = np.take(data,indices,axis=0)[0]
    plt.scatter([k[0] for k in points],[k[1] for k in points], color=colors[i])
    
plt.show()



plt.scatter([k[0] for k in data],[k[1] for k in data])
plt.show()

