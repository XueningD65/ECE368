import numpy as np
import matplotlib.pyplot as plt
import lab_2.reg.util as util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    fig = plt.figure()
    prior_mean = np.array([0, 0])
    prior_cov = np.array([[beta, 0], [0, beta]])

    # plot the true value of a
    plt.plot([-0.1], [-0.5], marker = '.', markersize = 8)

    x, y = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    contour = []
    for i in range(100):
        contour.append(util.density_Gaussian(prior_mean,
                                             prior_cov,
                                             np.concatenate((X[i].reshape(100, 1), Y[i].reshape(100, 1)), 1)))
    plt.contour(X, Y, contour)
    plt.xlabel("$a_0$")
    plt.ylabel("$a_1$")
    plt.title("Prior Distribution")
    #plt.savefig("/Users/dongxuening/Desktop/ECE368/lab_2/reg/saved_file/prior.pdf")
    #plt.show()
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    # calculate the prior mean and covariance
    prior_mean = np.array([0, 0])
    prior_cov = np.array([[beta, 0], [0, beta]])
    Cov_prior_inv = np.linalg.inv(prior_cov)
    #print(Cov_prior_inv)

    # the noise
    Cov_w_inv = 1/sigma2
    mean_w = 0

    X = np.array([[1, x[idx].item()] for idx in range(len(x))])
    #print(X)

    sigma_a_D = np.linalg.inv(Cov_prior_inv + np.dot(np.dot(X.T, Cov_w_inv),X))
    mu_a_D = np.dot(sigma_a_D, np.dot(np.dot(X.T, Cov_w_inv), z))
    mu_a_D = np.array([mu_a_D[0].item(), mu_a_D[1].item()])
    print(sigma_a_D)
    print(mu_a_D)

    fig = plt.figure()

    # plot the true value of a
    plt.plot([-0.1], [-0.5], marker='.', markersize=8)

    xi, yi = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(xi, yi)

    contour = []
    for i in range(100):
        contour.append(util.density_Gaussian(mu_a_D,
                                             sigma_a_D,
                                             np.concatenate((X[i].reshape(100, 1), Y[i].reshape(100, 1)), 1)))
    plt.contour(X, Y, contour)
    plt.xlabel("$a_0$")
    plt.ylabel("$a_1$")
    plt.title("Posterior Distribution with sample size: "+str(len(x)))
    # plt.savefig("/Users/dongxuening/Desktop/ECE368/lab_2/reg/saved_file/posterior"+str(len(x))+".pdf")
    # plt.show()
   
    return (mu_a_D,sigma_a_D)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    X = np.array([[1, x[idx]] for idx in range(len(x))])
    sigma_W = sigma2

    mu_pred = np.dot(X, mu)
    sigma_pred = sigma_W + np.dot(X, np.dot(Cov, X.T))

    fig = plt.figure()
    plt.scatter(x_train, z_train, color = 'black')
    plt.errorbar(x, mu_pred, yerr=np.sqrt(np.diag(sigma_pred)))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.title("Prediction Distribution with sample size: " + str(ns))
    plt.savefig("/Users/dongxuening/Desktop/ECE368/lab_2/reg/saved_file/predict"+str(ns)+".pdf")
    plt.show()
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    global ns
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    #
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    #

   

    
    
    

    
