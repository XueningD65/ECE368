import numpy as np
import matplotlib.pyplot as plt
import util
import copy

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    count_male, mu_male_height, mu_male_weight = 0, 0, 0
    count_female, mu_female_height, mu_female_weight = 0, 0, 0

    male_heights, female_heights, male_weights, female_weights = [], [], [], []

    # y = 1 -> male and y = 2 -> female
    # x = [height weight]
    for idx, label in enumerate(y):
        if label == 1:
            count_male += 1

            # record height of male
            male_heights.append(x[idx][0])

            # record weight of male
            male_weights.append(x[idx][1])
        else:
            count_female += 1

            # record height of female
            female_heights.append(x[idx][0])

            # record weight of male
            female_weights.append(x[idx][1])


    mu_male_height = np.mean(np.array(male_heights))
    mu_male_weight = np.mean(np.array(male_weights))

    mu_male = np.array([mu_male_height, mu_male_weight])

    mu_female_height = np.mean(np.array(female_heights))
    mu_female_weight = np.mean(np.array(female_weights))

    mu_female = np.array([mu_female_height, mu_female_weight])

    # print("male: height", mu_male_height, "weight:", mu_male_weight)
    # print("female: height", mu_female_height, "weight:", mu_female_weight)
    print(mu_male)
    print(mu_female)

    sigma = np.array([[0, 0], [0, 0]])
    sigma_male = np.array([[0, 0], [0, 0]])
    sigma_female = np.array([[0, 0], [0, 0]])

    for idx in range(len(x)):

        if y[idx] == 1:

            temp_res = np.subtract(x[idx], mu_male)
            sigma_male = np.add(sigma_male, np.matmul(np.transpose([temp_res]), [temp_res]))

        else:
            temp_res = np.subtract(x[idx], mu_female)
            sigma_female = np.add(sigma_female, np.matmul(np.transpose([temp_res]), [temp_res]))

        sigma = np.add(sigma, np.matmul(np.transpose([temp_res]), [temp_res]))
        #print(temp_res, np.transpose([temp_res]), sigma)

    cov = sigma / len(y)
    cov_male = sigma_male / count_male
    cov_female = sigma_female / count_female

    print(cov)
    print(cov_male)
    print(cov_female)

    # plot all datapoints
    plt.scatter(male_heights, male_weights, color='blue')
    plt.scatter(female_heights, female_weights, color='red')

    # plot gradient
    x_grid = np.linspace(50, 80, 100)
    y_grid = np.linspace(80, 280, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    male, female = [], []

    for i in range(len(y)):
        grid = np.concatenate((X[0].reshape(100, 1), Y[i].reshape(100, 1)), 1)
        male.append(util.density_Gaussian(mu_male, cov, grid))
        female.append(util.density_Gaussian(mu_female, cov, grid))

    # plot the contours
    plt.contour(X, Y, male, colors='b')
    plt.contour(X, Y, female, colors='r')

    bound = np.array(male) - np.array(female)
    plt.contour(X, Y, bound, 0)

    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Decision Boundary with Contours in 2D - LDA')
    #plt.savefig("lda.pdf")
    #plt.show()

    # plot all datapoints
    plt.scatter(male_heights, male_weights, color='blue')
    plt.scatter(female_heights, female_weights, color='red')

    # plot gradient
    x_grid = np.linspace(50, 80, 100)
    y_grid = np.linspace(80, 280, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    male, female = [], []

    # QDA
    for i in range(len(y)):
        grid = np.concatenate((X[0].reshape(100, 1), Y[i].reshape(100, 1)), 1)
        male.append(util.density_Gaussian(mu_male, cov_male, grid))
        female.append(util.density_Gaussian(mu_female, cov_female, grid))

    # plot the contours
    plt.contour(X, Y, male, colors='b')
    plt.contour(X, Y, female, colors='r')

    bound = np.array(male) - np.array(female)
    plt.contour(X, Y, bound, 0)

    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Decision Boundary with Contours in 2D - QDA')
    #plt.savefig("qda.pdf")
    #plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)
    #return mu_male,mu_female, cov, 0, 0
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate

    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """

    # some generally used terms
    cov_male_inv, cov_female_inv = np.linalg.inv(cov_male), np.linalg.inv(cov_female)
    cov_inv = np.linalg.inv(cov)

    # for LDA
    alpha_m = - 0.5 * np.dot(np.dot(np.transpose(mu_male), cov_inv), mu_male)
    alpha_f = - 0.5 * np.dot(np.dot(np.transpose(mu_female), cov_inv), mu_female)

    beta_m = np.dot(np.transpose(mu_male), cov_inv)
    beta_f = np.dot(np.transpose(mu_female), cov_inv)


    incorrect = 0
    for idx, x_val in enumerate(x):
        LDA_male = np.dot(beta_m, x_val.T) + alpha_m
        LDA_female = np.dot(beta_f, x_val.T) + alpha_f

        if LDA_male < LDA_female and y[idx] == 1:
            incorrect += 1
        elif LDA_female < LDA_male and y[idx] == 2:
            incorrect += 1

    mis_lda = incorrect / len(y)

    # for QDA
    incorrect = 0
    for idx, x_val in enumerate(x):
        QDA_male = - np.log(np.linalg.det(cov_male)) - \
                   np.dot(x_val, np.dot(cov_male_inv, x_val.T)) + \
                   2 * np.dot(np.transpose(mu_male), np.dot(cov_male_inv, x_val.T)) - \
                   np.dot(np.transpose(mu_male), np.dot(cov_male_inv, mu_male))

        QDA_female = - np.log(np.linalg.det(cov_female)) - \
                     np.dot(x_val, np.dot(cov_female_inv, x_val.T)) + \
                     2 * np.dot(np.transpose(mu_female), np.dot(cov_female_inv, x_val.T)) - \
                     np.dot(np.transpose(mu_female), np.dot(cov_female_inv, mu_female))

        if QDA_male < QDA_female and y[idx] == 1:
            incorrect += 1
        elif QDA_female < QDA_male and y[idx] == 2:
            incorrect += 1

    mis_qda = incorrect / len(y)
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    # print(x_train[:2], y_train[:2])
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')

    # parameter estimation and visualization in LDA/QDA
    mu_male, mu_female, cov, cov_male, cov_female = discrimAnalysis(x_train,y_train)
    #
    # misclassification rate computation
    mis_LDA, mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    print("Missing rate LDA:", mis_LDA, "Missing rate QDA:", mis_QDA)
    

    
    
    

    
