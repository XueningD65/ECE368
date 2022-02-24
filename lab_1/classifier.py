import os.path
import numpy as np
import matplotlib.pyplot as plt
import util


def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set

    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the
    smoothed estimates of q_d
    """

    #probabilities_by_category = tuple()
    global laplace_smooth_spam, laplace_smooth_ham
    alpha = 1

    # get the counts for each word in the emails
    freq_spam = util.get_word_freq(file_lists_by_category[0]) # example: 'Subject:': 3675
    freq_ham = util.get_word_freq(file_lists_by_category[1])
    freq_total = util.get_word_freq(file_lists_by_category[0] + file_lists_by_category[1])

    total_spam, total_ham = sum(freq_spam.values()), sum(freq_ham.values())
    total_dict = len(freq_total.keys())

    #print(total_ham, total_spam, total_dict)

    laplace_smooth_spam = np.log(1 / (total_spam + alpha * total_dict))
    laplace_smooth_ham = np.log(1 / (total_ham + alpha * total_dict))

    #print("laplace:",laplace_smooth_ham, laplace_smooth_spam)

    for key in freq_spam.keys():
        freq_spam[key] = (freq_spam[key] + alpha) / (total_spam + alpha * total_dict)

    for key in freq_ham.keys():
        freq_ham[key] = (freq_ham[key] + alpha) / (total_ham + alpha * total_dict)
    #print(list(freq_spam.items())[:5])
    return (freq_spam, freq_ham)#probabilities_by_category


def classify_new_email(filename, probabilities_by_category, prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)],
    representing the log posterior probabilities
    """
    ### TODO: Write your code here

    freq_new = util.get_word_freq([filename])
    #print(freq_new)

    spam_prob, ham_prob = np.log(prior_by_category[0]), np.log(prior_by_category[1])
    for key in freq_new.keys():
        if key in probabilities_by_category[0].keys(): # belongs to SPAM
            spam_prob += np.log(probabilities_by_category[0][key]) * freq_new[key]
        elif key in probabilities_by_category[1].keys():
            spam_prob += laplace_smooth_spam * freq_new[key]
        else:
            pass

        if key in probabilities_by_category[1].keys():
            ham_prob += np.log(probabilities_by_category[1][key]) * freq_new[key]
        elif key in probabilities_by_category[0].keys():
            ham_prob += laplace_smooth_ham * freq_new[key]
        else:
            pass

    #print(ham_prob, spam_prob)
    classification = ""
    if threshold == 0:
        if ham_prob >= spam_prob:
            classification = "ham"
        else:
            classification = "spam"
    else:
        if spam_prob - np.log(threshold) >= ham_prob: # we want more ham than spam
            classification = "spam"
        else:
            classification = "ham"
    #print(classification)
    return (classification, (spam_prob, ham_prob))


if __name__ == '__main__':

    # folder for training and testing
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))

    # Learn the distributions
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2, 2])

    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam'
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham'
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam'
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham'

    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    type_1, type_2 = [], []
    tolerance = [1e-5 * (10 ** (i * 3)) for i in range(10)]
    tolerance.append(0)
    global threshold

    for val in tolerance:
        # Store the classification results
        performance_measures = np.zeros([2, 2])
        threshold = val

        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                      probabilities_by_category,
                                                      priors_by_category)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))
        print("Type 1 Error: %d | Type 2 Error: %d"%(totals[0] - correct[0], totals[1] - correct[1]))

        if val != 0:
            type_1.append(totals[0] - correct[0])
            type_2.append(totals[1] - correct[1])

    fig, axs = plt.subplots()
    axs.plot(type_1, type_2)
    axs.set_xlabel('Type 1 Error')
    axs.set_ylabel('Type 2 Error')
    fig.savefig("nbc.pdf")
    plt.show()

