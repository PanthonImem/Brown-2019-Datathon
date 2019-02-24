import pandas as pd
import numpy as np

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors

import itertools

NUM_HOTELS = 1057
NUM_USER_ACTIONS = 1577114
NUM_UNIQUE_USERS = 363619
NUM_PREDICTIONS = 7

activity = pd.read_csv('activity_data.csv')
hotel = pd.read_csv('hotel_data.csv')

# map user id to an index from 0 to NUM_UNIQUE_USERS
user_id_to_num = {}
num_to_user_id = {}
# map hotel id to an index form 0 to NUM_HOTELS
hotel_id_to_num = {}
num_to_hotel_id = {}
# map hotel id to hotel name
hotel_id_to_name = {}

for counter, i in enumerate(activity['user_id'].unique()):
    user_id_to_num[i] = counter
    num_to_user_id[counter] = i

for counter, i in enumerate(list(hotel['hotel_id'])):
    hotel_id_to_num[i] = counter
    num_to_hotel_id[counter] = i
    hotel_id_to_name[i] = hotel.iloc[counter][1]

def prepare_data(occurence, data):
    """
    This method changes data into one-hot type ready for training/testing a model.
    It first reduces data by using only users who occurs more than occurence in data.
    Then, it creates one-hot arrays of size NUM_HOTEL x NUM_SELECTED_USERS.
    If the user do any action to the hotel TripAdvisor webpage, then give 
    that [hotel, user] index 1, else 0.

    Inputs:
    occurence:: number of occurence of each user id in the data
                If that user occurs less than 'occurence', then delete him/her from the dataset
                This is to reduce training/testing data in a model.
    data:: dataset we want to use for a preparation

    Outputs:
    all_one_hots:: one-hot arrays of size NUM_HOTEL x NUM_SELECTED_USERS
    """
    all_users = data['user_id'].values
    map_id_to_num = np.vectorize(lambda x: user_id_to_num[x])
    all_users_dict = map_id_to_num(all_users)
    unique_user_counts = np.bincount(all_users_dict)
    selected_user_ids = []
    for counter, i in enumerate(list(unique_user_counts)):
        if i > occurence: selected_user_ids.append(num_to_user_id[counter])
    selected_activity = data.loc[data['user_id'].isin(selected_user_ids)]

    NUM_SELECTED_USER = len(selected_user_ids)
    selected_user_id_to_num = {}
    for counter, i in enumerate(selected_user_ids):
        selected_user_id_to_num[i] = counter

    hotel_ids = np.zeros(NUM_HOTELS)
    all_one_hots = np.zeros((NUM_HOTELS, NUM_SELECTED_USER))

    for counter, e in enumerate(list(hotel['hotel_id'])):
        hotel_ids[counter] = e
        cur = selected_activity.loc[data['hotel_id'] == e]
        if (cur.shape[0] == 0): 
            all_one_hots[counter] = np.zeros(NUM_SELECTED_USER)
            continue
        cur_one_hot = np.zeros(NUM_SELECTED_USER)
        for i in list(cur['user_id']):
            cur_one_hot[selected_user_id_to_num[i]] = 1
        all_one_hots[counter] = cur_one_hot

    return all_one_hots


def reduce_dimension(all_one_hots, batch_sz=200):
    """
    This method reduces the dimension of one-hot array from prepare_data() 
    using sklearn.decomposition.IncrementalPCA so that GaussianofMixture 
    can perform using the data. 

    Inputs:
    all_one_hots:: one-hot array (from prepare_data method)
    batch_sz:: batch size that we want the data to go through IncrementalPCA each time
                This will also be a final dimension of each hotel feature

    Outputs:
    reduced_one_hots:: reduced one-hot array with dimension of batch_sz
    """

    pca = IncrementalPCA()
    train_sz = all_one_hots.shape[0]
    last = 0
    for start, end in zip(range(0, train_sz - batch_sz, batch_sz), range(batch_sz, train_sz, batch_sz)):
        pca.partial_fit(all_one_hots[start:end, :])
        last = end
    pca.partial_fit(all_one_hots[last:, :])

    reduced_one_hots = np.array([])
    for start, end in zip(range(0, train_sz - batch_sz, batch_sz), range(batch_sz, train_sz, batch_sz)): 
        reduced = pca.transform(all_one_hots[start:end, :])
        if (reduced_one_hots.size == 0):
            reduced_one_hots = reduced
        else:
            reduced_one_hots = np.vstack((reduced_one_hots, reduced))
    reduced = pca.transform(all_one_hots[last:, :])
    reduced_one_hots = np.vstack((reduced_one_hots, reduced))

    return reduced_one_hots

def plot_results(X, Y_, means, covariances, index, title):
    """
    This method plots the gmm graph to see the clustering of data. 
    """
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    plt.xlim(-1., 1.)
    plt.ylim(-2., 3.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def gmm(one_hots): 
    """
    This method performs sklearn.mixture.GaussianMixture
    with one-hot hotel features (1 if the user view the hotel in TripAdvisor else 0)
    and plot the clustering using plot_result

    Inputs:
    one_hots:: one-hot hotel features (1 if the user view the hotel in TripAdvisor else 0)
    """
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(one_hots)
    plot_results(one_hots, gmm.predict(one_hots), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')
    plt.show()

def knn(one_hots, k=4, testing_data=None, predict=False):
    """
    This method performs sklearn.neighbors.NearestNeighbors
    on one-hot hotel features (1 if the user view the hotel in TripAdvisor else 0)
    with k neighbors. 
    If predict is True, we returns the predictions of the testing data,
    else we returns the nearest neighbors of each hotel.

    Inputs:
    one_hots:: one-hot hotel features (1 if the user view the hotel in TripAdvisor else 0)
    k:: number of neighbors to use in NearestNeighbors (default = 4)
    testing_data:: one-hot of hotel features of testing dataset (default = None)
    predict:: boolean indicating whether we want to test the model 

    Outputs:
    zip_recommend: NumPy array of indices of nearest neighbors and sorted distances
    """
    knn = NearestNeighbors(metric='cosine', algorithm='auto')
    knn.fit(one_hots)
    if (predict == False): 
        distances, indices = knn.kneighbors(one_hots, n_neighbors=k)
        zip_recommend = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())))
        return np.array(zip_recommend)

    distances, indices = knn.kneighbors(testing_data, n_neighbors=k)
    zip_recommend = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())))
    return np.array(zip_recommend) 

def get_recommendation(knn_data):
    """
    This method creates a csv file including 4 recommendations of similar hotels to each hotel
    and their distances got from NearestNeighbors model.

    Inputs:
    knn_data: NumPy array of indices of nearest neighbors and sorted distances of each hotel
    """
    recommend = hotel[['hotel_id', 'hotel_name']] 
    recommend['first recommendation'] = '0'
    recommend['distance 1'] = 0.000
    recommend['second recommendation'] = '0'
    recommend['distance 2'] = 0.000
    recommend['thrid recommendation'] = '0'
    recommend['distance 3'] = 0.000
    for i in range(NUM_HOTELS):
        each_hotel = knn_data[i]
        recommend.at[i, 'first recommendation'] = hotel_id_to_name[num_to_hotel_id[each_hotel[0][1]]]
        recommend.at[i, 'distance 1'] = each_hotel[1][1]
        recommend.at[i, 'second recommendation'] = hotel_id_to_name[num_to_hotel_id[each_hotel[0][2]]]
        recommend.at[i, 'distance 2'] = each_hotel[1][2]
        recommend.at[i, 'thrid recommendation'] = hotel_id_to_name[num_to_hotel_id[each_hotel[0][3]]]
        recommend.at[i, 'distance 3'] = each_hotel[1][3]
    recommend.to_csv('recommend.csv')

def create_labels(testing_labels_data):
    """
    This method create labels from testing_labels_data by listing the hotel_id.
    """
    labels = []
    for e in testing_labels_data:
        labels.append[e['hotel_id']]
    return np.array(labels)

def get_accuracy(predictions, labels):
    """
    This method returns accuracy of predictions compared to labels.
    This method is unfinished, but we intend to do Mean Average Precision(MAP).
    That is, we predict the next hotel a new user would view/click/book from 
    their previous records. If the label (i.e. the next hotel the user looks into)
    is in a prediction, we give points according to the rank of recommendations. 
    """
    assert(len(predictions) == len(labels))
    for num_hotel in range(len(labels)):
        each_predictions = predictions[e]
        for i in range(NUM_PREDICTIONS):
            pass

"""
=================================
RUN FUNCTIONS
=================================
"""
training_data = pd.read_csv('training_data.csv')
one_hots = prepare_data(7, training_data)
testing_features_data = pd.read_csv('testing_features.csv')
testing_labels_data = pd.read_csv('testing_labels.csv')
testing_features_one_hots = prepare_data(7, testing_features_data)
testing_labels = create_labels(testing_labels_data)
#reduce_dimension()
#one_hots = np.load('reduced-one-hot.npy')
#one_hots = np.load('one-hot.npy')
zip_knn = knn(one_hots, k=NUM_PREDICTIONS, testing_data=testing_features_one_hots, predict=True)
knn_predictions = np.load('knn.npy')
#get_recommendation(knn_data)
#gmm(one_hots)
