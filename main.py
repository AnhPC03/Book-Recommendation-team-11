# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import argparse
import timeit
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine, euclidean
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

books = pd.read_csv('books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True)
pd.set_option('display.max_colwidth', -1)

books.loc[books['ISBN'] == '0836218523', 'bookTitle'] = 'The Calvin & Hobbes Lazy Sunday Book'
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"

books.yearOfPublication = pd.to_numeric(books.yearOfPublication, errors='coerce')
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
books.yearOfPublication = books.yearOfPublication.astype(np.int32)

books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'

users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
users.Age = users.Age.astype(np.int32)
n_users = users.shape[0]
n_books = books.shape[0]

ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings_new = ratings_new[ratings_new.userID.isin(users.userID)]

ratings_explicit = ratings_new[ratings_new.bookRating != 0]
ratings_implicit = ratings_new[ratings_new.bookRating == 0]

users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]

counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]

mean = ratings_explicit.groupby(by="userID", as_index=False)['bookRating'].mean()
ratings_avg = pd.merge(ratings_explicit, mean, on='userID')
ratings_avg['adg_ratings'] = ratings_avg['bookRating_x'] - ratings_avg['bookRating_y']
ratings_avg_merge = pd.merge(ratings_avg, books, on='ISBN')

check = pd.pivot_table(ratings_avg,values='bookRating_x',index='userID',columns='ISBN')

final_user = pd.pivot_table(ratings_avg,values='bookRating_x',index='userID',columns='ISBN')
final_user.fillna(0, inplace=True)

def find_similarity(user_index, ratings, metric, k):
    similarities=[]
    indices=[]    
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    distances, indices = model_knn.kneighbors(ratings.iloc[user_index, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    return similarities,indices

def predict_similarity(user_index, item_index, ratings, metric, k, normalized, similarities, indices):
    wtd_sum = 0
    mean_user = 0
    sum_wt = np.sum(similarities)-1
    # print(similarities, indices)
    # anhpc03
    if (normalized == 1):
        userId = ratings.index[user_index]
        mean_user = ratings_avg.loc[ratings_avg['userID'] == userId]['bookRating_y'].unique()[0]
    for i in range(0, len(indices.flatten())):
        user_index_sim = indices.flatten()[i]
        if user_index_sim == user_index:
            continue
        else:
            userId_sim = ratings.index[user_index_sim]
            mean_user_sim = ratings_avg.loc[ratings_avg['userID'] == userId_sim]['bookRating_y'].unique()[0]
            product = (ratings.iloc[user_index_sim][item_index] - mean_user_sim) * similarities[i]
            wtd_sum += product
    prediction = int(mean_user + round(wtd_sum/sum_wt))
    return prediction

def recommendItem(user_id, ratings, metric, k, normalized):    
    if (user_id not in ratings.index.values) or type(user_id) is not int:
        print(f"User id should be a valid integer from this list :\n\n {re.sub('[chr(92)[chr(92)]]', '', np.array_str(ratings_matrix.index.values))}")
    else:
        prediction = []
        user_index = ratings.index.get_loc(user_id)
        similarities, indices = find_similarity(user_index, ratings, metric, k)
        for i in range(0, len(indices.flatten())):
            user_index_sim = indices.flatten()[i]
            if user_index_sim == user_index:
                continue
            else:
                for j in range(ratings.shape[1]):
                    if int(ratings.iloc[user_index_sim][j]) != 0 and int(ratings.iloc[user_index][j]) == 0:
                        prediction.append(predict_similarity(user_index, j, ratings, metric, k, normalized, similarities, indices))

        prediction = pd.Series(prediction)
        prediction = prediction.sort_values(ascending=False)
        recommended = prediction[:10]
        if normalized == 1:
            using_normalized = 'True'
        else:
            using_normalized = 'False'
        print(f"As metric = {metric}, k = {k}, normalized = {using_normalized}. For user has id {user_id}. These books are recommended...\n")
        for i in range(len(recommended)):
            print(f"{i+1}. {books.bookTitle[recommended.index[i]]}")



def evaluateRS(ratings, metric, k, normalized):
    number_users = ratings.shape[0]
    number_items = ratings.shape[1]
    prediction = np.zeros((number_users, number_items))
    prediction = pd.DataFrame(prediction)
    for i in range(50):
        similarities, indices = find_similarity(i, ratings, metric, k)
        for j in range(number_items):
            if int(ratings.iloc[i][j]) == 0:
                prediction[i][j] = 0
                continue
            else:
                prediction[i][j] = predict_similarity(i, j, ratings, metric, k, normalized, similarities, indices)

    MSE = mean_squared_error(ratings, prediction)
    RMSE = round(sqrt(MSE), 3)
    return RMSE


def allEvaluateRS(ratings):
    metrics = ['cosine']
    number_k = [10, 20, 30]
    number_normalized = [0, 1]
    for metric in metrics:
        for k in number_k:
            for normalized in number_normalized:
                RMSE = evaluateRS(ratings, metric, k, normalized)
                print(f'metric: {metric}, k = {k}, normalized = {normalized}. RMSE = {RMSE}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="ID of the user want to recommend")
    parser.add_argument("--metric", type=str, default='cosine', help="type of distance metric to caculate distance")
    parser.add_argument("--k", type=int, default=30, help="number of kneighbors of above user")
    parser.add_argument("--normalized", type=int, default=1, help="you want to normalize among users?")
    opt = parser.parse_args()
    user_id = opt.id
    metric = opt.metric
    k = opt.k
    normalized = opt.normalized
    print('\nWait for minutes...\n\n')
    start = timeit.default_timer()
    recommendItem(user_id, final_user, metric, k, normalized)
    stop = timeit.default_timer()
    print('\n\n')
    print(f'Time to predict: {stop - start}\n')
