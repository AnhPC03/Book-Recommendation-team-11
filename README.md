## About Project

Project was implemented by Team 11.

## Requirement

Project requires python 3.\
Run following command to setup environment:\
``` pip install scipy ```\
``` pip install pandas ```\
``` pip install numpy ```\
``` pip install scikit-learn ```


## Make Recommendation

To make recommendation for specified user, run following command:\
``` python main.py --id userID --metric 'cosine' --k 10 --normalized 1 ```
1. userID is specified id of user you want to make recommendation
2. metric is what distance metric (cosine, correlation, euclidean) you want to use in caculate k-NN
3. k is number of neighbors similarity with above user
4. normalized = 1 if you want normalize in prediction formula. 
