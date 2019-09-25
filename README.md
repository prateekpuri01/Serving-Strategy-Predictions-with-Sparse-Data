# ATP_strategy_predictions_sparse_data
I extend my analysis of optimal ATP serving strategies to include prediction for player matchups when data is sparse

As a reminder, as detailed in another repository, previously, I collected data on how a particular player served (FSWP, SSWP) against all other players he has played against to identify player matchups in which the player could benefit from a 'risky' strategy in which he hits two first serves on every point. This analysis required pooling from a player's previous match statistics againt other players in order to make a prediction. 

How can we make predictions when such information is unavailable?

To this end, to make predictions for when a player should go 'risky' against a certain opponent, I first collected matchup data for all matchups with match lengths greater than 3 involving the opponent (see below for why 3 was chosen as a cutoff). I then assigned each player in this matchup set to either 'risky' or 'bold' classifications based on the EM factors with the opponent. I then trained a logistic regression model to this data and optimized the model coefficients. I could then use this classifier to make predictions for when other players not included in the training set (no previous matches with the opponent) who could benefit from a risky strategy. 

The process is repeated for each potential opponent who had been previously ranked in the top 30. Below I give a brief explanation of the various steps involved in the process.

# Classifying which players would benefit from a risky strategy

From previous work, we have two indications of whether a player is likely to benefit from a risky serve strategy against a certain player: 1) Their matchup EM factor 2) the win percentage enhancement expected from switching to a risky strategy as predicted from a Monte Carlo (MC) simulation. 

If either of these quanitities is >0, the matchup is classified as benefit from a risky strategy. 

2) is more robust in that the MC simulations take into account factors such as player's first and second serve make percentages and subtle effects in the structure of tennis matches. On the other hand, the MC simulations are time consuming to produce and were only performed for players with a matchup history of at least five matches with a particular opponent. The reduced sample size also reduces the number of training points that will be available for our machine learning model. On the other hand, EM factors may be a less rigorous classifier but are easy and quick to compute for all matchups in the ATP database. 

Moreover, it turns out that 94% of the classifications of the MC predicitons match the classifications of the EM prediction. Given this agreement, EM factors were used for classification to increase training points and eventual ML accuracy. 

# Producing the machine learning model

Here are the steps taken in producing the machine learning model

Goal: Construct a classifier that can predict if a player (P1) who has never played an opponent (P2) will benefit from a risky strategy. 

Step 1: Identify all players who have a matchup history of at least 3 matches with P2. Call this set of players P_ML. If size(P_ML)<15, select another P2 to ensure we will have enough training points.  

Step 2: Calculate the matchup EM factor of all players in P_ML when playing P2. If >0, classify the player as 'risky'; if <=0 classify as 'safe'.
Step 3: No that we have our classification, we need to produce a list of features that can help predict these classifications. First produce a list of all active top30 ATP players. Let us refer to this set as P_active. 

Consider the matchups between all players in P_active for a given player in P_ML, and compute the following quantities for each matchup:

FSP*FSWP*
SSP*SSWP*

These values are the features that will define the player. If a player does not have a matchup history with a player in P_active, list these values as zero. 

Once these features have been computer for every player, remove feature columns that contain all zeroes - this effectively removes matchup statistics for players in P_active who have not played anyone in P_ML.

Step 4: Split the above data in test and train sets. Initialize a LogisticRegression (LR) classifier, and perform a grid search, bench marked by accuracy scores, to select which the optimal regularization parameter. During the grid search use a cv parameter of 3. 

Step 5: Train the model on the training data using optimzied parameters and generate an accuracy score for the test data (which has been shielded from model up until this point)
.
Step 6: Also construct a stratified dummy classifier, and apply the classifier to the test data for model evaluation purposes. 

Step 7: Repeat process for every P2 within the set P_active


# Producing the machine learning model

The following plot demonstrates the accuracy score for the LR models for all players in P_active as well as the scores for the simple dummy classifier. As seen from the plot, the two distributions are distinct at the 1% confidence level. 

Of course there are a few situations in which the LR is outperformed by even the dummy classifier. This is expected since the number of training points can sometimes be ~10, which is insufficient to produce reliable results, especially with feature sizes of ~100. 

However, by selecting models who outperformed the dummy classifier by at least 10% (accuracy score), we can list the top 10 player matchups where a player is expected to benefit from a risky strategy, ranked by the value or the LR classificaiton probability. Also, I restricted this list to include only models where the train score and the test score were within 20% of one another. This was done to reduce the presence of overfit models since overfitting is a concern here, especially when the number of training points is less than 20 or so. 

We can also inspect a AUC curve for a sample player in our dataset, Kei Nishikori.



# How to determine how much match history is needed for the EM metric to be a reliable classifier?

How much match history do you need to determine if a risky strategy is actually likely optimal? It's unclear whether an EM factor computed after one match is likely to be reflective of the EM factor that would be obtained from 10+ matches. To address this, I first restricted my data to players who have played each other 15 times. I calculated the EM factor for each match and then calculated what the matchup averaged EM factor was as a function of number of matches included in the average (0-15). I could then compare these values to the EM factor averaged over the first 15 values in each matchup. This gives me a sense of how quickly the EM factors converged to their average value as the match history evolved between two players. I was most interested in determining how many matches it took for the average EM factor to converge to being the same sign (positive or negative) and the 15 match average. Since is relevant since whether EM>0 determine the classificaiton for each player in the ML model. 

The following plot displays this convergence curve along with the mean accuracy of the LR model. As can be seen from the plot, after 3 matches, in 70% of cases, the classification has converged to what its 15-match classification would be. 

As a side note here, I calculated the EM factors here by averaging the EM factor from each individual match in a given matchup. However, in my classification metric, I'm actually considering the EM factor averaged across *all points* in a given matchup. These two quantities are slightly different since there are a different number of points in each match. However, I verified that the two averages produce the same classificaiton ('risky' vs 'safe') in over 90% of cases. The match-averaged values are easier to consider since they allow for simple filtering of player matchup when constructing the ML model, so they were employed here for convenience.

# Replacing Monte Carlo Modeling: Using Machine Learning instead

As noted above, the Monte Carlo classification is a fairly robust, but inconvenient, method to classify optimal serving strategy.
I tried to replace the MC classification with a Linear Regression ML model. 

I once again considered all top-30 active player matchups (P1 vs P2). For each matchup I calculate thf following features

P1 FSP*FSWP
P1 SSP*SSWP
P2 FSP*FSWP
P2 SSP*SSWP

Each matchup was also classified by a corresponding win percentage for P1. Using the winning percentage as a classifier, I split my data into training and test sets and fit the Linear Regression model to the training data. I then compare the test data predictions to test data actual winning percentages, with an R^2 value of 0.79, as compared to the MC R^2 value of 0.85.

This machine learning model was extremely quick to implement as compared to the MC method (seconds vs. hours), and produced similarly accurate results. 

Of course what I would really like do is change the feature values (SSP*SSWP->FSP*FSWP) to account for risky serving strategies and then see if my model can produce reliable, updated win percentages. 

I performed such a calculation and compared the results to the win percentage difference expected by my MC model. The plot below comapres the two results, with an R^2 value of 0.83. Thus, it seems like the machine learning model was able to reproduce the Monte Carlo results with reasonable accuracy while shedding hours of computation time. 














