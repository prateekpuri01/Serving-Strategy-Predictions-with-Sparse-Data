# ATP_ML_strategy_predictions_sparse_data

In this repository, I detail a recent project where I constructed a machine learning model to predict what serving strategy is optimal for a given ATP player, even when playing an opponent he has never faced before. 

Several of the concepts mentioned in this document have been described extensively in the following repository: https://github.com/prateekpuri01/ATP_serving_strategy. Please refer to this page for clarification.

Here is a list of acronyms that will be used in this document:

**FSP**: first serve make percentage <br/>
**SSP**: second serve make percentage <br/>
**FSWP**: percentage of points that are won when a server makes his first serve <br/>
**SSWP**: percentage of points that are won when a server makes his second serve <br/>
**EM**: FSP*FSWP-SSP*SSWP <br/>

As described in another repository, previously I collected data on how successfully a player served (FSWP, SSWP, SSP,FSP) against particular opponents in order to identify player matchups in which the player could benefit from a 'risky' strategy. A 'risky' strategy is defined as one where a player hits two first serves on all points instead of the conventional 'safe' strategy of hitting a first serve followed by a second serve. Whether a 'risky' or 'safe' strategy is optimal for a given matchup was found to be heavily opponent-dependent. Thus, making strategy predictions required collecting serving statistics from a player's previous matches against an opponent.

But how can we make strategy predictions when such information is unavailable? This situation arises often when a player is playing a particular opponent for the first time. 

To make predictions for when a player (P1) should go 'risky' against a certain opponent (call him P2), I followed the following roadmap:

1) I first collected match statistics for all player matchups involving P2 with total match lengths greater than 3 (see below for why 3 was chosen as a cutoff). This pool of players will be denoted as P_ML.

2) I then classified each player in P_ML as either 'risky' or 'bold' based on their EM factors with P2. 

3) I then trained various machine learning models using a set of features constructed for each player in P_ML, and then I optimized the model coefficients using cross-validation/grid search methods. I then used each optimized classifier to make predictions on whether P1 (no previous matches with P2) could benefit from a risky strategy. The three classifiers I considered were a Random Forest classifier, a KNN classifier, and a Logistic Regression model, all three of which performed similarly with an accuracy score of ~70% as compared to a score of ~60% for a stratified dummy classifier applied to the same data.  

The process was repeated for each potential opponent who has been ranked in the top 30 and is currently active. Below I give a brief explanation of the various steps involved in the process.

# Classifying which players would benefit from a risky strategy

From previous work, we have two indications of whether a player is likely to benefit from a risky serve strategy against a certain player: 

(1) their matchup EM factor <br/>
(2) the win percentage enhancement expected from switching to a risky strategy as predicted from a Monte Carlo (MC) simulation <br/>

If either of these quantities is > 0, the matchup is classified as benefiting from a risky strategy. 

(2) is more robust than (1) in that the MC simulations consider factors such as player's FSP and SSP distributions as well as subtle effects in the structure of tennis matches. However, the MC simulations are time consuming to produce and were only performed for players with a matchup history of at least five matches with a particular opponent. This reduced sample size also reduces the number of training points that will be available for a machine learning model based on this classification scheme. On the other hand, EM factors may be a less rigorous classification metric but are easy and quick to compute for all matchups in the ATP database. 

Moreover, upon further analysis, 94% of the classifications based on MC predictions match the classifications of the EM metric. Given this agreement, EM factors were used as classification metrics in our ML model to increase training points and eventual ML model accuracy. 

# Producing the machine learning model

Here are the steps taken in producing the machine learning model

**Goal**: Construct a classifier that can predict if a player (P1) who has never played an opponent (P2) will benefit from a risky strategy. 

**Step 1**: Identify all players who have a matchup history of at least 3 matches with P2. Call this set of players P_ML. If size(P_ML) <15, select another P2 to ensure we will have enough ML model training points.  

**Step 2**: Calculate the matchup EM factor of all players in P_ML when playing P2. If EM>0, classify the player as 'risky'; if EM<=0 classify as 'safe'.

**Step 3**: Now that we have our classification vector, we need to produce a list of features that can help predict these classifications. First produce a list of all active top-30 ATP players. Let us refer to this set as P_active. 

Consider the matchups between all players in P_active for a given player in P_ML and compute the following quantities for each matchup:

FSP x FSWP <br/>
SSP x SSWP <br/>

These are the feature values that will define the player. If a player does not have a matchup history with a player in P_active, we use the average values for these quantities, considering all matches the player in P_active has played against active top-30 players. 

Once these features have been computed for every player, store them as rows in a Pandas dataframe. Remove feature columns that contain all zeroes - this effectively removes matchup statistics for players in P_active who have not played anyone in P_ML.

**Step 4**: Split the above data into test and train sets (25/75 split, respectively). Initialize a Logistic Regression (LR) classifier, a Random Forest Classifier (RFC), and a K-nearest neighbors (KNN) classifier and perform a grid search for regularization parameter optimization, benchmarked by accuracy scores. During the grid search use a cv parameter of 3. 

**Step 5**: Train the model on the training data using optimized parameters and then generate an accuracy score for the test data (which has been shielded from model up until this point). <br/>

**Step 6**: Also construct a stratified dummy classifier and apply the classifier to the test data for model evaluation purposes. <br/>

**Step 7**: Repeat process for every P2 within the set P_active


# Evaluating the machine learning model

The following plot displays the test-set accuracy score distribution for the different ML models for all players in P_active as well as the analogous distribution for the simple dummy classifier. As seen from the plot, the ML distributions and the Dummy Classifier are distinct at the 1% confidence level. 

![](/data_visualizations/ML_vs_dummy_accuracy_scores.png?raw=true)

Of course, there are a few situations in which the ML models are outperformed by even the dummy classifier. This is expected since the number of training points can sometimes be ~10, which is insufficient to produce reliable results, especially with feature vector sizes of ~100. As can be seen from the plot, the three ML models produced similar accuracies.

Below, we list the top player matchups where a player who has never played an opponent is expected to benefit from a risky strategy, ranked by the mean classification probability probability from all three machine learning models. 

![](/data_visualizations/model_av_strat_predictions_table.png?raw=true)


# How much match history is needed for the EM metric to be a reliable classifier?

When calibrating our ML model, we relied on matchup-averaged EM factors to make train/test point classifications. But how much match history do you need to determine if a risky strategy is likely optimal? It's unclear whether an EM factor computed after one match is likely to be reflective of the average EM factor that would be obtained from 10+ matches. To address this, I considered matchup data from players who have played each other at least 15 times. 

I calculated the EM factor for each match in the matchup history. Afterwards, I calculated what the match-averaged EM factor was *as a function of the number of matches included in the average* (0-15). I could then compare these individual values to the 15-match-averged EM factor. This gives me a sense of how quickly the EM factors converged to their average value as the match history evolved between two players. I was most interested in determining how many matches it took for the average EM factor to converge to being the same sign (positive or negative) as the 15-match average. This is relevant since whether EM>0 determines the classification for each player in the ML model. 

The following plot displays this convergence curve along with the mean accuracy of the LR model. As can be seen from the plot, after 3 matches, in 70% of cases, the classification prediction has converged to what the eventual 15-match classification would be. 

![](/data_visualizations/strategy_convergence.png?raw=true)

Given that our ML models have a mean accuracy score of 75%, we estimate that if a player has played less than two matches versus a particular opponent, the LR model may provide a better estimate of which strategy is preferable than his own limited match statistics against that player would imply.

As a side note, I calculated the average EM factors here by averaging the EM factor from each individual match in a given matchup. However, in the classification metric used in the ML model, I'm actually considering the EM factor averaged across *all points* that have been played in the matchup. These two quantities are slightly different since there are a different number of points in each match. However, I verified that the two averages produce the same classification ('risky' vs 'safe') in over 90% of cases when considering the full history in a particular matchup, and thus their convergence rates are likely similar as well, although in the future, this point should be addressed. Convergence curves as a function of points played were more difficult to produce for technical reasons and were not pursued here. 

In reality, there are pros and cons to using point-averaged and match-averaged EM factors for classifications; however, point averages were chosen in my approach. 


# Replacing Monte Carlo Modeling: Using Machine Learning instead

As noted above, the Monte Carlo classification is a robust, but inconvenient, method to classify optimal serving strategy.
I therefore tried to replace the MC classification with a Linear Regression ML model. 

I once again considered all top-30 active player matchups (P1 vs P2). For each matchup I calculated the following features

P1 FSP x FSWP <br/>
P1 SSP x SSWP <br/>
P2 FSP x FSWP <br/>
P2 SSP x SSWP <br/>

Each matchup was also classified by a corresponding win percentage for P1. Using the winning percentage as a classifier, I split my data into training and test sets and fit the Linear Regression model to the training data. I then compared the test data predictions to the test data actual winning percentages, resulting in an R^2 value of 0.79, as compared to the MC model R^2 value of 0.85.

![](/data_visualizations/ML_vs_WLR_calibration.png?raw=true)

This machine learning model was extremely quick to implement as compared to the MC method (seconds vs. hours) and produced similarly accurate results. 

Of course, what I would really like do is change the feature values (SSP x SSWP --> FSP x FSWP) to account for risky serving strategies and then see if my model can produce reliable, updated win percentages. 

I performed such a calculation and compared the results to the win percentage difference expected by my MC model. The plot below compares the two results, with an R^2 value of 0.83. Thus, it seems like the machine learning model was able to reproduce the Monte Carlo results with reasonable accuracy while shedding hours of computation time. 

![](/data_visualizations/ML_vs_MC_calibration.png?raw=true)















