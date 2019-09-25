# ATP_strategy_predictions_sparse_data
I extend my analysis of optimal ATP serving strategies to include prediction for player matchups when data is sparse.
Several of the concepts mentioned in this document have been desribed extensively in the following repository: https://github.com/prateekpuri01/ATP_serving_strategy. Please refer to this page for clarification.

Here is a list of acronyms that will be used in this document:

**FSP** - first serve make percentage <br/>
**SSP** - second serve make percentage <br/>
**FSWP** - percentage of points that are won when a server makes his first serve <br/>
**SSWP** - percentage of points that are won when a server makes his second serve <br/>
**EM** - FSP*FSWP-SSP*SSWP <br/>

As detailed in the above-linked repository, previously I collected data on how well a particular player served (FSWP, SSWP, SSP,FSP) against all other players he has played against in his career in order to identify player matchups in which the player could benefit from a 'risky' strategy. A 'risky' strategy is defined as one where a player hits two first serves on all points instead of the conventional 'safe' strategy of hitting a first serve followed by a second serve. The optimal choice of strategy was found to be heavily opponent-dependent, and thus, this analysis required pooling a player's previous match statistics against other players in order to make a prediction. 

But how can we make predictions when such information is unavailable?

To make predictions for when a player (P1) should go 'risky' against a certain opponent (call him P2), I followed the following roadmap:

1) I first collected matchup data for all matchups with match lengths greater than 3 involving P2 (see below for why 3 was chosen as a cutoff). This pool of players will be denoted as P_ML.

2) I then classified each player in P_ML as either 'risky' or 'bold' based on their EM factors with P2. 

3) I then trained a Logistic Regression model using a set of features obtained for each player in P_ML, and then optimized the model coefficients using cross-validation/grid search methods. I then used the optimized classifier to make predictions on whether P1 (no previous matches with P2) could benefit from a risky strategy. In particular, my model identified players in P_ML who had features similar to P1, and then classified P1 based on the classifications of this subset of P_ML. 

The process was repeated for each potential opponent who has been ranked in the top 30 and is currently active. Below I give a brief explanation of the various steps involved in the process.

# Classifying which players would benefit from a risky strategy

From previous work, we have two indications of whether a player is likely to benefit from a risky serve strategy against a certain player: 

(1) their matchup EM factor <\br>
(2) the win percentage enhancement expected from switching to a risky strategy as predicted from a Monte Carlo (MC) simulation <\br>

If either of these quantities is > 0, the matchup is classified as benefiting from a risky strategy. 

(2) is more robust than (1) in that the MC simulations consider factors such as player's first and second serve make percentages and subtle effects in the structure of tennis matches. However, the MC simulations are time consuming to produce and were only performed for players with a matchup history of at least five matches with a particular opponent. The reduced sample size also reduces the number of training points that will be available for our machine learning model. On the other hand, EM factors may be a less rigorous classification metric but are easy and quick to compute for all matchups in the ATP database. 

Moreover, upon further analysis, 94% of the classifications based on MC predictions match the classifications of the EM predictions. Given this agreement, EM factors were used for classification to increase training points and eventual ML model accuracy. 

# Producing the machine learning model

Here are the steps taken in producing the machine learning model

**Goal**: Construct a classifier that can predict if a player (P1) who has never played an opponent (P2) will benefit from a risky strategy. 

**Step 1**: Identify all players who have a matchup history of at least 3 matches with P2. Call this set of players P_ML. If size(P_ML) <15, select another P2 to ensure we will have enough training points.  

**Step 2**: Calculate the matchup EM factor of all players in P_ML when playing P2. If EM>0, classify the player as 'risky'; if EM<=0 classify as 'safe'.

**Step 3**: Now that we have our classification vector, we need to produce a list of features that can help predict these classifications. First produce a list of all active top-30 ATP players. Let us refer to this set as P_active. 

Consider the matchups between all players in P_active for a given player in P_ML and compute the following quantities for each matchup:

FSP*FSWP <br/>
SSP*SSWP <br/>

These values are the features that will define the player. If a player does not have a matchup history with a player in P_active, list these values as zero. 

Once these features have been computed for every player, remove feature columns that contain all zeroes - this effectively removes matchup statistics for players in P_active who have not played anyone in P_ML.

**Step 4**: Split the above data into test and train sets (25/75 split, respectively). Initialize a Logistic Regression (LR) classifier, and perform a grid search for regularization paramter optimization, benchmarked by accuracy scores. During the grid search use a cv parameter of 3. 

**Step 5**: Train the model on the training data using optimized parameters and then generate an accuracy score for the test data (which has been shielded from model up until this point)
.
**Step 6**: Also construct a stratified dummy classifier and apply the classifier to the test data for model evaluation purposes. 

**Step 7**: Repeat process for every P2 within the set P_active


# Evaluating the machine learning model

The following plot demonstrates the test-set accuracy scores for the LR models for all players in P_active as well as the scores for the simple dummy classifier. As seen from the plot, the two distributions are distinct at the 1% confidence level. 

![](/data_visualizations/LR_vs_dummy_accuracy_score.png?raw=true)

Of course, there are a few situations in which the LR is outperformed by even the dummy classifier. This is expected since the number of training points can sometimes be ~10, which is insufficient to produce reliable results, especially with feature vector sizes of ~100. 

However, by selecting models who outperformed the dummy classifier by at least 10% (accuracy score), we can list the top 5 player matchups where a player is expected to benefit from a risky strategy, ranked by LR classification probability. Also, I restricted this list to include only models where the train score and the test score were within 10% of one another. This was done to reduce the presence of overfit models since overfitting is a concern here, especially when the number of training points is less than 20 or so. 
![](/data_visualizations/strat_predictions_table.png?raw=true)

To get a better sense for the model, we can also inspect an AUC curve for a sample player in our dataset, Kei Nishikori.

![](/data_visualizations/risk_strat_predict_ROC.png?raw=true)

# How to determine how much match history is needed for the EM metric to be a reliable classifier?

During our classificaiton, we relied on matchup-averaged EM factors. But how much match history do you need to determine if a risky strategy is likely optimal? It's unclear whether an EM factor computed after one match is likely to be reflective of the EM factor that would be obtained from 10+ matches. To address this, I first restricted my data to players who have played each other 15 times. 

I then calculated the EM factor for each match in the matchup history. Afterwards, I calculated what the matchup averaged EM factor was *as a function of the number of matches included in the average* (0-15). I could then compare these individual values to the 15-match-averged EM factor. This gives me a sense of how quickly the EM factors converged to their average value as the match history evolved between two players. I was most interested in determining how many matches it took for the average EM factor to converge to being the same sign (positive or negative) as the 15-match average. This is relevant since whether EM>0 determines the classification for each player in the ML model. 

The following plot displays this convergence curve along with the mean accuracy of the LR model. As can be seen from the plot, after 3 matches, in 70% of cases, the classification prediction has converged to what the eventual 15-match classification would be. 

![](/data_visualizations/strategy_convergence.png?raw=true)

Given that our ML models have a mean accuracy score of 75%, we can estimate that if a player has played less than two matches versus a particular opponent, the LR model may provide a better estimate of which strategy is preferable than his own limited match statistics against that player would imply.

As a side note, I calculated the EM factors here by averaging the EM factor from each individual match in a given matchup. However, in the classification metric used in the ML model, I'm actually considering the EM factor averaged across *all points* that have been played in the matchup. These two quantities are slightly different since there are a different number of points in each match. However, I verified that the two averages produce the same classification ('risky' vs 'safe') in over 90% of cases when considering the full history in a particular matchup, and thus their convergence rates are likely fairly similar as well. Convergence curves as a function of points played were more difficult to produce for technical reasons and were neglected here. 

There are pros and cons to using point-averaged and match-averaged EM factors for classificaitons; however point averages were chosen in my approach. 


# Replacing Monte Carlo Modeling: Using Machine Learning instead

As noted above, the Monte Carlo classification is a fairly robust, but inconvenient, method to classify optimal serving strategy.
I therfore tried to replace the MC classification with a Linear Regression ML model. 

I once again considered all top-30 active player matchups (P1 vs P2). For each matchup I calculated the following features

P1 FSP*FSWP
P1 SSP*SSWP
P2 FSP*FSWP
P2 SSP*SSWP

Each matchup was also classified by a corresponding win percentage for P1. Using the winning percentage as a classifier, I split my data into training and test sets and fit the Linear Regression model to the training data. I then compared the test data predictions to the test data actual winning percentages, resulting in an R^2 value of 0.79, as compared to the MC model R^2 value of 0.85.

![](/data_visualizations/ML_vs_WLR_calibration.png?raw=true)

This machine learning model was extremely quick to implement as compared to the MC method (seconds vs. hours) and produced similarly accurate results. 

Of course, what I would really like do is change the feature values (SSP*SSWP->FSP*FSWP) to account for risky serving strategies and then see if my model can produce reliable, updated win percentages. 

I performed such a calculation and compared the results to the win percentage difference expected by my MC model. The plot below compares the two results, with an R^2 value of 0.83. Thus, it seems like the machine learning model was able to reproduce the Monte Carlo results with reasonable accuracy while shedding hours of computation time. 

![](/data_visualizations/ML_vs_MC_calibration.png?raw=true)















