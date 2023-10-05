# Energy Use Prediction

Accurately predict the total energy use in a low energy building using the UCI data (open source) collected in 2017
●	This being a time series prediction project, we followed an end-to-end solution following a typical ML process including data sourcing, identifying data characteristics, data cleaning, data exploration, feature scaling, feature selection, creation of a performance analysis framework, model pre-processing, parameter optimization, running ML models, model selection and ultimate judgment
●	We carried out various time series specific analyses and explorations such as time series decomposition, lag variables, stationarity analysis and time-series component extraction 
●	Time series prediction models were chosen such as multi-linear regression (MLR), decision-tree regressor (DTR) and multi-layer perceptron (Neural networks) (MLP)
●	We implemented various feature scaling exercises based on model requirements and exploratory analysis. QuantileTransformer and MinMaxScaler are selected as it improves the distribution and range of our data without changing the relationships
●	The dataset is split into 70/30 train/test split, the training dataset is fit into feature scaling process, then the transformation is applied for the testing dataset
●	We created a performance analysis framework focusing on popular and relevant time-series performance metrics such as mean absolute error (MAE), mean absolute percentage error (MAPE), mean squared error (MSE), root mean squared error (RMSE), R-squared score. These were chosen based on target variable type, target variable scale, error values scale (homogeneity) and type of performance aspect (accuracy vs error)
●	GridSearchCV with k-fold cross validation are employed to tune parameters for the DTR model
●	Multi Layer Perceptron (MLP) - base model with selected features was chosen due to the best performance (higher accuracy, less errors); [ultimate judgment]
●	As part of the independent model testing, we focus on similar implementations across research papers/publications to compare the approach and performance

Important and Notable Highlights of The Approach
●	The data was imported as the ‘UCI_data.csv’ from the amazon s3 repository
●	The time-series components such as month, week, dayofweek, dayofmonth, hour, date and typeofday (weekday/weekend) were extracted as new columns
●	Explanatory variables (similar to a data dictionary) are added to the notebook for easy reference aiding in the variable understanding
●	The target column is computed as ‘total energy usage inclusive of appliance as well as light fixtures’ (a slight variant of the datasets found in similar analyses, where the energy usage of appliances and light fixtures were provided as separate target variables)
●	A statistical summary was generated for all the numeric variables (firstly temperature, humidity, then other variables, ending with the target variable [see notebook for detailed findings]
●	Distributions were explored for all variables (in order of temperature, humidity, weather, target)
●	The variables’ distributions were also compared relative to each other (on the same scale) [see notebook for detailed findings]
●	Bivariate analysis on scatter plots, correlation plots/tables and time series bivariates
○	Focus: time components of energy - explore all time components such as trends, seasonality, residuals, lag variables, stationarity etc.
○	Each of the trend, seasonality, residuals, and lag (t-1) variables are added to dataset as part of this analysis and were fed as inputs to the ML models (turned out that they are highly predictive and led to better accuracy)
●	Multiple scaling methods are explored (yeo-johnson, yeo-johnson-standard, quantile, robust, min-max). Quantile and Min-Max are selected.

Choice of ML Models (Rationale and Selection)
●	Common approaches are classical models that work on temporal variation (above time series details) can be used, but we decided to use the additional x variables to predict the energy usage
●	We used supervised ML models which included the below 3 things
○	x variables
○	time components as x variables (month, dayofmonth, hour, dayofweek, typeofday (weekday/weekend)), 7-day average. These have already been extracted during data cleaning and explored during data exploration
○	lagged values (t-1, t-2 values), etc. We have found out that t-1 is a good predictor
●	We also used neural network models (MLP) which can learn the relationships in data and predict energy usage (using its black box, hidden networks architecture)
●	Overall, we will predict using 2 supervised ML models (MLR and DTR) and then predict using a neural network model (MLP). We initially considered RandomForestRegressor, but the hyperparameter tuning took a lot of time, hence we decided to go with a single DTR
●	Base models are created, then incremental models (feature selection and hyperparameter tuning)
●	Finally, one model is chosen based on an ultimate judgment basis performance evaluation

Model Preprocessing
●	Created train-validation data splits (use full data in case of cross-validation). We created train/test datasets in 70/30 proportion and the full data for cross-validation
●	Some helper functions are defined to be used by the DTR model
●	We focused on all popular time series performance metrics, they are set up in the performance evaluation module
○	MAPE and MAE are highly popular performance metrics
○	RMSE is a good error indicator which is not sensitive to scale of error (due to the 'root' computation)
○	R-squared is a good metric which shows whether or not the model is a good fit for the observed values (we chose to use R-squared as the deciding metric))
●	Feature selection
○	Recursive feature elimination (RFE) is run on MLR for highly correlated features
○	We ran base model based on all variables
○	We ran incremental models by changing a combination of the below criteria:
■	We chose features based on exploratory analysis [see notebook for details]
■	We tuned hyperparameters to verify improvements in models

ML Model Implemented
●	We ran 3 ML models for our execution
○	Multi Linear Regression (Simple Supervised Learning Model)
○	Decision Tree Regressor (Simple Supervised Learning Model)
○	Multi Layer Perceptron (Neural Networks)
●	We ran base models for each, and then make incremental models by changing hyperparameters and using feature selection
●	We started off with train-test validation, and then also explored cross-validation after that

Performance Analysis
●	MultiLinearRegression
○	RFE is performed on MLR to extract the feature ranking. Interestingly, some time components which are indicated as insignificant such as hour (rank 24), typeofday (rank 31), etc. all are correlated to the target variable (from EDA and Correlation Matrix). It also happens for some temperature and humidity variables. Similarly, those weather variables are marked as not important [Appendix B table 1 and 2]
○	We firstly trained with all variables except for lagged and trend variables. However, the results obtained were extremely bad (r-squared is 0.14 and high error rate) [Appendix A]
○	Training with lagged and trend variables improved the results significantly (r-squared is 0.91 on test data; MAE, MAPE and RMSE is 17.07, 0.22 and 31.76 respectively)
○	Residuals are calculated (the difference between the real and predicted values), there is no doubt that the relationship between target and other features is not well presented as the residuals are not normally distributed around the horizontal line [Appendix C figure 1]
●	DecisionTreeRegressor
○	As obtained from MLR, better results will be achieved if all variables (including lagged and trend) are employed. We initially run the base model with all variables. The results are better, however, overfitting happens because of perfect results for training data (r-squared is 1.00, others are 0.00) while testing results are not good [Appendix A for more details]
○	We then tuned parameters with k-fold cross validation to identify the appropriate range of learning rates. However, due to time consuming, this process is not well performed. As a result, although the tree complexity is reduced, the incremental model doesn’t improve the result that much (overfitting still happens). There is even no difference when comparing two models using r-squared (0.96), MAE (8.61 and 8.77), MAPE (0.07) and RMSE (20.09 and 20.75)
●	MultiLayerPerceptron
○	We have run four models and usually got high MAPE. This is probably due to few errors when the target variable is scaled to nearly 0. However, other metrics (MAE, RMSE, R-squared) are well employed to evaluate this model
○	It is suggested that the incremental models with more dense layers don’t perform well due to overfitting. Although other performance metrics are pretty well (MAE is 0.08 and RMSE is 0.09 for testing data), the R-squared score is negative (-0.01)
○	Base models completely outperform with lower MAE and RMSE (0.01 on both training and testing dataset). Moreover, the base model on selected features [see Appendix B table 2 for list of features] achieves higher R-squared score compared to that of all variables (0.99 and 0.99 respectively). Probably, those features are correlated to the energy consumption, thereby impacting our model performance

Ultimate Judgment (Final Model Selection)
●	In order to compare the performance of each model, we employed RMSE and R-squared. The best model is the one with lowest RMSE and highest R-squared on both training and testing data
●	MLR shows a pretty high RMSE (33.95 and 31.76 on training and testing data respectively) with lower R-squared score (0.90 and 0.91 for training and testing). Moreover, the relationship between independent and target feature doesn’t fit into a linear model [Appendix C figure 1]
○	Heteroskedasticity happens with an unequal scatter of the residuals. Despite the normal distribution of residuals, some records with extremely high values reduces the performance [Appendix C figure 2]
○	Probably the validity of MLR suffers from outliers
●	DTR is overfitting with RMSE of 20.75 and R-squared of 0.96, which is an improvement over MLR but not by much
○	As obtained from EDA, temperature and and humidity variables are highly correlated among each other. This leads to the problem of multicollinearity when the tree greedily selects the best one out of them, which hides many aspects of the data. Observing the tree plot, some important temperature variables are selected last in the tree [see Notebook for the graphic representation]
○	Although parameter optimization is performed to increase the complexity, it appears that overfitting starts really quick due to regression problem with an increased computational complexity of the tree
●	As we already compared different models of MLP from the Performance Analysis. In this section, the best MLP model (base model with 1 dense layer (1000) with selected features) will be compared with MLR and DTR models. Obviously, it shows a significant reduction of the RMSE compared to MLR and DTR (0.01, 31.76 and 20.75 respectively on testing data). R-squared score is also improved (0.99, 0.96 and 0.91 for MLP, DTR and MLR respectively)
●	Reasons for MLP outperforming other models
○	MLR shows that the a non-linear model is a better choice, MLP can tackle that problem
○	Most of our input variables is continuous, which is more well fit into MLP model
○	MLP is known to provide an efficient mapping of input-output values based on its hidden layers providing a level of abstraction
○	It focuses on optimizing a loss metric, in this case MSE was used. Thus it iterated over 1500 epocs to minimize the error (MSE) and using the sigmoid optimizer
○	We explored multiple variations of different optimizers, learning rates, different count of dense layers and multiple epocs count in the process of identifying a really efficient base model. And running incremental models using selected features provided a better harmony between train-test performance as well as smaller errors
○	Thus, we have chosen ‘MLP base model with selected variables’ to be the most appropriate model to be used in a real world setting [Ultimate judgement]

Independent Model Evaluation (Comparison with Similar Published Projects)

For the purpose of evaluating the relative performance against similar implementations published online, we explored 2 implementations (a total 4 were explored and only 2 are reported due to space constraints)

Implementation 1 (https://www.kaggle.com/code/msand1984/appliance-energy-prediction/notebook) 

Approach similarities:
●	The project uses the same data to predict appliance energy usage using non-temporal models
●	Data characteristics are briefly explored, including missing values and summary statistics
●	Predictive Features are chosen basis of exploratory analysis
●	Time component variables are created (although minimal)
●	Variables distribution are plotted using histograms, feature scaling is carried out
●	Correlation plots are used to identify predictive features
●	Models such as linear regression, random forest regressor and MLP are used

Approach differences & challenges in comparing the models directly:
●	The target variable considered is appliances only (and total, which should include lights)
●	‘Lights’ energy usage is dropped early on, thus the predictions cannot be a direct apples-apples comparison (since our ‘total usage’ includes both appliances and lights usage both
●	Extensive time series variables such as lag, trend, and seasonality are not created (or used)

Performance comparison:
●	A variety of supervised models such as Ridge, Lasso, Ensemble (RFR), Support Vector, Gradient Boosting and MLP were used
●	Best model performance (r-squared) on test data is 0.57 (ExtraTreeRegressor)
●	Linear regression (r-squared) on test data is 0.12
●	MLP (r-squared) on test data is 0.24
●	Our models did significantly better compared to the best models in this approach (owing to better feature selection, feature engineering and hyperparameter tuning)

Implementation 2 (https://www.kaggle.com/code/abdulryhm999/random-forest-model-for-applianc-energy-prediction)

Approach similarities:
●	Predicts appliance usage using supervised ML models on the same data
●	Data preprocessing including null and unique values
●	Extracted various components from datetime such as date, hours, seconds, dayofweek, etc.
●	Explored trends at time component level (day level, dayofweek-hour, etc.)
●	Correlation plots to help with feature selection
●	All the same performance metrics were used to measure performance

Approach differences & challenges in comparing the models directly:
●	Outliers from the target variable were directly removed instead of retaining or imputing the observations
●	Only random forest model was built for the prediction (other models were not tried out)

Performance comparison:
●	Random forest accuracy (r-squared) came up to 0.7 on test data (0.96 on test data)
●	We didn't use a random forest model (ensemble), but used a decision tree regressor (non-ensemble) with hyperparameter tuning. The r-squared came up to 0.96 on test data.
●	Our best model (neural net) provided an accuracy (r-squared) of 0.99 which is better than the best model of this implementation

[2 more research publications were identified as part of research, but not excluded, given the page constraint]

Improvements and Further Analysis
●	Additional ML models can be used to check for better performance (either accuracy or time)
○	Temporal models may do really well as the time components and lag variables showed very high predictive power (likely to take less time than neural nets)
○	Ensemble models can also give higher accuracy (compared to non-MLP models), but we decided against them as they were taking too long
○	Other neural network techniques may provide better results.
○	Additional hyperparameters or hidden layers can be tuned on existing models and possible improvements can be explored (although overfitting can be a concern)
●	Grid search can be implemented to tune a variety of parameters (we only explored for DTR but not much due to execution time constraints)
●	Since the data is telemetric data and is likely collected on a regular basis, there may be scope to get more data to train on and compare the performance of the model (and retrain if needed)
●	Additional code to handle any potential missing values that may come up will surely help smooth running of the models

Appendix A

	MAE	MAPE	RMSE	R-squared
MLR - base model without lag and time components	Train	56.05	0.65	97.09	0.14
	Test	55.00	0.66	95.89	0.14
MLR - base model with all variables	Train	17.65	0.23	33.95	0.90
	Test	17.07	0.22	31.76	0.91
DTR - base model with all variables	Train	0.00	0.00	0.00	1.00
	Test	8.61	0.07	20.09	0.96
DTR - incremental model with all variables	Train	0.00	0.00	0.00	1.00
	Test	8.77	0.07	20.75	0.96
MLP - base model with all variables	Train	0.01	437971635693.26	0.01	0.98
	Test	0.01	129883159560.49	0.01	0.99
MLP - incremental model with all variables	Train	0.08	1145799352179.02	0.09	-0.00
	Test	0.08	381887042512.26	0.09	-0.00
MLP - base model with selected variables	Train	0.01	190147403696.55	0.01	0.99
	Test	0.01	103082365967.75	0.01	0.99
MLP - incremental model with selected variables	Train	0.08	1159663691759.86	0.10	-0.01
	Test	0.08	386507931526.41	0.09	-0.01

Appendix B

1. temp_laundryroom	9. hr_laundryroom	17. temp_ironingroom	25. date
2. temp_parentsroom	10. temp_livingroom	18. temp_officeroom	26. datetime
3. hr_teenagerroom2	11. temp_teenagerroom2	19. temp_dewPoint	27. month
4. hr_kitchen	12. week	20. hr_bathroom	28. dayofmonth
5. hr_livingroom	13. temp_bathroom	21. visibility	29. hr_outsideBuilding
6. hr_ironingroom	14. hr_parentsroom	22. hr_outside	30. hr_officeroom
7. temp_outsideBuilding	15. wind_speed	23. pressure	31. typeofday
8. temp_outside	16. temp_kitchen	24. hour	32. dayofweek
Table 1: Feature ranking from RFE algorithm with MultiLinearRegression

temp_kitchen	temp_teenagerroom2	hr_parentsroom	typeofday
temp_livingroom	temp_parentsroom	hr_teenagerroom2	hour
temp_laundryroom	temp_outside	hr_ironingroom	energy_lag1
temp_officeroom	temp_outsideBuilding	hr_outsideBuilding	energy_trend
temp_bathroom	temp_dewPoint	dayofmonth	energy_seasonality
temp_ironingroom	hr_outside	month	energy_residual
Table 2: Selected features from EDA and Correlation Matrix
