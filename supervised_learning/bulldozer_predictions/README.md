<h1>Prediction Bulldozer Prices</h1>

The prediction of bulldozer pricing is a popular dataset found on Kaggle that was part of a 2012 competition. I decided
to try this one out to become more familiar with datasets that have mixed variables (i.e. numerical, categorical). I followed along
the Udemy course `Complete Machine Learning & Data Science Bootcamp 2021` for part of this project. However, my preprocessing
of the data was partially different. For example, I used both ordinal categorical data, as well as one hot encoded
categorical data. Furthermore, I manually ordered the ordinal data to reflect what I beleive the data represents (i.e small -> medium -> large). 
Lastly, I didn't use specific variables that I thought wouldn't add much to the model, such as "machine id's". All this was to experiement
with the data and produce better results.

<h2>Installation</h2>

cd to the directory that contains the requirements.txt, activate a virtual environment and run the following in your terminal or command prompt.

`pip install -r requirements.txt`

Note: I used Python 3.8.

<h2>Usage</h2>

Simply run the main method to see the results. 

My Scoring Results <br>
Training Set: RMSLE 0.0924 <br>
Validation Set: RMSLE 0.2704

Note: Root Mean Squared Log Error is the error used in the competition.