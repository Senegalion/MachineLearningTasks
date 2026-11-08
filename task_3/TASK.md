# Person similarity
Your task is to assign movie recomendations basing on similarity between persons. In this approach you should first prepare a method allowing to compare people basing on their existing evaluations of the movies. Peaople are the more similar the more similar evaluations they have for the same movies. Such a similarity measure should allow you to identify persons of similar movie taste. To find the evaluatoin for some new movie you should use evaluations of the people with similar taste (proper aggregation method should be proposed). As a result you should submit the following files:

- submission.csv
- code.zip
- The submission.csv file should correspond with task.csv, but all the NULL values should be replaced by predicted movie evaluation values (0, 1, 2, 3, 4 or 5). The code.zip file should contain soruce code and resources used to generate submission.csv file basing on training data.

Hints:

- In this task movie features are not taken into account, only their evaluations.
- Person similiarity can be only evaluated basing on movies that both persons have seen.
- Persons of similar taste can be used to predict evaluation only if they have seen the analysed movie.
- To assess the generalizing abilities of the considered method (e.g. select hyperparameters), validation or cross-validation procedures should be used.


https://realpython.com/build-recommendation-engine-collaborative-filtering/#memory-based