# Decision trees
Your task is to assign movie recommendations using decision trees as classifiers applied for each person separately. You should test two solutions: single decision tree and random forest ensemble. Additionally, since decision trees are interpretable models, you should be able to visualize (draw an image) the induced decision tree. As a result you should submit the following files:

- submission_tree.csv
- submission_forest.csv
- code.zip
- tree.png

The submission_tree.csv file should correspond with task.csv, but all the NULL values should be replaced by predicted movie evaluation values (0, 1, 2, 3, 4 or 5) obtained using single decision tree. The submission_forest.csv file should be result of random forest classifier. Finally, the code.zip file should contain soruce code and resources used to generate submission_tree.csv and submission_orest.csv files basing on training data, while tree.png should present a trained decision tree for an arbitrarily selected person.

Hints:

- You can use all the features gathered in previous task (task_1 with KNN), decision trees are embedded feature selectors, so they should find suitable subsets of features automatically.
- You should prepare your own implementation of decision trees and random forests.
- We are interested only in binary decision trees (the node conditions can only be satisfied or not).
- In case of decision trees you should decide: what condition set is considered, how conditions are selected while the tree is induced and when the induction is stopped.
- In case of random forests you should decide: how the diverisity (independence) of classifieirs in the ensemble is achieved and how the final result is aggregated.
- To assess the generalizing abilities of classifiers, validation or cross-validation procedures can be used.
