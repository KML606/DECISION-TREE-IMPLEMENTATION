# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: KAMAL SAHU

*INTERN ID*: CT04DH499

*DOMAIN*: MACHINE LEARNING

*DURATION* : 4 WEEKS

*MENTOR*: NEELA SANTOSH
Title:
Build and Visualize a Decision Tree Model Using scikit-learn to Classify Student Outcomes

Description:
In this project, I developed a simple machine learning model using Python and the scikit-learn library to classify whether students would pass or fail based on their academic performance. I used Visual Studio Code (VS Code) as the development environment, which provided an efficient setup for writing, debugging, and running the Python code.

The goal was to implement a Decision Tree Classifier that could learn from a dataset of student scores and attendance, and then make predictions on unseen data. The dataset was manually created and kept intentionally small for simplicity and better understanding. It contained ten student records, each with three input features: Math score, Science score, and Attendance rate. The output variable was whether the student passed or failed, labeled as "Yes" or "No".

The code begins by importing necessary libraries including pandas for data handling, matplotlib for plotting, and modules from scikit-learn for building and evaluating the machine learning model. The dataset was then created using a Python dictionary and converted into a DataFrame using pandas.

Once the dataset was prepared, I printed an overview of the data, including the total number of students and how many of them had passed or failed. The features (Math_Score, Science_Score, and Attendance_Rate) were selected and the labels (Passed) were converted from categorical values (Yes/No) into numerical values (1/0) to make them compatible with the machine learning model.

Next, the data was split into training and testing sets using scikit-learn’s train_test_split() function. Seventy percent of the data was used for training and the remaining thirty percent for testing. This ensures that the model is evaluated on data it hasn't seen during training.

I then created and trained a Decision Tree Classifier using the entropy criterion. After training, the model was used to make predictions on the test set, and its accuracy was evaluated using accuracy_score(). The accuracy result indicated how well the model could predict student outcomes based on the test data.

To better understand the internal decision logic of the model, I included a manual explanation of how the decision tree split the data. For example, the tree first checks if the Science score is less than or equal to 46.5. If this condition is true, the model predicts the student will fail. Otherwise, it predicts a pass. This summary helps demystify the "black box" nature of machine learning models and provides insight into how decisions are made.

Finally, I visualized the entire decision tree using plot_tree() from matplotlib. The tree diagram clearly shows the decision splits, feature names, class outcomes, and how samples are classified. This step adds an intuitive and visual understanding of how the model works.

Overall, this project successfully demonstrated how to build, train, and visualize a basic classification model using a real-world scenario. It also highlighted the usefulness of VS Code for machine learning development, as it made the process of writing and running code seamless and interactive.

