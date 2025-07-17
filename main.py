import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = {
    'Math_Score': [85, 42, 78, 60, 90, 55, 40, 72, 88, 33],
    'Science_Score': [80, 45, 75, 58, 95, 50, 35, 70, 92, 30],
    'Attendance_Rate': [90, 60, 88, 70, 95, 65, 50, 85, 93, 45],
    'Passed': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
total_students = len(df)
passed_count = df['Passed'].value_counts().get('Yes', 0)
failed_count = df['Passed'].value_counts().get('No', 0)

print("STUDENT DATASET OVERVIEW")
print("----------------------------")
print(df)
print(f"\nTotal students: {total_students}")
print(f"Passed: {passed_count}")
print(f"Failed: {failed_count}")
X = df[['Math_Score', 'Science_Score', 'Attendance_Rate']]
y = df['Passed'].map({'Yes': 1, 'No': 0})  # Convert Yes/No to 1/0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTRAINING/TEST SPLIT")
print("----------------------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("----------------------------")
print(f"Predicted: {list(y_pred)}")
print(f"Actual:    {list(y_test.values)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDECISION TREE LOGIC SUMMARY")
print("--------------------------------")
print("Root split: Science_Score <= 46.5")
print("If True (Science <= 46.5): -> Class = Fail")
print("If False (Science > 46.5): -> Class = Pass")
print("\nSamples classified:")
print("Left Node -> 2 students (2 Fail, 0 Pass)")
print("Right Node -> 5 students (0 Fail, 5 Pass)")
plt.figure(figsize=(12, 8))
plot_tree(model,
          feature_names=['Math_Score', 'Science_Score', 'Attendance_Rate'],
          class_names=['Fail', 'Pass'],
          filled=True)
plt.title("Decision Tree - Student Pass/Fail Prediction")
plt.show()
