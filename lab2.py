# Simple ML program: Iris flower classification

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X = iris.data      # Features: measurements of petals & sepals
y = iris.target    # Labels: flower species

# 2. Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Try predicting a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Sepal length, sepal width, petal length, petal width
prediction = model.predict(sample)
print("Predicted class:", iris.target_names[prediction][0])