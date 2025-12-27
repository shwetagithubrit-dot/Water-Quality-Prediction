import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
df = pd.read_csv("water_pre.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]
# Build pipeline with:
# 1) Imputer -> fills NaN values
# 2) StandardScaler -> scales data
# 3) SVM classifier
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved successfully with NaN handling!")

