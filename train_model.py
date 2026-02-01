import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Generate Data
print("🔄 Generating synthetic training data...")
np.random.seed(42)
n_samples = 10000

data = {
    'attendance': np.random.randint(40, 100, n_samples),
    'marks': np.random.randint(30, 100, n_samples),
    'backlogs': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
    'hours': np.random.randint(1, 30, n_samples),
    'failures': np.random.randint(0, 5, n_samples),
    'year': np.random.choice([1, 2, 3, 4], n_samples),
    'skills': np.random.randint(1, 11, n_samples)
}
df = pd.DataFrame(data)

# 2. Define Logic
def define_risk(row):
    score = 0
    if row['attendance'] < 60: score += 2
    elif row['attendance'] < 75: score += 1
    
    if row['backlogs'] > 1:
        score += 1
        if row['year'] >= 3: score += 1 

    expected_skill = row['year'] * 2
    if row['skills'] < expected_skill: score += 2

    if row['marks'] < 50: score += 1

    if score >= 3: return 'High Risk'
    elif score == 2: return 'Medium Risk'
    else: return 'Low Risk'

df['risk_label'] = df.apply(define_risk, axis=1)

# 3. Train
X = df[['attendance', 'marks', 'backlogs', 'hours', 'failures', 'year', 'skills']]
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save
joblib.dump(model, 'risk_model.pkl')
print(f"✅ Model saved with accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")