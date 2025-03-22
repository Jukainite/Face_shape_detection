
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib



# Read from CSV
train_df = pd.read_csv("Train_data.csv")
test_df = pd.read_csv("Test_data.csv")

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)




# Separate features and labels, convert labels to numbers if needed

features = train_df.columns.difference(['label']).tolist()


# Use LabelEncoder for encoding the label
le = LabelEncoder()
y_train = le.fit_transform(train_df['label'])
y_test = le.transform(test_df['label'])

X_train = train_df[features].values
X_test = test_df[features].values

print("Labels:", le.classes_)


# Training and Evaluation of Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)

rf_model.fit(X_train, y_train)

# Prediction on test set
y_pred_rf = rf_model.predict(X_test)

#evaluation Metrics
print("=== Random Forest Classifier ===")
print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred_rf)))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))



rf_model_filename = "models/rf_model.pkl"


#Save the trained model
# joblib.dump(rf_model, rf_model_filename)

# joblib.dump(le, "models/label_encoder.pkl")
# print(f"Saved Random Forest model as: {rf_model_filename}")