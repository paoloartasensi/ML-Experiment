import pandas as pd
from sklearn.linear_model import LogisticRegression
import ftplib
import json
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


url = 'https://dev.paoloartasensi.it/python/csv/last_dataset.csv'
df = pd.read_csv(url)

df = df.drop(['mov', 'prob',], axis=1)
df = df.dropna()

###
# Define the features and target variable
X = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pitch', 'roll', 'BAR', 'totacc']]
y = df['status']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the logistic regression model with cross-validation
clf = LogisticRegression()
scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
clf.fit(X_scaled, y)

# Extract the coefficients and intercepts
coefficients = clf.coef_[0].tolist()
intercept = clf.intercept_.tolist()[0]

# Create the dictionary for the JSON file
data = {
    'ax': coefficients[0],
    'ay': coefficients[1],
    'az': coefficients[2],
    'gx': coefficients[3],
    'gy': coefficients[4],
    'gz': coefficients[5],
    'pitch': coefficients[6],
    'roll': coefficients[7],
    'BAR': coefficients[8],
    'totacc': coefficients[9],
    'intercept': intercept
}

# Save the dictionary as a JSON file
with open('coefficients.json', 'w') as f:
    json.dump(data, f)

# Upload the JSON file to FTP server
ftp_server = ftplib.FTP('185.114.108.114')
ftp_server.login('niotron2023', 'csv_wow_2023!')

with open('coefficients.json', 'rb') as jsonfile:
    ftp_server.storbinary('STOR coefficients.json', jsonfile)
    
ftp_server.quit()

#end of code