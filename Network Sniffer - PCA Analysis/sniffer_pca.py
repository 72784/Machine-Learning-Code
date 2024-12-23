import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ipaddress import ip_address

file_path = 'packet_data_copy.csv' 
data = pd.read_csv(file_path)

def extract_ack(packet_data):
    # Ensure packet_data is a string before applying regex
    if isinstance(packet_data, str):
        match = re.search(r'ACK: (\d)', packet_data)
        if match:
            return int(match.group(1))
    return 0  # Return 0 if no ACK value is found or if packet_data is not a string

cleaned_data = []

for _, row in data.iterrows():
    # Extract packet size (using 'header_length')
    packet_size = row['header_length']
    
    # Convert Source IP to integer (use 'src' column)
    src_ip = ip_address(row['src']).packed
    src_ip_int = int.from_bytes(src_ip, 'big')
    
    # Convert Destination IP to integer (use 'target' column)
    dst_ip = ip_address(row['target']).packed
    dst_ip_int = int.from_bytes(dst_ip, 'big')
    
    # Extract Source Port (using 'packet_data9' column)
    src_port = row['packet_data9'] if 'packet_data9' in row else None
    
    # Extract Destination Port (using 'packet_data10' column)
    dst_port = row['packet_data10'] if 'packet_data10' in row else None
    
    # Collect cleaned feature data
    cleaned_data.append([packet_size, src_ip_int, dst_ip_int, src_port, dst_port])

cleaned_df = pd.DataFrame(cleaned_data, columns=["Packet_Size", "Source_IP", "Destination_IP", "Source_Port", "Destination_Port"])

cleaned_df['ACK'] = data['packet_data13'].apply(extract_ack)


cleaned_df = cleaned_df.dropna()


features = cleaned_df.drop('ACK', axis=1)  # Features excluding 'ACK'
target = cleaned_df['ACK']  # 'ACK' as the target variable


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest (Original Data)")
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled)


X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, target, test_size=0.2, random_state=10)


rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train_pca)


y_pred_pca = rf_pca.predict(X_test_pca)
conf_matrix_pca = confusion_matrix(y_test_pca, y_pred_pca)
disp_pca = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_pca)
disp_pca.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest (PCA Data)")
plt.show()


print("Classification Report (Original Data):")
print(classification_report(y_test, y_pred))

print("Classification Report (PCA Data):")
print(classification_report(y_test_pca, y_pred_pca))
