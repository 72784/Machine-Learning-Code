import csv
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
from ipaddress import ip_address

# CSV file setup
csv_filename = "packet_data.csv"
csv_header = ["Packet_Size", "Source_IP", "Destination_IP", "Source_Port", "Destination_Port"]

# Write the header to the CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

# Extract features from packets
def extract_features(packet):
    features = []
    if IP in packet:
        features.append(len(packet))  # Packet size in bytes
        features.append(packet[IP].src)  # Source IP as string
        features.append(packet[IP].dst)  # Destination IP as string
        if TCP in packet:
            features.append(packet[TCP].sport)  # Source port
            features.append(packet[TCP].dport)  # Destination port
        elif UDP in packet:
            features.append(packet[UDP].sport)  # Source port
            features.append(packet[UDP].dport)  # Destination port
        else:
            features.extend([None, None])  # Placeholder if no TCP/UDP
    return features


def packet_sniffer(packet):
    print(packet)
    features = extract_features(packet)
    if features:
        # Append the features to the CSV file
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(features)

# Start sniffing
print("Sniffing packets and saving to CSV...")
sniff(prn=packet_sniffer, count=100, timeout=10)

print(f"Packet data saved to {csv_filename}")
