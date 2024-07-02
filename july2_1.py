import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    def __init__(self, sig_r, sig_a, sig_e_sqr):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.zeros((3, 3))  # Measurement noise covariance
        self.R[0, 0] = sig_r**2 * np.cos(self.Sf[4])**2 * np.sin(self.Sf[3])**2 + self.Sf[0]**2 * np.cos(self.Sf[4])**2 * np.cos(self.Sf[3])**2 + sig_a**2 + self.Sf[0]**2 * np.sin(self.Sf[4])**2 * np.sin(self.Sf[3])**2 * sig_e_sqr
        self.R[1, 1] = sig_r**2 * np.cos(self.Sf[4])**2 * np.cos(self.Sf[3])**2 + self.Sf[0]**2 * np.cos(self.Sf[4])**2 * np.sin(self.Sf[3])**2 + sig_a**2 + self.Sf[0]**2 * np.sin(self.Sf[4])**2 * np.cos(self.Sf[3])**2 * sig_e_sqr
        self.R[2, 2] = sig_r**2 * np.cos(self.Sf[4])**2 * np.sin(self.Sf[3])**2 + self.Sf[0]**2 * np.cos(self.Sf[4])**2 * np.cos(self.Sf[3])**2 + sig_a**2 + self.Sf[0]**2 * np.sin(self.Sf[4])**2 * np.sin(self.Sf[3])**2 * sig_e_sqr
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)
        
    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sp:", self.Sp)
        print("Pp:", self.Pp)

    def update_step(self, Z, r, e, a):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("Pf:", self.Pf)

    def gating(self, Z, r, e, a):
        Inn = Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        d2 = np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold

def generate_hypotheses(clusters, targets):
    hypotheses = []
    for cluster in clusters:
        for target in targets:
            hypotheses.append((cluster, target))
    return hypotheses

def compute_hypothesis_likelihood(hypothesis, filter_instance):
    cluster, target = hypothesis
    Z = np.array([[cluster[0]], [cluster[1]], [cluster[2]]])
    Inn = Z - np.dot(filter_instance.H, target)
    S = np.dot(filter_instance.H, np.dot(filter_instance.Pf, filter_instance.H.T)) + filter_instance.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood

def jpda(measurements, targets, filter_instance):
    clusters = []
    for measurement in measurements:
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if filter_instance.gating(Z, measurement[0], measurement[1], measurement[2]).item():
            clusters.append(measurement)
    hypotheses = generate_hypotheses(clusters, targets)
    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, filter_instance) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)
    
    # Handle division by zero
    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]

    best_hypothesis = hypotheses[np.argmax(marginal_probabilities)]
    return best_hypothesis

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x) * 180 / np.pi
    if az < 0:
        az += 360
    return r, az, el

def cart2sph2(x, y, z, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / np.pi)
        az.append(math.atan2(y[i], x[i]) * 180 / np.pi)
        if az[i] < 0:
            az[i] += 360
    return r, az, el

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((x, y, z, mt))
    return measurements

def form_measurement_groups(measurements, max_time_diff=50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]
    
    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]
    
    if current_group:
        measurement_groups.append(current_group)
        
    return measurement_groups

def chi_square_clustering(group, filter_instance):
    clusters = []
    for i, measurement in enumerate(group):
        Z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]])
        if filter_instance.gating(Z, measurement[0], measurement[1], measurement[2]).item():
            clusters.append(measurement)
    return clusters

# Define measurement noise parameters
sig_r = 30
sig_a = 5 / 1000  # 5 milliradians to radians
sig_e_sqr = 5 / 1000  # 5 milliradians to radians

# Create an instance of the CVFilter class
kalman_filter = CVFilter(sig_r, sig_a, sig_e_sqr)

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_52_test.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Form measurement groups based on time
measurement_groups = form_measurement_groups(measurements)

csv_file_predicted = "ttk_52_test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values
measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Initial targets list
targets = []

# Iterate through measurement groups
for group in measurement_groups:
    for i, (x, y, z, mt) in enumerate(group):
        r, az, el = cart2sph(x, y, z)
        if i == 0:
            kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            targets.append(kalman_filter.Sf)
        elif i == 1:
            Z = np.array([[x], [y], [z]])
            if kalman_filter.gating(Z, x, y, z):
                prev_x, prev_y, prev_z = group[i-1][:3]
                dt = mt - group[i-1][3]
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                vz = (z - prev_z) / dt
                kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
                targets.append(kalman_filter.Sf)
            else:
                continue  # Move to the next measurement if gating test fails
        else:
            clusters = chi_square_clustering(group, kalman_filter)
            if clusters:
                best_hypothesis = jpda(clusters, targets, kalman_filter)
                Z = np.array([[best_hypothesis[0][0]], [best_hypothesis[0][1]], [best_hypothesis[0][2]]])
                
                kalman_filter.predict_step(mt)
                kalman_filter.update_step(Z, r, el, az)

                # Append data for plotting
                time_list.append(mt)
                r_list.append(r)
                az_list.append(az)
                el_list.append(el)

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[0], label='filtered range (track id 31)', color='red', marker='*')
plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1], label='measured range (track id 31)', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()
