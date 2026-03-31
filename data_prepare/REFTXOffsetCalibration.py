#%% md
# # Reference Transmitter Offset Compensation
#%% md
# ## Without Offset Compensation
#%%
!mkdir dichasus
!wget --content-disposition https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/darus-2202/2 -P dichasus # dichasus-0152
#%%
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import json
%matplotlib inline
#%% md
# ### Loading the dataset
#%%
def record_parse_function(proto):
    record = tf.io.parse_single_example(proto, {
        "csi": tf.io.FixedLenFeature([], tf.string, default_value = ''),
        "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value = '')
    })

    csi = tf.io.parse_tensor(record["csi"], out_type = tf.float32)
    csi = tf.signal.fftshift(csi, axes = 1)
    csi = tf.complex(csi[:,:,0], csi[:,:,1])
    pos_tachy = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type = tf.float64), (3))

    # We only care about x/y position dimensions, z is always close to 0
    return csi, pos_tachy[:2]

datafiles = ["dichasus/dichasus-0152.tfrecords"]
dataset = tf.data.TFRecordDataset(datafiles).map(record_parse_function)
dataset = dataset.cache()
#%% md
# ### Filtering: Datapoints around a certain target position
#%%
# Generate filter function that returns true if datapoint was measured somewhere around given target location
def filter_around_point(point, radius):
    def f(csi, pos):
        return tf.norm(pos - tf.constant(point, dtype = tf.float64)) < radius
    return f

# Generate three filter positions
points_left = dataset.filter(filter_around_point((-3.0, 1.0), 0.15))
points_frontal = dataset.filter(filter_around_point((-3.0, 0.0), 0.15))
points_right = dataset.filter(filter_around_point((-3.0, -1.0), 0.15))
#%%
assignments = [
    [28,5,10,14,6,2,16,18],
    [19,4,23,17,20,11,9,27],
    [31,29,0,13,1,12,3,7],
    [30,26,21,25,22,15,24,8]
]

antennacount = np.sum([len(line) for line in assignments])
antennapos = np.zeros((antennacount, 2), dtype = int)

for y in range(len(assignments)):
    for x in range(len(assignments[y])):
        antennapos[assignments[y][x]] = np.asarray((x, y))
#%%
def plot_array_phases(dataset, subcarrier_start, subcarrier_count, title):
    dataset_phases = []

    # Collect the received phases by averaging over the given subcarrier range
    for csi, pos in dataset:
        datapoint_phases = np.sum(csi.numpy()[:, subcarrier_start:subcarrier_start + subcarrier_count], axis = 1)
        datapoint_phases = datapoint_phases * np.conj(datapoint_phases[0])
        dataset_phases.append(datapoint_phases)

    # Assign the received phases to the antennas at the respective positions in the array
    antenna_phases_frontview = np.zeros((len(assignments), len(assignments[0])), dtype = np.complex128)
    for antenna, phase in enumerate(np.sum(dataset_phases, axis = 0)):
        pos = antennapos[antenna]
        antenna_phases_frontview[pos[1], pos[0]] = phase

    plt.title(title)
    plt.xlabel("Antenna Position X")
    plt.ylabel("Antenna Position Y")
    plt.imshow(np.angle(antenna_phases_frontview), cmap = plt.get_cmap("twilight"))
    plt.colorbar(shrink = 0.7)
    plt.show()
#%%
plot_array_phases(points_frontal, 512, 8, "Frontal")
plot_array_phases(points_left, 512, 8, "Left")
plot_array_phases(points_right, 512, 8, "Right")
#%% md
# ## Phase and Timing Offset Compensation
#%% md
# ### Loading Offsets from the provided JSON file
#%%
!wget https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-015x/reftx-offsets-dichasus-015x.json -P dichasus
#%%
offsets = None
with open("dichasus/reftx-offsets-dichasus-015x.json", "r") as offsetfile:
    offsets = json.load(offsetfile)
#%% md
# ### Visualizing the Phase Correction Terms
#%%
sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(1024, dtype = np.float32) / 1024.0, axes = 0)
cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(1024, dtype = np.float32), axes = 0)
correction_phase = (sto_offset + cpo_offset).numpy()
plt.figure(figsize=(8, 5))
plt.xlabel("Subcarrier Index")
plt.ylabel("Phase Correction Term [rad]")
plt.plot(np.transpose(correction_phase))
plt.show()
#%% md
# ### Applying Correction Terms
#%%
def apply_calibration(csi, pos):
    sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
    cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
    csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))
    return csi, pos
#%%
points_left_calibrated = points_left.map(apply_calibration)
points_frontal_calibrated = points_frontal.map(apply_calibration)
points_right_calibrated = points_right.map(apply_calibration)
#%%
plot_array_phases(points_frontal_calibrated, 512, 8, "Frontal")
plot_array_phases(points_left_calibrated, 512, 8, "Left")
plot_array_phases(points_right_calibrated, 512, 8, "Right")
#%% md
# ## Classical Angle of Arrival (AoA) Estimation
#%%
# Apply phase calibration to complete dataset
dataset_calibrated = dataset.map(apply_calibration)
#%%
antenna_distance = 0.118

def estimate_frequency(samples):
    # Kay's Single Frequency Estimator:
    # S. Kay, "A fast and accurate single frequency estimator" in IEEE Transactions on Acoustics, Speech, and Signal Processing,
    # vol. 37, no. 12, pp. 1987-1990, Dec. 1989, doi: 10.1109/29.45547.
    N = len(samples)
    w = (3 / 2 * N) / (N**2 - 1) * (1 - ((np.arange(0, N - 1) - (N / 2 - 1)) / (N / 2))**2)
    return sum([w[t] * np.angle(np.conj(samples[t]) * samples[t + 1]) for t in np.arange(0, N - 1)])

def estimate_angles(dataset, subcarrier_start, subcarrier_count, fcarrier, bandwidth):
    positions = []
    angle_estimates = []
    
    for csi, pos in dataset:
        subcarriers = csi.shape[1]
        csi_mean = tf.math.reduce_sum(csi[:,subcarrier_start:subcarrier_start + subcarrier_count], axis = 1).numpy()
        wavelength = 299792458 / (fcarrier + bandwidth * (-subcarriers / 2 + subcarrier_start + subcarrier_count / 2) / subcarriers)
        phase_diff = np.mean([estimate_frequency(csi_mean[row]) for row in assignments])
        wavelength_diff = phase_diff * wavelength / antenna_distance / (2 * np.pi)

        angle_estimate = None
        if wavelength_diff <= -1:
            angle_estimate = -np.pi / 2
        elif wavelength_diff >= 1:
            angle_estimate = np.pi / 2
        else:
            angle_estimate = np.arcsin(wavelength_diff)

        positions.append(pos.numpy())
        angle_estimates.append(angle_estimate)
        
    return np.vstack(positions), np.hstack(angle_estimates)
#%%
positions, angle_estimates = estimate_angles(dataset_calibrated, 512, 8, 1.272e9, 50e6)

plt.figure(figsize=(10,10))
plt.hexbin(positions[:,0], positions[:,1], C = np.rad2deg(angle_estimates), gridsize = 25)
plt.xlabel("Transmitter Position X")
plt.ylabel("Transmitter Position Y")
plt.colorbar(label = "AoA Estimate [deg]")
plt.show()
#%%
true_angles = np.arctan2(-positions[:,1], -positions[:,0] + 0.2)
est_error = np.rad2deg(np.angle(np.exp(1.0j * angle_estimates) * np.conj(np.exp(1.0j * true_angles))))
plt.xlabel("AoA Estimate Error")
plt.ylabel("Number of Occurences")
plt.hist(est_error, bins = 100)
plt.show()

print("MAE:", np.mean(np.abs(est_error)), "degrees")