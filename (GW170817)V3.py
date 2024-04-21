from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
import numpy as np
import matplotlib.pyplot as plt

def gravitational_wave_freq(t, Mc, tc):
    # Ensure that Mc is positive to avoid division by zero or negative roots
    if Mc <= 0:
        return np.nan
    term = (tc - t)
    mask = term > 0  # Only evaluate the model where (tc - t) is positive
    freq = np.empty_like(term)
    freq[~mask] = np.nan  # Set non-physical values to NaN
    freq[mask] = (1/np.pi) * ((5/256) * (1/Mc**(5/3)) * term[mask]**(-5/8))**(3/8)
    return freq

# Fetch the data as before...
event_time = 1187008882.4
start = event_time - 30
end = event_time + 30
data = TimeSeries.fetch_open_data('L1', start, end)

# Whitening the data
whitened_data = data.whiten()

# Refining the band-pass filtering parameters
filtered_data = whitened_data.bandpass(30, 300)

# Apply a tukey window to reduce edge effects
tukey_window = tukey(data.size, alpha=1.0/8)
windowed_data = filtered_data * tukey_window

# Perform a Q-transform with optimized parameters
q_transform = windowed_data.q_transform(frange=(30, 300), qrange=(100, 110))

# Extract times and frequencies from the Q-transform object
times = q_transform.times.value

# Determine the time window of interest (850s to 880s after the event time)
time_window_mask = (times >= event_time - 30) & (times <= event_time)
selected_times = times[time_window_mask]
selected_q_transform = q_transform[time_window_mask]

# Check if selected times are empty
if selected_times.size == 0:
    raise ValueError("No data found in the selected time window. Check your event_time and time window bounds.")

# Extract the frequencies corresponding to the peak energy at each time step
peak_frequencies = np.array([row.argmax() for row in selected_q_transform])

# Initialize RANSAC regressor with appropriate settings
ransac = RANSACRegressor(residual_threshold=30)  # Adjust this threshold as needed

# Fit the RANSAC regressor to find inliers
ransac.fit(selected_times.reshape(-1, 1), peak_frequencies)

# Get the inlier mask and filter the data points
inlier_mask = ransac.inlier_mask_
inlier_times = selected_times[inlier_mask]
inlier_freqs = peak_frequencies[inlier_mask]

# Now use curve_fit on the inliers to fit your non-linear model
initial_guess_Mc = 1.4  # Starting guess for the chirp mass
# Ensure tc is greater than any inlier time for the model to be valid
initial_guess_tc = np.max(inlier_times) + 1  # One second after the last inlier time

# Perform the curve fitting, making sure that the bounds allow tc to be greater than any inlier time
params, params_covariance = curve_fit(
    gravitational_wave_freq,
    inlier_times,
    inlier_freqs,
    p0=[initial_guess_Mc, initial_guess_tc],
    bounds=([1, np.max(inlier_times) + 0.1], [3, np.max(inlier_times) + 30])  # Example bounds, adjust as needed
)

# Plot the RANSAC inliers and the curve fit
plt.figure(figsize=(10, 6))
plt.scatter(selected_times, peak_frequencies, label='Data', color='grey', alpha=0.5)
plt.scatter(inlier_times, inlier_freqs, label='Inliers', color='blue')
plt.plot(selected_times, gravitational_wave_freq(selected_times, *params), label='Fitted function', color='red')
plt.xlabel('Time (s) from event')
plt.ylabel('Frequency (Hz)')
plt.title('RANSAC Fit of Gravitational Wave Frequency Model to Data')
plt.legend()
plt.show()

print("Fitted parameters (Chirp Mass, Coalescence Time):", params)

# Define the event time for GW170817
event_time = 1187008882.4  # GW170817 event time
start2 = event_time - 2048  # Start time of the 2048 seconds of data before the event
end2 = event_time           # End time is the event time

# Use the fetch_open_data method to get data from the LIGO Hanford and Livingston detectors
data_hanford = TimeSeries.fetch_open_data('H1', start2, end2)
data_livingston = TimeSeries.fetch_open_data('L1', start2, end2)

# Compute the Amplitude Spectral Density (ASD) for both Hanford and Livingston
# Amplitude spectral density (ASD) is the square root of the power spectral density (PSD),
asd_hanford = data_hanford.asd(fftlength=16, method='median')
asd_livingston = data_livingston.asd(fftlength=16, method='median')

# Plot the ASD for both LIGO detectors
plt.figure(figsize=(10, 6))
plt.loglog(asd_hanford.frequencies, asd_hanford, label='LIGO Hanford')
plt.loglog(asd_livingston.frequencies, asd_livingston, label='LIGO Livingston')

# The horizontal axis represents the frequency of the noise in the detectors, measured in Hertz (Hz). 
plt.xlim(10, 2048)  # Focus on the most sensitive frequency band. 
# The vertical axis shows the strain sensitivity of the detectors, also on a logarithmic scale.
# It is measured in units of strain per square root of Hz (Hz^{-1/2}). 
# Lower values mean the detector is more sensitive to potential gravitational wave signals at that frequency.

plt.ylim(1e-24, 1e-19)  # Set y-axis limits to reasonable ASD values
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude Spectral Density [Hz$^{-1/2}$]')
plt.title('LIGO Sensitivity Curve around GW170817')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.show()

# Save the filtered time series as an audio file
whitened_data.write("GW170817_audio.wav", format='wav')
