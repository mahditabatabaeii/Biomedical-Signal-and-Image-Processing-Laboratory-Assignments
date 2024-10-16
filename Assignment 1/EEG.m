% Load the .mat file
load('Lab 1_data/EEG_sig.mat');

%% Part 1

% Extract the sampling frequency and channel labels from the 'des' variable
fs = des.samplingfreq;            % Sampling frequency (Hz)
channel_labels = des.channelnames;  % Channel labels

% Extract the signal data for the 5th channel (row 5 of matrix Z)
eeg_channel_5 = Z(5, :);

% Generate time vector in seconds
time = (0:length(eeg_channel_5)-1) / fs;

% Plot the signal for the 5th channel
figure;
plot(time, eeg_channel_5);

% Set axis labels and title
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title(['EEG Signal - Channel: ', channel_labels{5}]);

% Adjust the figure size manually to better show the signal's variation over time
set(gcf, 'Position', [100, 100, 1000, 400]);  % Adjust figure size

% Display the plot
grid on;

%% Part 2

% Define time intervals for analysis
intervals = {[0 15], [18 40], [45 50], [50 max(time)]};  % Time intervals in seconds
interval_labels = {'0-15s', '18-40s', '45-50s', '50s to end'};

% Plot each time interval
figure;
for i = 1:length(intervals)
    % Find the indices corresponding to the time range
    idx_range = find(time >= intervals{i}(1) & time <= intervals{i}(2));
    
    % Extract the corresponding signal and time
    time_interval = time(idx_range);
    signal_interval = eeg_channel_5(idx_range);
    
    % Plot the signal for the given time interval
    subplot(length(intervals), 1, i);
    plot(time_interval, signal_interval);
    xlabel('Time (s)');
    ylabel('Amplitude (\muV)');
    title(['EEG Signal - Channel: ', channel_labels{5}, ' (', interval_labels{i}, ')']);
    
    % Adjust plot appearance
    grid on;
end

% Adjust figure size
set(gcf, 'Position', [100, 100, 1000, 800]);  % Adjust figure size

%% Part 3

eeg_channel_10 = Z(10, :);  % Change 10 to any other channel number if needed

% Plot the entire signal for the 5th channel
figure;
subplot(2, 1, 1);  % First plot for 5th channel
plot(time, eeg_channel_5);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title(['EEG Signal - Channel 5 (', channel_labels{5}, ')']);
grid on;

% Plot the entire signal for the 10th channel
subplot(2, 1, 2);  % Second plot for 10th channel
plot(time, eeg_channel_10);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title(['EEG Signal - Channel 10 (', channel_labels{10}, ')']);
grid on;

% Adjust figure size
set(gcf, 'Position', [100, 100, 1000, 600]);  % Adjust figure size

%% Part 4

% Define parameters for the display function
offset = max(max(abs(Z))) / 5;  % Offset to separate the signals for clarity
feq = 256;  % Frequency for displaying the time axis (if not provided, set it manually)

% Call the display function to plot all channels with clear labeling
disp_eeg(Z, offset, feq, channel_labels);
xlim('tight')
% Customize the figure for a more visually appealing output
title('EEG Signal Display - All Channels');
xlabel('Time (s)');
ylabel('Amplitude (\muV) with Offset for Each Channel');

grid on;
set(gca, 'FontSize', 12);  % Set font size for better readability

% Adjust figure size for better presentation
set(gcf, 'Position', [100, 100, 1200, 800]);  % Make the figure larger for easier viewing

%% Part 6

% Time vector in seconds
time = (0:length(eeg_channel_5)-1) / fs;

% Define the time intervals in seconds (start and end times)
intervals = [2 7; 30 35; 42 47; 50 55];  % Four intervals, each 5 seconds long

% Define the number of FFT points and frequency vector for DFT
nfft = 1024;  % Number of points for FFT
freq = (0:nfft-1)*(fs/nfft);  % Frequency vector

% Plot the time-domain and frequency-domain signals for each interval
figure;
for i = 1:size(intervals, 1)
    % Find the indices corresponding to the time range
    idx_range = find(time >= intervals(i, 1) & time <= intervals(i, 2));
    
    % Extract the corresponding signal and time for the current interval
    time_interval = time(idx_range);
    signal_interval = eeg_channel_5(idx_range);
    
    % Compute the Discrete Fourier Transform (DFT) of the signal
    signal_dft = fft(signal_interval, nfft);
    
    % Plot the time-domain signal
    subplot(4, 2, 2*i-1);  % Odd-numbered subplots for time-domain signals
    plot(time_interval, signal_interval);
    xlabel('Time (s)');
    ylabel('Amplitude (\muV)');
    title(['Time-Domain Signal (C3) - Interval ', num2str(intervals(i, 1)), ' to ', num2str(intervals(i, 2)), ' s']);
    grid on;
    
    % Plot the frequency-domain signal (magnitude of DFT)
    subplot(4, 2, 2*i);  % Even-numbered subplots for frequency-domain signals
    plot(freq(1:nfft/2), abs(signal_dft(1:nfft/2)));  % Plot up to Nyquist frequency (fs/2)
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title(['Frequency-Domain Signal (C3) - Interval ', num2str(intervals(i, 1)), ' to ', num2str(intervals(i, 2)), ' s']);
    grid on;
end

% Adjust figure size for better viewing
set(gcf, 'Position', [100, 100, 1200, 800]);  % Make the figure larger

%% Part 7

% Define parameters for Welch's method (pwelch)
window_size = 256;  % Window size for pwelch
noverlap = window_size / 2;  % Overlap between segments
nfft = 512;  % Number of FFT points

% Plot the frequency-domain signals for each interval using pwelch
figure;
for i = 1:size(intervals, 1)
    % Find the indices corresponding to the time range
    idx_range = find(time >= intervals(i, 1) & time <= intervals(i, 2));
    
    % Extract the corresponding signal for the current interval
    signal_interval = eeg_channel_5(idx_range);
    
    % Compute and plot the power spectral density using pwelch
    subplot(4, 1, i);
    [pxx, f] = pwelch(signal_interval, window_size, noverlap, nfft, fs);
    
    % Plot the PSD
    plot(f, 10*log10(pxx));  % Convert to decibels
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    title(['PSD - Interval ', num2str(intervals(i, 1)), ' to ', num2str(intervals(i, 2)), ' s']);
    grid on;
    
    % Set x-axis limit to show useful frequency range (0 to 60 Hz)
    xlim([0 128]);  % Adjust based on the relevant frequency range
end

% Adjust figure size for better viewing
set(gcf, 'Position', [100, 100, 1200, 800]);  % Make the figure larger

%% Part 8

% Define parameters for spectrogram
window_length = 128;  % Window length (L)
noverlap = 64;        % Number of points in overlap (N_overlap)
nfft = 128;           % Number of DFT points (nfft)
window = hamming(window_length);  % Use Hamming window

% Plot the time-frequency spectrum (spectrogram) for each interval
figure;
for i = 1:size(intervals, 1)
    % Find the indices corresponding to the time range
    idx_range = find(time >= intervals(i, 1) & time <= intervals(i, 2));
    
    % Extract the corresponding signal for the current interval
    signal_interval = eeg_channel_5(idx_range);
    
    % Compute and plot the spectrogram
    subplot(2, 2, i);
    spectrogram(signal_interval, window, noverlap, nfft, fs, 'yaxis');
    
    % Set axis labels and title
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(['Spectrogram (Channel 5 - C3) - Interval ', num2str(intervals(i, 1)), ' to ', num2str(intervals(i, 2)), ' s']);
    colorbar;  % Display colorbar to indicate power magnitude
    ylim([0 60]);  % Limit frequency axis to 60 Hz (relevant EEG frequencies)
end

% Adjust figure size for better viewing
set(gcf, 'Position', [100, 100, 1200, 800]);  % Make the figure larger

%% Part 9

% Define the second interval (30 to 35 seconds)
start_idx = round(30 * fs) + 1;
end_idx = round(35 * fs);
signal_interval = eeg_channel_5(start_idx:end_idx);

% Design a low-pass Butterworth filter
cutoff_freq = 64; % Cutoff frequency in Hz (below Nyquist of the downsampled rate)
order = 4;  % Filter order
[b, a] = butter(order, cutoff_freq/(fs/2));  % Normalize by Nyquist frequency

% Apply the low-pass filter
filtered_signal = filtfilt(b, a, signal_interval);

% Downsample the signal (downsampling factor)
downsample_factor = 2; % Change this factor to adjust the downsampling
downsampled_signal = downsample(filtered_signal, downsample_factor);
fs_downsampled = fs / downsample_factor; % New sampling frequency

% Time vector for the downsampled signal
time_downsampled = (0:length(downsampled_signal)-1) / fs_downsampled;

% Compute DFT of the downsampled signal
nfft = 128;  % Number of FFT points
dft_signal = fft(downsampled_signal);
L = length(dft_signal);
f = fs*(-L/2:L/2-1)/L;
freq = 128 * (0:(L/2)) / L; % Frequency vector

% Compute STFT of the downsampled signal
window_length = 128;  % Window length for STFT
noverlap = 64;        % Number of points in overlap
window = hamming(window_length);  % Use Hamming window

% Plot results
figure;

% Plot the time-domain signal
subplot(3, 1, 1);
plot(time_downsampled, downsampled_signal);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title('Time-Domain Signal (Downsampled)');
grid on;

% Plot the frequency spectrum (DFT)
subplot(3, 1, 2);
plot(freq(1:320), abs(dft_signal(1:320)));  % Magnitude in dB
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
xlim('tight')
title('Frequency Spectrum (DFT)');
grid on;
xlim([0 60]);  % Limit to 60 Hz

% Plot the STFT
subplot(3, 1, 3);
spectrogram(downsampled_signal, window, noverlap, nfft, 128, 'yaxis');
xlabel('Time (seconds)', 'Interpreter', 'latex');
ylabel('Frequency (Hz)', 'Interpreter', 'latex');
title('Spectrogram (STFT) - Downsampled', 'Interpreter', 'latex');

% Adjust figure size for better viewing
set(gcf, 'Position', [100, 100, 1200, 800]);  % Make the figure larger
