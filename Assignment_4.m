%% 1-1

% Load the EEG ERP data
load('ERP_EEG.mat', 'ERP_EEG');  % Assumes the EEG data is stored in ERP_EEG

% Sampling frequency
fs = 240; % Hz

% Define the values of N for averaging
N_values = 100:100:2500;

% Time vector for 1 second (assuming data starts from stimulus onset)
t = (0:size(ERP_EEG, 1)-1) / fs;

% Initialize a figure for plotting
figure;
hold on;
title('Averaged EEG Response for Different N Values');
xlabel('Time (seconds)');
ylabel('Amplitude (μV)');

% Loop over different values of N
for N = N_values
    % Randomly select N trials from the 2550 available trials
    selected_trials = ERP_EEG(:, [1:N]);
    
    % Compute the average response across the N selected trials
    avg_response = mean(selected_trials, 2);
    
    % Plot the averaged response
    plot(t, avg_response, 'DisplayName', sprintf('N = %d', N), LineWidth=1.5);
end

% Add legend and display the plot
legend show;
hold off;

%% 1-2

% Define the range for N values
N_values = 1:2550;

% Preallocate an array to store the maximum absolute amplitude for each N
max_amplitude = zeros(1, length(N_values));

% Loop over different values of N
for N = N_values
    % Randomly select N trials from the 2550 available trials
    selected_trials = ERP_EEG(:, [1:N]);
    
    % Compute the average response across the N selected trials
    avg_response = mean(selected_trials, 2);
    
    % Calculate the maximum absolute amplitude of the averaged response
    max_amplitude(N) = max(abs(avg_response));
end

% Plot the maximum absolute amplitude versus N
figure;
plot(N_values, max_amplitude, LineWidth=1.5);
title('Maximum Absolute Amplitude vs. Number of Averaged Trials (N)');
xlabel('Number of Averaged Trials (N)');
ylabel('Maximum Absolute Amplitude (μV)');
grid on;

%% 1-3

% Preallocate an array to store RMSE between N-th and (N-1)-th averaged patterns
rmse_values = zeros(1, length(N_values) - 1);

% Initialize the previous average response for N=1
prev_avg_response = mean(ERP_EEG(:, [1:N]), 2);

% Loop over values of N from 2 to 2550
for N = 2:length(N_values)
    % Randomly select N trials from the 2550 available trials
    selected_trials = ERP_EEG(:, randperm(2550, N));
    
    % Compute the average response across the N selected trials
    avg_response = mean(selected_trials, 2);
    
    % Calculate the RMSE between the current and previous averaged response
    rmse_values(N-1) = sqrt(mean((avg_response - prev_avg_response).^2));
    
    % Update the previous average response
    prev_avg_response = avg_response;
end

% Plot RMSE versus N
figure;
plot(N_values(2:end), rmse_values);
title('RMSE Between Consecutive Averaged Patterns vs. Number of Averaged Trials (N)');
xlabel('Number of Averaged Trials (N)');
ylabel('RMSE (μV)');
grid on;

%% 1-5

% Define the number of trials for each condition
N_all = 2550;
N0 = 1000;
N_third = round(N0 / 3);

% Calculate the average response for each condition

% 1. Average response using all trials (N = 2550)
avg_response_all = mean(ERP_EEG(:, [1:N_all]), 2);

% 2. Average response using N0/3  trials
avg_response_N0_third = mean(ERP_EEG(:, [1:N_third]), 2);

% 3. Average response using N0 trials
avg_response_N0 = mean(ERP_EEG(:, randperm(2550, N0)), 2);

% 4. Average response using N0 / 3 trials
avg_response_N0_third_random = mean(ERP_EEG(:, randperm(2550, N_third)), 2);

% Plotting the responses
figure;
plot(t, avg_response_all, 'DisplayName', 'N = 2550', LineWidth=1.5);
hold on;
plot(t, avg_response_N0_third, 'DisplayName', sprintf('N = 1000/3 not Random'), LineWidth=1.5);
plot(t, avg_response_N0, 'DisplayName', sprintf('N = 1000  Random'), LineWidth=1.5);
plot(t, avg_response_N0_third_random, 'DisplayName', sprintf('N = 1000/3 Random'), LineWidth=1.5);
hold off;

% Customize the plot
title('Comparison of Averaged EEG Responses for Different Trial Counts');
xlabel('Time (seconds)');
ylabel('Amplitude (μV)');
legend show;
grid on;

%% Q2.0 - Load data
clc; close all;

load("SSVEP_EEG.mat");
Fs = 250; 

%% Q2.1
% Design a bandpass filter to keep frequencies between 1 Hz and 40 Hz
filtered_SSVEP = zeros(size(SSVEP_Signal));

for i = 1 : size(SSVEP_Signal,1)
    figure;
    bandpass(SSVEP_Signal(i,:),[1 40],Fs);
    filtered_SSVEP(i,:) = bandpass(SSVEP_Signal(i,:),[1 40],Fs);
end

%% Q2.2
T = 5;
epochs = zeros([size(filtered_SSVEP,1) T*Fs 15]);

for i = 1 : 15
    ind_s = Event_samples(i);
    ind_f = ind_s + T*Fs;
    epochs(:,:,i) = filtered_SSVEP(:,ind_s:ind_f-1);
end

%% Q2.3
% Channel labels
strs = ["Pz","Oz","P7","P8","O2","O1"];

figure("units","normalized","OuterPosition",[0 0 1 1]);

for i = 1 : 15
    subplot(5,3,i);
    hold on
    for j = 1 : 6
        [pxx,f] = pwelch(epochs(j,:,i),[],[],[],Fs);
        plot(f,pxx,'LineWidth',1.5);
        title("Welch PSD Estimate for Event"+ num2str(i));
        xlabel("f[Hz]");ylabel("Power/frequency(dB/Hz)");
    end
    xlim([0 40]);
end
legend(strs);



%% 3-1
% Load EEG signal data
load('FiveClass_EEG.mat', 'X', 'trial', 'y'); % EEG data with variables X, trial, and y

% Filter parameters
N = 4; % Filter order
Apass = 1; % Passband Ripple (dB)
fs = 250; % Sampling frequency (update if different)
channels = 30;
bands = {'delta', 'theta', 'alpha', 'beta'};
freq_ranges = [1, 4; 4, 8; 8, 13; 13, 30];

% Filter EEG data for each frequency band
filtered_X = struct();
for b = 1:length(bands)
    Fpass1 = freq_ranges(b, 1);
    Fpass2 = freq_ranges(b, 2);
    h = fdesign.bandpass('N,Fp1,Fp2,Ap', N, Fpass1, Fpass2, Apass, fs);
    Hd = design(h, 'cheby1');
    filtered_X.(bands{b}) = zeros(size(X,1), channels);
    for c = 1:channels
        filtered_X.(bands{b})(:, c) = filter(Hd, X(:, c));
    end
end

% Plot filtered signals
t = 0:1/fs:5 - 1/fs;
figure;
titles = {'Original', 'Delta', 'Theta', 'Alpha', 'Beta'};
plot_signals = [{X}, filtered_X.delta, filtered_X.theta, filtered_X.alpha, filtered_X.beta];
for i = 1:length(plot_signals)
    subplot(length(plot_signals), 1, i);
    plot(t, plot_signals{i}(1:5*fs, 1));
    title(titles{i});
    grid minor;
end

%% 3-2
trial_duration = 10 * fs + 1;
num_trials = 200;
trial_data = struct();
for b = 1:length(bands)
    trial_data.(bands{b}) = zeros(trial_duration, channels, num_trials);
    for i = 1:num_trials
        trial_data.(bands{b})(:, :, i) = filtered_X.(bands{b})(trial(i):trial(i) + trial_duration - 1, :);
    end
end

%% 3-3
squared_data = struct();
for b = 1:length(bands)
    squared_data.(bands{b}) = trial_data.(bands{b}).^2;
end

%% 3-4
class_count = 5;
summed_data = struct();
avg_data = struct();
sizes = zeros(1, class_count);
for b = 1:length(bands)
    summed_data.(bands{b}) = zeros(trial_duration, channels, class_count);
    avg_data.(bands{b}) = zeros(trial_duration, channels, class_count);
end

for i = 1:num_trials
    class = y(i);
    for b = 1:length(bands)
        summed_data.(bands{b})(:, :, class) = summed_data.(bands{b})(:, :, class) + squared_data.(bands{b})(:, :, i);
    end
    sizes(class) = sizes(class) + 1;
end

for b = 1:length(bands)
    for i = 1:class_count
        avg_data.(bands{b})(:, :, i) = summed_data.(bands{b})(:, :, i) / sizes(i);
    end
end

%% 3-5
window = ones(1, 200) / sqrt(200);
filtered_avg = struct();
for b = 1:length(bands)
    % Adjust filtered_avg to match the length of 'valid' convolution output
    filtered_avg.(bands{b}) = zeros(trial_duration - length(window) + 1, channels, class_count);
    for i = 1:class_count
        for j = 1:channels
            % Use 'valid' to avoid size mismatch errors
            filtered_avg.(bands{b})(:, j, i) = conv(avg_data.(bands{b})(:, j, i), window, 'valid');
        end
    end
end

%% 3-6
% Adjust time vector to match the length of 'valid' convolution output
t_adjusted = 0:1/fs:(size(filtered_avg.delta, 1) - 1) / fs;

figure;
for b = 1:length(bands)
    subplot(2, 2, b);
    for i = 1:class_count
        plot(t_adjusted, filtered_avg.(bands{b})(:, 13, i)); hold on;
        grid minor;
    end
    title(bands{b});
    legend(arrayfun(@(x) ['class ' num2str(x)], 1:class_count, 'UniformOutput', false));
end
