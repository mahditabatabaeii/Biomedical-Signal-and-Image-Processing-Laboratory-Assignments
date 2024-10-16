% Load the EMG signals from the .mat file
load('sig_EMG.mat');  % Replace with actual filename if different

% Assuming the file contains variables 'EMG1', 'EMG2', and 'EMG3' for the three subjects
% EMG1: Healthy subject
% EMG2: Neuropathy patient
% EMG3: Myopathy patient

% If the variables are in a different format, adjust the following assignments
EMG1 = sig_EMG(1, :);  % Healthy subject
EMG2 = sig_EMG(2, :);  % Neuropathy
EMG3 = sig_EMG(3, :);  % Myopathy

% Sampling frequency and time vector
Fs = 4000;  % Sampling rate after downsampling to 4 kHz
t1 = (0:length(EMG1)-1) / Fs;
t2 = (0:length(EMG2)-1) / Fs;
t3 = (0:length(EMG3)-1) / Fs;

% Plot the EMG signals
figure;
subplot(3,1,1);
plot(t1, EMG1);
title('EMG Signal - Healthy Subject');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3,1,2);
plot(t2, EMG2);
title('EMG Signal - Neuropathy Patient');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3,1,3);
plot(t3, EMG3);
title('EMG Signal - Myopathy Patient');
xlabel('Time (s)');
ylabel('Amplitude');

% Zoom into specific regions for analysis
disp('Use the zoom tool in the figure to analyze specific parts of the signal.');
zoom on;
