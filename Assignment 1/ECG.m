%% ECG - P.1.1

% Loading the Data
load('ECG_sig.mat');
signal = Sig;
fs = sfreq;
time = (0:length(signal)-1) / fs;

% Ploting the signals
figure;
subplot(2,1,1);
plot(time, signal(:,1), 'b', 'DisplayName', 'Lead 1');

subplot(2,1,2);
plot(time, signal(:,2), 'r', 'DisplayName', 'Lead 2');
xlabel('Time (seconds)');
ylabel('Amplitude');
title('ECG Signals for Lead 1 and Lead 2');
legend;
hold off;

%% ECG - P.1.2

r_peaks = ATRTIMED;

r_peak = r_peaks(1);

window = 1.2;

r1 = r_peaks(1);
r2 = r_peaks(2);

% Beat 1
beat_start = max(1, round((r_peak - window) * fs));
beat_end = min(length(time), round((r_peak + window) * fs));
beat_time = time(beat_start:beat_end);
beat_signal = signal(beat_start:beat_end, 1);

% Beat 2
beat2_start = max(1, round((r2 - window) * fs));
beat2_end = min(length(time), round((r2 + window) * fs));
beat2_time = time(beat2_start:beat2_end);
beat2_signal = signal(beat2_start:beat2_end, 2);

% Plot beat 1 from Lead 1
figure;
subplot(2,1,1);
plot(beat_time, beat_signal, 'b');
xlabel('Time (seconds)');
title('ECG Beat 1 with P, Q, R, S, and T waves');

% Plot beat 2 from Lead 2
subplot(2,1,2);
plot(beat2_time, beat2_signal, 'r');
xlabel('Time (seconds)');
ylabel('Amplitude');
title('ECG Beat 2 with P, Q, R, S, and T waves');

%% ECG - P.2

r_peaks = ATRTIMED;
annotations = ANNOTD;
annotation_labels = {'NOTQRS', 'NORMAL', 'LBBB', 'RBBB', 'ABERR', 'PVC', ...
                     'FUSION', 'NPC', 'APC', 'SVPB', 'VESC', 'NESC', 'PACE', ...
                     'UNKNOWN', 'NOISE', '', 'ARFCT', '', 'STCH', 'TCH', 'SYSTOLE', ...
                     'DIASTOLE', 'NOTE', 'MEASURE', 'PWAVE', 'BBB', 'PACESP', ...
                     'TWAVE', 'RHYTHM', 'UWAVE'};

% Ploting the Signals 
figure;
subplot(2,1,1)
plot(time, signal(:,1), 'b');
hold on;

for i = 1:length(r_peaks)
    r_time = r_peaks(i);
    r_amplitude = signal(round(r_time * fs), 1);
    plot(r_time, r_amplitude, 'ro', 'DisplayName', 'R-peak');

    anomaly_code = annotations(i);
    if anomaly_code <= length(annotation_labels)
        anomaly_label = annotation_labels{anomaly_code + 1};
        text(r_time, r_amplitude, [' ' anomaly_label], 'FontSize', 8, 'Color', 'r');
    end
end

subplot(2,1,2)
plot(time, signal(:,2), 'b');
hold on;

for i = 1:length(r_peaks)
    r_time = r_peaks(i);
    r_amplitude = signal(round(r_time * fs), 2);
    plot(r_time, r_amplitude, 'ro', 'DisplayName', 'R-peak');

    anomaly_code = annotations(i);
    if anomaly_code <= length(annotation_labels)
        anomaly_label = annotation_labels{anomaly_code + 1};
        text(r_time, r_amplitude, [' ' anomaly_label], 'FontSize', 8, 'Color', 'r');
    end
end

xlabel('Time (seconds)');
ylabel('Amplitude');
title('ECG Signal with R-Peaks and Annotations');
legend('ECG Signal', 'R-peaks');
hold off;

%% ECG - P.3

window = 0.5;

% Anomaly Types to Plot
anomalies_to_plot = [1, 4, 5, 8, 11, 28];

% Plotting Anomalies
figure;
for i = 1:length(anomalies_to_plot)
    anomaly_code = anomalies_to_plot(i);
    anomaly_label = annotation_labels{anomaly_code + 1};
    
    anomaly_indices = find(annotations == anomaly_code);
    num_beats_to_plot = min(3, length(anomaly_indices));
 
    for j = 1:num_beats_to_plot
        r_peak = r_peaks(anomaly_indices(j));
        beat_start = max(1, round((r_peak - window) * fs));
        beat_end = min(length(time), round((r_peak + window) * fs));
        beat_time = time(beat_start:beat_end);
        beat_signal = signal(beat_start:beat_end, 1); % Lead 1

        subplot(length(anomalies_to_plot), num_beats_to_plot, (i-1)*num_beats_to_plot + j);
        plot(beat_time, beat_signal, 'b');
        title([anomaly_label ' (Beat ' num2str(j) ')']);
        xlabel('Time (s)');
        ylabel('Amplitude');
    end
end

sgtitle('ECG Beats for Selected Anomalies (NORMAL, ABERR, PVC, APC, NESC, RHYTHM)');

%% ECG - P.4

normal_times = ATRTIMED(61:63);
abnormal_times = ATRTIMED(640:642); 
beat_window = 0.5; 

normal_start_idx = max(1, round((normal_times(1) - beat_window) * sfreq));
normal_end_idx = min(length(signal), round((normal_times(end) + beat_window) * sfreq));
normal_segment = signal(normal_start_idx:normal_end_idx, :);
normal_time_segment = time(normal_start_idx:normal_end_idx);

abnormal_start_idx = max(1, round((abnormal_times(1) - beat_window) * sfreq));
abnormal_end_idx = min(length(signal), round((abnormal_times(end) + beat_window) * sfreq));
abnormal_segment = signal(abnormal_start_idx:abnormal_end_idx, :);
abnormal_time_segment = time(abnormal_start_idx:abnormal_end_idx);

% Plots
figure;
sgtitle('Normal Beats (Time, FFT, and Spectrogram)');

% Lead 1 - Time Domain
subplot(3, 2, 1);
plot(normal_time_segment, normal_segment(:, 1), 'b');
title('Time Domain - Normal (Lead 1)');
xlabel('Time (s)');
ylabel('Amplitude');

% Lead 1 - FFT
subplot(3, 2, 3);
n = length(normal_segment(:, 1));
f = (0:n-1)*(sfreq/n); % Frequency vector
fft_signal = abs(fft(normal_segment(:, 1)));
plot(f, fft_signal);
title('Frequency Domain (FFT) - Normal (Lead 1)');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
xlim([0, sfreq/2]);

% Lead 1 - Spectrogram
subplot(3, 2, 5);
spectrogram(normal_segment(:, 1), 128, 120, 128, sfreq, 'yaxis');
title('Spectrogram - Normal (Lead 1)');

% Lead 2 - Time Domain
subplot(3, 2, 2);
plot(normal_time_segment, normal_segment(:, 2), 'r');
title('Time Domain - Normal (Lead 2)');
xlabel('Time (s)');
ylabel('Amplitude');

% Lead 2 - FFT
subplot(3, 2, 4);
n = length(normal_segment(:, 2));
f = (0:n-1)*(sfreq/n); % Frequency vector
fft_signal = abs(fft(normal_segment(:, 2)));
plot(f, fft_signal);
title('Frequency Domain (FFT) - Normal (Lead 2)');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
xlim([0, sfreq/2]);

% Lead 2 - Spectrogram
subplot(3, 2, 6);
spectrogram(normal_segment(:, 2), 128, 120, 128, sfreq, 'yaxis');
title('Spectrogram - Normal (Lead 2)');

% Plot abnormal beats for both leads
figure;
sgtitle('Abnormal Beats (Time, FFT, and Spectrogram)');

% Lead 1 - Time Domain
subplot(3, 2, 1);
plot(abnormal_time_segment, abnormal_segment(:, 1), 'b');
title('Time Domain - Abnormal (Lead 1)');
xlabel('Time (s)');
ylabel('Amplitude');

% Lead 1 - FFT
subplot(3, 2, 3);
n = length(abnormal_segment(:, 1));
f = (0:n-1)*(sfreq/n);
fft_signal = abs(fft(abnormal_segment(:, 1)));
plot(f, fft_signal);
title('Frequency Domain (FFT) - Abnormal (Lead 1)');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
xlim([0, sfreq/2]);

% Lead 1 - Spectrogram
subplot(3, 2, 5);
spectrogram(abnormal_segment(:, 1), 128, 120, 128, sfreq, 'yaxis');
title('Spectrogram - Abnormal (Lead 1)');

% Lead 2 - Time Domain
subplot(3, 2, 2);
plot(abnormal_time_segment, abnormal_segment(:, 2), 'r');
title('Time Domain - Abnormal (Lead 2)');
xlabel('Time (s)');
ylabel('Amplitude');

% Lead 2 - FFT
subplot(3, 2, 4);
n = length(abnormal_segment(:, 2));
f = (0:n-1)*(sfreq/n); % Frequency vector
fft_signal = abs(fft(abnormal_segment(:, 2)));
plot(f, fft_signal);
title('Frequency Domain (FFT) - Abnormal (Lead 2)');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
xlim([0, sfreq/2]);

% Lead 2 - Spectrogram
subplot(3, 2, 6);
spectrogram(abnormal_segment(:, 2), 128, 120, 128, sfreq, 'yaxis');
title('Spectrogram - Abnormal (Lead 2)');






