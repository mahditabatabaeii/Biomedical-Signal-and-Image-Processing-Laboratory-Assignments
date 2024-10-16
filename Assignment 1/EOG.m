%% EOG - P.1
% loading the Siganls 
load('EOG_sig.mat');
signal = Sig;
fs = fs;
time = (0:length(signal)-1) / fs;

% Ploting the Signals
figure;
plot(time, signal(1,:), 'b', 'DisplayName', 'Left Eye');
hold on;
plot(time, signal(2,:), 'r', 'DisplayName', 'Right Eye');
xlabel('Time (seconds)');
ylabel('Amplitude');
title('EOG Signals for Left and Right Eye');
legend;
hold off;

%% EOG - P.2

% Fourier Transform of the Signals
n = length(signal);
frequencies = (0:n-1)*(fs/n);

% Performing FFT
fft_left_eye = abs(fft(signal(1,:)));
fft_right_eye = abs(fft(signal(2,:)));

% Ploting the Frequency Spectrum
figure;
subplot(2, 1, 1);
plot(frequencies, fft_left_eye);
xlim([0 30])
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('DFT of Left Eye Signal');

subplot(2, 1, 2);
plot(frequencies, fft_right_eye, 'r');
xlim([0 30])
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('DFT of Right Eye Signal');

%% EOG - P.2.2

% Spectrogram for Left Eye
figure;
subplot(2,1,1);
spectrogram(signal(1,:), 256, 128, 256, fs, 'yaxis');
title('Spectrogram of Left Eye EOG Signal');
colorbar;

% Spectrogram for Right Eye
subplot(2,1,2);
spectrogram(signal(2,:), 256, 128, 256, fs, 'yaxis');
title('Spectrogram of Right Eye EOG Signal');
colorbar;

xlabel('Time (s)');
ylabel('Frequency (Hz)');
