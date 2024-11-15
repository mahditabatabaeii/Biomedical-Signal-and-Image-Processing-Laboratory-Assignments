%% Section 1 
clc; close all;

load('normal.mat');

% Parameters
Fs = 250;
t = normal(:,1);
t_total = length(sig) / Fs;
sig = normal(:,2);
sig2 = randn(1,1000);
%% Q1.1

% Time segments for clean and noisy parts
t_clean_start = 5;
t_clean_end = 15; 
t_noisy_start = 250;
t_noisy_end = 260; 

% Clean and noisy signal segments
sig_clean = sig(t_clean_start*Fs+1 : t_clean_end*Fs);
sig_noisy = sig(t_noisy_start*Fs+1 : t_noisy_end*Fs);

% Clean signal segment
figure;
subplot(2, 2, 1);
plot(t(t_clean_start*Fs+1 : t_clean_end*Fs), sig_clean);
title('Clean Signal Segment');
xlabel('Time (s)');
ylabel('Amplitude (V)');

% Noisy signal segment
subplot(2, 2, 2);
plot(t(t_noisy_start*Fs+1 : t_noisy_end*Fs), sig_noisy);
title('Noisy Signal Segment');
xlabel('Time (s)');
ylabel('Amplitude (V)');

% PSD for clean signal segment
subplot(2, 2, 3);
window = gausswin(128); 
noverlap = 64;          
nfft = 128;        
[p,f] = pwelch(sig_clean', window, noverlap, nfft, Fs);
plot(f,db(p))
title('Power Spectrum of Clean Signal Segment');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
xlim([0,120]);

% PSD for noisy signal segment
subplot(2, 2, 4);
[p, f] = pwelch(sig_noisy', window, noverlap, nfft, Fs);
plot(f,db(p))
title('Power Spectrum of Noisy Signal Segment');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
xlim([0,120]);



%% Q1.2

n = size(normal, 1);

% Frequency limits
low_freq = 1 * n / Fs;
high_freq = 90 * n / Fs;

% FFT of the signal and isolate positive frequencies
normal_fft = abs(fftshift(fft(normal(:, 2))));
normal_half = normal_fft(n/2+1:n);

% Power of the signal
normal_pwr = sum(normal_half(low_freq:n/2).^2);

% Bandstop filter to attenuate frequencies between 1 and 90 Hz
[b, a] = butter(3, [1/(Fs/2) 40/(Fs/2)], 'stop');
normal_filtered = filter(b, a, normal_half(low_freq:n/2));

% Power of the filtered signal
filter_pwr = sum(normal_filtered.^2);

% Frequency response of the filter
figure;
subplot(2, 1, 1);
freqz(b, a, 50);
title("Frequency Response of Bandstop Filter");
grid minor;

% Impulse response of the filter
subplot(2, 1, 2);
impz(b, a, 50);
title("Impulse Response of Bandstop Filter");
grid minor;

%% Q1.3
% Filter the clean and noisy segments
clean_filtered = filter(b, a, clean_signal);
noisy_filtered = filter(b, a, noisy_signal);

figure;
% PSD of filtered clean signal
subplot(2, 1, 1);
[p, f] = pwelch(clean_filtered', [], [], [], Fs);
plot(f,db(p))
title("Clean ECG Signal (Filtered)");
grid minor;
xlim([0 120]);

% PSD of filtered noisy signal
subplot(2, 1, 2);
[p, f] = pwelch(noisy_filtered', [], [], [], Fs);
plot(f,db(p))
title("Noisy Signal (Filtered)");
grid minor;
xlim([0 120]);
