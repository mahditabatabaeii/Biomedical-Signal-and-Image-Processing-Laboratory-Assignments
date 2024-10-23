%% Part 1-1

% Assume that the loaded data is stored in a variable called 'recorded'
ecg_mother = load('Lab3_data\data\mecg1.dat');  % Mother ECG signal
ecg_fetus = load('Lab3_data\data\fecg1.dat');  % Fetus ECG signal
noise = load('Lab3_data\data\noise1.dat'); % Noise signal

% Combine the signals to create the mixed signal
mixed_signal = ecg_mother + ecg_fetus + noise;

% Smapling Frequency
fs = 256;

% Create the time axis (assuming a sampling frequency of 256 Hz)
t = (0:length(ecg_mother)-1) / fs;  

% Plot the mother ECG signal
subplot(4,1,1);
plot(t, ecg_mother);
title('Mother ECG Signal');
xlabel('Time (seconds)');
ylabel('Voltage (mV)');

% Plot the fetus ECG signal
subplot(4,1,2);
plot(t, ecg_fetus);
title('Fetus ECG Signal');
xlabel('Time (seconds)');
ylabel('Voltage (mV)');

% Plot the noise signal
subplot(4,1,3);
plot(t, noise);
title('Noise Signal');
xlabel('Time (seconds)');
ylabel('Voltage (mV)');

% Plot the mixed signal
subplot(4,1,4);
plot(t, mixed_signal);
title('Mixed Signal');
xlabel('Time (seconds)');
ylabel('Voltage (mV)');

%% Part 1-2

% Plot the power spectrum of the mother ECG signal
figure;
subplot(4,1,1);
pwelch(ecg_mother, [], [], [], fs);
title('Power Spectrum of Mother ECG Signal');

% Plot the power spectrum of the fetus ECG signal
subplot(4,1,2);
pwelch(ecg_fetus, [], [], [], fs);
title('Power Spectrum of Fetus ECG Signal');

% Plot the power spectrum of the noise signal
subplot(4,1,3);
pwelch(noise, [], [], [], fs);
title('Power Spectrum of Noise Signal');

% Plot the power spectrum of the mixed signal
subplot(4,1,4);
pwelch(mixed_signal, [], [], [], fs);
title('Power Spectrum of Mixed Signal');

%% Part 1-3

% Calculate the mean of the signals
mean_mother = mean(ecg_mother);
mean_fetus = mean(ecg_fetus);
mean_noise = mean(noise);
mean_mixed = mean(mixed_signal);

% Calculate the variance of the signals
var_mother = var(ecg_mother);
var_fetus = var(ecg_fetus);
var_noise = var(noise);
var_mixed = var(mixed_signal);

% Display the results without rounding
disp('Mean and Variance of Signals:');
fprintf('Mother ECG - Mean: %g, Variance: %g\n', mean_mother, var_mother);
fprintf('Fetus ECG - Mean: %g, Variance: %g\n', mean_fetus, var_fetus);
fprintf('Noise - Mean: %g, Variance: %g\n', mean_noise, var_noise);
fprintf('Mixed Signal - Mean: %g, Variance: %g\n', mean_mixed, var_mixed);

%% Part 1-4-1

% Plot the histograms
figure;

% Histogram of the mother ECG signal
subplot(4,1,1);
hist(ecg_mother, 50);  % 50 bins for the histogram
title('Histogram of Mother ECG Signal');
xlabel('Amplitude');
ylabel('Frequency');

% Histogram of the fetus ECG signal
subplot(4,1,2);
hist(ecg_fetus, 50);  % 50 bins for the histogram
title('Histogram of Fetus ECG Signal');
xlabel('Amplitude');
ylabel('Frequency');

% Histogram of the noise signal
subplot(4,1,3);
hist(noise, 50);  % 50 bins for the histogram
title('Histogram of Noise Signal');
xlabel('Amplitude');
ylabel('Frequency');

% Histogram of the mixed signal
subplot(4,1,4);
hist(mixed_signal, 50);  % 50 bins for the histogram
title('Histogram of Mixed Signal');
xlabel('Amplitude');
ylabel('Frequency');

%% Part 1-4-2

% Calculate the kurtosis (4th moment) of the signals
kurt_mother = kurtosis(ecg_mother);
kurt_fetus = kurtosis(ecg_fetus);
kurt_noise = kurtosis(noise);
kurt_mixed = kurtosis(mixed_signal);

% Display the results
disp('Kurtosis (4th Moment) of Signals:');
fprintf('Mother ECG: %.4f\n', kurt_mother);
fprintf('Fetus ECG: %.4f\n', kurt_fetus);
fprintf('Noise: %.4f\n', kurt_noise);
fprintf('Mixed Signal: %.4f\n', kurt_mixed);

%% Part 2-1

% Load the data from dat.X (assuming it's a .mat file)
X = load('Lab3_data\data\X.dat');  % This loads the data into the workspace

% Call the custom plot function 'ch3plot' with the loaded data
plot3ch(X);

% Perform Singular Value Decomposition (SVD)
[U, S, V] = svd(X);

%% Part 2-2

% Call the custom plotting function dv3plot
for v = 1:length(V)
    plot3dv(V(:,v), S(:,v));
    hold on
end
%% Part 2-3-1

t = (0:length(U)-1) / fs;  % Time vector in seconds

% Plot the first three columns of matrix U
figure;
subplot(3,1,1);
plot(t, U(:,1)); % Plot the first column of U
title('First Column of U');
xlabel('Time (seconds)');
ylabel('Value');

subplot(3,1,2);
plot(t, U(:,2)); % Plot the second column of U
title('Second Column of U');
xlabel('Time (seconds)');
ylabel('Value');

subplot(3,1,3);
plot(t, U(:,3)); % Plot the third column of U
title('Third Column of U');
xlabel('Time (seconds)');
ylabel('Value');

%% Part 2-3-2

% Extract the singular values from the diagonal of matrix S
singular_values = diag(S);

% Plot the eigenspectrum using the stem function
figure;
stem(singular_values, 'filled');
xlim([0,4]);  
title('Eigenspectrum (Singular Values)');
xlabel('Index');
ylabel('Singular Value');

%% Part 2-4

% Modify the singular values matrix S
S_mod = zeros(size(S));   % Create a new matrix S_mod initialized to zeros
S_mod(2,2) = S(2,2);      % Keep only the singular value for the fetus component (index 3)

% Reconstruct the matrix using the modified S matrix
X_Reconstructed = U * S_mod * V';  % Reconstructed matrix with only the fetal component

% Plot the original mother ECG signal
subplot(3,1,1);
plot(t, X_Reconstructed(:,1));
title('Reconstructed ECG Signal channel 1');
xlabel('Index');
ylabel('Amplitude');

% Plot the original fetal ECG signal
subplot(3,1,2);
plot(t, X_Reconstructed(:,2));
title('Reconstructed ECG Signal channel 2');
xlabel('Index');
ylabel('Amplitude');

% Plot the reconstructed fetal ECG signal
subplot(3,1,3);
plot(t, X_Reconstructed(:,3));
title('Reconstructed ECG Signal channel 3');
xlabel('Index');
ylabel('Amplitude');

%% %%%%%%%%%%%%%%% Q3 %%%%%%%%%%%%%%%%
% Load data
load('mecg1.dat');
load('fecg1.dat');
load('noise1.dat');
load('X.dat');

Fs = 256;
t = (0:length(X)-1)./Fs;

%% Q3.1
% Perform ICA on X'T
[W, ZHAT] = ica(X'); 

% Pseudo-inverse of W 
Winv = pinv(W);

% Save the results
save('ICA_output','W','ZHAT','Winv');

%% Q3.2
% Scatter plot
plot_title = "Mixed signal";
plot3ch(X, Fs, plot_title);

figure;
col = ["blue", "green", "red"];

% Plot the columns of Winv
for i = 1:size(Winv, 2)
    plot3dv(Winv(:, i), 1, col(i)); 
    title('Component vectors');
    xlabel('x'); ylabel('y'); zlabel('z');
end

% Save the figure
savefig('component_vectors.fig');

%% Q3.3
% Three rows components of ZHAT over time
figure;
for i = 1:size(ZHAT,1)
    hold on;
    plot(t, i*5 + ZHAT(i,:))  
    xlabel('Time (s)')
end

% Label each component
yticks([5 10 15]);
yticklabels({"Component 1", "Component 2", "Component 3"});
ytickangle(30);
title('Components');

% Reconstruct the signal using only component 3
n = 3; 
X_reconstructed = Winv(:,n) * ZHAT(n,:);

% Save the reconstructed signal
save('X_reconstructed_ICA','X_reconstructed');

%% Q3.4
% Plot the reconstructed signals
figure;
hold on;
for i = 1:3
    plot(t, i*6 + X_reconstructed(i,:));  
    xlabel("Time (s)");
    ylabel("Amplitude");
end

yticks([1 2 3].*6);
yticklabels({"Channel 1", "Channel 2", "Channel 3"});
ytickangle(30);
title('X reconstructed');

hold off;

%% %%%%%%%%%%%%%%% Q4 %%%%%%%%%%%%%%%%
% SVD decomposition of X
[U, S, V] = svd(X); 

% Perform ICA on X'T
[W, ZHAT] = ica(X'); 
A = pinv(W);

%% Q4.1
Fs = 256;
load('X.dat')
plot_title = "Mixed signal";
plot3ch(X,Fs,plot_title)

load('X_reconstructed_SVD')
plot_title = "Mixed signal SVD";
plot3ch(X_reconstructed,Fs,plot_title)

load('X_reconstructed_ICA')
plot_title = "Mixed signal ICA";
plot3ch(X_reconstructed',Fs,plot_title)

figure;
plot3dv(A(:,1), [], 'r');
hold on;
plot3dv(A(:,2), [], 'g');
plot3dv(A(:,3), [], 'b'); 
title('Directions of matrix A columns (ICA mixing matrix)');
xlabel('Ch1'); ylabel('Ch2'); zlabel('Ch3');
hold off;

figure;
plot3dv(V(:,1), [], 'r'); 
hold on;
plot3dv(V(:,2), [], 'g');
plot3dv(V(:,3), [], 'b');
title('Directions of matrix V columns (SVD right singular vectors)');
xlabel('Ch1'); ylabel('Ch2'); zlabel('Ch3');
hold off;

% Calculate angles between the columns of matrix V
angle_V_1_2 = dot(V(:,1), V(:,2));
angle_V_1_3 = dot(V(:,1), V(:,3));
angle_V_3_2 = dot(V(:,3), V(:,2));
norm_V_1 = norm(V(:,1));
norm_V_2 = norm(V(:,2));
norm_V_3 = norm(V(:,3));

% Calculate angles between the columns of matrix A
angle_A_1_2 = dot(A(:,1), A(:,2));
angle_A_1_3 = dot(A(:,1), A(:,3));
angle_A_3_2 = dot(A(:,3), A(:,2));
norm_A_1 = norm(A(:,1));
norm_A_2 = norm(A(:,2));
norm_A_3 = norm(A(:,3));

disp('Angles between the columns of matrix V (SVD):');
disp(['Angle between column 1 and 2: ', num2str(angle_V_1_2)]);
disp(['Angle between column 1 and 3: ', num2str(angle_V_1_3)]);
disp(['Angle between column 3 and 2: ', num2str(angle_V_3_2)]);

disp('Norms of the columns of matrix V:');
disp(['Norm of column 1: ', num2str(norm_V_1)]);
disp(['Norm of column 2: ', num2str(norm_V_2)]);
disp(['Norm of column 3: ', num2str(norm_V_3)]);

disp('Angles between the columns of matrix A (ICA):');
disp(['Angle between column 1 and 2: ', num2str(angle_A_1_2)]);
disp(['Angle between column 1 and 3: ', num2str(angle_A_1_3)]);
disp(['Angle between column 3 and 2: ', num2str(angle_A_3_2)]);

disp('Norms of the columns of matrix A:');
disp(['Norm of column 1: ', num2str(norm_A_1)]);
disp(['Norm of column 2: ', num2str(norm_A_2)]);
disp(['Norm of column 3: ', num2str(norm_A_3)]);

%% Q4.2
fs = 256;
t = 0 : 1/fs : 2560/fs - 1/fs;

% Load the signals
load('fecg2.dat');
load('SVD_Output');
load('ICA_Output');

% Plot the signals for comparison
figure;
% The ideal signal
subplot(3,1,1)
plot(t, fecg2); 
title('Ideal Signal (fecg2)');
grid minor;
xlabel('Time (s)');
ylabel('Amplitude');
% The SVD-denoised signal
load('X_reconstructed_SVD')
subplot(3,1,2)
plot(t, X_reconstructed(:,1));
title('Denoised by SVD');
grid minor;
xlabel('Time (s)');
ylabel('Amplitude');
% The ICA-denoised signal
load('X_reconstructed_ICA')
subplot(3,1,3)
plot(t, X_reconstructed(2,:)); 
title('Denoised by ICA');
grid minor;
xlabel('Time (s)');
ylabel('Amplitude');

% Save the figure
savefig('Signal_Comparison_SVD_ICA.fig');  % Save as .fig
saveas(gcf, 'Signal_Comparison_SVD_ICA.png');  % Save as .png

%% Q4.3
% Load the outputs
load('SVD_Output');
load('ICA_Output');

ang_vec = zeros(3,3,2);
norm_vec = zeros(2,3);


for i = 1:3
    for j = 1:3
        % W 
        ang_vec(i,j,1) = real(acosd((Winv(:,i)' * Winv(:,j))/(norm(Winv(:,i)')*norm(Winv(:,j)))));
        
        % V
        ang_vec(i,j,2) = real(acosd((V(:,i)' * V(:,j))/(norm(V(:,i)')*norm(V(j,:)))));
    end
end

load('fecg2.dat')
load('X_reconstructed_SVD')
r_SVD = corrcoef(fecg2,X_reconstructed(:,1))

load('X_reconstructed_ICA')
r_ICA = corrcoef(fecg2,X_reconstructed(1,:))
