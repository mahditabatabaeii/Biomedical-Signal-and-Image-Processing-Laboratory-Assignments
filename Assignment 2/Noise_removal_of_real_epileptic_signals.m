% Extract signals 1 and 2
signal_1  = load('Lab2_2\NewData1.mat').EEG_Sig;
signal_2  = load('Lab2_2\NewData2.mat').EEG_Sig;
Electrodes = load('Lab2_2\Electrodes.mat');
labels = Electrodes.Electrodes.labels;

%% Part 1
signal = signal_1;

offset = max(max(abs(signal)))/1;
feq = Fs;
disp_eeg(signal,offset,feq,labels);
xlim("tight")
title("Signal 2")

%% Part 3

Pest = size(signal,1);
[F,W,K] = COM2R(signal,Pest);
Z = W*signal; 

%% Part 4

% plot time domain of the components
offset = max(max(abs(Z)))/1;
feq = Fs;
ElecName = 1:32;
disp_eeg(Z,offset,feq,ElecName);
xlim('tight')
title('Components')
ylabel('Components N');

% fft
figure('units','normalized','outerposition',[0 0 1 1])
for i = 1:21
    subplot(7,3,i)
    windowL = gausswin(128);
    overlap = length(windowL)/2;
    L = length(Z);
    f = (0:(L/2-1))*Fs/L;
    [pxx, f] = pwelch(Z(i,:),windowL,overlap,f,Fs);
    plot(f,pxx,'LineWidth',1.5,'Color','#0072BD');
    xlim([0 70]);
    title("Component " + i)
    
    if(mod(i,3) == 1)
        subplot(7,3,i)
        ylabel('Magntiude');
    end
end

subplot(7,3,19); xlabel('Frequency(Hz)');
subplot(7,3,20); xlabel('Frequency(Hz)');
subplot(7,3,21); xlabel('Frequency(Hz)');

% topography
elocsX = Electrodes.Electrodes.X;
elocsY = Electrodes.Electrodes.Y;
elabels = Electrodes.Electrodes.labels;
figure('units','normalized','outerposition',[0 0 1 1])
nsub = 1;
for i = 1:21
    subplot(4,6,nsub)
    plottopomap(elocsX,elocsY,elabels,F(:,i))
    nsub = nsub + 1;
    title("Component " + i)
end

%% Part 5

rmv_comp = [4, 10];
Z(rmv_comp,:) = [];
F(:,rmv_comp) = [];

% reverse problem
X_Den = F*Z;

% plot
offset = max(max(abs(X_Den)))/1;
feq = Fs;
ElecName = Electrodes.Electrodes.labels;
disp_eeg(X_Den,offset,feq,ElecName);
xlim('tight')
title('Denoised')

%% Part 6

% plot
offset = max(max(abs(X_Den)))/1;
disp_eeg(X_Den,offset,feq,ElecName);
title('Denoised')
xlim("tight")

offset = max(max(abs(signal)))/1;x
disp_eeg(signal,offset,feq,ElecName);
title('Raw')
xlim("tight")
