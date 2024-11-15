%% 2-1

load("Lab 5_data/n_424.mat");
fs=250;

normal_1=n_424(8210:10710);
abnormal_1=n_424(31442:33942);

normal_2=n_424(1:2500);
abnormal_2=n_424(37211:39711);

t_normal_1=0:1/fs:length(normal_1)/fs-1/fs;
t_abnormal_1=0:1/fs:length(abnormal_1)/fs-1/fs;

t_normal_2=0:1/fs:length(normal_2)/fs-1/fs;
t_abnormal_2=0:1/fs:length(abnormal_2)/fs-1/fs;

subplot(2,1,1)
plot(t_normal_1,normal_1);
xlabel('time(second)')
title("normal 1")
subplot(2,1,2)
plot(t_abnormal_1,abnormal_1);
xlabel('time(second)')
title("abnormal 1")

figure
subplot(2,1,1)
plot(t_normal_2,normal_2);
xlabel('time(second)')
title("normal 2")
subplot(2,1,2)
plot(t_abnormal_2,abnormal_2);
xlabel('time(second)')
title("abnormal 2")

figure
subplot(4,1,1)
pwelch(normal_1)
title("normal 1")
subplot(4,1,2)
pwelch(abnormal_1)
title("abnormal 1")
subplot(4,1,3)
pwelch(normal_2)
title("normal 2")
subplot(4,1,4)
pwelch(abnormal_2)
title("abnormal 2")

%% 2-3

labels=zeros(59,11);
for i=1:59
    t_start=(i-1)*5*250;
    t_end=((i-1)*5+10)*250;
    labels(i,2)=t_end;
    if((t_start>=1 && t_end<27249))
        labels(i,1)=1;
    elseif((t_start>=27249 && t_end<53673))
        labels(i,1)= 2;
    elseif((t_start>=53673 && t_end<55134))
        labels(i,1)= 4;
    elseif((t_start>=55134 && t_end<58288))
        labels(i,1)= 5;
    elseif((t_start>=58288))
        labels(i,1)= 6;
    else
        labels(i,1)= 0;
    end 
end


%% 2-4

d = designfilt('bandpassiir', 'FilterOrder', 2, 'HalfPowerFrequency1', 10, 'HalfPowerFrequency2', 30, 'DesignMethod', 'butter', 'SampleRate', fs);
filtered_signal=filtfilt(d, n_424);
for i=1:59
    if(labels(i,1)==1 || labels(i,1)==2)
        signal = n_424(labels(i,2)-10*250+1:labels(i,2));
        filtered_signal=filtfilt(d, signal);
        labels(i,3)=meanfreq(signal);
        labels(i,4)=medfreq(signal);
        labels(i,5)=sum(filtered_signal.^2,'all');
    end
end

normal_features=labels(2:20,:);
VFIB_features=labels(23:41,:);

%% 2-5

figure

histogram(normal_features(:,3).','FaceColor', 'blue','BinWidth',0.005)
hold on
histogram(VFIB_features(:,3),'FaceColor', 'red','BinWidth',0.005)
legend("normal","VFIB")
title("mean frequency")
hold off

figure

histogram(normal_features(:,4).','FaceColor', 'blue','BinWidth',0.005)
hold on
histogram(VFIB_features(:,4),'FaceColor', 'red','BinWidth',0.005)
legend("normal","VFIB")
title("median frequency")
hold off

figure

histogram(normal_features(:,5).','FaceColor', 'blue','BinWidth',10^5)
hold on
histogram(VFIB_features(:,5),'FaceColor', 'red','BinWidth',10^5)
legend("normal","VFIB")
title("band power")
hold off

%% 2-6

[alarm_mean,t_mean] = va_detect(n_424 , fs, 'mean2');

%% 2-7

targets = labels((labels(:,1) == 1 | labels(:,1) == 2) , 1) == 2;
outputs_mean = alarm_mean((labels(:,1) == 1 | labels(:,1) == 2)) == 1;

cm1 = confusionmat(targets, outputs_mean);
confusionchart(cm1)

%% 2-7

targets_all= labels(: , 1);
outputs_mean_all= alarm_mean+1;

figure
cm1_all = confusionmat(targets_all, outputs_mean_all);
confusionchart(cm1_all)
title("mean")


%% 2-8

for i=1:59
    if(labels(i,1)==1 || labels(i,1)==2)
        signal = n_424(labels(i,2)-10*250+1:labels(i,2));
        labels(i,6)=min(signal);
        labels(i,7)=max(signal);
        labels(i,8)=labels(i,7)-labels(i,6);
        labels(i,9)=mean(findpeaks(signal),'all');
        labels(i,10)=sum(signal==0);
        labels(i,11)=var(signal);
    end
end

normal_features=labels(2:20,[6:11]);
VFIB_features=labels(23:41,[6:11]);

%% 2-9

figure

histogram(normal_features(:,1),'FaceColor', 'blue','BinWidth',20)
hold on
histogram(VFIB_features(:,1),'FaceColor', 'red','BinWidth',20)
legend("normal","VFIB")
title("min amplitude")
hold off

figure

histogram(normal_features(:,2),'FaceColor', 'blue','BinWidth',30)
hold on
histogram(VFIB_features(:,2),'FaceColor', 'red','BinWidth',30)
legend("normal","VFIB")
title("max amplitude")
hold off

figure

histogram(normal_features(:,3).','FaceColor', 'blue','BinWidth',30)
hold on
histogram(VFIB_features(:,3),'FaceColor', 'red','BinWidth',30)
legend("normal","VFIB")
title("peak to peak")
hold off

figure

histogram(normal_features(:,4).','FaceColor', 'blue','BinWidth',8)
hold on
histogram(VFIB_features(:,4),'FaceColor', 'red','BinWidth',8)
legend("normal","VFIB")
title("mean R")
hold off

figure

histogram(normal_features(:,5).','FaceColor', 'blue','BinWidth',3)
hold on
histogram(VFIB_features(:,5),'FaceColor', 'red','BinWidth',3)
legend("normal","VFIB")
title("zero crossing")
hold off

figure

histogram(normal_features(:,6).','FaceColor', 'blue','BinWidth',10^4)
hold on
histogram(VFIB_features(:,6),'FaceColor', 'red','BinWidth',10^4)
legend("normal","VFIB")
title("var")
hold off

%% 2-10

[alarm_zerocross,t_zerocross] = va_detect(n_424,fs,'zero-crossing');
[alarm_peak,t_peak] = va_detect(n_424,fs, 'max-min');

%% 2-11

targets = labels((labels(:,1) == 1 | labels(:,1) == 2) , 1) == 2;
outputs_zero_score = alarm_zerocross((labels(:,1) == 1 | labels(:,1) == 2)) == 1;
outputs_peak = alarm_peak((labels(:,1) == 1 | labels(:,1) == 2)) == 1;

figure
cm1 = confusionmat(targets, outputs_peak);
confusionchart(cm1)

figure
cm2 = confusionmat(targets, outputs_zero_score);
confusionchart(cm2)

%% 2-11

targets_all= labels(: , 1);
alarm_zerocross_all= alarm_zerocross+1;
outputs_peak_all = alarm_peak+1;

cm1_all = confusionmat(targets_all, alarm_zerocross_all);
confusionchart(cm1_all)
title("zero-crossing")
    
figure
cm2_all = confusionmat(targets_all, outputs_peak_all);
confusionchart(cm2_all)
title("peak to peak")

%% 2-14

[alarm_max,t_max] = va_detect(n_424,fs,'max');
targets_all= labels(: , 1);
outputs_max_all= alarm_max+1;

targets = labels((labels(:,1) == 1 | labels(:,1) == 2) , 1) == 2;
outputs_max = alarm_max((labels(:,1) == 1 | labels(:,1) == 2)) == 1;

figure
cm1 = confusionmat(targets, outputs_max);
confusionchart(cm1)

figure
cm_all = confusionmat(targets_all, outputs_max_all);
confusionchart(cm_all)
title("max amplitude")
