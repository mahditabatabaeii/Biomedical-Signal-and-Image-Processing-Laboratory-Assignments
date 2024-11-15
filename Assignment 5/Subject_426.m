% Part 15
load("Lab 5_data/n_426.mat");
fs=250;

labels=zeros(59,11);
for i=1:59
    t_start=(i-1)*5*250;
    t_end=((i-1)*5+10)*250;
    labels(i,2)=t_end;
    if((t_start>=1 && t_end<26432))
        labels(i,1)=1;
    elseif((t_start>=26432))
        labels(i,1)= 2;
    else
        labels(i,1)= 0;
    end 
end

%% 

d = designfilt('bandpassiir', 'FilterOrder', 2, 'HalfPowerFrequency1', 10, 'HalfPowerFrequency2', 30, 'DesignMethod', 'butter', 'SampleRate', fs);
filtered_signal=filtfilt(d, n_426);
for i=1:59
    if(labels(i,1)==1 || labels(i,1)==2)
        signal = n_426(labels(i,2)-10*250+1:labels(i,2));
        filtered_signal=filtfilt(d, signal);
        labels(i,3)=meanfreq(signal);
        labels(i,4)=medfreq(signal);
        labels(i,5)=sum(filtered_signal.^2,'all');
    end
end

normal_features=labels(2:20,:);
VFIB_features=labels(23:59,:);

%%

for i=1:59
    if(labels(i,1)==1 || labels(i,1)==2)
        signal = n_426(labels(i,2)-10*250+1:labels(i,2));
        labels(i,6)=min(signal);
        labels(i,7)=max(signal);
        labels(i,8)=labels(i,7)-labels(i,6);
        labels(i,9)=mean(findpeaks(signal),'all');
        labels(i,10)=sum(signal==0);
        labels(i,11)=var(signal);
    end
end

normal_features=labels(2:20,[6:11]);
VFIB_features=labels(23:59,[6:11]);

%%

[alarm_zero_crossing,t_zero_crossing] = va_detect(n_426,fs,'zero-crossing');

targets = labels((labels(:,1) == 1 | labels(:,1) == 2) , 1) == 2;
outputs_zero = alarm_zero_crossing((labels(:,1) == 1 | labels(:,1) == 2)) == 1;

figure
cm1 = confusionmat(targets, outputs_zero);
confusionchart(cm1)

targets_all= labels(: , 1);
outputs_zero_cross_all= alarm_zero_crossing+1;

figure
cm_all = confusionmat(targets_all, outputs_zero_cross_all);
confusionchart(cm_all)
title("max amplitude")