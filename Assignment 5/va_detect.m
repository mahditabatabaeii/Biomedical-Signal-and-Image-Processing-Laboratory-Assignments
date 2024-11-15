function [alarm,t] = va_detect(ecg_data,Fs,choice)
%VA_DETECT  ventricular arrhythmia detection skeleton function
%  [ALARM,T] = VA_DETECT(ECG_DATA,FS) is a skeleton function for ventricular
%  arrhythmia detection, designed to help you get started in implementing your
%  arrhythmia detector.
%
%  This code automatically sets up fixed length data frames, stepping through 
%  the entire ECG waveform with 50% overlap of consecutive frames. You can customize 
%  the frame length  by adjusting the internal 'frame_sec' variable and the overlap by
%  adjusting the 'overlap' variable.
%
%  ECG_DATA is a vector containing the ecg signal, and FS is the sampling rate
%  of ECG_DATA in Hz. The output ALARM is a vector of ones and zeros
%  corresponding to the time frames for which the alarm is active (1) 
%  and inactive (0). T is a vector the same length as ALARM which contains the 
%  time markers which correspond to the end of each analyzed time segment. If Fs 
%  is not entered, the default value of 250 Hz is used. 

  %  Template Last Modified: 3/4/06 by Eric Weiss, 1/25/07 by Julie Greenberg


%  Processing frames: adjust frame length & overlap here
%------------------------------------------------------
frame_sec = 10;  % sec
overlap = 0.5;    % 50% overlap between consecutive frames


% Input argument checking
%------------------------
if nargin < 2
    Fs = 250;  % default sample rate
end;
if nargin < 1
    error('You must enter an ECG data vector.');
end;
ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector


% Initialize Variables
%---------------------
frame_length = round(frame_sec*Fs);  % length of each data frame (samples)
frame_step = round(frame_length*(1-overlap));  % amount to advance for next data frame
ecg_length = length(ecg_data);  % length of input vector
frame_N = floor((ecg_length-(frame_length-frame_step))/frame_step); % total number of frames
alarm = zeros(frame_N,1);	% initialize output signal to all zeros
t = ([0:frame_N-1]*frame_step+frame_length)/Fs;

% Analysis loop: each iteration processes one frame of data
%----------------------------------------------------------

d = designfilt('bandpassiir', 'FilterOrder', 2, 'HalfPowerFrequency1', 10, 'HalfPowerFrequency2', 30, 'DesignMethod', 'butter', 'SampleRate', Fs);

if choice == "bandpower"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        filtered_seg=filtfilt(d, seg);
        band_power=sum(filtered_seg.^2,'all');
        %  Decide whether or not to set alarm . . .
        if band_power<2*10^6
            alarm(i) = 1;
        end
    end
elseif choice == "med"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        med_seg = medfreq(seg);
        %  Decide whether or not to set alarm . . .
        if med_seg>0.07
            alarm(i) = 1;
        end
    end
elseif choice == "mean"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        med_seg = meanfreq(seg);
        %  Decide whether or not to set alarm . . .
        if med_seg>0.07
            alarm(i) = 1;
        end
    end
elseif choice == "mean2"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        med_seg = meanfreq(seg);
        %  Decide whether or not to set alarm . . .
        if med_seg>0.02
            alarm(i) = 1;
        end
    end
elseif choice == "max"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        max_seg = max(seg);
        %  Decide whether or not to set alarm . . .
        if max_seg<325
            alarm(i) = 1;
        end
    end
elseif choice == "mean-R"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        mean_R = mean(findpeaks(seg));
        %  Decide whether or not to set alarm . . .
        if mean_R > -28
            alarm(i) = 1;
        end
    end
elseif choice == "max-min"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        peak = max(seg)-min(seg);
        %  Decide whether or not to set alarm . . .
        if peak < 315
            alarm(i) = 1;
        end
    end
elseif choice == "zero-crossing"
    for i = 1:frame_N
        %  Get the next data segment
        seg = ecg_data(((i-1)*frame_step+1):((i-1)*frame_step+frame_length));
        %  Perform computations on the segment . . .
        zero_cross = sum(seg==0);
        %  Decide whether or not to set alarm . . .
        if zero_cross > 10
            alarm(i) = 1;
        end
    end
end