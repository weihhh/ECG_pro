%data_cut=filter_data(1:5*10.^7);
clear all;
load('original_signal.mat');
spike=original_signal;
fs=64000;
gate=std(spike)*5;
lengthT=1;% spike waveform length time 2ms
pregateR=0.35;%proportion
deadT=0;%Refractory period
%wfpara=[gata,lengthT,pregateRmdeadT];%waveform paramiters have 4 factors
lengthN=round(fs*lengthT/1000);
segN=round(lengthN*pregateR);
pregateN=round(lengthN*pregateR);
posgateN=lengthN-pregateN-1;
deadN=round(deadT*fs/1000);

tscale=(1:lengthN)/fs*1000;
ls=length(spike);
midvalue=mean(spike);

if gate>midvalue
    ascend=1;
else
    ascend=0;
end

ind=1;
%index=[];
%waveforms=[];
endpos=ls-2*lengthN-1;%end position
ti=lengthN;

%detecting waveforms according to the alignment
datashift=spike-gate;
while ti<endpos
    if datashift(ti)*datashift(ti+1)<=0
        PatGates(ind)=ti;
        ind=ind+1;
    end
    ti=ti+1;
end

ti=ind-1;
PatGates=fliplr(PatGates);

index=PatGates;
%duplip=find(diff(PatGates)<=lengthN/4);
%index(duplip)=[];

ti=pregateN+1;
data=spike;
while ti<endpos
    IsSpike=0;
    if ascend
        if data(ti-1)<gate && data(ti)>=gate
            IsSpike=1;
        end
    elseif data(ti-1)>gate && data(ti)<gate %threshold below average
        IsSpike=1;
    end
    
    if IsSpike==1
        index(ind)=ti;
        ind=ind+1;
        ti=ti+posgateN+deadN;
    else
        ti=ti+1;
    end
end


waveforms=zeros(length(index),lengthN);
for ti=1:length(index)
    pos=index(ti);
    waveforms(ti,:)=spike((pos-pregateN):(pos+posgateN));
end

