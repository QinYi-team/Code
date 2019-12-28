function xf=lpf(x,fp)

% A length N=231 Remez lowpass filter with filtering procedure 
% for long length data > (N-1)*3+1
% fp - Filter cutoff frequency (0.005>fp>0.05)

N=231-1;
h=firpm(N,[0 .001 fp .5]*2,[1 1 0 0],'l');
xf=filtfilt(h,1,x);


