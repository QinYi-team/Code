load 'voie1'
x = v1;
Fs = 1;

nlevel = 7;

% Pre-whitening the signal (optional)
prewith = input('Do you want to prewhiten the signal ? (no = 0 ; yes = 1): ');
if prewith == 1
   x = x - mean(x);
   Na = 100;
   a = lpc(x,Na);
   x = fftfilt(a,x);
   x = x(Na+1:end);			% it is very important to remove the transient of the whitening filter, otherwise the SK will detect it!!
end

% Fast Kurtogram (fb-based and stft-based)
c = Fast_Kurtogram(x,nlevel,Fs);


