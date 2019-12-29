function snr=SNR(I,In)
% Calculate Signal to Noise Ratio (SNR)
% I :original signal
% In:noisy signal(ie. Original signal + noise signal)
% snr=10*log10(sigma2(I2)/sigma2(I2-I1))

snr = 10*log10(sum(abs(I(:)).^2)/sum(abs(I(:)-In(:)).^2));
