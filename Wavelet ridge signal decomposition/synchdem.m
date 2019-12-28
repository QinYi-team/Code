function [xi,Ai,phi]=synchdem(x,omega_i,fp);
% Synchronous demodulation of the component "xi" with the frequency 
% vector "omega_i" from the initial composition x.
% fp - Lowpass filter cutoff frequency (0.005>fp>0.1)


x=x(:); om=omega_i(:);
xH  = hilbfir(x); % Hilbert transform via FIR filter
%[x,xH]  = hilbturner(x); % Hilbert transform via Turner filter
A2=(xH.^2 + x.^2); A=sqrt(A2); Am=mean(A(200:length(x)-200));  % A, amplitude,   
cs=cumtrapz(om); xc =Am*cos(cs); xs =Am*sin(cs);
x1=x.*xc; x2=x.*xs; x3=xH.*xc; x4=xH.*xs;
Acos=(x1+x4)/Am; Asin=(x3-x2)/Am;

AcosM=lpf(Acos,fp); AsinM=lpf(Asin,fp);
%AcosM=ilpf(Acos,fp); AsinM=ilpf(Asin,fp);
Ai=sqrt(abs((AcosM).^2+(AsinM).^2));
phi=atan2(AsinM,AcosM);    % phase shift correction  
xi=Ai.*cos(cumtrapz(om)+phi);



