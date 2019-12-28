function xH=hilbfir(x)
x=x(:);
% A length N=231 Remez Hilbert transformer with filtering procedure
% for long length data > N*3+1
% fp - Filter cutoff frequency (0.01>fp>0.5)
%
% ? 2011 Michael Feldman
% For use with the book "HILBERT TRANSFORM APPLICATION
% IN MECHANICAL VIBRATION", John Wiley & Sons, 2011
%

N=231-1;
fp=0.01;
h=firpm(N,[fp 0.5-fp]*2,[1 1],'Hilbert');
xH0=filter(h,1,x);

% 90 degree - phase forward and reverse digital filtering
n=ceil((N+1)/2);
l=length(x); a=1;
b=h(:);
xH=filter(b,a,x);
xH=xH(n:l-n-1);
xb=x(l:-1:l-3*N);
xHb=-filter(b,a,xb);
xHb=xHb(3*N:-1:n);
xH=[xH; xHb(3*N+2-3*n:3*N-n+1)];
  