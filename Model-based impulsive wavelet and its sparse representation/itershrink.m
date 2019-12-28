function [x1,x2,cost] = itershrink(x, scales, wn, sigma, fws, wavsupport, rwc, p1, A2, A2T, p2, lam1, lam2, mu1, mu2, itn)
% INPUT
% x : input signal signal
% wn, sigma, fws: parameters of shock wavelet
% lam1, lam2, mu1, mu2: regular parameters and Lagrangian parameters
% itn: iterative number

% OUTPUT
%   x1,x2 : two output components
%   cost :  cost function in iteration
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

c1 = shockwaveletcwt(x,scales,wn,sigma,fws,wavsupport);
c2 = A2T(x);

d1 = shockwaveletcwt(zeros(size(x)),scales,wn,sigma,fws,wavsupport);
d2 = A2T(zeros(size(x)));

T1 = lam1/(2*mu1);
T2 = lam2/(2*mu2);

N = length(x);
A = 1.1*max(abs(x));

C1 = (1/mu1)/(p1/mu1 + p2/mu2);
C2 = (1/mu2)/(p1/mu1 + p2/mu2);

for k = 1:itn
   
    u1 = soft(c1 + d1, T1) - d1;
    u2 = soft(c2 + d2, T2) - d2;
    
    c = x - shockwaveleticwt(u1,scales,wn,sigma,fws,wavsupport,rwc) - A2(u2);
  
    d1 = C1* shockwaveletcwt(c,scales,wn,sigma,fws,wavsupport);
    d2 = C2*A2T(c);
    
    c1 = d1 + u1;
    c2 = d2 + u2;
    
    cost(k) = lam1*sum(abs(c1(:))) + lam2*sum(abs(c2(:)));

end

x1 =  shockwaveleticwt(c1,scales,wn,sigma,fws,wavsupport,rwc);
x2 = A2(c2);


function rx = shockwaveleticwt(wcoefs,scales,wn,sigma,fws,wavsupport,rwc)  %%%This function and shockwaveletcwt can be improved by using the algorithm based on FFT
 pa=-sigma*wn;
pb=wn*sqrt(1-sigma^2);
[nLevel,SigLen]=size(wcoefs);       
rw = zeros(nLevel, SigLen);        

t = 0:1/fws:wavsupport; 
wav=sin(pb*t).*exp(pa*t);
th=max(wav);
for m = 1:nLevel
    a = scales(m);
    t = -a*wavsupport:1/fws:a*wavsupport;
    N=length(t);
    N1=round(N/2);
    
    t = 0:1/fws:a*wavsupport;
    wav=sin(pb*t/a).*exp(pa*t/a)/th;
    tt=-a*wavsupport:1/fws:-1/fws;
    fwav=sin(pb*tt/a).*exp(-pa*tt/a)/th;
    wav=[fwav wav];
    
    temp = conv(wcoefs(m,:),wav) / sqrt(abs(a));
    d=(length(temp)-SigLen)/2;
    first = floor(d)+1;
    rw(m,:)=temp(first:first+SigLen-1)/a^2;
end
rx=rwc*sum(rw);
