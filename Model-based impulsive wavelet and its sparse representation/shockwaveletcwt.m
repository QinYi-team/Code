function wcoefs = shockwaveletcwt(Sig,Scales,wn,sigma,fws,wavsupport)
%============================================================%
%  Continuous Model-based impulsive Wavelet Transform                
%-----------Input--------------%
%    Sig: Input signal                                          
%    Scales: the vector of scales 
%    wn: central frequency  
%    sigma: decay rate 
%    fws: sampling frequency of wavelet  
%    wavsupport:support of wavelet 
%-----------output--------------%
%    wcoefs
%============================================================%
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

pa=-sigma*wn;
pb=wn*sqrt(1-sigma^2);
nLevel=length(Scales);                  
SigLen = length(Sig);                   
wcoefs = zeros(nLevel, SigLen);        

t = 0:1/fws:wavsupport; 
wav=sin(pb*t).*exp(pa*t);
th=max(wav);
for m = 1:nLevel                         
    a = Scales(m);
    t = -a*wavsupport:1/fws:a*wavsupport;
    N=length(t);
    N1=round(N/2);
    
    t = 0:1/fws:a*wavsupport;
    wav=sin(pb*t/a).*exp(pa*t/a)/th;
    tt=-a*wavsupport:1/fws:-1/fws;
    fwav=sin(pb*tt/a).*exp(-pa*tt/a)/th;
    wav=[fwav wav];
    Morl=fliplr(wav);
    temp = conv(Sig,Morl) / sqrt(abs(a));
    d=(length(temp)-SigLen)/2;
    first = floor(d)+1;
    wcoefs(m,:)=temp(first:first+SigLen-1);
end