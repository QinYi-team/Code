
function [x] = iMFLexiWT(x,w,N,p,q,r1,s1,r2,s2,F)
% w : Coefficients from RanDwt
% N : length of the output
% p,q,r1,s1,r2,s2 : sampling parameters of wavelet filters
% F : filters from 'CreateFilters2'
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

% make sure 'x' is a row vector of even length
N = N + mod(N,2);  
X = zeros(1,N);
J = size(w,1)-1;
XXX=fft(x)/sqrt(N);
%the sampling factors

PQ = zeros(J,2);
RS1 = zeros(J,2); %the matrix including ri1 and si1,第一列是ri1，第二列是si1
RS2 = zeros(J,2); %the matrix including ri2 and si2,第一列是ri2，第二列是si2
for k = 1:J,
    p0 = ceil(N * ((p/q)^k) ); p0 = p0 + mod(p0,2); % make sure p0 is even
    PQ(k,1) = p0;
    RS1(k,1) = round(N * ( (p/q)^(k-1) ) * r1/s1/2);
    RS2(k,1) = round(N * ( (p/q)^(k-1) ) * r2/s2/2);  
end
PQ(1,2) = N;
PQ(2:end,2) = PQ(1:end-1,1);
RS1(1:end,2) = PQ(1:end,2)/2;   
RS2(1:end,2) = PQ(1:end,2)/2;  
 
PQ = [1 1;PQ];
PQk = 1; 


for n = 1:J,
    PQk = PQk*PQ(n,1)/PQ(n,2);
    
    pp = PQ(n+1,1);qq = PQ(n+1,2);
    rr1 = RS1(n,1);ss1 = RS1(n,2);
    rr2 = RS2(n,1);ss2 = RS2(n,2);
        
    G1 = conj(F{n,1});
    G2 = conj(F{n,2});
    f = F{n,3};
   
    %%%%Wavelet coefficients obtained by G1
    % positive frequencies
    N1 = N*PQk*rr1/ss1/2;
    N1 = round(N1);
    d = mod(f{1},N1);
    
    
    sub = (fft(w{n,1} - 1i*w{n,3},N1))/sqrt(2*N1);
    sub = circshift(sub,-d);
    sub = sub.';    
    dd = length(G1);
    X(1+(f{1}:f{1}+dd-1)) = X(1+(f{1}:f{1}+dd-1)) + G1.*sub(1:dd);
    
    % negative frequencies
    g1 = N - f{1};    
    g4 = N - f{1} - dd + 1;     
    sub = (fft(w{n,1} + 1i*w{n,3},N1))/sqrt(2*N1);
    sub = circshift(sub,d-1);
    sub = sub.';    
    X(1+(g4:g1)) = X(1+(g4:g1)) + conj(G1(end:-1:1)).*sub(end-dd+1:end);
    
    
    %%%%Wavelet coefficients obtained by G2
    % positive frequencies
    N1 = N*PQk*rr2/ss2/2;
    N1 = round(N1);
    d = mod(f{3},N1);
    
    
    sub = (fft(w{n,2} - 1i*w{n,4},N1))/sqrt(2*N1);
    sub = circshift(sub,-d);
    sub = sub.';    
    dd = length(G2);
    X(1+(f{3}:f{3}+dd-1)) = X(1+(f{3}:f{3}+dd-1)) + G2.*sub(1:dd);
    
    % negative frequencies
    g1 = N - f{3};    
    g4 = N - f{3} - dd + 1;     
    sub = (fft(w{n,2} + 1i*w{n,4},N1))/sqrt(2*N1);
    sub = circshift(sub,d-1);
    sub = sub.';    
    X(1+(g4:g1)) = X(1+(g4:g1)) + conj(G2(end:-1:1)).*sub(end-dd+1:end); 
end
sub = fft(w{J+1,1})/sqrt(length(w{J+1,1}));
H = F{J+1,1};  
X(1:f{2}) = X(1:f{2}) + sub(1:f{2}).*H(1:f{2});
X(end-f{2}+2:end) = X(end-f{2}+2:end) + sub(end-f{2}+2:end).*H(f{2}:-1:2);
X(N/2+1)=X(N/2+1)/2;
x = ifft(X)*sqrt(N);
x = real(x);