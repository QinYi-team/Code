
function [w] = MFLexiWT(x,p,q,r1,s1,r2,s2,J,F)
% x : input signal
% p,q,r1,s1,r2,s2 : sampling parameters of wavelet filters
% J : number of levels (redundant -- to be removed)
% F : filters from 'CreateFilters2'
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

% make sure 'x' is a row vector of even length
x = x(:);x = x.';
L = length(x); 
N = L + mod(L,2); % x should be even length
x = [x zeros(1,N-L)];
clear L;

if (N * ((p/q)^J))*r1/s1/2 < 2,
    error('Too many subbands -- Reduce ''J''');
end
    
X = fft(x)/sqrt(N);

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
             
    G1 = F{n,1};
    G2 = F{n,2};
    f = F{n,3};
    
    %%%%Wavelet coefficients obtained by G1
    % positive frequencies
    dd = length(G1);
    sub = G1(1:end).*X(1+(f{1}:f{1}+dd-1));
    N1 = N*PQk*rr1/ss1/2;
    N1 = round(N1);
    sub2 = zeros(1,N1);    
    sub2(1:length(sub)) = sub;
    d = mod(f{1},N1);    
    sub2 = circshift(sub2.',d);
    sub2 = ifft(sub2);
    
    % negative frequencies
    g1 = N - f{1};    
    g4 = N - f{1} - dd + 1;
    sub = conj(G1(end:-1:1)).*X(1+(g4:g1));
    sub3 = zeros(1,N1);
    sub3(end-length(sub)+1:end) = sub;
    sub3 = circshift(sub3.',-(d-1));
    sub3 = ifft(sub3);
    w{n,1} = (sub2 + sub3)*sqrt(N1/2);  
    w{n,3} = 1i*(sub2 - sub3)*sqrt(N1/2);
  
     %%%%Wavelet coefficients obtained by G1
    dd2 = length(G2);
    sub = G2(1:end).*X(1+(f{3}:f{3}+dd2-1));
    N2 = N*PQk*rr2/ss2/2;
    N2 = round(N2);
    sub2 = zeros(1,N2);    
    sub2(1:length(sub)) = sub;
    d = mod(f{3},N2);    
    sub2 = circshift(sub2.',d);
    sub2 = ifft(sub2);
    
    % negative frequencies
    g1 = N - f{3};    
    g4 = N - f{3} - dd2 + 1;
    sub = conj(G2(end:-1:1)).*X(1+(g4:g1));
    sub3 = zeros(1,N2);
    sub3(end-length(sub)+1:end) = sub;
    sub3 = circshift(sub3.',-(d-1));
    sub3 = ifft(sub3);
    w{n,2} = (sub2 + sub3)*sqrt(N2/2);  
    w{n,4} = 1i*(sub2 - sub3)*sqrt(N2/2);
end
H = F{J+1};
N1 = PQ(J+1,1); 
sub = [X(1+(0:f{2}-1)).*H(1+(0:f{2}-1)) 0 X(end-f{2}+2:end).*H(f{2}:-1:2)];
sub2 = ifft(sub)*sqrt(N1);
w{J+1,1} = real(sub2);