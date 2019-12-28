function [F] = CreateFilters2(N,p,q,r1,s1,r2,s2,bet,alp,J)
% N : input signal length
% % p,q,r1,s1,r2,s2 : sampling parameters of wavelet filters
% bet, alp : filter parameters
% J : number of levels
% =========================================================================
%                          Written by Yi Qin
% =========================================================================

% make sure 'x' is a row vector of even length
N = N + mod(N,2); % N should be even

if (N * ((p/q)^J))*r1/s1/2 < 2,
    error('Too many subbands -- Reduce ''J''');
end

PQ = zeros(J,2);%the matrix including pi and qi,第一列是pi，第二列是qi
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

Hk = ones(1,N);

for n = 1:J,
    if n == J-10,
        a=1;
    end
    PQk = PQk*PQ(n,1)/PQ(n,2);  %第一个PQK为1
    
    pp = PQ(n+1,1);qq = PQ(n+1,2);
    rr1 = RS1(n,1);ss1 = RS1(n,2);
    rr2 = RS2(n,1);ss2 = RS2(n,2);
        

    f{1} = (1-bet)*N/2;
    f{2} = (N/2*pp/qq);
    f{3} = alp*N/2 ;
    f{4} = (rr1/ss1)*(N/2);
    f{5}= N/2;      
    
    for k = 1:5,
        f{k} = (f{k}*PQk);
    end
    [H,G1,G2] = MakeFilters(f);
    
    f{1} = ceil(f{1});  
    f{2} = floor(f{2});
    f{3} = ceil(f{3});  
    f{4} = floor(f{4});
    f{5} = ceil(f{5});
    % filter for positive frequencies
    Gk1 = Hk(1+(f{1}:f{4})).*G1(1:end);
    Gk2= Hk(1+(f{3}:f{5})).*G2(1:end);

    dd1 = min(rr1,length(Gk1));
    dd2 = min(rr2,length(Gk2));
    F{n,1} = Gk1(1:dd1);
    F{n,2} = Gk2(1:dd2);
    F{n,3} = f;
    
    %update the lowpass filter
    Hk(1+(f{1}:f{2})) = Hk(1+(f{1}:f{2})).*H(1+(f{1}:f{2})); 
    Hk(1+(f{2}+1:f{5}))=0;
    
end
F{J+1,1} = Hk(1:f{2});  % save low-pass filter 滤波器组最后一个存的是低通滤波器

%%%%%

function [H, G1,G2] = MakeFilters(f)
% Make frequency responses

% MAKE H0
w = (0:floor(f{2}));
k_pass = (w < f{1});                 % pass-band
k_trans = (w >= f{1});    % transition-band
b = (f{2}-f{1})/pi;
w_scaled = (w - f{1})/b;

H = zeros(size(w));
H(k_pass) = 1;
H(k_trans) = (1+cos(w_scaled(k_trans))) .* sqrt(2-cos(w_scaled(k_trans)))/2;
seq = sqrt(1 - H(k_trans).^2);

w = (ceil(f{1}):floor(f{4}));
k_pass = (w <= f{3}) & (w >= f{2});                 % pass-band
k_trans1 = (w <= f{2});    % transition-band1
k_trans2 = (w >= f{3});    % transition-band2

b = (f{4}-f{3})/pi;
if b > 0,
    w_scaled = (w - f{3})/b;
else
    w_scaled = 0*w;
end
   
G1 = zeros(size(w));
G1(k_pass) = 1;
G1(k_trans1) = seq;
G1(k_trans2) = (1+cos(w_scaled(k_trans2))) .* sqrt(2-cos(w_scaled(k_trans2)))/2;

seq = sqrt(1 - G1(k_trans2).^2);

w = (ceil(f{3}):ceil(f{5}));
G2 = zeros(size(w));
k_pass = (w > f{4});                 % pass-band
k_trans = (w <= f{4});    % transition-band
G2(k_pass) = 1;
G2(k_trans)=seq;

