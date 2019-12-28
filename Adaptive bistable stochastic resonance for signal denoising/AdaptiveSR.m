% =========================================================================
%                          Written by Yi Qin
% =========================================================================
function Y=AdaptiveSR(x,fs,ratio)
%x---------noisy signal 含噪信号
%fs--------sampling frequency原始采样频率
%ratio-----scale ratio变尺度比
%Y---------denoised signal输出降噪后信号
N=length(x);
fs1=fs/ratio;
t1=(0:N-1)'/fs1;
D1=fdomatrix(N,1);
D2=sdomatrix(N,1);
Tspan = [0 t1(end)];
IC = [0 1]; 
OY=x;
Y=x;
X(1)= -1;
X(2)=0.2;
for i=1:5
x1=x;
[T YY] = ode45(@(tt,y) myodefunc(tt,y,t1,x1,X(1),X(2)),Tspan,IC);
Y=YY(:,1);
Y=resample(Y,N,length(Y));
n=5;
[SWA,SWD] = swt(Y,n,'db5');
SWD=zeros(n,N);
Y1= iswt(SWA,SWD,'db5')';
Y=Y1;
err=sum((Y-OY).^2)/sum(OY.^2);
if err<0.08
    break;
end
OY=Y;
B=D2*Y+D1*Y-Y;
A=[Y Y.^3];
X=linsolve(A,B);
X=A\B;
end

function dy = myodefunc(t,y,t1,x,a,b)
f =interp1(t1,x,t);
dy(1,1) = y(2,1);
dy(2,1) = a*y(1,1)+b*y(1,1).^3+f-y(2,1);

function D=fdomatrix(n,fs)
D = diag(-ones(n - 1, 1), 0) + diag(ones(n - 2, 1), 1);
D = [D, zeros(n - 1, 1)]; 
D(end, end) = 1;
D(n,1)=1;
D(n,end)=-1;
D=fs*[D(:,2:n) D(:,1)];

function D=sdomatrix(n,fs)
D = diag(ones(n - 2, 1), 0) + diag(-2*ones(n - 3, 1), 1) + diag(ones(n - 4, 1), 2); 
D = [D, zeros(n - 2, 2)]; 
D(end, end - 1) = -2; D(end - 1, end - 1) = 1;
D(end, end) = 1;
D(n-1, 1) = 1;
D(n-1, end-1) = 1;
D(n-1, end) = -2;
D(n, 1) = -2;
D(n, 2) = 1;
D(n, end) = 1;
D=fs*fs*[D(:,2:n) D(:,1)];