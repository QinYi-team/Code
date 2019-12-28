function [A,W]=teodemodu(x)
%Demodulation by TEO
%A---------Amplitude ·ùÖµ
%W---------Nomarlized frequency ¹éÒ»»¯ÆµÂÊ
%x---------Analyzed signal ´ı½âµ÷ĞÅºÅ
% =========================================================================
%                          Written by Yi Qin
% =========================================================================
%---------------------±ß½ç¾µÏñÑÓÍØ------------------%
N1=length(x);
N2=floor(N1/30);
x1=[x(N2:-1:2),x,x(N1-1:-1:N1-N2+1)];
x=x1;
%--------------------------------------------------%
s=[x(end-1) x(end) x x(1) x(2)]; %%¶Ô³ÆÑÓÍØ
N=length(s);
n=3:N-2;
load lowfir;
tex=s(n).^2-s(n+1).*s(n-1);
% tex=filtfilt(b,1,tex);       %texÂË²¨
m=2:N;
y0=s(m)-s(m-1);
n1=2:N-2;
tey=y0(n1).^2-y0(n1+1).*y0(n1-1);
% tey=filtfilt(b,1,tey);       %teyÂË²¨
n2=1:N-4;
temp=1-((tey(n2)+tey(n2+1))./(4*tex));
if any(abs(temp)>1)
    disp('abnormal'); 
    abindex=find(abs(temp)>1);
    for i=1:length(abindex)
        if abindex(i)<floor(N/2)
            k=1;
            while abs(temp(abindex(i)+k))>1
                k=k+1;
                if abindex(i)+k>N
                    disp('no point found');  
                    break;
                end
            end
            temp(abindex(i))= temp(abindex(i)+k);
        end
        if abindex(i)>=floor(N/2)
            k=1;
            while abs(temp(abindex(i)-k))>1
                k=k+1;
                if abindex(i)-k==0
                    disp('no point found');  
                    break;
                end
            end
            temp(abindex(i))= temp(abindex(i)-k);
        end
    end
end
W=acos(temp)/(2*pi);
% W(1)=W(3);
% W(2)=W(3);
% W(end)=W(end-2);
% W(end-1)=W(end-2);
A=sqrt(abs(tex./(1-temp.^2)));
W=W(N2:N1+N2-1);
A=A(N2:N1+N2-1);
% A(1)=A(3);
% A(2)=A(3);
% A(end)=A(end-2);
% A(end-1)=A(end-2);