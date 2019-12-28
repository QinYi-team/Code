% =========================================================================
%                          Written by Yi Qin
% =========================================================================
function b=ReHankel(a)
[L,K]=size(a);
c=zeros(1,L+K-1);
% for n=1:L+K-1
%     if n<K+1
%         for i=1:n
%             c(n)=c(n)+a(i,n-i+1);
%         end
%         b(n)=c(n)/n;
%     elseif n>L
%         for i=1-K+n:L
%             c(n)=c(n)+a(i,n-i+1);
%         end
%         b(n)=c(n)/(K+L-n);
%     else
%         for i=1:K
%             c(n)=c(n)+a(n-i+1,i);
%         end
%         b(n)=c(n)/K;
%     end
%    
% end
for n=1:L+K-1
    if n<L
        for i=1:n
            c(n)=c(n)+a(i,n-i+1);
        end
        b(n)=c(n)/n;
    elseif n>K
        for i=(n+1-K):L
            c(n)=c(n)+a(i,n-i+1);
        end
        b(n)=c(n)/(L-(n-K));
    else
        for i=1:L
            c(n)=c(n)+a(i,n-i+1);
        end
        b(n)=c(n)/L;
    end
end

    
        














