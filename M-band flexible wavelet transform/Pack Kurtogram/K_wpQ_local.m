%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [K,KQ] = K_wpQ_local(x,h,g,h1,h2,h3,nlevel,opt,level)%计算每层的峭度，二分层峭度存储在K中，三分层峭度存储在KQ中,level=nlevel时计算第一层，然后按步长-1，level减小，计算层递增

[a,d] = DBFB(x,h,g);                    % perform one analysis level into the analysis tree二分滤波并降采样后的低频信号a与高频信号d

N = length(a);                       
d = d.*(-1).^(1:N)';%高通信号*（-j）^n

K1 = kurt(a(length(h):end),opt);%??
K2 = kurt(d(length(g):end),opt);

if level > 2%倒数第三层前进行三分段
   [a1,a2,a3] = TBFB(a,h1,h2,h3);%以上一层的低频部分为基准分三段滤波
   [d1,d2,d3] = TBFB(d,h1,h2,h3);%以上一层的高频部分为基准分三段滤波
   %三分段后的峭度
   Ka1 = kurt(a1(length(h):end),opt);
   Ka2 = kurt(a2(length(h):end),opt);
   Ka3 = kurt(a3(length(h):end),opt);
   Kd1 = kurt(d1(length(h):end),opt);
   Kd2 = kurt(d2(length(h):end),opt);
   Kd3 = kurt(d3(length(h):end),opt);
else%倒数第一二层不计算三分段峭度，用0值代替
   Ka1 = 0;
   Ka2 = 0;
   Ka3 = 0;
   Kd1 = 0;
   Kd2 = 0;
   Kd3 = 0;
end

if level == 1
   K =[K1*ones(1,3),K2*ones(1,3)];%倒数第一层二分段峭度存储，每段由三个单元组成
   KQ = [Ka1 Ka2 Ka3 Kd1 Kd2 Kd3];%倒数第一层三分段峭度值存储在KQ中，实际值为零，每段长度为一个单元
end

if level > 1
   [Ka,KaQ] = K_wpQ_local(a,h,g,h1,h2,h3,nlevel,opt,level-1);%循环计算每层低频部分分为二段及三段的峭度,相当于level=level-1，包含嵌套循环
   [Kd,KdQ] = K_wpQ_local(d,h,g,h1,h2,h3,nlevel,opt,level-1);%循环计算每层高频部分分为二段及三段的峭度
   
   K1 = K1*ones(1,length(Ka));
   K2 = K2*ones(1,length(Kd));
   K = [K1 K2; Ka Kd];
   
   Long = 2/6*length(KaQ);
   Ka1 = Ka1*ones(1,Long);
   Ka2 = Ka2*ones(1,Long);
   Ka3 = Ka3*ones(1,Long);
   Kd1 = Kd1*ones(1,Long);
   Kd2 = Kd2*ones(1,Long);
   Kd3 = Kd3*ones(1,Long);
   KQ = [Ka1 Ka2 Ka3 Kd1 Kd2 Kd3; KaQ KdQ];
end

if level == nlevel
   K1 = kurt(x,opt);
   K = [ K1*ones(1,length(K));K];%未分解前即原始信号的峭度存储在K中第一行
   
   [a1,a2,a3] = TBFB(x,h1,h2,h3);%原始信号三分段滤波
   Ka1 = kurt(a1(length(h):end),opt);
   Ka2 = kurt(a2(length(h):end),opt);
   Ka3 = kurt(a3(length(h):end),opt);
   Long = 1/3*length(KQ);
   Ka1 = Ka1*ones(1,Long);
   Ka2 = Ka2*ones(1,Long);
   Ka3 = Ka3*ones(1,Long);   
   KQ = [Ka1 Ka2 Ka3; KQ(1:end-2,:)];
end

% --------------------------------------------------------------------