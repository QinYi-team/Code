% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 辅助函数——用来画出相关函数

% 相关画图
string = '画图结果保存';

x1 = -20:0.1:20;

%% 组合画图
x = -6:0.1:6;
% softplus
y_softplus  = softplus(x, nn);
yd_softplus = dev_softplus(x, nn);
% hexpo
y_hexpo  = hexpo(x, nn);
yd_hexpo = dev_hexpo(x, nn);
% swish
y_swish  = swish(x, nn);
yd_swish = dev_swish(x, nn);
% hreltanh_opt-1
nn.net{1}.thp = 0;
nn.net{1}.thn = -1.5;
nn.net{1}.err = zeros(size(x));
y_reltanh1  = hreltanh_opt(x, nn,1);
nn = dev_hreltanh_opt(x, nn,1);
yd_reltanh1 = nn.net{1}.d;
% hreltanh_opt-2
nn.net{1}.thp = 0.5;
nn.net{1}.thn = -1000;
nn.net{1}.err = zeros(size(x));
y_reltanh2  = hreltanh_opt(x, nn,1);
nn = dev_hreltanh_opt(x, nn,1);
yd_reltanh2 = nn.net{1}.d;
% ELU
y_ELU  = ELU(x, nn);
yd_ELU = dev_ELU(x, nn);
% tanh
y_tanh  = tanh(x);
yd_tanh = dev_tanh(x);
% RelU
y_ReLU  = ReLU(x, nn);
yd_ReLU = dev_ReLU(x, nn);
% LRelU
y_LReLU  = LReLU(x, nn);
yd_LReLU = dev_LReLU(x, nn);
% 汇总
y  = [y_reltanh1;y_reltanh2;y_tanh;y_ReLU;y_LReLU;y_softplus;y_ELU;y_hexpo;y_swish];
y_yd = 1111*ones(1,size(y,2));
yd = [yd_reltanh1;yd_reltanh2;yd_tanh;yd_ReLU;yd_LReLU;yd_softplus;yd_ELU;yd_hexpo;yd_swish];

xlswrite('结果保存\函数画图.xls',[y;y_yd;yd])



%% softplus
% softplus原函数
fs = 30;
x = -11:0.1:10;
nn.net{1}.thp = 0;
nn.net{1}.thn = -2;
nn.net{1}.err = ones(size(x));
y  = softplus(x, nn);
yd = dev_softplus(x, nn);


figure(1);
plot(x,y,'b','Linewidth',2);
str=strcat(string,'\','softplus');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('0.999*y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

%softplus 导数
figure(2);
plot(x,yd,'b','Linewidth',2);
str=strcat(string,'\','softplus 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1.05]); 
xlabel('x','FontSize',fs);
ylabel('yd*0.99','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');


%% hexpo
% hexpo原函数
fs = 30;
x = -11:0.1:10;
nn.net{1}.thp = 0;
nn.net{1}.thn = -2;
nn.net{1}.err = ones(size(x))
y  = hexpo(x, nn);
yd = dev_hexpo(x, nn);


figure(1);
plot(x,y,'b','Linewidth',2);
str=strcat(string,'\','hexpo ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -10 10]); 
xlabel('x','FontSize',fs);
ylabel('0.999*y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% hexpo导数
figure(2);
plot(x,yd,'b','Linewidth',2);
str=strcat(string,'\','hexpo 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -10 10]); 
xlabel('x','FontSize',fs);
ylabel('yd*0.99','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');









%% Swish
% Swish原函数
fs = 30;
x = -11:0.1:10;
nn.net{1}.thp = 0;
nn.net{1}.thn = -2;
nn.net{1}.err = ones(size(x))
y  = swish(x, nn);
yd = dev_swish(x, nn);


figure(1);
plot(x,y,'b','Linewidth',2);
str=strcat(string,'\','swish ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -3 3]); 
xlabel('x','FontSize',fs);
ylabel('0.999*y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% Swish导数
figure(2);
plot(x,yd,'b','Linewidth',2);
str=strcat(string,'\','swish 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1 1.05]); 
xlabel('x','FontSize',fs);
ylabel('yd*0.99','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');





%% hreltanh_opt
% hreltanh_opt原函数
fs = 30;
x = -11:0.1:10;
nn.net{1}.thp = 0;
nn.net{1}.thn = -2;
nn.net{1}.err = ones(size(x))
y  = hreltanh_opt(x, nn,1);
nn = dev_hreltanh_opt(x, nn,1);
yd = nn.net{1}.d;

figure(1);
plot(x,y,'b','Linewidth',2);
str=strcat(string,'\','hreltanh_opt ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('0.999*y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% hreltanh_opt导数
figure(2);
plot(x,yd,'b','Linewidth',2);
str=strcat(string,'\','hreltanh_opt 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1.05]); 
xlabel('x','FontSize',fs);
ylabel('yd*0.99','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');




%% reltanh_opt
% reltanh_opt原函数
fs = 30;
x = -11:0.1:10;
nn.net{1}.thp = 5;
nn.net{1}.thn = -10;
nn.net{1}.err = ones(size(x))
y  = reltanh_opt(x, nn,1);
nn = dev_reltanh_opt(x, nn,1);
yd = nn.net{1}.d;

figure(1);
plot(x,y,'b','Linewidth',2);
str=strcat(string,'\','reltanh_opt ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('0.999*y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% reltanh_opt导数
figure(2);
plot(x,yd,'b','Linewidth',2);
str=strcat(string,'\','reltanh_opt 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1.05]); 
xlabel('x','FontSize',fs);
ylabel('yd*0.99','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');



%% PReLU
% PReLU原函数
fs = 30;
x = -10:0.1:10;
nn.net{1}.k_PReLU = 0.25;
nn.net{1}.err = ones(size(x))
nn.opts.lr = 0.005
y  = PReLU(x, nn,1);
nn = dev_PReLU(x, nn,1);
yd = nn.net{1}.d;

figure(1);
plot(x,y,'b','Linewidth',2);
str=strcat(string,'\','PReLU ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('0.999*y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% PReLU导数
figure(2);
plot(x,yd,'b','Linewidth',2);
str=strcat(string,'\','PReLU 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1.05]); 
xlabel('x','FontSize',fs);
ylabel('yd*0.99','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');





%% ELU
% ELU原函数
fs = 30;
x = -10:0.1:10;
nn.opts.k_god = 1;
y  = ELU(x, nn);
yd = dev_ELU(x, nn);

figure(1);
plot(x,y,'b','Linewidth',2);
str=strcat(string,'\','ELU ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('0.999*y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% ELU导数
figure(2);
plot(x,yd,'b','Linewidth',2);
str=strcat(string,'\','ELU 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1.05]); 
xlabel('x','FontSize',fs);
ylabel('yd','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');




%% tanh
fs = 30;

x = -10:0.1:10;
nn.opts.th_god = 2.5;
%nn.opts.k_god  = 0.1;
y0  = tanh(x);
y00 = dev_tanh(x);
y1  = itanh_opt(x, nn);
y11 = dev_itanh_opt(x, nn);

% {
% tanh原函数
figure(1);
plot(x,y0,'b','Linewidth',2);
str=strcat(string,'\','tanh ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% tanh 导数
figure(2);
plot(x,y00,'b','Linewidth',2);
str=strcat(string,'\','tanh 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');


%% itanh_opt
fs = 30;

% itanh_opt 原函数
figure(3);
plot(x,y0,'r--','Linewidth',2);
hold on;
plot(x,y1,'Linewidth',2);
hold off;
legend('Tanh','Itanh',4);
str=strcat(string,'\','itanh_opt');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');


% itanh_opt 导数
figure(4);
plot(x,y00,'r--','Linewidth',2);
hold on;
plot(x,y11,'Linewidth',2);
hold off;
legend('Tanh','Itanh',1);
str=strcat(string,'\','itanh_opt 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');


%% ReLU
nn.opts.k_god = 0.05;
y2  = ReLU(x, nn);
y22 = dev_ReLU(x, nn);


% tanh原函数
figure(1);
plot(x,y2,'b','Linewidth',2);
str=strcat(string,'\','ReLU ');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -2 7]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

% tanh 导数
figure(2);
plot(x,y22,'b','Linewidth',2);
str=strcat(string,'\','ReLU 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

%}

%% itanh_opt  两条斜率
fs = 30;

y0  = tanh(x);
y00 = dev_tanh(x);

nn.opts.th_god = 1.15;
%nn.opts.k_god  = 0.1;
y4  = itanh_opt(x, nn);
y44 = dev_itanh_opt(x, nn);

nn.opts.th_god = 2.2;
%nn.opts.k_god  = 0.1;
y5  = itanh_opt(x, nn);
y55 = dev_itanh_opt(x, nn);


% itanh_opt 原函数
figure(4);
plot(x,y0,'r--','Linewidth',2);
hold on;
plot(x,y4,'Linewidth',2);
hold on;
plot(x,y5,'b--','Linewidth',2);
hold off;
L = legend('Tanh','Itanh (1.15)','Itanh (2.2)',4);
set(L,'FontName','Times New Roman','FontSize',fs-5,'LineWidth',1.5); 
str=strcat(string,'\','itanh_opt 双斜线');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -1.5 1.5]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');


% itanh_opt 导数
figure(4);
plot(x,y00,'r--','Linewidth',2);
hold on;
plot(x,y44,'Linewidth',2);
hold on;
plot(x,y55,'b--','Linewidth',2);
hold off;
L = legend('Tanh','Itanh (1.15)','Itanh (2.2)',1);
set(L,'FontName','Times New Roman','FontSize',fs-5,'LineWidth',1.5); 
str=strcat(string,'\','itanh_opt 双斜线 导数');  
set(gca,'FontName','Times New Roman','FontSize',fs,'LineWidth',1.5); 
axis([-7 7 -0.1 1]); 
xlabel('x','FontSize',fs);
ylabel('y','FontSize',fs);
saveas(gcf, str, 'tif');
saveas(gcf, str, 'pdf');

