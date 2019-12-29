% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
%% 此函数用于单一矩阵输入的，横坐标是迭代次数，纵坐标是性能的画图子函数调用
function nnfigure(figure_data , address_2th , figure_title , x_label , y_label)


num_point=length(figure_data);
x=1:num_point;
y=figure_data;
set(gcf, 'PaperPosition', [100 100 64 48]); 

if size(y,1)>1
    plot(x,y(1,:),'r','Linewidth',5);
    hold on;
    plot(x,y(2,:),'b','Linewidth',5);
    hold off;
else 
    plot(x,y,'r','Linewidth',5);
end

title(figure_title,'FontSize',70,'fontweight','bold');
xlabel(x_label,'FontSize',70);
ylabel(y_label,'FontSize',70);
str=strcat(address_2th ,'\',figure_title);     % 因为最后图片也是一部分地址，因此还需要加一条“\”
set(gca,'FontName','Times New Roman','FontSize',50,'LineWidth',5);     % 设置坐标轴的（是坐标轴）的线宽，文字大小等
saveas(gcf, str, 'fig');
saveas(gcf, str, 'tif');
%saveas(gcf, str, 'png');