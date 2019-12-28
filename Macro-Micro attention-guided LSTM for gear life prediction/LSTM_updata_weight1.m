% =========================================================================
%                          Written by Yi Qin and Sheng Xiang
% =========================================================================
function  [w,b,uo,wo,bo,uc,wc,bc,ui,wi,...
    bi,uf,wf,bf]=LSTM_updata_weight1...
    (t,yita,Error,w,b,uo,wo,bo,uc,wc,...
    bc,ui,wi,bi,uf,wf,bf,Sc,...
    C,i,f,c,o,h,train_data,B,A2,l,cell_num)
%% number of cells
input_num=size(uf,2);
cell_num=size(uf,1);
output_num=size(w,1);
%% weight update
for n=1:output_num
    for m=1:cell_num
        delta_bo(m,:)=Error(n,1)*w(n,m)*o(m)...
            *(1-o(m))*tanh(c(m,t));
        delta_uo(m,:)=delta_bo(m,:)*(B.*train_data(:,t)');
        
        delta_bc(m,:)=Error(n,1)*w(n,m)*...
            i(m)*(1-tanh(Sc(m))*tanh(Sc(m)))...
            *o(m)*(1-tanh(c(m,t))*...
            tanh(c(m,t)));
        delta_uc(m,:)=delta_bc(m,:)*(B.*train_data(:,t)');
        
        delta_bi(m,:)=Error(n,1)*w(n,m)*C(m)...
            *i(m)*(1-i(m))*o(m)*(1-...
            tanh(c(m,t))*tanh(c(m,t)));
        delta_ui(m,:)=delta_bi(m,:)*(B.*train_data(:,t)');
        if t~=1
            delta_wo(m,:)=delta_bo(m,:)*(h(:,t-1)'.*(1+exp(h(:,t-1)'*train_data(cell_num,t))/A2/sqrt(l+cell_num)));
            delta_wc(m,:)=delta_bc(m,:)*(h(:,t-1)'.*(1+exp(h(:,t-1)'*train_data(cell_num,t))/A2/sqrt(l+cell_num)));
            delta_wi(m,:)=delta_bi(m,:)*(h(:,t-1)'.*(1+exp(h(:,t-1)'*train_data(cell_num,t))/A2/sqrt(l+cell_num)));
            delta_bf(m,:)=Error(n,1)*w(n,m)*c(m,t-1)*f(m)*...
                (1-f(m))*o(m)*(1-tanh(c(m,t))*tanh(c(m,t)));
            delta_uf(m,:)=delta_bf(m,:)*train_data(:,t)';
            delta_wf(m,:)=delta_bf(m,:)*(h(:,t-1)'.*(1+exp(h(:,t-1)'*train_data(cell_num,t))/A2/sqrt(l+cell_num)));
        end
    end
    delta_w(n,:)=Error(n,1)*h(:,t)';
    delta_b(n,:)=Error(n,1);
    bo=bo-yita*delta_bo;
    uo=uo-yita*delta_uo;
    
    bc=bc-yita*delta_bc;
    uc=uc-yita*delta_uc;
    
    bi=bi-yita*delta_bi;
    ui=ui-yita*delta_ui;
    
    if t~=1
        wo=wo-yita*delta_wo;
        wc=wc-yita*delta_wc;
        wi=wi-yita*delta_wi;
        bf=bf-yita*delta_bf;
        uf=uf-yita*delta_uf;
        wf=wf-yita*delta_wf;
    end
end
w=w-yita*delta_w;
b=b-yita*delta_b;


end