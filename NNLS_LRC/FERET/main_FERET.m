clear
clc
close all

load('..\FERET_40x40.mat')

ClassNum = length(unique(gnd));
EachClassNum = 7;

Tr_num = 1:4;
w_val = 0.9:-0.1:0.1;
accuracy = zeros(length(w_val),length(Tr_num));

ii=1;
for train_num = Tr_num
    
    temp = zeros(1,EachClassNum);
    temp(1:train_num) = 1;
    
    train_ind = logical(repmat(temp,1,ClassNum));
    test_ind = ~train_ind;
    
    train_data = fea(:,train_ind);
    train_label = gnd(:,train_ind);
    
    test_data = fea(:,test_ind);
    test_label = gnd(:,test_ind);
    
    train_tol = length(train_label);
    test_tol = length(test_label);
    
    train_norm = normc(train_data);
    test_norm = normc(test_data);
    
    pre_label=zeros(1,test_tol);
    
    jj = 1;
    for w=w_val
        for i=1:test_tol
            
            y=test_norm(:,i);
            
            LRC_res = LRC(train_norm,ClassNum,train_label,y);
            NNLS_res = NNLS(train_norm,ClassNum,train_label,y);
            
            residual=(1-w)*LRC_res+w*NNLS_res;
            [~,ind]=min(residual);
            pre_label(i)=ind;
            
        end        
        accuracy(jj,ii)=sum(pre_label==test_label)/test_tol
        jj = jj+1;
    end
    ii = ii+1;
end