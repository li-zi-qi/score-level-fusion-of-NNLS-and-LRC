function residual=NNLS(trainsample,ClassNum,train_label,y)

train_tol=size(trainsample,2);
options.TolX=1e-4;

residual=zeros(1,ClassNum);
xp = lsqnonneg(trainsample,y,options);

for j=1:ClassNum
    mmu=zeros(train_tol,1);
    ind=(j==train_label);
    mmu(ind)=xp(ind);
    residual(j)=norm(y-trainsample*mmu);
end

res_max=max(residual);
res_min=min(residual);
residual=(residual-res_min)/(res_max-res_min);