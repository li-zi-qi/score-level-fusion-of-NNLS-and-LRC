function residual=LRC(train_data,ClassNum,train_label,y)

residual=zeros(1,ClassNum);

for j=1:ClassNum
    X = train_data(:,j==train_label);
    beta = pinv(X'*X)*X'*y;
    test_project = X*beta;
    residual(j) = norm(y-test_project);
end

res_max = max(residual);
res_min = min(residual);
residual=(residual-res_min)/(res_max-res_min);