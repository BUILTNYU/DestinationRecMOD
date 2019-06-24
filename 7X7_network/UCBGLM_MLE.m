function F=UCBGLM_MLE(N,thetat,X,Y,t)

F=zeros(1,N);
for i=1:t-1
    F=F+(Y(i,1)-(1+exp(-thetat*X(i,:)'))^-1)*X(i,:);
end
    