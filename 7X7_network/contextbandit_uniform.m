function [thetat,Arms,Regret,Y,MaxUtil,ChosenUtil]=contextbandit_uniform(T,tau,Ratings,RouteCost,RealTheta,poi,alpha)

% Matrices preparation
XX=ones(poi,3);
X=ones(T,3);
Y=zeros(T,1);
V=zeros(3,3);   %this is for 3 dimensions: constant, ratings, routecost

Theta=zeros(T,3);
Util=zeros(poi,1);
MaxUtil=zeros(T,1);
ChosenUtil=zeros(T,1);
Regret=zeros(T,1);
Arms=zeros(T,1);
Count=zeros(poi,1);

%Initialization (make every alternative be chosen at least once)
t=1;
while t<=poi
    for v=1:poi
        XX(v,2)=Ratings(v,1);
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=RealTheta*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);

    if RouteCost(t,t)<999
        X(t,2)=Ratings(t,1);
        X(t,3)=RouteCost(t,t);
        ChosenUtil(t,1)=RealTheta*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1; %simulating real response
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);

        Count(t,1)=Count(t,1)+1;
        V=V+(X(t,:)')*X(t,:);

        if rand<Pr
            Y(t,1)=1;
        else
            Y(t,1)=0;
        end
    else
        t=t-1;
    end
    t=t+1;
end

%Random choice of alternative during remained learning period "tau"
while t<=tau
    for v=1:poi
        XX(v,2)=Ratings(v,1);
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=RealTheta*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);

    d=randi([1,poi]);
    if RouteCost(d,t)<999
        X(t,2)=Ratings(d,1);
        X(t,3)=RouteCost(d,t);
        ChosenUtil(t,1)=RealTheta*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1; %simulating real response
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);

        Count(d,1)=Count(d,1)+1;
        V=V+(X(t,:)')*X(t,:);

        if rand<Pr
            Y(t,1)=1;
        else
            Y(t,1)=0;
        end
    else
        t=t-1;
    end
    t=t+1;
end

F=@(thetat)UCBGLM_MLE(size(RealTheta,2),thetat,X,Y,t-1);
thetat=fsolve(F,zeros(1,size(RealTheta,2)));
Theta(t-1,:)=thetat;

for t=tau+1:T
    %update X
    Xta=ones(3,poi);
       
    %select arm
    armst=zeros(poi,1);
    maxa=-inf;
    
    for v=1:poi
        XX(v,2)=Ratings(v,1);
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=RealTheta*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);
    
    for i=1:poi
        Xta(2,i)=Ratings(i,1);
        Xta(3,i)=RouteCost(i,t);
        
        if Xta(3,i)~=999
            armst(i,1)=Theta(t-1,:)*Xta(:,i)+alpha*sqrt(Xta(:,i)'*V^-1*Xta(:,i));
        else
            armst(i,1)=-9999;
        end
        
        if armst(i,1)>maxa
            maxa=armst(i,1);
            X(t,1)=1;
            X(t,2)=Xta(2,i);
            X(t,3)=Xta(3,i);
            Arms(t,1)=i;
        end
    end
    
    Count(Arms(t,1),1)=Count(Arms(t,1),1)+1;
    
    %simulate observation of acceptance
    ChosenUtil(t,1)=RealTheta*X(t,:)';
    Pr=(1+exp(-ChosenUtil(t,1)))^-1;
    Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);
    if rand<=Pr
        Y(t,1)=1;
    else
        Y(t,1)=0;
    end
      
    F=@(thetat)UCBGLM_MLE(size(RealTheta,2),thetat,X,Y,t);
    thetat=fsolve(F,zeros(1,size(RealTheta,2)));
    Theta(t,:)=thetat;
    
    V=V+(X(t,:)')*X(t,:);
end