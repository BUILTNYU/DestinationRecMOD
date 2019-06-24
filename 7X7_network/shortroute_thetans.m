function [Arms,Regret,Y,MaxUtil,ChosenUtil]=shortroute_thetans(T,Ratings,ConstRealPool,RouteCost,RealTheta,poi)

% Matrices preparation
XX=ones(poi,3);
X=ones(T,3);
Y=zeros(T,1);

Util=zeros(poi,1);
MaxUtil=zeros(T,1);
ChosenUtil=zeros(T,1);
Regret=zeros(T,1);
Arms=zeros(T,1);
Count=zeros(poi,1);

% Recommending alternative with least route cost increasing
t=1;
while t<=T
    for v=1:poi
        XX(v,2)=Ratings(v,1);
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=[ConstRealPool(v,t),RealTheta(1,2:3)]*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);

    RouteCostT(:,1)=RouteCost(:,t);
    for i=1:poi
        RouteCostT(i,2)=i;
    end
    SRouteCost=sortrows(RouteCostT,1,"ascend");
        
    % Tiebreaker fot alternatives with the same route cost increasing
    q=0;
    r=1;
    while q==0
        if SRouteCost(r+1,1)==SRouteCost(1,1)
            r=r+1;
        else
            q=q+1;
        end
    end
    d=SRouteCost(randi(r),2);   % Randomly pick one of alternatives with the least increasing cost
    
    X(t,2)=Ratings(d,1);
    X(t,3)=RouteCost(d,t);
    ChosenUtil(t,1)=[ConstRealPool(d,t),RealTheta(1,2:3)]*X(t,:)';
    Pr=(1+exp(-ChosenUtil(t,1)))^-1; 
    Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);
    Arms(t,1)=d;
    Count(d,1)=Count(d,1)+1;
    if rand<Pr
        Y(t,1)=1;
    else
        Y(t,1)=0;
    end
    t=t+1;
end