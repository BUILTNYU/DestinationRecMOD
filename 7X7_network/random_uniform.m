function [Arms,Regret,Y,MaxUtil,ChosenUtil]=random_uniform(T,Ratings,RouteCost,RealTheta,poi)

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

% Random recommendation
t=1;
while t<=T
    for v=1:poi
        XX(v,2)=Ratings(v,1);
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=RealTheta*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);

    d=randi([1,poi]);       % pick a random alternative
    if RouteCost(d,t)<999
        X(t,2)=Ratings(d,1);
        X(t,3)=RouteCost(d,t);
        ChosenUtil(t,1)=RealTheta*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1; 
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);
        Arms(t,1)=d;
        Count(d,1)=Count(d,1)+1;
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