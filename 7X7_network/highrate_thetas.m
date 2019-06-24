function [Arms,Regret,Y,MaxUtil,ChosenUtil]=highrate_thetas(T,Ratings,ConstFlux,RouteCost,RealTheta,poi)

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

% Recommending alternative with highest rating
t=1;
while t<=T
    for v=1:poi
        XX(v,2)=Ratings(v,1);
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=[ConstFlux(v,1),RealTheta(1,2:3)]*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);

    for i=1:poi
        Ratings(i,2)=i;
    end
    SRatings=sortrows(Ratings,1,"descend");
    d=SRatings(1,2);    % Alternative with the highest rating
    f=SRatings(2,2);    % Alternative with the 2nd highest rating
    if RouteCost(d,t)<999   % Verify the most highly rated alternative
        X(t,2)=Ratings(d,1);
        X(t,3)=RouteCost(d,t);
        ChosenUtil(t,1)=[ConstFlux(d,1),RealTheta(1,2:3)]*X(t,:)';
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
        X(t,2)=Ratings(f,1);    % Bring the alternative with the 2nd highest rating
        X(t,3)=RouteCost(f,t);
        ChosenUtil(t,1)=[ConstFlux(f,1),RealTheta(1,2:3)]*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1; 
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);
        Arms(t,1)=f;
        Count(f,1)=Count(f,1)+1;
        if rand<Pr
            Y(t,1)=1;
        else
            Y(t,1)=0;
        end
    end
    t=t+1;
end