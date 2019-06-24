function [thetat,Arms,Regret,Y,MaxUtil,ChosenUtil]=contextbandit_thetans(T,tau,Ratings,ConstReal,RouteCost,RealTheta,poi,alpha)

% Matrices preparation
XX=ones(poi,3);
X=ones(T,3);
Y=zeros(T,1);
V=zeros(3,3);   %this is for 3 dimensions: constant, ratings, routecost
XF=zeros(T,poi+2);
YF=zeros(T,1);

% Struct preparation: accumulate information regarding alternatives
ARM=struct;
for a=1:poi
    ARM(a).ThetaReal=[0,RealTheta(1,2:3)];
    ARM(a).rating=Ratings(a,1);
    ARM(a).ConstEst=zeros(1,1);
    ARM(a).ChosenConst=zeros(1,1);
    ARM(a).X=ones(1,3);
    ARM(a).Y=zeros(1,1);
    ARM(a).count=0;
end

% theta=zeros(T,3);
ThetaF=zeros(T,poi+2);
Util=zeros(poi,1);
MaxUtil=zeros(T,1);
ChosenUtil=zeros(T,1);
Regret=zeros(T,1);
Arms=zeros(T,1);

%Initialization (make every alternative be chosen at least once)
t=1;
while t<=poi
    for a=1:poi
        ARM(a).ThetaReal(1,1)=ConstReal(a,t);
    end
    for v=1:poi
        XX(v,2)=ARM(v).rating;
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=ARM(v).ThetaReal*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);

    if RouteCost(t,t)<999
        X(t,2)=ARM(t).rating;
        X(t,3)=RouteCost(t,t);
        ChosenUtil(t,1)=ARM(t).ThetaReal*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1; %simulating real response
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);

        ARM(t).count=ARM(t).count+1;
        ARM(t).X(ARM(t).count,1:3)=X(t,1:3);
        ARM(t).ChosenConst(ARM(t).count,1)=ARM(t).ThetaReal(1,1);
        V=V+(X(t,:)')*X(t,:);

        if rand<Pr
            Y(t,1)=1;
            ARM(t).Y(ARM(t).count,1)=1;
        else
            Y(t,1)=0;
            ARM(t).Y(ARM(t).count,1)=0;
        end
    else
        t=t-1;
    end
    t=t+1;
end

%Random choice of alternative during remained learning period "tau"
while t<=tau
    for a=1:poi
        ARM(a).ThetaReal(1,1)=ConstReal(a,t);
    end
    for v=1:poi
        XX(v,2)=ARM(v).rating;
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=ARM(v).ThetaReal*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);

    d=randi([1,poi]);
    if RouteCost(d,t)<999
        X(t,2)=ARM(d).rating;
        X(t,3)=RouteCost(d,t);
        ChosenUtil(t,1)=ARM(d).ThetaReal*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1; %simulating real response
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);

        ARM(d).count=ARM(d).count+1;
        ARM(d).X(ARM(d).count,1:3)=X(t,1:3);
        ARM(d).ChosenConst(ARM(d).count,1)=ARM(d).ThetaReal(1,1);
        V=V+(X(t,:)')*X(t,:);

        if rand<Pr
            Y(t,1)=1;
            ARM(d).Y(ARM(d).count,1)=1;
        else
            Y(t,1)=0;
            ARM(d).Y(ARM(d).count,1)=0;
        end
    else
        t=t-1;
    end
    t=t+1;
end

% Building (POI+2)*tau vector of XF, (POI+2)*1 vector of YF
ttt=0;
for a=1:poi
    if ARM(a).count>0
        for b=1:ARM(a).count
            ttt=ttt+1;
            XF(ttt,a)=1;
            XF(ttt,poi+1:poi+2)=ARM(a).X(b,2:3);
            YF(ttt,1)=ARM(a).Y(b,1);
        end
    end
end

% F=@(thetat)myfun49(POI,thetat,XF,YF,ttt);
% thetat=fsolve(F,zeros(1,POI+2));
% ThetaF(t-1,:)=thetat;

G=@(theta)((YF'-(1+exp(-theta*XF')).^(-1))*XF);
thetat=fsolve(G,zeros(1,poi+2));
ThetaF(t-1,:)=thetat;

for t=tau+1:T
    for a=1:poi
        ARM(a).ThetaReal(1,1)=ConstReal(a,t);
    end
    
    %update X
    Xta=ones(3,poi);
       
    %select arm
    armst=zeros(poi,1);
    maxa=-inf;
    
    for v=1:poi
        XX(v,2)=ARM(v).rating;
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=ARM(v).ThetaReal*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);
    
    for i=1:poi
        Xta(2,i)=Ratings(i,1);
        Xta(3,i)=RouteCost(i,t);
        
        if Xta(3,i)~=999
            armst(i,1)=[ThetaF(t-1,i),ThetaF(t-1,poi+1:poi+2)]*Xta(:,i)+alpha*sqrt(Xta(:,i)'*V^-1*Xta(:,i));
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
    
    ARM(Arms(t,1)).count=ARM(Arms(t,1)).count+1;
    ARM(Arms(t,1)).X(ARM(Arms(t,1)).count,1:3)=X(t,1:3);
    ARM(Arms(t,1)).ChosenConst(ARM(Arms(t,1)).count,1)=ARM(Arms(t,1)).ThetaReal(1,1);
    
    %simulate observation of acceptance
    ChosenUtil(t,1)=ARM(Arms(t,1)).ThetaReal*X(t,:)';
    Pr=(1+exp(-ChosenUtil(t,1)))^-1;
    Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);
    if rand<=Pr
        Y(t,1)=1;
        ARM(Arms(t,1)).Y(ARM(Arms(t,1)).count,1)=1;
    else
        Y(t,1)=0;
        ARM(Arms(t,1)).Y(ARM(Arms(t,1)).count,1)=0;
    end
    
    % Augment XF and YF
    XF(t,Arms(t,1))=1;
    XF(t,poi+1:poi+2)=X(t,2:3);
    YF(t,1)=Y(t,1);
    
    GG=@(theta)((YF'-(1+exp(-theta*XF')).^(-1))*XF);
    thetat=fsolve(GG,zeros(1,poi+2));
    ThetaF(t,:)=thetat;
        
    %update V
    V=V+(X(t,:)')*X(t,:);
end

for a=1:poi
    ARM(a).ConstEst(1,1)=ThetaF(t,a);
end