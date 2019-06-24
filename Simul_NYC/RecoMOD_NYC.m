function [Thetat,Arms,Regret,Y,MaxUtil,ChosenUtil]=RecoMOD_NYC(alpha,T,tau,lambda,vehcap,ttc,gamma,poi,ALTS,PassPool)

%INPUTS
% alpha: exploration factor
% T: number of trials
% tau: number of initial random draws (tau<T) to initiate algorithm
% lambda: arrival rate
% pcap: passenger capacity on shuttle
% ttc: conversion factor from distance (km) to time (min)
% gamma: degree of congestion
% poi: number of recommending point of interest (<=1130)
% ALTS: subset of alternative pool (n=POI)
% PassPool: subset of passenger pool

% Prepare matrices
XX=ones(poi,3);         % feature vector for POI alternatives
Util=zeros(poi,1);      % utility values for POI alternatives
V=zeros(3,3);           % Variance matrix of feature vectors of chosen alternatives (for 3 dimensions: constant, ratings, routecost)
RouteCost=zeros(poi,T); % Providing RouteCost derived by insertion heuristics for each trial

% Archiving outputs for entire simulation
X=ones(T,3);            % feature vectors
Y=zeros(T,1);           % users' response (Accept: 1, Reject: 0)
Arms=zeros(T,1);        % chosen alternative
MaxUtil=zeros(T,1);     % maximum utility among alternatives at trial
ChosenUtil=zeros(T,1);  % real utility of chosen alternative at trial
Regret=zeros(T,1);      % calculated regret (difference between MaxUtil and ChosenUtil at trial)

% Archiving ouputs for each alternatives when chosen
ARM=struct;

for a=1:poi
    ARM(a).ThetaReal=[0,2,-6];      % real theta for each alternative [position for constant, coefficient of rating, coefficient of travel time]
    ARM(a).rating=ALTS(a,2);        % rating
    ARM(a).estconst=zeros(1,1);     % estimated varied alternative-specific constant
    ARM(a).chosenconst=zeros(1,1);  % real constant
    ARM(a).X=ones(1,3);             % feature vectors
    ARM(a).Y=zeros(1,1);            % acceptance
    ARM(a).count=0;                 % number of chosen times
end

% Randomly draw user-alternative-specific constants for each trial and simulation
ConstReal=zeros(poi,T);
for p=1:poi
    ConstReal(p,:)=norminv(rand([1,T]),ALTS(p,5),3);
end

%Initialization (make every alternative be chosen at least once)
t=1;
while t<=poi
    % Arranging randomized constants to alternatives
    for a=1:poi
        ARM(a).ThetaReal(1,1)=ConstReal(a,t);
    end
    % Calculate increasing route costs for all alternatives
    [RouteCost(:,t)]=InsHeurstic(lambda,vehcap,ttc,gamma,PassPool,ALTS);
    for v=1:poi
        XX(v,2)=ARM(v).rating;
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=ARM(v).ThetaReal*XX(v,:)';
    end
    % Find the alternative with the maximum utility
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);
    % Recommend t-th alternative
    if RouteCost(t,t)<999
        X(t,2)=ARM(t).rating;
        X(t,3)=RouteCost(t,t);
        ChosenUtil(t,1)=ARM(t).ThetaReal*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1;    %simulating real response from logit model
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);
        
        % Update the information
        ARM(t).count=ARM(t).count+1;
        ARM(t).X(ARM(t).count,1:3)=X(t,1:3);
        ARM(t).chosenconst(ARM(t).count,1)=ARM(t).ThetaReal(1,1);
        V=V+(X(t,:)')*X(t,:);
        
        % Monte Carlo simulation
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
    % Arranging randomized constants to alternatives
    for a=1:poi
        ARM(a).ThetaReal(1,1)=ConstReal(a,t);
    end
    % Calculate increasing route costs for all alternatives
    [RouteCost(:,t)]=InsHeurstic(lambda,vehcap,ttc,gamma,PassPool,ALTS);
    for v=1:poi
        XX(v,2)=ARM(v).rating;
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=ARM(v).ThetaReal*XX(v,:)';
    end
    % Find the alternative with the maximum utility
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);
    % Randomly choose d-th alternative
    d=randi([1,poi]);
    if RouteCost(d,t)<999
        X(t,2)=ARM(d).rating;
        X(t,3)=RouteCost(d,t);
        ChosenUtil(t,1)=ARM(d).ThetaReal*X(t,:)';
        Pr=(1+exp(-ChosenUtil(t,1)))^-1; 
        Regret(t,1)=MaxUtil(t,1)-ChosenUtil(t,1);
        ARM(d).count=ARM(d).count+1;
        ARM(d).X(ARM(d).count,1:3)=X(t,1:3);
%         ARM(d).V=ARM(d).V+(X(t,:)')*X(t,:);
        ARM(d).chosenconst(ARM(d).count,1)=ARM(d).ThetaReal(1,1);
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

% Building POI*tau vector of XF, POI*1 vector of YF
XF=zeros(tau,poi+2);    % augmented feature vectors (POI constants and coeffieints of 2 feature)
YF=zeros(tau,1);        % users' response

% Reordering and augmenting feature vectors
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

% Estimate theta based on learning during tau
G=@(theta)((YF'-(1+exp(-theta*XF')).^(-1))*XF);
Thetat=fsolve(G,zeros(1,poi+2));
thetaF(t,:)=Thetat;

% Recommending the best alternative after tau
for t=tau+1:T
    for a=1:poi
        ARM(a).ThetaReal(1,1)=ConstReal(a,t);
    end
    % Prepare Xta (temporary X for POI alternatives in trial t)
    XX=ones(poi,3);
    % Prepare matrix for alternative evaluation
    armst=zeros(poi,1);
    maxa=-inf;

    [RouteCost(:,t)]=InsHeurstic(lambda,vehcap,ttc,gamma,PassPool,ALTS);
    for v=1:poi
        XX(v,2)=ARM(v).rating;
        XX(v,3)=RouteCost(v,t);
        Util(v,1)=ARM(v).ThetaReal*XX(v,:)';
    end
    [~,maxarm]=max(Util);
    MaxUtil(t,1)=Util(maxarm,1);
    % Evaluate alternatives
    for i=1:poi
        % Bring features
%         Xta(i,2)=ALTS(i,2);
%         Xta(i,3)=RouteCost(i,t);
        % If recommending the location of passenger, exterminate
        % Otherwise, calculate the systematic utility and exploration term
        if XX(i,3)~=999
            armst(i,1)=[thetaF(t-1,i),thetaF(t-1,poi+1:poi+2)]*XX(i,:)'+alpha*sqrt(XX(i,:)*V^-1*XX(i,:)');
        else
            armst(i,1)=-9999;
        end
        % Find the alternative with the maximum value
        if armst(i,1)>maxa
            maxa=armst(i,1);
            X(t,1)=1;
            X(t,2)=XX(i,2);
            X(t,3)=XX(i,3);
            Arms(t,1)=i;
        end
    end
    
    % Update the information: number of trials recommended, feature at recommendation, user-and alternative-constant at recommendation
    ARM(Arms(t,1)).count=ARM(Arms(t,1)).count+1;
    ARM(Arms(t,1)).X(ARM(Arms(t,1)).count,1:3)=X(t,1:3);
    ARM(Arms(t,1)).chosenconst(ARM(Arms(t,1)).count,1)=ARM(Arms(t,1)).ThetaReal(1,1);
    
    % Simulate observation of acceptance
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

    % Estimate theta based on learning during tau
    GG=@(theta)((YF'-(1+exp(-theta*XF')).^(-1))*XF);
    Thetat=fsolve(GG,zeros(1,poi+2));
    thetaF(t,:)=Thetat;
    
    %update V
    V=V+(X(t,:)')*X(t,:);
end

% Archiving finally estimated constants
for a=1:poi
    ARM(a).estconst(1,1)=thetaF(t,a);
end