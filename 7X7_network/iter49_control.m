% % % Before running, load('sampledata_49.mat');

% % dataset49.mat: inputs
%       Ratings49: rating of 49 zones on 7x7 network
%       ConstFlux49: assumed mean of constants of each zone which may be perceived differently by individuals

% % Constants and increase of route cost data were generated in advance to maintain the same setting along simulations of different combinations
%       ConstRealPool: 50000 combinations of generated constants covering 50000 trials (T*w=1000*50=50000)
%       IncRouteCost: 50000 combinations of increasing route cost precalculated covering 50000 trials (T*w=1000*50=50000)

% % Spatial information of zones (used in generating ConstRealPool and IncRouteCost)
%       Zdist49: distance matrix among 49 zones on 7x7 network
%       Zcoord49: coordinations of 49 zones on 7x7 network

% % This dataset was prepared and generated with this fixed parameters
%       lambda = 1;  % lambda: arrival rate
%       pcap = 4;    % pcap: passenger capacity on shuttle
%       ttc = 0.1;   % ttc: converter from distance (mi) to time (min)

function [Y]=iter49_control(w,A,B,Ratings49,ConstFlux49,ConstRealPool,IncRouteCost)
% w: number of conducting simulations per case
% A: type of constants 
%   1: uniform (theta)
%   2: alternative specific constant (ASC, theta_s), various constant for zones but fixed over the population
%   3: heterogenous alternative specific constant (hASC, theta_n,s), various constant for zones and the population
% B: type of recommendation
%   1: random
%   2: highest rating
%   3: least route cost increasing
%   4: RecMOD

% Simulation parameters
T=1000;         % number of trials in a simulation
tau=200;        % lenght of learning period in a simulation (tau<T)
poi=size(Ratings49,1);   % number of candidate alternatives
RealTheta = [-8,2,-6];      % assumed actual coefficient

% Varied parameter
Alpha=[1.5];    % set of exploration factor (alpha>0)

% % Or, run simulations with different alpha
% Alpha=[0;0.5;1;1.5;2];

% Prepare matrices
Theta=zeros(poi+2,w*size(Alpha,1));  % Estimated constants and coefficients 
Arms=zeros(T,w*size(Alpha,1));    % Chosen arm at trials in simulations
Regret=zeros(T,w*size(Alpha,1));    % Regret at trials in simulations
Y=zeros(T,w*size(Alpha,1));     % Acceptance (1)/Rejection (0) at trials in simulations
Util=zeros(T,2*w*size(Alpha,1));    % Values of utility function of chosen and optimal alternative

if B==1 % 1. Random recommendation
    for k=1:size(Alpha,1)
        for j=1:w
            RouteCost=IncRouteCost(:,((j-1)*T+1:j*T));
            if A==1 %1. theta
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=random_uniform(T,Ratings49,RouteCost,RealTheta,poi);
            elseif A==2 % 2. ASC (theta_s for s zones)
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=random_thetas(T,Ratings49,ConstFlux49,RouteCost,RealTheta,poi);
            elseif A==3 % 3. hASC (theta_n,s for s zones and n users)
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=random_thetans(T,Ratings49,ConstRealPool,RouteCost,RealTheta,poi);
            end
            Arms(:,(k-1)*w+j) = Arms(:,1);      % Indicate which alternative was recommended
            Regret(:,(k-1)*w+j) = Reg(:,1);     % Calculated regret
            Y(:,(k-1)*w+j) = Accept(:,1);       % Indicate whether recommendation was accepted or rejected
            Util(:,2*(k-1)*w+(2*j-1)) = MaxUtil(:,1);   % Maximally available systematic utility from alternatives
            Util(:,2*(k-1)*w+(2*j)) = ChosenUtil(:,1);  % Utility of recommended alternative
        end
    end

elseif B==2 % Recommending zone with highest rating
    for k=1:size(Alpha,1)
        for j=1:w
            RouteCost=IncRouteCost(:,((j-1)*T+1:j*T));
            if A==1 
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=highrate_uniform(T,Ratings49,RouteCost,RealTheta,poi);
            elseif A==2 
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=highrate_thetas(T,Ratings49,ConstFlux49,RouteCost,RealTheta,poi);
            elseif A==3 
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=highrate_thetans(T,Ratings49,ConstRealPool,RouteCost,RealTheta,poi);
            end
            Arms(:,(k-1)*w+j) = Arms(:,1);
            Regret(:,(k-1)*w+j) = Reg(:,1);
            Y(:,(k-1)*w+j) = Accept(:,1);
            Util(:,2*(k-1)*w+(2*j-1)) = MaxUtil(:,1);
            Util(:,2*(k-1)*w+(2*j)) = ChosenUtil(:,1);
        end
    end

elseif B==3 % Recommending zone with smallest route cost increasing
    for k=1:size(Alpha,1)
        for j=1:w
            RouteCost=IncRouteCost(:,((j-1)*T+1:j*T));
            if A==1 
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=shortroute_uniform(T,Ratings49,RouteCost,RealTheta,poi);
            elseif A==2 
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=shortroute_thetas(T,Ratings49,ConstFlux49,RouteCost,RealTheta,poi);
            elseif A==3 
                [Arms,Reg,Accept,MaxUtil,ChosenUtil]=shortroute_thetans(T,Ratings49,ConstRealPool,RouteCost,RealTheta,poi);
            end
            Arms(:,(k-1)*w+j) = Arms(:,1);
            Regret(:,(k-1)*w+j) = Reg(:,1);
            Y(:,(k-1)*w+j) = Accept(:,1);
            Util(:,2*(k-1)*w+(2*j-1)) = MaxUtil(:,1);
            Util(:,2*(k-1)*w+(2*j)) = ChosenUtil(:,1);
        end
    end

elseif B==4 % Using Contextual Bandit
    for k=1:size(Alpha,1)
        for j=1:w
            RouteCost=IncRouteCost(:,((j-1)*T+1:j*T));
            if A==1 
                [Thetatt,Arms,Reg,Accept,MaxUtil,ChosenUtil]=contextbandit_uniform(T,tau,Ratings49,RouteCost,RealTheta,poi,Alpha(k,1));
                Thetat=[zeros(1,poi-1),Thetatt];
            elseif A==2 
                [Thetat,Arms,Reg,Accept,MaxUtil,ChosenUtil]=contextbandit_thetas(T,tau,Ratings49,ConstFlux49,RouteCost,RealTheta,poi,Alpha(k,1));
            elseif A==3 
                [Thetat,Arms,Reg,Accept,MaxUtil,ChosenUtil]=contextbandit_thetans(T,tau,Ratings49,ConstRealPool,RouteCost,RealTheta,poi,Alpha(k,1));
            end
            Theta(:,((k-1)*w+(j-1))+1) = Thetat';       % Indicate estimated theta
            Arms(:,(k-1)*w+j) = Arms(:,1);
            Regret(:,(k-1)*w+j) = Reg(:,1);
            Y(:,(k-1)*w+j) = Accept(:,1);
            Util(:,2*(k-1)*w+(2*j-1)) = MaxUtil(:,1);
            Util(:,2*(k-1)*w+(2*j)) = ChosenUtil(:,1);
        end
    end
end

flname="result(%d,%d).xlsx";
flname1=sprintf(flname,A,B);
xlswrite(flname1,Theta,"theta");
xlswrite(flname1,Arms,"arm");
xlswrite(flname1,Regret,"regret");
xlswrite(flname1,Y,"accept");
xlswrite(flname1,Util,"utility");