function [ALTS,PassPool] = PoolGen(poi,Alternatives,Points_NTA,Proportion)

% Pool of restaurants (POI, Point Of Interest): Randomly gathered from Yelp API
% Tagging random numbers to restaurants and select pool of restaurant
q = size(Alternatives,2)+1;
for p=1:size(Alternatives,1)
    Alternatives(p,q) = rand;
end
Alternatives=sortrows(Alternatives,size(Alternatives,2));
ALTS=Alternatives(1:poi,:);

% Pool of passengers: 1000 points for 35 pseudo-NTA zones (not exactly matched to official NTA zones)
% Tagging random numbers to passenger points
for i=1:35000
    Points_NTA(i,6)=rand;
end

k=1;
PassPool=[];

% Pick passengers regarding the proportion of number of originated internal trips in Manhattan
% Derived from MATSIM synthesized population (C2SMART Center project)
for i=1:35
    for j=1:1000
        if Points_NTA((i-1)*1000+j,6)<Proportion(i,1)
            PassPool(k,:)=Points_NTA((i-1)*1000+j,:);
            k=k+1;
        end
    end
end