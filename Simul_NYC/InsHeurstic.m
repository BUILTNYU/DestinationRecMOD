function [IncRouteCost]=InsHeurstic(lambda,vehcap,ttc,gamma,PassPool,ALTS)

%INPUTS
% lambda: mean passenger arrival rate
% pcap: vehicle's passenger capacity
% ttc: travel time cost
% gamma: degree of congestion
% PassPool: pool of passenger (Pickup and drop-off points)
% ALTS: alternative pool

%OUTPUTS
% IncRouteCost: increase in route cost of a destination, 

% count available alternatives and passenger points
poi = size(ALTS,1);
psgpts = size(PassPool,1);

initvloc = randi([1,psgpts]);       % generate vehicle original location
pnum = poissinv(rand,lambda);       % generate number of passengers
Broute = [];                        % initialize vehicle route

%generate locations
if pnum > 0
    ExLoc = zeros(2*pnum,2);  % locations of each existing passenger
    for i = 1:2*pnum
        ExLoc(i,1) = i;
        ExLoc(i,2) = randi([1,psgpts]);
    end
    for i = 1:pnum
        while ExLoc(i,1) == ExLoc(i+pnum,1) % prevent departing and arriving at the same point
            ExLoc(i,1) = randi([1,psgpts]);
            ExLoc(i+pnum,1) = randi([1,psgpts]);
        end
    end
    
    % verifying repeated points
    ExLoc2 = sortrows(ExLoc,2);
    ExLoc2(1,3)=1;
    for i = 2:2*pnum
        % labeling to identify unique passenger points
        if ExLoc(i-1,2) == ExLoc(i,2)
            ExLoc2(i,3) = ExLoc2(i-1,3);
        else
            ExLoc2(i,3) = ExLoc2(i-1,3)+1;
        end
    end
    ExLoc3=sortrows(ExLoc2,1);
    ExLoc4=[ExLoc2(:,3),ExLoc2(:,2)];   %remove duplicated passenger points to build OD distance matrix
    ExLoc4=unique(ExLoc4,'rows');
    numpts=size(ExLoc4,1);
    %distance array from initial vehicle location to passenger points
    Distvloc = zeros(2*pnum,1);
    for i = 1:numpts
        Distvloc(i,1) = ttc*LatLongDist(PassPool(initvloc,2),PassPool(initvloc,3),PassPool(ExLoc4(i,2),2),PassPool(ExLoc4(i,2),3))*gamma;
    end
    
    %distance matrix among simulated passenger points
    Dist = zeros(numpts,numpts);
    for i = 1:numpts
        for j = 1:numpts
            Dist(i,j) = ttc*LatLongDist(PassPool(ExLoc4(i,2),2),PassPool(ExLoc4(i,2),3),PassPool(ExLoc4(j,2),2),PassPool(ExLoc4(j,2),3))*gamma;
        end
    end
           
    %double insertion heuristic to create route
    Broute = [0;1;pnum+1];      %"Before"-route
    Bload = [0;1;0];            %load after visiting node
    BRouteCost = Distvloc(ExLoc3(Broute(2,1),3),1)+Dist(ExLoc3(Broute(2,1),3),ExLoc3(Broute(3,1),3));   %cost of Broute
    for i = 2:pnum
        %Insert P, then D
        mincost = inf;
        minroute = Broute;
        rs = size(Broute,1);
        for j = 1:rs
            for k = j+1:rs+1
                TempLoad = [Bload(1:j,1); Bload(j,1)+1; Bload(j+1:rs,1)+1];
                TempRoute = [Broute(1:j,1); i; Broute(j+1:rs,1)];
                TempLoad = [TempLoad(1:k,1); TempLoad(k,1)-1; TempLoad(k+1:rs+1,1)-1];
                TempRoute = [TempRoute(1:k,1); i+pnum; TempRoute(k+1:rs+1,1)];
                if max(TempLoad) <= vehcap %vehicle capacity restriction
                    TempCost = Distvloc(ExLoc3(TempRoute(2,1),3),1);
                    rs2 = size(TempRoute,1);
                    for l = 2:rs2-1
                        TempCost = TempCost+Dist(ExLoc3(TempRoute(l,1),3),ExLoc3(TempRoute(l+1,1),3));
                    end
                    if TempCost < mincost
                        mincost = TempCost;
                        minroute = TempRoute;
                        minload = TempLoad;
                    end
                end
            end
        end
        Broute = minroute;      %update Broute
        Bload = minload;        %update load profile of Broute
        BRouteCost = mincost;   %update cost of Broute
    end

    %random trim of early legs before the 1st drop-off point
    count = 1;
    samenum = 1;
    while samenum == 1
        if Broute(count+1,1) <= pnum
           count = count+1;
        else
            samenum = 0;
        end
    end

    rcount = zeros(count,1);
    rcount(1,1) = Distvloc(ExLoc3(Broute(2,1),3),1);
    for i = 2:count
        rcount(i,1) = Dist(ExLoc3(Broute(i,1),3),ExLoc3(Broute(i+1,1),3));
    end
    rcount = rcount/sum(rcount);
    rcountcdf = rcount;
    for i = 2:size(rcount,1)
        rcountcdf(i,1) = rcountcdf(i-1,1)+rcount(i,1);
    end

    leg = 1;
    legseed = rand;
    stop = 0;
    i = 1;
    while stop == 0
        if legseed < rcountcdf(i,1)
            leg = i;
            stop = 1;
        end
        if i == size(rcount,1)
            stop = 1;
        else
            i = i+1;
        end
    end
    
    %keep information about route after trimmed point
    rs = size(Broute,1);
    vloc2 = Broute(leg,1);
    Broute = Broute(leg:rs,1);
    Bload = Bload(leg:rs,1);
    Broute(1,1) = 0;
    rs = size(Broute,1);
    if leg > 1
        BRouteCost = Dist(ExLoc3(vloc2,3),ExLoc3(Broute(2,1),3));
    elseif leg == 1
        BRouteCost = Distvloc(ExLoc3(Broute(2,1),3),1);
    end
    for i = 2:rs-1
        BRouteCost = BRouteCost+Dist(ExLoc3(Broute(i,1),3),ExLoc3(Broute(i+1,1),3));
    end
end

%generate a new passenger
pzone = randi([1,psgpts]);
if pnum > 0 %with existing passengers
    ExLoc(2*pnum+1,:) = [2*pnum+1, pzone];
    % verifying whether pzone is already included in matrix
    for i=1:numpts
        if pzone == ExLoc4(i,2)
            ExLoc3(2*pnum+1,:) = [2*pnum+1,pzone,i];
            break
        end
    end    
    % augment distance matrix
    if size(ExLoc3,1)==2*pnum
        % expand distance matrix for a new passenger and alternatives
        ExLoc3(2*pnum+1,:) = [2*pnum+1,pzone,numpts+1];
        ExLoc4(numpts+1,:)=[numpts+1,pzone];
        for i = 1:numpts
            Dist(numpts,i) = ttc*LatLongDist(PassPool(ExLoc4(i,2),2),PassPool(ExLoc4(i,2),3),PassPool(pzone,2),PassPool(pzone,3))*gamma;
            Dist(i,numpts) = Dist(numpts,i);
            for j = 1:poi
                Dist(numpts+10+j,i) = ttc*LatLongDist(PassPool(ExLoc4(i,2),2),PassPool(ExLoc4(i,2),3),ALTS(j,4),ALTS(j,3))*gamma;
                Dist(i,numpts+10+j) = Dist(numpts+10+j,i);
            end
        end
        Distvloc = [Distvloc;ttc*LatLongDist(PassPool(pzone,2),PassPool(pzone,3),PassPool(initvloc,2),PassPool(initvloc,3)*gamma)];    
    else
        % just adding distance matrix for POIs
        for i = 1:numpts
            for j = 1:poi
                Dist(numpts+10+j,i) = ttc*LatLongDist(PassPool(ExLoc4(i,2),2),PassPool(ExLoc4(i,2),3),ALTS(j,4),ALTS(j,3))*gamma;
                Dist(i,numpts+10+j) = Dist(numpts+10+j,i);
            end
        end
    end
    
    %determine extra cost
    IncRouteCost = zeros(poi,1);
    for d = 1:poi
        if d ~= pzone
            TempLoc = [ExLoc; [2*pnum+2,numpts+10+d]];
            %Insert P, then D
            mincost = inf;
            minroute = Broute;
            rs = size(Broute,1);
            if leg == 1
                TempCost = Distvloc(ExLoc3(Broute(2,1),3),1);
            else
                TempCost = Dist(vloc2, ExLoc3(Broute(2,1),3));
            end
            for j=1:rs
                for k=j+1:rs+1
                    TempLoad=[Bload(1:j,1); Bload(j,1)+1; Bload(j+1:rs,1)+1];
                    TempRoute=[Broute(1:j,1); 2*pnum+1; Broute(j+1:rs,1)];
                    TempLoad=[TempLoad(1:k,1); TempLoad(k,1)-1; TempLoad(k+1:rs+1,1)-1];
                    TempRoute=[TempRoute(1:k,1); 2*pnum+2; TempRoute(k+1:rs+1,1)];

                    if max(TempLoad)<=vehcap
                        if leg == 1
                            TempCost = Distvloc(ExLoc3(Broute(2,1),3),1);
                        else
                            TempCost = Dist(ExLoc3(vloc2,3),ExLoc3(Broute(2,1),3));
                        end
                        rs2=size(TempRoute,1);
                        for l=2:rs2-1
                            if l==k
                                TempCost=TempCost+Dist(ExLoc3(TempRoute(l,1),3),numpts+10+d);
                            elseif l==k+1
                                TempCost=TempCost+Dist(numpts+10+d,ExLoc3(TempRoute(l+1,1),3));
                            else
                                TempCost=TempCost+Dist(ExLoc3(TempRoute(l,1),3),ExLoc3(TempRoute(l+1,1),3));
                            end
                        end
                        if TempCost<mincost
                            mincost=TempCost;
                            minroute=TempRoute;
                            minload=TempLoad;
                            minExLoc=TempLoc;
                        end
                    end

                end
            end
            IncRouteCost(d,1)=mincost-BRouteCost;
        elseif d == pzone %avoid recommending the same place where new passenger is standing
            IncRouteCost(d,1)=999;
        end
    end
else  % without existing passengers
    for d=1:poi
        if d~=pzone
            IncRouteCost(d,1)=(LatLongDist(PassPool(pzone,2),PassPool(pzone,3),PassPool(initvloc,2),PassPool(initvloc,3))+LatLongDist(PassPool(pzone,2),PassPool(pzone,3),ALTS(d,4),ALTS(d,3)))*ttc*gamma;
        elseif d == pzone
            IncRouteCost(d,1)=999;
        end
    end

end