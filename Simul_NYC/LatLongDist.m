% Calculated two different points with latitude and longitude
function [distance] = LatLongDist(lat1,long1,lat2,long2)

dlat = lat2/180*pi-lat1/180*pi;
dlong = long2/180*pi-long1/180*pi;

a = (sin(dlat/2))^2 + (sin(dlong/2))^2*cos(lat1/180*pi)*cos(lat2/180*pi);
c = 2*atan2(sqrt(a),sqrt(1-a));
distance = 6371*c;