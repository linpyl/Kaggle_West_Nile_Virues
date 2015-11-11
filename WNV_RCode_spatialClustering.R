library(data.table)
library(plyr)
library(ggplot2)
library(ggmap)

#***************   Breaking apart date *****************
train <- fread("train.csv")

train[,dMonth:=substr(train$Date,6,7)]
train[,dYear:=substr(train$Date,1,4)]
train[,dDay:=substr(train$Date,9,10)]

# Summing number of mosquitos based on date species and trap
f1 <- formula(NumMosquitos~Date+Species+Block+Trap+Latitude+Longitude+dMonth+ dDay + dYear)
tempDF1 <- aggregate(f1,data=train,FUN=sum)
tempDF1 <- tempDF1[order(tempDF1$Date),]


# summing the positive occurences based on same criterion
f2 <- formula(WnvPresent~Date+Species+Block+Trap+Latitude+Longitude+dMonth+ dDay + dYear)
tempDF2 <- aggregate(f2,data=train,FUN=sum)
tempDF2 <- tempDF2[order(tempDF2$Date),]

#New Data set
newTrain <- merge(tempDF1,tempDF2)
newTrain$WnvPresent[newTrain$WnvPresent>0] <- 1

wnv2007 <- subset(newTrain, newTrain$dYear=="2007")

# Wrote CSV to play with the data this morning
# was able to get response rate up to about 9% without much effort
write.csv(newTrain, file="train2.csv")


# ***************  Spatial Geodesic latitude longitude clustering *******
# Function to retrieve Haversine distance
geo.dist = function(df) {
      require(geosphere)
      d <- function(i,z){         # z[1:2] contain long, lat
            dist <- rep(0,nrow(z))
            dist[i:nrow(z)] <- distHaversine(z[i:nrow(z),1:2],z[i,1:2])
            return(dist)
      }
      dm <- do.call(cbind,lapply(1:nrow(df),d,df))
      return(as.dist(dm))
}


cords <- data.frame(long =train$Longitude, lat=train$Latitude)
cords <- unique(cords)


d <- geo.dist(cords)
hc     <- hclust(d) # hierarchical clustering
plot(hc)                 # dendrogram suggests 4 clusters
cords$clust <- cutree(hc,k=4)

#km <- kmeans(geo.dist(cords),centers=3)
#cords$clust <- km$cluster

#***********  Create the map of all unique data points **********
# ************** 4 clusters ***************
# getting the map
mapWnv <- get_map(location = c(lon = mean(cords$long), lat = mean(cords$ lat)), zoom = 10,
                      maptype = "satellite", scale = 2)

# plotting the map with trap locations
ggmap(mapWnv) + ggtitle("Clustered by Longitude & Latitude")+
geom_point(data=cords, aes(x=long, y=lat, color=factor(clust)), size=3)+
      scale_color_discrete("Cluster")+
      coord_fixed()

# ****** subset by WnvPresent ****
wnvPos <- subset(newTrain, newTrain$WnvPresent==1)
# creating data frame for the coordinates
cords.wnvP <- data.frame(long =wnvPos$Longitude, lat=wnvPos$Latitude)
cords.wnvP <- unique(cords.wnvP)

d.pos <- geo.dist(cords.wnvP)
hc2     <- hclust(d.pos) # hierarchical clustering
plot(hc2)            # dendrogram suggests 4 clusters
cords.wnvP$clust <- cutree(hc2,k=4)

#***********  Create the map of all unique data points w/WnvPresent = 1 **********
# ************** 4 clusters ***************
# Must have already got the map above


# plotting the map with trap locations
ggmap(mapWnv) + ggtitle("Clustered Long & Lat W/Wnv Present")+
      geom_point(data=cords, aes(x=long, y=lat, color=factor(clust)), size=4)+
      scale_color_discrete("Cluster")+
      coord_fixed()

#**** Map of both clusters and WnvPresent ****

ggmap(mapWnv) +  ggtitle("Wnv Present overlayed on Trap clusters") +
      geom_point(data=cords, aes(x=long, y=lat, color=factor(clust)), size=4)+
      geom_point(data = cords.wnvP, aes(x = long, y = lat, fill= 'Wnv Present'), size = 3)+
      scale_color_discrete("Cluster")+
      coord_fixed()


      
