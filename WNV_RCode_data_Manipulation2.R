require(timeDate)
require(plyr)
require(ISLR)
require(leaps)
require(data.table)
require(ggplot2)
require(ggmap)

# ************************
# Moving average function
# BECAREFUL Returns a TS object
ma <- function(x,n=5){filter(x,rep(1/n,n), sides=1)}

# ************************************************
# Shifts forward (positive number) or backwards(negative number)
shift<-function(x,shift_by){
      stopifnot(is.numeric(shift_by))
      stopifnot(is.numeric(x))
      
      if (length(shift_by)>1)
            return(sapply(shift_by,shift, x=x))
      
      out<-NULL
      abs_shift_by=abs(shift_by)
      if (shift_by > 0 )
            out<-c(tail(x,-abs_shift_by),rep(NA,abs_shift_by))
      else if (shift_by < 0 )
            out<-c(rep(NA,abs_shift_by), head(x,-abs_shift_by))
      else 
            out<-x
      out
}

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

#***********************************************************************

# Determining length of day by Forsythe's CBM Model
# Longitude and Latitude for OHare airport
# Longitude -87.900 Latitude 41.983
weather <- read.csv("weather.csv")
weather <- weather[weather$Station %in% c(1),]
weather$Date <- as.timeDate(weather$Date) 
avgT <- (weather$Tmax + weather$Tmin)/2

weather <- cbind(weather, avgT)
# Returns the integer value for the day of the year
dayYear <- dayOfYear(weather$Date)

#*****************************
# Forsythe's CBM model for day length
d <- dayYear

# Calculating epsilon in the model
e = asin(0.39795*cos(0.2163108 + 
                           2*atan(0.9671396*tan(0.00860*(dayYear-186)))))
# Converting latitude from degrees to radians
l = (41.983*pi)/180

# calculate length of day
# note the constant 0.8333 is converted to radians in the formula
#*****************  Should be Significant to Formula ************
dayLength <- 24 - (24/pi)* acos((sin(0.8333*pi/180)+
                                       sin(l)*sin(e))/(cos(l)*cos(e)))

#****************************************
# Add length of day to weather data frame
weather <- cbind(weather,dayLength)

#*****************************
#Moving average for day length
dayLength.ma4wk <- as.vector(ma(weather$dayLength,7))

#**********************************
#Add lagged day length To day frame
# 4 week lag and 5 week lag
dayL.lag.ma4wk <- shift(dayLength.ma4wk,-28)
dayL.lag.ma5wk <- shift(dayLength.ma4wk,-35)
weather <- cbind(weather,dayL.lag.ma4wk,dayL.lag.ma5wk)
weather$dayLength.ma4 = SMA(weather$dayLength, n=28)

#*******************************************
#Calculate average between 4 and 5 week MA's
#******** Should be significant to formula **********
avgDL.lag <- (dayL.lag.ma4wk+dayL.lag.ma5wk)/2

weather <- cbind(weather,avgDL.lag)
#***************************
#Calculate Relative humidity
weather$Tavg = as.numeric(as.character(weather$Tavg))
temperatureC = (weather$Tavg - 32)/1.8
DewPointC = (weather$DewPoint - 32)/1.8
weather$relHum = 100*(exp((17.625*DewPointC)/(243.04+DewPointC))/exp((17.625*temperatureC)/(243.04+temperatureC))) 
weather$relHum.ma4 = SMA(weather$relHum, n=28)


#precipitation 4 weeks 28 days
weather$PrecipTotal = as.numeric(as.character(weather$PrecipTotal))
weather$PrecipTotal[is.na(weather$PrecipTotal)] = 0.001
Precip.ma70 <- as.vector(ma(weather$PrecipTotal,28))

weather <- cbind(weather, Precip.ma70)

## Heavy rain: if there was heavy rain in the current day as well as the past 6 days (1=YES, 0=NO):
## 55mm = 2.165inch
HeavyRain.threshold = 2.165
tmp.1 = ifelse(weather$PrecipTotal>=2.165,1,0)
tmp.2 = rep(0,length(tmp.1))
for (i in 7:length(tmp.1)) {
    tmp.2[i] = sum(tmp.1[(i-6):i])
}
weather$HeavyRain = ifelse(tmp.2>0,1,0)


# Calculate 2 week temperature moving averages
# Maybe necessary to change to 18 day average
Tmax.ma2 <- as.vector(ma(weather$Tmax,14))
Tmin.ma2 <- as.vector(ma(weather$Tmin,14))
Tavg.ma2 <- as.vector(ma(weather$avgT,14))

#add to data frame
weather <- cbind(weather,Tmax.ma2,Tmin.ma2,Tavg.ma2)

# Add wind MA's
weather$AvgSpeed = as.numeric(as.character(weather$AvgSpeed))
AvgSpeed.ma2 <- as.vector(ma(weather$AvgSpeed,14))
ResultSpeed.ma2 <- as.vector(ma(weather$ResultSpeed,14))
weather <- cbind(weather,AvgSpeed.ma2,ResultSpeed.ma2)

LowWind.threshold = 3.72
weather$LowWind.byMean = ifelse(weather$AvgSpeed < LowWind.threshold, 1, 0)

## Weekly Moving Average of Dew Point
weather$DewPoint.ma1 = SMA(weather$DewPoint, n=7)



write.csv(weather, file="weather_new2.csv")

#======================================
train <- read.csv("train.csv")
# Summing number of mosquitos based on date species and trap
f1 <- formula(NumMosquitos~Date+Species+Latitude+Longitude)
tempDF1 <- aggregate(f1,data=train,FUN=sum)
tempDF1 <- tempDF1[order(tempDF1$Date),]

f2 <- formula(WnvPresent~Date+Species+Latitude+Longitude)
# summing the positive occurences based on same criterion
tempDF2 <- aggregate(f2,data=train,FUN=sum)
tempDF2 <- tempDF2[order(tempDF2$Date),]

newTrain <- merge(tempDF1,tempDF2)
newTrain$WnvPresent[newTrain$WnvPresent>0] <- 1

# Keeping only important attributes
#num.train = newTrain[c(1,3,6,8,9,11,12)]
newTrain$Date = as.Date(newTrain$Date, "%m/%d/%Y")

#weather <- read.csv("weather_new2.csv")
weather$Date = as.Date(weather$Date)

#train.num = merge(num.train, weather, by.x="Date", by.y="Date")
train.num = merge(newTrain, weather, by.x="Date", by.y="Date")

# ************************************
# Add trap clusters to data frame

cords <- data.frame(long =train.num$Longitude, lat=train.num$Latitude)
cords <- unique(cords)


d <- geo.dist(cords)
hc     <- hclust(d) # hierarchical clustering
plot(hc)                 # dendrogram suggests 4 clusters
cords$clust <- cutree(hc,k=8)

# **********************
# getting the map but not necessary
mapWnv <- get_map(location = c(lon = mean(cords$long), lat = mean(cords$ lat)), zoom = 10,
                  maptype = "satellite", scale = 2)

# plotting the map with trap locations
ggmap(mapWnv) + ggtitle("Clustered by Longitude & Latitude")+
      geom_point(data=cords, aes(x=long, y=lat, color=factor(clust)), size=3)+
      scale_color_discrete("Cluster")+
      coord_fixed()

# **********************************************************************
# Add spatial geodesic clusters to data
cluster <- ifelse(train.num$Latitude %in% cords$lat & train.num$Longitude %in% cords$long, cords$clust,0)
train.num <- cbind(train.num, cluster)


#==========================================================================================
# Data manipulation: compute the counts of HotSpot and log(HotSpot)
#==========================================================================================
## HotSpot for specificied Species
temp = newTrain
temp$Month = month(temp$Date)
temp$WnvPresent = as.numeric(as.character(temp$WnvPresent))
temp$Latitude = round(temp$Latitude,3)
temp$Longitude = round(temp$Longitude,3)

HotSpot = aggregate(WnvPresent ~ Latitude+Longitude+Species+Month, temp, FUN=sum)
colnames(HotSpot) = c("Latitude", "Longitude", "Species","Month","HotSpot")
HotSpot$log.HotSpot  = round(ifelse(HotSpot$HotSpot==0,0,log10(HotSpot$HotSpot)),4)

write.csv(HotSpot, file="hotspot.csv")

train.num$Month = month(train.num$Date)
train.num$Latitude = round(train.num$Latitude,3)
train.num$Longitude = round(train.num$Longitude,3)

## Hot Spots
temp = merge.data.frame(train.num, HotSpot, by=intersect(c("Latitude","Longitude","Species","Month"), 
                                                         c("Latitude","Longitude","Species","Month")))
train.num = temp

## HotSpot for UNspecificied Species-------------------------------------------------------------------------------------------
HotSpot.unspecificied = aggregate(WnvPresent ~ Latitude+Longitude+Month, temp, FUN=sum)                                       #
colnames(HotSpot.unspecificied) = c("Latitude", "Longitude","Month","HotSpot")                                                #
HotSpot.unspecificied$log.HotSpot  = round(ifelse(HotSpot.unspecificied$HotSpot==0,0,log10(HotSpot.unspecificied$HotSpot)),4) #
write.csv(HotSpot, file="hotspot_unspecificied.csv")                                                 #
##-----------------------------------------------------------------------------------------------------------------------------
#==========================================================================================
## Data manipulation:
#  Define LowWind: Label as 1 if there is LowWind(in weather table) during the days 
#  between 2 traps(in train table)
#==========================================================================================
temp.wind = cbind(weather[,c("Date","LowWind.byMean")], index=seq(1,nrow(weather)))
trap.date = data.frame(trap.date=unique(train.num$Date), LowWind.byMean=0)
trap.date.list = split(trap.date, year(unique(train.num$Date)))

for (i in 1:4) {
    for (j in 2:nrow(trap.date.list[[i]])) {
        a = temp.wind[temp.wind[,1] %in% trap.date.list[[i]][j-1,1],]$index + 1
        b = temp.wind[temp.wind[,1] %in% trap.date.list[[i]][j,1],]$index
        trap.date.list[[i]][j,2] = sum(temp.wind[a:b,2])
    }
}

trap.date = unsplit(trap.date.list, f=year(unique(train.num$Date)))
trap.date$LowWind.byMean[trap.date$LowWind.byMean>0] = 1
train.temp = merge(train.num, trap.date, by.x="Date", by.y="trap.date")




train.temp = train.temp[,c("Date","Month","Species","Latitude","Longitude",
                         "Tmax","Tmax.ma2","Tmin","Tmin.ma2","Tavg","Tavg.ma2","avgT",
                         "DewPoint","DewPoint.ma1",
                         "PrecipTotal","Precip.ma70","HeavyRain","relHum","relHum.ma4",
                         "ResultSpeed","ResultSpeed.ma2","ResultDir","AvgSpeed","AvgSpeed.ma2","LowWind.byMean.y",
                         "dayLength","dayL.lag.ma4wk","dayL.lag.ma5wk","avgDL.lag","dayLength.ma4",
                         "cluster","HotSpot","log.HotSpot",
                         "NumMosquitos","WnvPresent")]
train.temp$Month = as.factor(train.temp$Month)
train.temp$HeavyRain = as.factor(train.temp$HeavyRain)
train.temp$LowWind.byMean.y = as.factor(train.temp$LowWind.byMean.y)
train.temp$cluster = as.factor(train.temp$cluster)
train.temp$WnvPresent = as.factor(train.temp$WnvPresent)
#**************************************************************
#**************************************************************

write.csv(train.temp, file="train_num_Mosq.csv")

#**************************************************************
#**************************************************************


