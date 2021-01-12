data <- read.csv("Y:/CAU/기초과학/숙제2/covid-seoul-using-r/서울시 코로나19 확진자 현황.csv")

#데이터의 컬럼 명 확인

colnames(data) # "연번" "확진일" "환자번호" "국적" "환자정보" "지역" "여행력"   "접촉력"   "조치사항" "상태"     "이동경로" "등록일"   "수정일"   "노출여부"

#불필요한 컬럼 삭제

covid_data <- data[,-c(11:14)]
covid_data

#확진일을 날짜로 변환
#covid_data$확진일 <- as.POSIXct(covid_data$확진일, format = "%m.%d.")
covid_data$확진일 <- as.Date(covid_data$확진일, format = "%m.%d.")
summary(covid_data)
head(covid_data)

#공백 제거
covid_data <- as.data.frame(apply(covid_data,2,function(x)gsub('\\s+', '',x)))
summary(covid_data)

#분석할 내용

#(1) 각 구별 감염자

library(plyr)
covid_gu <- count(covid_data, "지역")
#NA값 삭제
covid_gu <- covid_gu[-c(1, 16),]
#frequ을 확진자로 변경
names(covid_gu)[2] <- c("확진자")
covid_gu
colnames(covid_gu)
#시각화
library(ggplot2)
ggplot(covid_gu, aes(지역, 확진자)) + geom_bar(stat='identity')

# 보기 쉽게 만들기 위해 정렬 및 색칠
colormap <- scale_fill_grey(start=0.7, end=0)
pal <- brewer.pal(11,"RdYlGn")
ggplot(covid_gu, aes(reorder(지역, -확진자), 확진자)) + geom_bar(stat='identity')
ggplot(covid_gu, aes(reorder(지역, -확진자), 확진자)) + geom_bar(stat='identity', aes(fill= 확진자)) + scale_fill_gradient2(low='yellow', high='red')

#최근 한달 구별 확진자 수
covid_month_gu <- count(subset(covid_data, 확진일 > "2020-11-18"),"지역")
covid_month_gu <- covid_month_gu[-c(1, 16), ]
names(covid_month_gu)[2] <- c("확진자")
covid_month_gu
ggplot(covid_month_gu, aes(reorder(지역, -확진자), 확진자)) + geom_bar(stat='identity', aes(fill= 확진자)) + scale_fill_gradient2(low='yellow', high='red')

#각 지역별 한달이내 확진자 수 비율
지역 <- covid_gu$지역
비율 <- covid_month_gu$확진자 / covid_gu$확진자
covid_gu_ratio <- data.frame(지역, 비율)
covid_gu_ratio
summary(covid_gu_ratio)
ggplot(covid_month_gu, aes(reorder(지역, -비율), 비율)) + geom_bar(stat='identity', aes(fill= 비율)) + scale_fill_gradient2(low='white', high='grey')


#(2) 해외 여행 국가별 감염자 수

covid_travel <- count(covid_data[covid_data$여행력 !='',], "여행력")
names(covid_travel)[2] = c("확진자")
covid_travel <- covid_travel[order(covid_travel$확진자, decreasing =TRUE),]
covid_travel
top_5 <- head(covid_travel)

ggplot(top_5, aes(reorder(여행력, -확진자),확진자)) + geom_bar(stat= "identity", fill="blue")

#(3) 접촉력 불확실 환자 기간별 증가 현황 및 예측
covid_data
uncertain <- count(subset(covid_data, 접촉력 == "감염경로조사중" & 확진일 < "2020-12-12"),"확진일")
names(uncertain)[2] <- c("감염경로불확실")
uncertain$확진일 <- as.Date(uncertain$확진일)
#uncertain$확진일 <- as.POSIXct(uncertain$확진일)
uncertain
class(uncertain$확진일)
library(scales)
ggplot(uncertain, aes(확진일,감염경로불확실, group = 1)) +geom_point(color = "red") + geom_line(color = "orange") + scale_x_date(breaks="month", labels=date_format("%b"))

library(TTR)
library(forecast)
install.packages("xts")
library(xts)
#convert to time series 
uncertain_ts <- ts(uncertain$감염경로불확실)
uncertain_ts
uncertain_decompose <- decompose(uncertain_ts)
uncertain_arima <- auto.arima(uncertain_ts)
summary(uncertain_arima)
uncertain_forecast <- forecast(uncertain_arima, h =10)
as.numeric(uncertain_forecast$mean)
plot(uncertain_forecast)

uncertain_recent <- count(subset(covid_data, 접촉력 == "감염경로조사중" & 확진일 >= "2020-12-12"),"확진일")
names(uncertain_recent)[2] <- c("실제 수")
uncertain_recent_numeric <- 
uncertain_recent$예측 <- as.numeric(uncertain_forecast$mean)[1:7]
uncertain_recent

#(4) 확진자 수 예측

covid_day <- count(covid_data, "확진일")
covid_day$확진일 <- as.Date(covid_day$확진일)
names(covid_day)[2] <- c("확진자수")

day <- ggplot(covid_day, aes(확진일, 확진자수, group = 1)) + geom_line(color = "black") 
#covid_day_ts <- ts(covid_day$확진자수, frequency = 120)
tail(covid_day)
#covid_day_decompose <- decompose(covid_day_ts)
#summary(covid_day_decompose)
#plot(covid_day_decompose)
covid_day_ts <- ts(covid_day$확진자수, frequency = 30)
covid_day
covid_day_decompose <- decompose(covid_day_ts)
summary(covid_day_decompose)
plot(covid_day_decompose)


covid_day_arima <- auto.arima(covid_day_ts)
summary(covid_day_arima)
covid_forecast <- forecast(covid_day_arima, h = 120)

head(as.numeric(covid_forecast$mean))
covid_forecast
plot(covid_forecast)

