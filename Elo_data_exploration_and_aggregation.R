library(ggplot2)
library(dplyr)
library(tidyr)
library(gmodels)
library(anytime)
library(rpart)
library(partykit)
library(xgboost)
library(Matrix)
library(caret)
library(glmnet)
library(ranger)
library(MASS)
library(solitude)


#for caret if cannot load automaticly: 
install.packages(c("lattice","ModelMetrics","recipes"))

#expand memory limits
memory.limit()
memory.limit(size=75000)

#load data
train<-read.csv("C:/Users/liz/Desktop/train.csv",stringsAsFactors = FALSE)
hist<-read.csv("C:/Users/liz/Desktop/historical_transactions.csv")
merch<-read.csv("C:/Users/liz/Desktop/merchants.csv")
new<-read.csv("C:/Users/liz/Desktop/new_merchant_transactions.csv")
test<-read.csv("C:/Users/liz/Desktop/test.csv",stringsAsFactors = FALSE)

##############################
#Data Explore#
##############################
#visualize distribution of target value
hist(train$target)

#Compare variables in Train and Test dataset
barplot(table(train$first_active_month),col="red",main= "Distribution of first active month in Train and Test Data")
barplot(table(test$first_active_month),col="blue",add=T)

hist(train$feature_1,col="red",main="Distribution of feature1 in Train and Test Data")
hist(test$feature_1,col="blue",add=T)

hist(train$feature_2,col="red",main="Distribution of feature2 in Train and Test Data")
hist(test$feature_2,col="blue",add=T)

hist(train$feature_3, col="red",main="Distribution of feature3 in Train and Test Data")
hist(test$feature_3, col="blue",add=T)


#number of unique card id in historical data 
length(unique(hist$card_id))  #325540
#number of unique merchants in historical data
length(unique(hist$merchant_id))  # 326312

head(hist$month_lag)
head(hist$purchase_date)

head(new$month_lag)
head(new$purchase_date)

max(new$month_lag) #2
min(new$month_lag) #1

min(hist$month_lag) #-13
max(hist$month_lag) #0


#check na in the hist dataset
s=matrix(0,14,1)
for (i in 1:14){
  s[i,1]=sum(is.na(hist[,i]))
}

print(s)

# I found that only catgory 2 has missing values
sum(is.na(hist$category_2))/nrow(hist) # 0.09 missing 

#check na in the new dataset
s=matrix(0,14,1)
for (i in 1:14){
  s[i,1]=sum(is.na(new[,i]))
}

print(s)

# I found that only catgory 2 has missing values
sum(is.na(new$category_2))/nrow(new) # 0.05 missing 

#check na in merch dataset
s=matrix(0,22,1)
for (i in 1:22){
  s[i,1]=sum(is.na(merch[,i]))
}

print(s)

# 13 na values in aveerage sales lag 3, lag6, and lag 12
# 11887 na values in category 2 


summary(hist)
unique(hist$category_2)

summary(merch)
boxplot(merch$avg_sales_lag3)
boxplot(merch$avg_purchases_lag3)

#########################
#Data Aggregation#
#########################

# combine the hist and new transactions

joinhist_new<-full_join(hist,new,by=c('authorized_flag','card_id','city_id','category_1','installments','category_3',
                                      'merchant_category_id','merchant_id','month_lag','purchase_amount','purchase_date',
                                      'category_2','state_id','subsector_id'))

head(joinhist_new)

trans_merch<-arrange(joinhist_new,card_id) %>% left_join(merch,by='merchant_id')

list=trans_merch$subsector_id.x==trans_merch$subsector_id.y
table(list)["FALSE"]

for (j in 1:length(list)) {
  
  if (list[j]=="FALSE"){
    print(trans_merch$subsector_id.x[j])
    print(trans_merch$subsector_id.y[j])
    
  }
  
}

# After checking the columns with same name between these two dataset, I decide to keep both because they have values different from each other( find jounals to back u up)
all(trans_merch$category_1.x==trans_merch$category_1.y) #F 9.76% different values, 89.73% same values
all(trans_merch$city_id.x==trans_merch$city_id.y) #F 29.05% different values 
all(trans_merch$merchant_category_id.x==trans_merch$merchant_category_id.y) #F 15.92% different values
all(trans_merch$category_2.x==trans_merch$category_2.y) #F 2.577% different values
all(trans_merch$state_id.x==trans_merch$state_id.y) #F 13.53% different values
all(trans_merch$subsector_id.x==trans_merch$subsector_id.y) #F 12.12% different values

sum_trans_merch<-arrange(joinhist_new,card_id) %>% left_join(merch,by=c('merchant_id','city_id','category_1','merchant_category_id','category_2','state_id',
                                                                        'subsector_id'))

sum_trans_merch$authorized_flag<-ifelse(sum_trans_merch$authorized_flag=='Y',1,0)

#aggregate transaction records

id_aml <- sum_trans_merch%>% group_by(card_id) %>% summarize_at(c("city_id","merchant_category_id","state_id","subsector_id","merchant_group_id","active_months_lag3","active_months_lag6","active_months_lag12"),funs(min,max,mean,sd,n_distinct),na.rm=TRUE)

#active_month_lag3 and active_month_lag6 shows very little variations, consider to remove them
#id_aml <- sum_trans_merch%>% group_by(card_id) %>% summarize_at(c("city_id","merchant_category_id","state_id","subsector_id","merchant_group_id","active_months_lag12"),funs(min,max,mean,sd,n_distinct),na.rm=TRUE)

sum_num<- sum_trans_merch%>% group_by(card_id)%>%summarize_at(c("purchase_amount","month_lag","numerical_1","numerical_2","avg_sales_lag3","avg_sales_lag6","avg_sales_lag12","avg_purchases_lag3","avg_purchases_lag6","avg_purchases_lag12"), funs(min,max,mean,sd,sum),na.rm=TRUE)

flag_inst<- sum_trans_merch%>% group_by(card_id) %>% summarize_at(c("authorized_flag",'installments'),funs(sum),na.rm=TRUE)%>% setNames(., c("card_id", "authorized_flag_sum", "installments_sum"))

c1<- sum_trans_merch %>% group_by(card_id,category_1) %>% summarize(num=n())%>% spread(category_1, num) %>% setNames(., c("card_id", "N_c1", "Y_c1"))
c1[is.na(c1)]<-0

c2<- sum_trans_merch %>% group_by(card_id,category_2) %>% summarize(num=n())%>% spread(category_2, num) 
c2<-c2[,-7]
c2[is.na(c2)]<-0

c3<- sum_trans_merch %>% group_by(card_id,category_3) %>% summarize(num=n())%>% spread(category_3, num) %>% setNames(., c("card_id", "A_c3","B_c3","C_c3")) 
c3<-c3[,-5]
c3[is.na(c3)]<-0

c4<- sum_trans_merch %>% group_by(card_id,category_4) %>% summarize(num=n())%>% spread(category_4, num) %>% setNames(., c("card_id", "N_c4", "Y_c4","NA_c4"))
c4<-c4[,-4]
c4[is.na(c4)]<-0

mrs<- sum_trans_merch %>% group_by(card_id,most_recent_sales_range) %>% summarize(num=n())%>% spread(most_recent_sales_range, num) %>% setNames(., c("card_id", "A_sales","B_sales","C_sales","D_sales","E_sales","NA_sales"))
mrs<-mrs[,-7]
mrs[is.na(mrs)]<-0

mrp<- sum_trans_merch %>% group_by(card_id,most_recent_purchases_range) %>% summarize(num=n())%>% spread(most_recent_purchases_range, num) %>% setNames(., c("card_id", "A_purchases","B_purchases","C_purchases","D_purchases","E_purchases","NA_purchases"))
mrp<-mrp[,-7]
mrp[is.na(mrp)]<-0

#number of unique categories and merchants 
uniq_cm <- sum_trans_merch %>% group_by(card_id)%>%
  summarise_at(vars(starts_with("merchant_"),starts_with("category")), n_distinct, na.rm = TRUE)
colnames(uniq_cm)[2:8]<-paste( colnames(uniq_cm)[2:8], "uniqnum",sep = "_")

#the time lap of purchase and the average time lap 
timelap <- sum_trans_merch %>%
  group_by(card_id) %>% summarise(no_trans=n(), pur_lap = as.integer(diff(range(as.Date(purchase_date)))),
                                  avg_pur_lap = as.integer(mean(abs(diff(order(as.Date(purchase_date)))))))

#most recent transaction date
mostrecent_trans<- sum_trans_merch %>% group_by(card_id) %>% summarise(mrec_trans=max(as.Date(purchase_date)))
mostrecent_trans<- mostrecent_trans %>% mutate(rectyear=substring(mrec_trans,1,4),rectmonth=substring(mrec_trans,6,7), rectday=weekdays(mrec_trans))

#days between first active month and the most recent purchase date
# year and month of activation time 


train<-train %>% left_join(mostrecent_trans, by="card_id") %>%
  mutate(year=substring(first_active_month,1,4),month=substring(first_active_month,6,7), 
         active_length=as.integer(mrec_trans-anydate(first_active_month)))

fulltrain<- train %>% left_join(id_aml,by="card_id") %>% 
  left_join(sum_num,by="card_id") %>% 
  left_join(flag_inst,by="card_id") %>%
  left_join(c1,by="card_id") %>% left_join(c2,by="card_id") %>% 
  left_join(c3,by="card_id") %>% left_join(c4,by="card_id") %>%
  left_join(mrs,by="card_id") %>% left_join(mrp,by="card_id") %>%
  left_join(uniq_cm,by="card_id") %>% 
  left_join(timelap,by="card_id")

colnames(fulltrain)[108]<-"c2_1"
colnames(fulltrain)[109]<-"c2_2"
colnames(fulltrain)[110]<-"c2_3"
colnames(fulltrain)[111]<-"c2_4"
colnames(fulltrain)[112]<-"c2_5"


#use same data set to aggregate test set
fulltest<- test %>% left_join(mostrecent_trans, by="card_id") %>%
  mutate(year=substring(first_active_month,1,4),month=substring(first_active_month,6,7), 
         active_length=as.integer(mrec_trans-anydate(first_active_month)))%>%
  left_join(id_aml,by="card_id") %>% 
  left_join(sum_num,by="card_id") %>% 
  left_join(flag_inst,by="card_id") %>%
  left_join(c1,by="card_id") %>% left_join(c2,by="card_id") %>% 
  left_join(c3,by="card_id") %>% left_join(c4,by="card_id") %>%
  left_join(mrs,by="card_id") %>% left_join(mrp,by="card_id") %>%
  left_join(uniq_cm,by="card_id") %>% 
  left_join(timelap,by="card_id")

colnames(fulltest)[107]<-"c2_1"
colnames(fulltest)[108]<-"c2_2"
colnames(fulltest)[109]<-"c2_3"
colnames(fulltest)[110]<-"c2_4"
colnames(fulltest)[111]<-"c2_5"


#####################################
#aggregated dataset exploration
#####################################

fulltrain$feature_1<-as.factor(fulltrain$feature_1)
fulltrain$feature_2<-as.factor(fulltrain$feature_2)
fulltrain$feature_3<-as.factor(fulltrain$feature_3)

fulltrain$rectmonth<-as.integer(fulltrain$rectmonth)
fulltrain$rectyear<-as.integer(fulltrain$rectyear)
fulltrain$year<-as.integer(fulltrain$year)
fulltrain$month<-as.integer(fulltrain$month)

summary(fulltrain)

#visulize the relationship between target and other factors
ggplot(data=fulltrain, aes(x=first_active_month,y=target))+geom_bar(stat="identity")
ggplot(data=fulltrain, aes(x=mrec_trans,y=target))+geom_bar(stat="identity")
ggplot(data=fulltrain, aes(x=rectday,y=target))+geom_bar(stat="identity")
ggplot(data=fulltrain, aes(x=active_length,y=target))+geom_bar(stat="identity")


#plot on mean target value for each entry of first active month
target_mean_fam<-fulltrain %>% group_by(first_active_month) %>% summarize_at("target", funs(mean),na.rm=TRUE)
ggplot(target_mean_fam,aes(x=first_active_month,y=target,group=1))+geom_point()+geom_line()
#plot on mean target value for each entry of most recent transaction date
target_mean_mrectr<-fulltrain %>% group_by(mrec_trans) %>% summarize_at("target", funs(mean),na.rm=TRUE)
ggplot(target_mean_mrectr,aes(x=mrec_trans,y=target,group=1))+geom_point()+geom_line()

#there is a significant drop in mean target value, check if there are outliers
summary(fulltrain$target)
boxplot(fulltrain$target,main="target distribution")
which.min(fulltrain$target)
# while this value is not related with the drops,so the drop could due to small observation number

#decision tree

f_dt<-as.formula(paste("target~",paste(names(fulltrain)[!names(fulltrain) %in% c("target","first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
#control=rpart.control(minsplit=1, minbucket=1, cp=0.001)
fit <- rpart(f_dt,method="anova", data=fulltrain)
plot(fit, uniform=TRUE, 
     main="Regression Tree for loyalty score")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

plot(as.party(fit))# a better decision tree plot 

fit$variable.importance #varaible importance 

#better plot
vi<-as.data.frame(fit$variable.importance)
vi<-cbind(vi,rownames(vi))
colnames(vi)<-c("importance","var_name")
ggplot(vi)+geom_bar(aes(x=reorder(var_name, importance),y=importance),stat = "identity")+coord_flip()+
  labs(title="Variable Importance",y="Importance",x="Variable Names")+
  theme(plot.title = element_text(size=20),axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20))

#get xgboost importance graph of variables
set.seed(100)
samp<-sample(nrow(fulltrain_upd),size=floor(0.8*nrow(fulltrain_upd)),replace=F)
subtrain<-fulltrain_upd[samp,]
subtest<-fulltrain_upd[-samp,]

tr_label=subtrain[,6]
ts_label=subtest[,6]
tr_data=subtrain[,-c(1,2,7,6,10)]
ts_data=subtest[,-c(1,2,7,6,10)]

tr_ma<-xgb.DMatrix(data=as.matrix(tr_data),label=tr_label,missing = c(NA,NaN,Inf))
ts_ma<-xgb.DMatrix(data=as.matrix(ts_data),label=ts_label,missing = c(NA,NaN,Inf))

set.seed(100)
m_xgb <- xgb.train(params=params, data=tr_ma, nrounds=2000, 
                   watchlist=list(val=ts_ma,train=tr_ma),
                   print_every_n = 200, early_stopping_rounds = 200)

m_imp<-xgb.importance(feature_names=colnames(tr_ma),model=m_xgb)

#install Ckmeans.1d.dp package first, then run the following codes
xgb.ggplot.importance(m_imp, top_n=50)

####################################################
#data processing and feature selection of fulltrain
####################################################

##check for non informative variables

uniq_prob<-data.frame(matrix(0,nrow=ncol(fulltrain),ncol=3),stringsAsFactors = F)
for(j in 1: ncol(fulltrain)){
  uniq_frq<-table(fulltrain[,j])/nrow(fulltrain)
  uniq_prob[j,1]<-max(uniq_frq)
  uniq_prob[j,2]<-max(uniq_frq)/max(uniq_frq[uniq_frq!=max(uniq_frq)])
  uniq_prob[j,3]<-colnames(fulltrain)[j]
}

# remove variables where the most frequent unique value occures more than 70% times of the whole occurances, 
# and 20 more times frequent than the second unique value

rm_feat<-uniq_prob[(uniq_prob[,1]>0.7)& (uniq_prob[,2]>20),]

#update fulltrain

fulltrain_new<-fulltrain[,-which(colnames(fulltrain)==rm_feat$X3[1])]

for (j in 2:nrow(rm_feat)){
  
  fulltrain_new<-fulltrain_new[,-which(colnames(fulltrain_new)==rm_feat$X3[j])]
  
}


##check for multilinearity 

add.col<-function(df, new.col) {n.row<-dim(df)[1]
length(new.col)<-max(n.row,length(new.col))
cbind(df, new.col)}

#transfer date from chacahter to numeric 
fulltrain_new$rectdaynum=rep(0,nrow(fulltrain_new))
for (i in 1:nrow(fulltrain_new)) {
  if (fulltrain_new$rectday[i]=="Monday"){
    fulltrain_new$rectdaynum[i]=1
  }
  
  else if (fulltrain_new$rectday[i]=="Tuesday"){
    fulltrain_new$rectdaynum[i]=2
  }
  else if (fulltrain_new$rectday[i]=="Wednesday"){
    fulltrain_new$rectdaynum[i]=3
  }
  else if (fulltrain_new$rectday[i]=="Thursday"){
    fulltrain_new$rectdaynum[i]=4
  }
  else if (fulltrain_new$rectday[i]=="Friday"){
    fulltrain_new$rectdaynum[i]=5
  }
  else if (fulltrain_new$rectday[i]=="Saturday"){
    fulltrain_new$rectdaynum[i]=6
  }
  else if (fulltrain_new$rectday[i]=="Sunday"){
    fulltrain_new$rectdaynum[i]=7
  }
  
}

fulltrain_new$rectmonth<-as.integer(fulltrain_new$rectmonth)
fulltrain_new$rectyear<-as.integer(fulltrain_new$rectyear)
fulltrain_new$year<-as.integer(fulltrain_new$year)
fulltrain_new$month<-as.integer(fulltrain_new$month)

hf_cor<-cor(fulltrain_new[,-c(1,2,7,6,10)], use="pairwise.complete.obs")

hf_cor<-hf_cor[!apply( hf_cor,1, function(x) all(is.na(x))), !apply( hf_cor,2, function(x) all(is.na(x))) ]
hf_cor[is.na(hf_cor)]<-0
diag(hf_cor)<-0

final_rm<-{}
lapcor=max(abs(hf_cor))

while (lapcor>0.8){
  
  highcor_var<-which(hf_cor==lapcor,arr.ind=T)
  a_ave_cor<- rep(0,nrow(highcor_var))
  b_ave_cor<- rep(0,nrow(highcor_var)) 
  value_comp<- data.frame("a_value"=rep(0,nrow(highcor_var)),"b_value"=rep(0,nrow(highcor_var)),
                          "rm_value"=rep(0,nrow(highcor_var)),"rm_row"=rep(0,nrow(highcor_var)),
                          "rm_col"=rep(0,nrow(highcor_var)))
  
  for (j in 1:nrow(highcor_var)){
    
    value_comp$a_value[j]<-rownames(highcor_var)[j]
    value_comp$b_value[j]<-rownames(highcor_var)[which(highcor_var[j,1]==highcor_var[,2] & highcor_var[j,2]==highcor_var[,1])]
    a_ave_cor[j]<- mean(abs(hf_cor[highcor_var[j,1],-c(highcor_var[j,2],highcor_var[j,1])]))
    b_ave_cor[j]<- mean(abs(hf_cor[highcor_var[j,2],-c(highcor_var[j,1],highcor_var[j,2])]))
    
    if (a_ave_cor[j]> b_ave_cor[j]){
      
      value_comp$rm_value[j]<- value_comp$a_value[j] 
      
    } else {
      
      value_comp$rm_value[j]<- value_comp$b_value[j]
      
    }
    value_comp$rm_row[j]=which(rownames(hf_cor)==value_comp$rm_value[j])
    value_comp$rm_col[j]=which(colnames(hf_cor)==value_comp$rm_value[j])      
    
  }  
  
  final_rm<-add.col(final_rm,value_comp$rm_value)
  
  hf_cor=hf_cor[-c(unique(value_comp$rm_row)),-c(unique(value_comp$rm_col))]
  
  lapcor=max(hf_cor[which(abs(hf_cor)<lapcor)])
  
}

#remove variabls appeared in final_rm 

final_rm<-matrix(final_rm,nrow=128,ncol=1)
final_rm<-unique(final_rm[!is.na(final_rm)])

fulltrain_new<-fulltrain_new[,-which(colnames(fulltrain_new)==final_rm[1])]

for (j in 2:length(final_rm)){
  
  fulltrain_new<-fulltrain_new[,-which(colnames(fulltrain_new)==final_rm[j])]
  
}

###############################################
### Average monthly perchase amount ratio ####
##############################################

#transform perchase amount
hist$purchase_amount_new<- round(hist$purchase_amount/0.00150265118 + 497.06, digits=2)
head(hist$purchase_amount_new)

#calculate monthly change of purchase amount for each card

hist2<-arrange(hist, card_id, purchase_date) %>% group_by(card_id,month_lag)%>% 
  summarise(mean_purch=mean(purchase_amount_new))

hist2$purc_monthly_rate<-{}
hist2$purc_monthly_rate[1]=0

for (i in 2:nrow(hist2)){
  if (hist2$card_id[i]==hist2$card_id[i-1]){
    hist2$purc_monthly_rate[i]=(hist2$mean_purch[i]/hist2$mean_purch[i-1])/(hist2$month_lag[i]-hist2$month_lag[i-1])
  }else{
    hist2$purc_monthly_rate[i]=0
  }
}

summary(hist2$purc_monthly_rate)
hist(hist2$purc_monthly_rate) # visualize monthly rate change

# notice that there are outlier, decide to remove them

hist2upd<-hist2  
hist2upd[which(hist2$purc_monthly_rate<=quantile(hist2$purc_monthly_rate,probs=0.999) | hist2$purc_monthly_rate>=0.0001),]

#averaging purchase rate by card ID
hist2upd<-hist2upd %>% group_by(card_id) %>% summarise(ave_month_rate=mean(purc_monthly_rate))

fulltrain<- fulltrain %>% left_join(hist2upd, by="card_id")



