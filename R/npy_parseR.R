library(RcppCNPy)

# In Python:
# import numpy as np
# from scipy.sparse import csr_matrix
# 
# print "started"
# loader = np.load("/home/jche/Data/day_samp_new_0604.npy")
# print "loaded"
# np.save("/home/jche/Desktop/day_samp_new_0604.npy", csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']).toarray())

#If sample too big
#dat <- foo[sample(nrow(foo),size=100000,replace=FALSE),]

myfunc <- function(day){
  day <- sprintf("%02d", day)
  file_in <- paste("Data/Day-Hour-Results-", day, ".npy", sep="")
  data <- npyLoad(file_in)
  data <- data.frame(data)
  data <- data %>%
    mutate(Hour = 1*X2+2*X3+3*X4+4*X5+5*X6+6*X7+7*X8+8*X9+9*X10+10*X11+11*X12+12*X13+13*X14+14*X15+15*X16+16*X17+17*X18+18*X19+19*X20+20*X21+21*X22+22*X23+23*X24) %>%
    mutate(Day = day) %>%
    select(Day, Hour, X25, X26) %>%
    mutate(Filtered = !X26) %>%
    mutate(Success = X25) %>%
    mutate(fn = X25==1 & X26==0) %>%
    mutate(tp = X25==1 & X26==1) %>%
    group_by(Hour) %>%
    summarise(number=n(), 
              filter_prop = sum(Filtered)/n(),
              recall = sum(tp)/(sum(tp)+sum(fn)),
              success_rate = sum(Success)/n()) %>%
    mutate(score = 127000*filter_prop-5200-850000*(1-recall))
  return(data)
}

df <- myfunc(5)
for(i in 6:25){
  df <- df %>%
    bind_rows(myfunc(i))
}

df <- df %>%
  mutate(hour_of_week = seq(0,503))

ggplot(df, aes(x=hour_of_week, y=success_rate)) +
  geom_line(colour="red") +
  scale_x_continuous(breaks=c(0,24,48,72,96,120,144,168),labels=c('June 19','June 20','June 21','June 22','June 23','June 24','June 25','June 26')) +
  geom_vline(xintercept = c(0,24,48,72,96,120,144,168), colour="gray", linetype = "longdash") +
  ggtitle("") +
  ylab("") +
  #ylim(c(-35000,35000)) +
  xlab("") +
  theme(axis.text=element_text(size=18))


df2 <- df %>%
  group_by(Hour) %>%
  summarise(ave_score = mean(score),
            sd = sd(score))

# BOOTSTRAPPING: we find we're close enough to normal, so we're good
library(boot)
boot_sd <- function(data, indices){
  d <- data[indices] # allows boot to select sample
  return(sd(d))
}
results <- boot(data=df$score, statistic=boot_sd, 
                R=1000)
plot(results)
boot.ci(results, type="bca")
# END BOOTSTRAPPING

ggplot(df2, aes(x=Hour, y=ave_score)) +
  geom_bar(stat="identity") +
  geom_errorbar(aes(ymin=ave_score-sd, ymax=ave_score+sd), width=.5)

# Hourly proportions
df <- df %>%
  group_by(Hour) %>%
  summarise(number = sum(number))
ggplot(df, aes(x=Hour, y=number)) +
  geom_bar(stat="identity")


data <- npyLoad("Data/output_neg_newer.npy")
data <- data.frame(data)
temp <- data %>%
  mutate(Hour = 1*X61+2*X62+3*X63+4*X64+5*X65+6*X66+7*X67+8*X68+9*X69+10*X70+11*X71+12*X72+13*X73+14*X74+15*X75+16*X76+17*X77+18*X78+19*X79+20*X80+21*X81+22*X82+23*X83) %>%
  select(Hour) %>%
  group_by(Hour) %>%
  summarise(number = n())
tally(temp$X67)
  
