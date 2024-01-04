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
    mutate(fn = X25==1 & X26==0) %>%
    mutate(tp = X25==1 & X26==1) %>%
    group_by(Hour) %>%
    summarise(num=n(), 
              filter_prop = sum(Filtered)/n(),
              recall = sum(tp)/(sum(tp)+sum(fn))) %>%
    mutate(score = 127000*filter_prop-5200-850000*(1-recall))
  return(data)
}

df <- myfunc(5)
for(i in 6:25){
  df <- df %>%
    bind_rows(myfunc(i))
}

df <- read.csv("Data/Oversampled-XGB-optimal.csv")
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

