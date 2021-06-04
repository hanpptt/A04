library(glmnet)

data<-read.csv("Documents/a04/data_oversampling2.csv")
y<-as.matrix(data[,1])
x<-as.matrix(data[,2:103])

myfit <- glmnet(x,y)
pdf("Documents/a04/lambda.pdf") # 好像参数的确是lambda 不是 lamda
plot(myfit, xvar = "lambda", label = TRUE)
dev.off()

# Min.pdf 中虚线表示  最小lamda纳入变量
myfit2 <- cv.glmnet(x,y)
pdf("Documents/a04/min.pdf")
plot(myfit2)
abline(v=log(c(myfit2$lambda.min,myfit2$lambda.1se)),lty="dashed")
dev.off()

# 打印最小lambda下的最佳变量it2$lambda.min
coe <- coef(myfit, s = myfit2$lambda.min)
act_index <- which(coe != 0)
act_coe <- coe[act_index]
row.names(coe)[act_index]
write.table(row.names(coe)[act_index], "Documents/a04/lasso-rows.txt",row.names=FALSE)
