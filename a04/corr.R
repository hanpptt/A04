# 热图 #
options(stringsAsFactors = F)
options(warn = -1)

# 安装pheatmap包
#install.packages("pheatmap")

library(pheatmap)
library(Matrix)

data<-read.csv("Documents/a04/corr_data.csv")
corr<-cor(data)
corr<-as.matrix(corr)

pdf("Documents/a04/corr.pdf")
pheatmap(corr,fontsize=5,shown_colnames=F)
dev.off()
