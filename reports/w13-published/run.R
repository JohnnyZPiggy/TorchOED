out <- Sys.getenv("R_ORACLE_OUT")
options(digits=17)
tryCatch({
  library(mixexp)
  des<-SCD(12)
  write.csv(des, file.path(out,"mix-scd12.csv"), row.names=FALSE)
}, error=function(e) writeLines(paste0("ERROR: ",conditionMessage(e)),file.path(out,"mix-scd12.csv")))
tryCatch({
  library(mixexp)
  des<-Xvert(3,uc=c(.8,.95,.50),lc=c(0,.10,.05),ndm=1,plot=FALSE)
  write.csv(des, file.path(out,"mix-xvert.csv"), row.names=FALSE)
}, error=function(e) writeLines(paste0("ERROR: ",conditionMessage(e)),file.path(out,"mix-xvert.csv")))
tryCatch({
  library(FrF2)
  value <- pb(8,default.levels=c("current","new"))
  value <- as.data.frame(lapply(value, function(x) as.numeric(x)*2-3))
  write.csv(value, file.path(out,"fr-pb8.csv"), row.names=FALSE)
}, error=function(e) writeLines(paste0("ERROR: ",conditionMessage(e)),file.path(out,"fr-pb8.csv")))
tryCatch({
  library(FrF2)
  test <- pb(40)
  test <- as.data.frame(lapply(test, function(x) as.numeric(x)*2-3))
  write.csv(test, file.path(out,"fr-pb40.csv"), row.names=FALSE)
}, error=function(e) writeLines(paste0("ERROR: ",conditionMessage(e)),file.path(out,"fr-pb40.csv")))
