out <- Sys.getenv("R_ORACLE_OUT")
library(CAs)
data("colbournBigFrame", package = "CAs")
str(eCAN(3, 10, 3))
arrays <- jsonlite::fromJSON("reports/w17-ca-bestknown/arrays.json")
ecan <- list()
verify <- list()
for (name in names(arrays)) {
  x <- as.integer(strsplit(name, ".", fixed=TRUE)[[1]])
  t <- x[1]; v <- x[2]; k <- x[3]
  ecan[[name]] <- tryCatch(data.frame(t=t,v=v,k=k,eCAN=eCAN(t,k,v)$CAN,error=""),
    error=function(e) data.frame(t=t,v=v,k=k,eCAN=NA,error=conditionMessage(e)))
  verify[[name]] <- tryCatch(data.frame(t=t,v=v,k=k,covered=caverify::ca_verify(arrays[[name]],t,v)$covered,error=""),
    error=function(e) data.frame(t=t,v=v,k=k,covered=NA,error=conditionMessage(e)))
}
write.csv(do.call(rbind,ecan),file.path(out,"ecan.csv"),row.names=FALSE)
write.csv(do.call(rbind,verify),file.path(out,"verify.csv"),row.names=FALSE)
