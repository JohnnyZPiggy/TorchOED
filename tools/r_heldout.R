library(AlgDesign)
set.seed(20261108)
cases <- list(H1=c(15,"DAI"),H2=c(16,"DI"),H3=c(10,"DAI"),H4=c(24,"D")); out <- list()
number <- function(x) if (is.null(x)) "null" else sprintf("%.17g",x)
for (name in names(cases)) {
  data <- read.csv(paste0("reports/w3-heldout/",name,"_F.csv"))
  for (criterion in strsplit(cases[[name]][2],"")[[1]]) {
    fit <- optFederov(~.,data,nTrials=as.integer(cases[[name]][1]),criterion=criterion,nRepeats=5)
    fields <- vapply(c("D","A","I"),function(k) paste0('"',k,'":',number(fit[[k]])),"")
    out <- c(out,paste0('{"case":"',name,'","criterion":"',criterion,'","rows":[',paste(fit$rows,collapse=","),'],',paste(fields,collapse=","),'}'))
  }
}
writeLines(paste0('[',paste(out,collapse=","),']'),"reports/w3-heldout/algdesign_out.json")
capture.output(print(sessionInfo()),print(packageVersion("AlgDesign")),file="reports/w3-heldout/sessionInfo.txt")
