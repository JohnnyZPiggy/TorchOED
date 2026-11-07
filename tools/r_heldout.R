library(AlgDesign)
set.seed(20261112)
cases <- list(H5=c(30,"D"),H6=c(20,"DI"),H7=c(14,"ID"),H8=c(16,"DAI")); out <- list()
dir.create("reports/w3-heldout/final",showWarnings=FALSE)
number <- function(x) if (is.null(x)) "null" else sprintf("%.17g",x)
for (name in names(cases)) {
  data <- read.csv(paste0("reports/w3-heldout/",name,"_F.csv"))
  for (criterion in strsplit(cases[[name]][2],"")[[1]]) {
    args <- list(frml=~.,data=data,nTrials=as.integer(cases[[name]][1]),criterion=criterion,nRepeats=5)
    if (name=="H6") { args$augment <- TRUE; args$rows <- scan("reports/w3-heldout/H6_fixed.json",what="",quiet=TRUE); args$rows <- as.integer(strsplit(gsub("[^0-9,]","",paste(args$rows,collapse="")),",")[[1]])+1L }
    fit <- do.call(optFederov,args)
    fields <- vapply(c("D","A","I"),function(k) paste0('"',k,'":',number(fit[[k]])),"")
    out <- c(out,paste0('{"case":"',name,'","criterion":"',criterion,'","rows":[',paste(fit$rows,collapse=","),'],',paste(fields,collapse=","),'}'))
  }
}
writeLines(paste0('[',paste(out,collapse=","),']'),"reports/w3-heldout/final/algdesign_out.json")
capture.output(print(sessionInfo()),print(packageVersion("AlgDesign")),file="reports/w3-heldout/final/sessionInfo.txt")
