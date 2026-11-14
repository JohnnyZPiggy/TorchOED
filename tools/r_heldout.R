if (identical(commandArgs(trailingOnly=TRUE), "H14")) {
  library(skpr)
  outdir <- "reports/w7-heldout"
  capture.output(print(sessionInfo()), print(packageVersion("skpr")), file=file.path(outdir, "sessionInfo.txt"))
  set.seed(20261114)
  whole <- gen_design(data.frame(A=-1:1), ~ A + I(A^2), trials=6)
  fit <- gen_design(read.csv(file.path(outdir, "H14.csv")),
                    ~ (A + B + C)^2 + I(A^2) + I(B^2) + I(C^2), trials=24,
                    splitplotdesign=whole, varianceratio=1)
  plot <- as.integer(sub("\\..*$", "", rownames(fit)))
  stopifnot(identical(tabulate(plot), rep(4L, 6)), nrow(fit) == 24)
  write.csv(data.frame(plot, fit[, c("A", "B", "C")]), file.path(outdir, "skpr.csv"), row.names=FALSE)
  quit(save="no")
}
library(AlgDesign)
set.seed(20261112)
if (identical(commandArgs(trailingOnly=TRUE), "H11")) {
  outdir <- "reports/w6-heldout"
  capture.output(print(sessionInfo()), print(packageVersion("AlgDesign")), file=file.path(outdir,"sessionInfo.txt"))
  grid <- read.csv(file.path(outdir,"H11.csv"))
  fit <- optBlock(~quad(A,B,C), withinData=grid, blocksizes=c(6,6,6))
  key <- function(rows) do.call(paste, rows)
  ids <- vapply(fit$Blocks, function(block) paste0("[",paste(match(key(block), key(grid))-1L,collapse=","),"]"), "")
  writeLines(paste0("[",paste(ids,collapse=","),"]"), file.path(outdir,"algdesign_out.json"))
  quit(save="no")
}
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
