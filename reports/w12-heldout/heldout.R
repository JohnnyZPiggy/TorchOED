out <- Sys.getenv("R_ORACLE_OUT")
tryCatch({
  x <- FrF2::pb(nruns = 28, randomize = FALSE)
  x <- as.data.frame(lapply(x, function(v) as.integer(v)*2-3))
  write.csv(x, file.path(out, "H31.csv"), row.names=FALSE)
}, error=function(e) cat("H31 NOT_RUN: ", conditionMessage(e), "\n", sep=""))
tryCatch({
  library(AlgDesign)
  set.seed(20261126)
  fit <- AlgDesign::optBlock(~ quad(x1, x2),
    withinData = read.csv("reports/w12-heldout/H32-candidates.csv"),
    blocksizes = c(4, 4, 4), criterion = "D")
  x <- do.call(rbind, Map(function(rows, block) data.frame(block, rows),
                         fit$Blocks, seq_along(fit$Blocks)))
  write.csv(x[,c("block","x1","x2")], file.path(out, "H32.csv"), row.names=FALSE)
}, error=function(e) cat("H32 NOT_RUN: ", conditionMessage(e), "\n", sep=""))
