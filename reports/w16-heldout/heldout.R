library(FrF2)
library(skpr)
out <- Sys.getenv("R_ORACLE_OUT")
tryCatch({
 d <- FrF2::FrF2(nruns = 64, nfactors = 10, blocks = 4, randomize = FALSE)
 x <- as.data.frame(lapply(d[, -1], function(z) as.numeric(as.character(z))))
 x$block <- as.integer(d[[1]])
 write.csv(x, file.path(out, "H43-R.csv"), row.names=FALSE)
}, error=function(e) write.csv(data.frame(error=conditionMessage(e)), file.path(out,"H43-R.csv"), row.names=FALSE))
tryCatch({
 design <- read.csv("reports/w16-heldout/H44-design.csv")
 r <- skpr::eval_design_mc(design = design, model = ~A + B + C, alpha = 0.05, nsim = 10000, glmfamily = "poisson", anticoef = c(0.5, 0.5, -0.25, 0))
 write.csv(r[r$type == "parameter.power.mc", ], file.path(out,"H44-R.csv"), row.names=FALSE)
}, error=function(e) write.csv(data.frame(error=conditionMessage(e)), file.path(out,"H44-R.csv"), row.names=FALSE))
