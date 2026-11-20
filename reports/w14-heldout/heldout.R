library(support)
S <- as.matrix(read.csv("reports/w14-heldout/S.csv", header=FALSE))
set.seed(20261127)
X <- support::sp(n = 15, p = 3, dist.samp = S, par.flg = FALSE)$sp
write.table(format(X, digits=17), file=file.path(Sys.getenv("R_ORACLE_OUT"), "H39-R.csv"), sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)
