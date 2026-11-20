out <- Sys.getenv("R_ORACLE_OUT")
set.seed(20261203); x <- MaxPro::MaxProLHD(n = 20, p = 3)$Design
write.table(x, file.path(out,"H34-R.csv"), sep=",", row.names=FALSE, col.names=FALSE)
