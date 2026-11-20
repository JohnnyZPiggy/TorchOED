out <- Sys.getenv("R_ORACLE_OUT"); options(digits=17)
library(support)
F <- as.matrix(read.csv("reports/w13-fff-ext/sample.csv",header=FALSE))
set.seed(20261119); X <- support::sp(n=10,p=2,dist.samp=F,par.flg=FALSE)$sp
rd <- tools::Rd_db("support")[["sp.Rd"]]; code <- capture.output(tools::Rd2ex(rd))
first <- which(grepl("D <- sp(",code,fixed=TRUE))[1]
e <- new.env(); set.seed(20261119)
eval(parse(text=code[(first-2):first]),e)
result <- list(run_id=Sys.getenv("GITHUB_RUN_ID"),source_sha=Sys.getenv("GITHUB_SHA"),R_version=R.version.string,support=as.character(packageVersion("support")),design=X,example=list(rd_file="sp.Rd",rd2ex_line=first,rd2ex=code,call=code[first],design=e$D$sp,status="not comparable: independent standard normal target is not a subset of [0,1]^2"))
writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,digits=17),file.path(out,"raw.json"))
