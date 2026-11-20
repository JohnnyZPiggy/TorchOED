out <- Sys.getenv("R_ORACLE_OUT")
options(digits=17)
library(lhs); library(MaxPro); library(DiceDesign)
p <- "reports/w13-spacefill-ext"
metrics <- function(x) list(min_distance=DiceDesign::mindist(x), phi_p=DiceDesign::phiP(x,p=15), cd2=DiceDesign::discrepancyCriteria(x,type="C2")$DisC2, maxpro=MaxPro::MaxProMeasure(x))
rows <- lapply(c("random","maximin","maxpro","uniform"), function(m) metrics(as.matrix(read.csv(file.path(p,paste0(m,".csv")),header=FALSE))))
names(rows) <- c("random","maximin","maxpro","uniform")
set.seed(20261119); a <- lhs::maximinLHS(8,3)
set.seed(20261119); b <- MaxPro::MaxProLHD(n=8,p=3)$Design
examples <- list()
for (pkg in c("lhs","MaxPro","DiceDesign")) {
  fun <- switch(pkg,lhs="maximinLHS",MaxPro="MaxProLHD",DiceDesign="discrepancyCriteria")
  rdfile <- paste0(fun,".Rd")
  rd <- tools::Rd_db(pkg)[[rdfile]]
  code <- capture.output(tools::Rd2ex(rd))
  # Rd2ex also includes the usage in comments: select the first executable call.
  first <- which(grepl(paste0(fun,"("),code,fixed=TRUE) & !grepl("^#",code))[1]
  call <- code[first]
  shape <- switch(pkg,lhs=c(4,3),MaxPro=c(10,4),DiceDesign=c(40,2)) # first Rd examples at these pins
  if (any(vapply(list(c(10,2),c(20,3),c(7,2)),function(s) identical(s,shape),logical(1)))) {
    examples[[pkg]] <- list(rd_file=rdfile,rd2ex_line=first,call=call,status="skipped: held-out size"); next
  }
  e <- new.env(); set.seed(20261119)
  if (pkg=="DiceDesign") {
    start <- which(code=="dimension <- 2")[1]
    eval(parse(text=code[start:(first-1)]),e)
  }
  value <- eval(parse(text=call),e)
  design <- if(pkg=="MaxPro") value$Design else if(pkg=="lhs") value else e$X
  examples[[pkg]] <- list(rd_file=rdfile,rd2ex_line=first,rd2ex=code,call=call,design=design,metrics=metrics(design))
}
result <- list(run_id=Sys.getenv("GITHUB_RUN_ID"),source_sha=Sys.getenv("GITHUB_SHA"),R_version=R.version.string,versions=lapply(c("lhs","MaxPro","DiceDesign"),function(p) as.character(packageVersion(p))),diagnostics=rows,quality=list(maximin=a,maxpro=b),examples=examples)
# Full-precision data; workflow records sessionInfo separately in the artifact.
writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,digits=17),file.path(out,"raw.json"))
