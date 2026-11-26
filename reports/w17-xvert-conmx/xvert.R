library(mixexp)
options(digits = 17)
out <- Sys.getenv("R_ORACLE_OUT")
helptext <- capture.output(tools::Rd2txt(utils:::.getHelpFile(help("Xvert", package="mixexp"))))
# Xvert's installed help names lb/ub/coef; conmx is internal, not a public argument.
writeLines(c(helptext, "", "Installed Xvert source (internal conmx convention):",
             deparse(mixexp::Xvert)), file.path(out, "xvert_help.txt"))
regions <- list(
  R1=list(nfac=3, lc=c(.1,.1,.1), uc=c(.7,.6,.5),
          nlc=1, lb=.2, ub=.75, coef=matrix(c(1,1,0), nrow=1)),
  # sum(x)=1: x1-x2 >= 0 iff 2*x1+x3+x4 >= 1.
  # Positive bounds avoid Xvert's deletion of conmx rows with zero RHS.
  R2=list(nfac=4, lc=rep(.05,4), uc=rep(.5,4),
          nlc=1, lb=1, ub=2, coef=matrix(c(2,0,1,1), nrow=1)))
for (name in names(regions)) {
  tryCatch({
    vertices <- do.call(mixexp::Xvert, c(regions[[name]], list(ndm=0, plot=FALSE)))
    write.csv(vertices, file.path(out, paste0(name, ".csv")), row.names=FALSE)
  }, error=function(e) writeLines(conditionMessage(e), file.path(out, paste0(name, ".csv"))))
}
