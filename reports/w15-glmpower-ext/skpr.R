library(skpr)
out <- Sys.getenv("R_ORACLE_OUT")
inputs <- read.csv("reports/w15-glmpower-ext/designs.csv")
pack <- function(r, call, nsim) {
  attrs <- attributes(r)
  list(status="RECORDED", call=call, nsim=nsim, power=as.data.frame(r),
       attribute_names=names(attrs), attributes=attrs[intersect(names(attrs), c("anticoef", "alpha", "nsim"))])
}
results <- list()
for (name in c("G1", "G2", "G3")) {
  design <- inputs[inputs$case == name, if (name == "G3") c("X1", "X2") else c("X1", "X2", "X3")]
  model <- if (name == "G3") ~X1 + X2 + X1:X2 else ~X1 + X2 + X3
  family <- if (name == "G2") "poisson" else "binomial"
  beta <- switch(name, G1=c(0, 0.5, 1, 0), G2=c(1, 0.25, 0.5, 0), G3=c(0.5, 1, 0.5, 0.25))
  call <- substitute(eval_design_mc(design, model, alpha=0.05, nsim=10000, glmfamily=FAMILY, anticoef=BETA, firth=FALSE, parallel=FALSE), list(FAMILY=family, BETA=beta))
  results[[name]] <- tryCatch({set.seed(1); pack(eval(call), paste(deparse(call), collapse="\n"), 10000)},
                            error=function(e) list(status="NOT_RUN", error=conditionMessage(e), call=paste(deparse(call), collapse="\n")))
}
binomial_call <- function(x) {
  if (!is.call(x)) return(NULL)
  if (identical(x[[1]], as.name("eval_design_mc")) && identical(x$glmfamily, "binomial")) return(x)
  for (part in as.list(x)[-1]) {found <- binomial_call(part); if (!is.null(found)) return(found)}
  NULL
}
results$RD1 <- tryCatch({
  con <- textConnection("example", "w"); tools::Rd2ex(tools::Rd_db("skpr")[["eval_design_mc.Rd"]], out=con); close(con)
  expressions <- parse(text=example, keep.source=TRUE); env <- new.env(); Sys.setenv(IN_PKGDOWN="true"); set.seed(1)
  for (i in seq_along(expressions)) {
    expr <- expressions[[i]]; target <- binomial_call(expr)
    if (!is.null(target)) {
      expression_text <- paste(as.character(attr(expressions, "srcref")[[i]]), collapse="\n")
      r <- eval(expr, env); write.csv(eval(target[[2]], env), file.path(out, "rd1_design.csv"), row.names=FALSE)
      result <- pack(r, expression_text, eval(target$nsim, env)); break
    }
    eval(expr, env)
  }
  result
}, error=function(e) list(status="NOT_RUN", error=conditionMessage(e)))
jsonlite::write_json(results, file.path(out, "skpr.json"), dataframe="rows", auto_unbox=TRUE, pretty=TRUE, digits=17)
