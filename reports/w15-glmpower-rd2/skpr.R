library(skpr)
out <- Sys.getenv("R_ORACLE_OUT")
f <- tempfile(fileext = ".R")
tools::Rd2ex(tools::Rd_db("skpr")[["eval_design_mc.Rd"]], out = f)
exprs <- parse(f, keep.source=TRUE)
binomial_call <- function(x) {
  if (!is.call(x)) return(NULL)
  if (identical(x[[1]], as.name("eval_design_mc")) && identical(x$glmfamily, "poisson")) return(x)
  for (part in as.list(x)[-1]) {found <- binomial_call(part); if (!is.null(found)) return(found)}
  NULL
}
i <- which(vapply(exprs, function(e) !is.null(binomial_call(e)), TRUE))[1]
target <- binomial_call(exprs[[i]])
expression_text <- paste(as.character(attr(exprs, "srcref")[[i]]), collapse="\n")
code <- c('library(skpr); set.seed(1); Sys.setenv(IN_PKGDOWN="true")',
  vapply(exprs[seq_len(i)], function(e) paste(deparse(e), collapse="\n"), ""))
last <- exprs[[i]]
name <- if (as.character(last[[1]]) %in% c("<-", "=")) as.character(last[[2]]) else ".rd2"
if (name == ".rd2") code[length(code)] <- paste(".rd2 <-", code[length(code)])
export <- sprintf('r <- %s; attrs <- attributes(r)
design <- attr(r, "runmatrix"); if (is.null(design)) design <- %s
write.csv(design, file.path(Sys.getenv("R_ORACLE_OUT"), "rd2_design.csv"), row.names=FALSE)
jsonlite::write_json(list(power=as.data.frame(r), attribute_names=names(attrs),
  attributes=attrs[intersect(names(attrs), c("anticoef", "alpha", "model_matrix", "parameter_analysis_method_string", "effect_analysis_method_string"))], nsim=%s),
  file.path(Sys.getenv("R_ORACLE_OUT"), "child.json"), dataframe="rows", auto_unbox=TRUE, digits=17)',
  name, paste(deparse(target[[2]]), collapse="\n"), paste(deparse(target$nsim), collapse="\n"))
child <- tempfile(fileext=".R"); writeLines(c(code, export), child)
err <- tempfile(); log <- tempfile()
status <- system2("Rscript", child, stdout=log, stderr=err)
record <- if (status == 0) c(list(status="RECORDED", expression=expression_text),
  jsonlite::read_json(file.path(out,"child.json"), simplifyVector=TRUE)) else
  list(status="NOT_RUN", expression=expression_text, error=paste(readLines(err), collapse="\n"))
unlink(file.path(out,"child.json"))
jsonlite::write_json(record, file.path(out,"skpr.json"), dataframe="rows", auto_unbox=TRUE, pretty=TRUE, digits=17)
