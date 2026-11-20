library(skpr)
out <- Sys.getenv("R_ORACLE_OUT")
cand <- read.csv("reports/w15-skpr-ssp/candidates.csv")
set.seed(1)
calls <- expression(gen_design(cand, ~A, trials = 2),
  gen_design(cand, ~A + B, trials = 4, splitplotdesign = wp, blocksizes = 2),
  gen_design(cand, ~A + B + x + I(x^2), trials = 8, splitplotdesign = sp,
             blocksizes = 2, add_blocking_columns = TRUE))
wp <- eval(calls[[1]])
sp <- eval(calls[[2]])
ss <- eval(calls[[3]])
record <- function(d) list(rows = d, rownames = rownames(d),
  model = deparse(attr(d, "model")), varianceratio = 1)
ss1 <- c(list(status = "RECORDED", calls = lapply(calls, deparse)), record(ss))
rd <- tools::Rd2ex(tools::Rd_db("skpr")[["gen_design.Rd"]])
exprs <- parse(text = rd)
depths <- list()
depth <- function(e) {
  if (is.symbol(e)) return(if (is.null(depths[[as.character(e)]])) 0 else depths[[as.character(e)]])
  if (!is.call(e) || as.character(e[[1]]) != "gen_design") return(0)
  a <- as.list(e)[-1]
  if (is.null(a$splitplotdesign)) return(0)
  1 + depth(a$splitplotdesign)
}
target <- NULL
for (i in seq_along(exprs)) {
  e <- exprs[[i]]
  assigned <- is.call(e) && as.character(e[[1]]) %in% c("<-", "=")
  value <- if (assigned) e[[3]] else e
  d <- depth(value)
  if (assigned) depths[[as.character(e[[2]])]] <- d
  if (d >= 2) { target <- i; break }
}
ss2 <- list(status = "NOT_RUN", error = "no nested split-plot example in gen_design.Rd")
if (!is.null(target)) {
  code <- c("library(skpr); set.seed(1)", vapply(exprs[seq_len(target)], function(e) paste(deparse(e), collapse="\n"), ""))
  last <- exprs[[target]]
  name <- if (as.character(last[[1]]) %in% c("<-", "=")) as.character(last[[2]]) else ".ss2"
  if (name == ".ss2") code[length(code)] <- paste(".ss2 <-", code[length(code)])
  code <- c(code, sprintf('jsonlite::write_json(list(rows=%s, rownames=rownames(%s)), "%s", auto_unbox=TRUE)', name, name, file.path(out,"ss2.json")))
  path <- file.path(out, "ss2-input.R"); writeLines(code, path)
  status <- system2("Rscript", path, stdout=file.path(tempdir(),"ss2.out"), stderr=file.path(tempdir(),"ss2.err"))
  ss2 <- if (status == 0) c(list(status="RECORDED", call=deparse(value)), jsonlite::read_json(file.path(out,"ss2.json"))) else list(status="NOT_RUN", error=paste(readLines(file.path(tempdir(),"ss2.err")),collapse="\n"))
  unlink(file.path(out,"ss2.json"))
}
jsonlite::write_json(list(SS1=ss1, SS2=ss2), file.path(out,"skpr.json"), auto_unbox=TRUE, pretty=TRUE, digits=17)
