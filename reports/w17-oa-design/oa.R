out <- Sys.getenv("R_ORACLE_OUT")
cases <- list(rep(2,7), rep(2,11), rep(2,15), rep(3,4), rep(3,13),
              rep(4,5), rep(5,6), c(4,4,rep(2,9)))
arrays <- list()
gwlp <- list()
for (i in seq_along(cases)) {
  name <- paste0("C", i)
  tryCatch({
    a <- DoE.base::oa.design(nlevels=cases[[i]], randomize=FALSE)
    x <- sapply(a, function(col) as.integer(factor(col, levels=levels(col)))-1L)
    arrays[[name]] <- data.frame(case=name, row=rep(seq_len(nrow(x)), ncol(x)),
      column=rep(seq_len(ncol(x)), each=nrow(x)), value=as.vector(x), error="")
    gwlp[[name]] <- list(values=unname(DoE.base::GWLP(a)))
  }, error=function(e) {
    arrays[[name]] <<- data.frame(case=name, row=NA, column=NA, value=NA,
                                 error=conditionMessage(e))
    gwlp[[name]] <<- list(error=conditionMessage(e))
  })
}
write.csv(do.call(rbind, arrays), file.path(out,"arrays.csv"), row.names=FALSE)
jsonlite::write_json(gwlp, file.path(out,"gwlp.json"), auto_unbox=TRUE, pretty=TRUE, digits=16)
