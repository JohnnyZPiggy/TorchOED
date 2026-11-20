library(FrF2)
out <- Sys.getenv("R_ORACLE_OUT")
rd <- tools::Rd_db("FrF2")[["FrF2.Rd"]]
examples <- rd[[which(vapply(rd, function(x) identical(attr(x, "Rd_tag"), "\\examples"), logical(1)))]]
lines <- strsplit(paste(unlist(examples), collapse=""), "\n")[[1]]
rdline <- lines[grep("FrF2\\(.*blocks\\s*=", lines)][1]
stopifnot(!is.na(rdline))
calls <- c(MA1="FrF2(nruns=32, nfactors=13)", MA2="FrF2(nruns=32, nfactors=14)",
 MA3="FrF2(nruns=64, nfactors=12)", BL1="FrF2(nruns=16, nfactors=6, blocks=2)",
 BL2="FrF2(nruns=16, nfactors=5, blocks=4)", BL3="FrF2(nruns=32, nfactors=7, blocks=4)", RD1=trimws(rdline))
rows <- list(); metadata <- list()
for (name in names(calls)) {
 expr <- parse(text=calls[[name]])[[1]]; expr$randomize <- FALSE
 metadata[[name]] <- tryCatch({
  d <- eval(expr); info <- design.info(d)
  blocked <- grepl("blocked", info$type, fixed=TRUE)
  factors <- d[, if (blocked) 2:ncol(d) else seq_len(ncol(d)), drop=FALSE]
  x <- as.data.frame(lapply(factors, function(z) as.numeric(as.character(z))))
  stopifnot(all(as.matrix(x) %in% c(-1,1)))
  names(x) <- paste0("x", seq_len(ncol(x)))
  x <- cbind(case=name, block=if (blocked) as.integer(d[[1]]) else 0L, x)
  for (j in seq_len(14)) if (!paste0("x",j) %in% names(x)) x[[paste0("x",j)]] <- NA
  rows[[name]] <- x
  list(status="RECORDED", call=paste(deparse(expr),collapse=" "), nfactors=ncol(factors),
   info=info[intersect(names(info),c("generators","gen","aberration","aliased","block.gen","type"))])
 }, error=function(e) list(status="NOT_RUN", call=paste(deparse(expr),collapse=" "), error=conditionMessage(e)))
}
write.csv(do.call(rbind, rows), file.path(out,"designs.csv"), row.names=FALSE, na="")
jsonlite::write_json(list(RD1_Rd_line=rdline,cases=metadata), file.path(out,"frf2.json"), auto_unbox=TRUE,pretty=TRUE)
