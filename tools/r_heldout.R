if (identical(commandArgs(trailingOnly=TRUE), "A11")) {
  outdir <- "reports/w11-mixai"
  capture.output(print(sessionInfo()), print(packageVersion("AlgDesign")), print(args(AlgDesign::optFederov)), file=file.path(outdir,"sessionInfo.txt"))
  candidates <- read.csv(file.path(outdir,"candidates_r1.csv"))
  for (crit in c("A", "I")) tryCatch({
    set.seed(11)
    design <- AlgDesign::optFederov(~ -1 + x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3, data=candidates, nTrials=10, criterion=crit)
    write.csv(data.frame(candidate=design$rows,design$design),file.path(outdir,paste0("optfederov_",tolower(crit),".csv")),row.names=FALSE)
  }, error=function(e) message(crit, ": ", conditionMessage(e)))
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "K11")) {
  library(skpr)
  outdir <- "reports/w11-skpr"
  capture.output(print(sessionInfo()), print(packageVersion("skpr")), print(args(skpr::gen_design)), print(formals(skpr::gen_design)$varianceratio), file=file.path(outdir,"sessionInfo.txt"))
  stopifnot(getRversion() == "4.6.1", packageVersion("skpr") == "1.9.2")
  record <- function(label, e) cat(label, conditionMessage(e), "\n", file=file.path(outdir,"calls.txt"), append=TRUE)
  set.seed(20261120)
  cand <- expand.grid(W = c(-1, 0, 1), X1 = c(-1, 1), X2 = c(-1, 1))
  htc <- tryCatch(withCallingHandlers({
    htc <- gen_design(unique(cand["W"]), model = ~W, trials = 6, blocksizes = 3, repeats = 10)
    htc
  }, warning=function(w) record("htc warning:",w)), error=function(e) { record("htc error:",e); NULL })
  full <- tryCatch(withCallingHandlers({
    full <- gen_design(cand, model = ~W + X1 + X2 + X1:X2, trials = 24, splitplotdesign = htc, blocksizes = 4, repeats = 10)
    full
  }, warning=function(w) record("full warning:",w)), error=function(e) { record("full error:",e); NULL })
  if (!is.null(htc) && !is.null(full)) write.csv(data.frame(run=seq_len(nrow(full)), full[,c("W","X1","X2")], rowname=rownames(full)),file.path(outdir,"skpr.csv"),row.names=FALSE)
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "M11")) {
  outdir <- "reports/w11-mixref"
  capture.output(print(sessionInfo()), file=file.path(outdir,"sessionInfo.txt"))
  for (pkg in c("mixexp","AlgDesign")) if (requireNamespace(pkg, quietly=TRUE)) capture.output(print(packageVersion(pkg)), print(args(if (pkg=="mixexp") mixexp::Xvert else AlgDesign::optFederov)), file=file.path(outdir,"sessionInfo.txt"), append=TRUE)
  if (requireNamespace("mixexp", quietly=TRUE)) tryCatch({
  a <- mixexp::Xvert(nfac=3, lc=c(.2,.1,.1), uc=c(.6,.5,.6), ndm=0, plot=FALSE)
  b <- mixexp::Xvert(nfac=4, lc=rep(.1,4), uc=c(.6,.5,.4,.3), ndm=0, plot=FALSE)
  a <- data.frame(region="R1", a); b <- data.frame(region="R3", b)
  for (name in setdiff(names(b),names(a))) a[[name]] <- NA
  write.csv(rbind(a[,names(b)],b),file.path(outdir,"xvert.csv"),row.names=FALSE,na="")
  }, error=function(e) message("Xvert: ",conditionMessage(e)))
  if (requireNamespace("AlgDesign", quietly=TRUE)) tryCatch({
  candidates <- read.csv(file.path(outdir,"candidates_r1.csv")); set.seed(11)
  design <- AlgDesign::optFederov(~ -1 + x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3, data=candidates, nTrials=10, criterion="D")
  write.csv(data.frame(candidate=design$rows,design$design),file.path(outdir,"optfederov.csv"),row.names=FALSE)
  }, error=function(e) message("optFederov: ",conditionMessage(e)))
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "K7")) {
  library(daewr)
  outdir <- "reports/w10-dsd"
  rd <- tools::Rd_db("daewr")[["DefScreen.Rd"]]
  usage <- Filter(function(x) identical(attr(x, "Rd_tag"), "\\usage"), rd)[[1L]]
  calls <- lapply(c(6L, 7L), function(k) {
    args <- list(m=k, c=0, center=0)
    if ("randomize" %in% names(formals(daewr::DefScreen))) args$randomize <- FALSE
    as.call(c(list(quote(daewr::DefScreen)), args))
  })
  capture.output(print(sessionInfo()), print(packageVersion("daewr")),
                 cat(trimws(paste(unlist(usage), collapse="")), "\n"),
                 for (call in calls) cat(deparse(call), "\n"),
                 if (!"randomize" %in% names(formals(daewr::DefScreen)))
                   cat("No randomize argument; rows used as returned.\n"),
                 file=file.path(outdir, "sessionInfo.txt"))
  rows <- Map(function(call, k) data.frame(k, lapply(eval(call), as.integer)), calls, c(6L, 7L))
  rows[[1L]]$G <- NA_integer_
  write.csv(do.call(rbind, rows), file.path(outdir, "daewr_dsd.csv"), row.names=FALSE, na="")
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "K6")) {
  outdir <- "reports/w10-rref"
  save_csv <- function(rows, file) write.csv(rows, file.path(outdir, file),
                                            row.names=FALSE, na="")
  combine <- function(rows, meta) {
    columns <- unique(unlist(lapply(rows, names)))
    do.call(rbind, lapply(rows, function(x) {
      for (name in setdiff(columns, names(x))) x[[name]] <- NA
      x[, c(meta, setdiff(columns, meta)), drop=FALSE]
    }))
  }
  errors <- FALSE
  attempt <- function(file, work) tryCatch(work(), error=function(e) {
    errors <<- TRUE
    message(file, ": ", conditionMessage(e))
  })
  packages <- c("FrF2", "DoE.base", "rsm")
  available <- packages[vapply(packages, requireNamespace, logical(1), quietly=TRUE)]
  capture.output(print(sessionInfo()),
                 for (pkg in available) cat(pkg, as.character(packageVersion(pkg)), "\n"),
                 if ("FrF2" %in% available)
                   writeLines(grep("starting with words of length|Chen, J[.]",
                                   capture.output(tools::Rd2txt(
                                     tools::Rd_db("FrF2")[["CatalogueAccessors.Rd"]],
                                     options=list(width=180))), value=TRUE)),
                 file=file.path(outdir, "sessionInfo.txt"))
  attempt("frf2_catlg.csv", function() {
    table <- FrF2::catlg
    rows <- data.frame(runs=integer(), k=integer(), name=character(),
                       gen=character(), WLP=character())
    for (m in 3:6) for (k in (m+1):min(11, 2^m-1)) {
      name <- sprintf("%d-%d.1", k, k-m)
      entry <- table[[name]]
      if (!is.null(entry)) rows[nrow(rows)+1, ] <- list(
        entry$nruns, entry$nfac, name, paste(entry$gen, collapse=";"),
        paste(entry$WLP, collapse=";"))
    }
    save_csv(rows, "frf2_catlg.csv")
  })
  attempt("frf2_pb.csv", function() {
    rows <- lapply(c(12, 20, 24), function(N) {
      design <- FrF2::pb(nruns=N, randomize=FALSE)
      data.frame(N, lapply(design, function(x) as.integer(as.character(x))))
    })
    save_csv(combine(rows, "N"), "frf2_pb.csv")
  })
  attempt("rsm_ccd.csv", function() {
    designs <- list(A=rsm::ccd(2, n0=c(3,3), alpha="rotatable", randomize=FALSE),
                    B=rsm::ccd(4, n0=c(4,2), alpha="orthogonal", randomize=FALSE),
                    C=rsm::ccd(5, n0=c(2,2), alpha="faces", randomize=FALSE))
    rows <- lapply(names(designs), function(case) {
      design <- designs[[case]]
      x <- as.data.frame(design)[attr(design, "rsdes")$primary]
      data.frame(case, block=design$Block, x)
    })
    save_csv(combine(rows, c("case", "block")), "rsm_ccd.csv")
  })
  attempt("rsm_bbd.csv", function() {
    rows <- lapply(c(3, 5), function(k) {
      design <- rsm::bbd(k, n0=3, block=FALSE, randomize=FALSE)
      data.frame(k, as.data.frame(design)[attr(design, "rsdes")$primary])
    })
    save_csv(combine(rows, "k"), "rsm_bbd.csv")
  })
  attempt("rsm_bbd67.csv", function() {
    rows <- lapply(c(6, 7), function(k) {
      design <- rsm::bbd(k, n0=3, block=FALSE, randomize=FALSE)
      data.frame(k, as.data.frame(design)[attr(design, "rsdes")$primary])
    })
    save_csv(combine(rows, "k"), "rsm_bbd67.csv")
  })
  quit(save="no", status=as.integer(errors))
}
if (identical(commandArgs(trailingOnly=TRUE), "K4")) {
  library(skpr)
  outdir <- "reports/w8-skpr"
  dir.create(outdir, showWarnings=FALSE)
  capture.output(print(sessionInfo()), print(packageVersion("skpr")), file=file.path(outdir, "sessionInfo.txt"))
  stopifnot(getRversion() == "4.6.1", packageVersion("skpr") == "1.9.2")
  extracted <- tempfile()
  tools::Rd2ex(tools::Rd_db("skpr")[["gen_design.Rd"]], out=extracted)
  examples <- readLines(extracted)
  first <- match("splitplotcandidateset = expand.grid(Altitude = c(-1, 1),", examples)
  last <- grep("^ *splitplotdesign *= *", examples)[1L]
  writeLines(sprintf("%d %d", first, last), file.path(outdir, "example-lines.txt"))
  published <- readLines("tools/r_heldout.R")
  start <- match("  set.seed(20261118)", published) + 1L
  stopifnot(identical(examples[first:last], published[start:(start + last - first)]))
  # skpr 1.9.2 (GPL-3), gen_design.Rd, Examples lines 217–230.
  # https://github.com/cran/skpr/blob/1.9.2/man/gen_design.Rd#L217-L230
  set.seed(20261118)
splitplotcandidateset = expand.grid(Altitude = c(-1, 1),
                                    Range = as.factor(c("Close", "Medium", "Far")),
                                    Power = c(1, -1))
hardtochangedesign = gen_design(splitplotcandidateset, model = ~Altitude,
                               trials = 11, repeats = 10)

#Now we can use the D-optimal blocked design as an input to our full design.

#Here, we add the easy to change factors from the candidate set to the model,
#and input the hard-to-change design along with the new number of trials. `gen_design` will
#automatically allocate the runs in the blocks in the most balanced way possible.

designsplitplot = gen_design(splitplotcandidateset, ~Altitude + Range + Power, trials = 33,
                             splitplotdesign = hardtochangedesign, repeats = 10)
  plot <- as.integer(sub("\\..*$", "", rownames(designsplitplot)))
  write.csv(splitplotcandidateset, file.path(outdir, "candidates.csv"), row.names=FALSE)
  write.csv(data.frame(plot, designsplitplot[, names(splitplotcandidateset)]),
            file.path(outdir, "skpr.csv"), row.names=FALSE)
  writeLines(as.character(c(ncol(model.matrix(~Altitude + Range + Power, designsplitplot)),
                            formals(skpr::gen_design)$varianceratio, tabulate(plot))),
             file.path(outdir, "model.txt"))
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "K5")) {
  library(skpr)
  outdir <- "reports/w9-skpr-i"
  capture.output(print(sessionInfo()), print(packageVersion("skpr")), file=file.path(outdir, "sessionInfo.txt"))
  stopifnot(getRversion() == "4.6.1", packageVersion("skpr") == "1.9.2")
  set.seed(20261119)
  splitplotcandidateset = expand.grid(Altitude = c(-1, 1),
                                      Range = as.factor(c("Close", "Medium", "Far")),
                                      Power = c(1, -1))
  hardtochangedesign = gen_design(splitplotcandidateset, model = ~Altitude,
                                 trials = 11, repeats = 10)
  designsplitplot = gen_design(splitplotcandidateset, ~Altitude + Range + Power, trials = 33,
                               splitplotdesign = hardtochangedesign, repeats = 10, optimality = "I")
  plot <- as.integer(sub("\\..*$", "", rownames(designsplitplot)))
  write.csv(data.frame(plot, designsplitplot[, names(splitplotcandidateset)]),
            file.path(outdir, "skpr.csv"), row.names=FALSE)
  writeLines(as.character(c(ncol(model.matrix(~Altitude + Range + Power, designsplitplot)),
                            formals(skpr::gen_design)$varianceratio, tabulate(plot))),
             file.path(outdir, "model.txt"))
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "H14")) {
  library(skpr)
  outdir <- "reports/w7-heldout"
  capture.output(print(sessionInfo()), print(packageVersion("skpr")), file=file.path(outdir, "sessionInfo.txt"))
  set.seed(20261114)
  whole <- gen_design(data.frame(A=-1:1), ~ A + I(A^2), trials=6)
  fit <- gen_design(read.csv(file.path(outdir, "H14.csv")),
                    ~ (A + B + C)^2 + I(A^2) + I(B^2) + I(C^2), trials=24,
                    splitplotdesign=whole, varianceratio=1)
  plot <- as.integer(sub("\\..*$", "", rownames(fit)))
  stopifnot(identical(tabulate(plot), rep(4L, 6)), nrow(fit) == 24)
  write.csv(data.frame(plot, fit[, c("A", "B", "C")]), file.path(outdir, "skpr.csv"), row.names=FALSE)
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "H19")) {
  library(AlgDesign)
  outdir <- "reports/w9-heldout"
  rd <- tools::Rd_db("AlgDesign")[["blockOpt.Rd"]]
  args <- Filter(function(x) identical(attr(x, "Rd_tag"), "\\arguments"), rd)[[1L]]
  criterion <- Filter(function(x) identical(attr(x, "Rd_tag"), "\\item") &&
                        identical(as.character(x[[1L]]), "criterion"), args)[[1L]]
  documented <- paste(as.character(criterion[[2L]]), collapse="")
  capture.output(print(sessionInfo()), print(packageVersion("AlgDesign")),
                 print(formals(optBlock)$criterion), cat(documented, "\n"),
                 file=file.path(outdir, "H19-session.txt"))
  if (grepl('"I"', documented, fixed=TRUE)) {
    set.seed(20261117)
    fit <- optBlock(~quad(A,B,C), withinData=read.csv(file.path(outdir, "H19.csv")),
                    blocksizes=c(8,8), criterion="I")
    write.csv(do.call(rbind, Map(function(rows, plot) data.frame(plot, rows),
                                fit$Blocks, seq_along(fit$Blocks))),
              file.path(outdir, "H19-design.csv"), row.names=FALSE)
  }
  quit(save="no")
}
if (identical(commandArgs(trailingOnly=TRUE), "H20")) {
  library(skpr)
  outdir <- "reports/w9-heldout"
  capture.output(print(sessionInfo()), print(packageVersion("skpr")), file=file.path(outdir, "H20-session.txt"))
  stopifnot(packageVersion("skpr") == "1.9.2", formals(gen_design)$varianceratio == 1)
  set.seed(20261117)
  wp <- gen_design(data.frame(A=c(-1,0,1)), ~A+I(A^2), trials=4, optimality="I")
  set.seed(20261117)
  fit <- gen_design(read.csv(file.path(outdir, "H20.csv")), ~(A+B+C)^2+I(A^2)+I(B^2)+I(C^2),
                    trials=12, splitplotdesign=wp, optimality="I")
  plot <- as.integer(sub("\\..*$", "", rownames(fit)))
  write.csv(data.frame(plot, fit[, c("A", "B", "C")]),
            file.path(outdir, "H20-design.csv"), row.names=FALSE)
  quit(save="no")
}
case <- commandArgs(trailingOnly=TRUE)
if (length(case) == 1L && case %in% c("H21", "H22", "H23")) {
  outdir <- "reports/w10-heldout"
  package <- if (case == "H21") "FrF2" else "rsm"
  library(package, character.only=TRUE)
  capture.output(print(sessionInfo()), print(packageVersion(package)),
                 file=file.path(outdir, paste0(case, "-session.txt")))
  if (case == "H21") {
    design <- FrF2::FrF2(nruns=16, nfactors=7, randomize=FALSE)
    capture.output(print(FrF2::generators(design)),
                   file=file.path(outdir, "H21-session.txt"), append=TRUE)
    rows <- data.frame(lapply(design, function(x) as.integer(as.character(x))))
  } else {
    design <- if (case == "H22") rsm::ccd(3, n0=c(4,2), alpha="orthogonal", randomize=FALSE)
              else rsm::bbd(4, n0=3, block=FALSE, randomize=FALSE)
    rows <- as.data.frame(design)[attr(design, "rsdes")$primary]
    if (case == "H22") rows$block <- as.integer(design$Block)
  }
  write.csv(rows, file.path(outdir, paste0(case, ".csv")), row.names=FALSE)
  quit(save="no")
}
library(AlgDesign)
set.seed(20261112)
if (identical(commandArgs(trailingOnly=TRUE), "H11")) {
  outdir <- "reports/w6-heldout"
  capture.output(print(sessionInfo()), print(packageVersion("AlgDesign")), file=file.path(outdir,"sessionInfo.txt"))
  grid <- read.csv(file.path(outdir,"H11.csv"))
  fit <- optBlock(~quad(A,B,C), withinData=grid, blocksizes=c(6,6,6))
  key <- function(rows) do.call(paste, rows)
  ids <- vapply(fit$Blocks, function(block) paste0("[",paste(match(key(block), key(grid))-1L,collapse=","),"]"), "")
  writeLines(paste0("[",paste(ids,collapse=","),"]"), file.path(outdir,"algdesign_out.json"))
  quit(save="no")
}
cases <- list(H5=c(30,"D"),H6=c(20,"DI"),H7=c(14,"ID"),H8=c(16,"DAI")); out <- list()
dir.create("reports/w3-heldout/final",showWarnings=FALSE)
number <- function(x) if (is.null(x)) "null" else sprintf("%.17g",x)
for (name in names(cases)) {
  data <- read.csv(paste0("reports/w3-heldout/",name,"_F.csv"))
  for (criterion in strsplit(cases[[name]][2],"")[[1]]) {
    args <- list(frml=~.,data=data,nTrials=as.integer(cases[[name]][1]),criterion=criterion,nRepeats=5)
    if (name=="H6") { args$augment <- TRUE; args$rows <- scan("reports/w3-heldout/H6_fixed.json",what="",quiet=TRUE); args$rows <- as.integer(strsplit(gsub("[^0-9,]","",paste(args$rows,collapse="")),",")[[1]])+1L }
    fit <- do.call(optFederov,args)
    fields <- vapply(c("D","A","I"),function(k) paste0('"',k,'":',number(fit[[k]])),"")
    out <- c(out,paste0('{"case":"',name,'","criterion":"',criterion,'","rows":[',paste(fit$rows,collapse=","),'],',paste(fields,collapse=","),'}'))
  }
}
writeLines(paste0('[',paste(out,collapse=","),']'),"reports/w3-heldout/final/algdesign_out.json")
capture.output(print(sessionInfo()),print(packageVersion("AlgDesign")),file="reports/w3-heldout/final/sessionInfo.txt")
