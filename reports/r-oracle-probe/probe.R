# rsm-only copy of tools/r_heldout.R K6 (Refs #372).
outdir <- Sys.getenv("R_ORACLE_OUT")
save_csv <- function(rows, file) write.csv(rows, file.path(outdir, file), row.names=FALSE, na="")
  combine <- function(rows, meta) {
    columns <- unique(unlist(lapply(rows, names)))
    do.call(rbind, lapply(rows, function(x) {
      for (name in setdiff(columns, names(x))) x[[name]] <- NA
      x[, c(meta, setdiff(columns, meta)), drop=FALSE]
    }))
  }
    designs <- list(A=rsm::ccd(2, n0=c(3,3), alpha="rotatable", randomize=FALSE),
                    B=rsm::ccd(4, n0=c(4,2), alpha="orthogonal", randomize=FALSE),
                    C=rsm::ccd(5, n0=c(2,2), alpha="faces", randomize=FALSE))
    rows <- lapply(names(designs), function(case) {
      design <- designs[[case]]
      x <- as.data.frame(design)[attr(design, "rsdes")$primary]
      data.frame(case, block=design$Block, x)
    })
    save_csv(combine(rows, c("case", "block")), "rsm_ccd.csv")
    rows <- lapply(c(3, 5), function(k) {
      design <- rsm::bbd(k, n0=3, block=FALSE, randomize=FALSE)
      data.frame(k, as.data.frame(design)[attr(design, "rsdes")$primary])
    })
    save_csv(combine(rows, "k"), "rsm_bbd.csv")
    rows <- lapply(c(6, 7), function(k) {
      design <- rsm::bbd(k, n0=3, block=FALSE, randomize=FALSE)
      data.frame(k, as.data.frame(design)[attr(design, "rsdes")$primary])
    })
    save_csv(combine(rows, "k"), "rsm_bbd67.csv")
for (name in c("rsm_ccd.csv", "rsm_bbd.csv", "rsm_bbd67.csv")) {
  stopifnot(identical(readBin(file.path(outdir, name), "raw", file.info(file.path(outdir, name))$size),
                      readBin(file.path("reports/w10-rref", name), "raw", file.info(file.path("reports/w10-rref", name))$size)))
  cat(name, "byte-identity PASS\n")
}
