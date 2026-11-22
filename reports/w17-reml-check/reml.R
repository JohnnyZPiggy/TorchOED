library(lmerTest)
options(digits=17)
out <- Sys.getenv("R_ORACLE_OUT")
data(Oats, package="nlme")
write.csv(Oats, file.path(out, "Oats.csv"), row.names=FALSE)
strip <- read.csv("reports/w17-reml-check/strip.csv")
strip$block <- factor(strip$block); strip$A <- factor(strip$A); strip$B <- factor(strip$B)
ctl <- lmerControl(optimizer="nloptwrap", optCtrl=list(ftol_abs=1e-14, xtol_abs=1e-14, maxeval=10^5))
fits <- list(Oats=lmer(yield ~ Variety * factor(nitro) + (1 | Block/Variety), Oats, REML=TRUE, control=ctl),
             strip=lmer(y ~ A * B + (1 | block) + (1 | block:A) + (1 | block:B), strip, REML=TRUE, control=ctl))
rows <- list()
for (name in names(fits)) {
  fit <- fits[[name]]; co <- summary(fit)$coefficients
  vc <- as.data.frame(VarCorr(fit))
  vals <- list(fixed=co[,1], SE=co[,2], df=co[,3], variance=vc$vcov, logLik=as.numeric(logLik(fit)))
  for (g in names(vals)) rows[[length(rows)+1]] <- data.frame(case=name, group=g, index=seq_along(vals[[g]]), value=vals[[g]])
}
write.csv(do.call(rbind, rows), file.path(out,"r.csv"), row.names=FALSE)
capture.output(sessionInfo(), file=file.path(out,"sessionInfo.txt"))
