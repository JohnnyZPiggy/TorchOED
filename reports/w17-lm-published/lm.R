library(rsm)
out <- Sys.getenv("R_ORACLE_OUT")
options(digits=17)
write.csv(warpbreaks, file.path(out, "warpbreaks.csv"), row.names=FALSE)
write.csv(ChemReact, file.path(out, "ChemReact.csv"), row.names=FALSE)
score_data <- function(fit, full) {
  cc <- coef(summary(fit)); a <- anova(fit); l <- anova(fit, full)
  list(coef=cc[,1], se=cc[,2], t=cc[,3], p_t=cc[,4],
    df_resid=df.residual(fit),
    anova=list(df=a$Df[-nrow(a)], ss=a$`Sum Sq`[-nrow(a)],
      f=a$`F value`[-nrow(a)], p=a$`Pr(>F)`[-nrow(a)], ss_resid=deviance(fit)),
    lack_of_fit=list(df_lof=l$Df[2], ss_lof=l$`Sum of Sq`[2],
      df_pe=df.residual(full), ss_pe=deviance(full), f=l$F[2], p=l$`Pr(>F)`[2]),
    residuals=residuals(fit), leverage=hatvalues(fit),
    studentized=rstandard(fit), cooks=cooks.distance(fit))
}
wb <- tryCatch(score_data(lm(breaks ~ wool + tension, warpbreaks),
  lm(breaks ~ wool:tension, warpbreaks)), error=function(e) list(error=conditionMessage(e)))
cr <- tryCatch({
  CR <- coded.data(ChemReact, x1 ~ (Time - 85)/5, x2 ~ (Temp - 175)/5)
  ans <- score_data(lm(Yield ~ Block + x1 + x2 + x1:x2 + I(x1^2) + I(x2^2), CR),
    lm(Yield ~ factor(interaction(Block, x1, x2)), CR))
  ans$rsm_anova_informational <- capture.output(summary(rsm(Yield ~ Block + SO(x1, x2), CR)))
  ans
}, error=function(e) list(error=conditionMessage(e)))
jsonlite::write_json(list(warpbreaks=wb, ChemReact=cr), file.path(out, "r.json"),
  digits=NA, auto_unbox=TRUE, pretty=TRUE, na="null")
