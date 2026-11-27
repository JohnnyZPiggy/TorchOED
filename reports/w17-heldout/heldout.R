library(DoE.base)
out <- Sys.getenv("R_ORACLE_OUT")
options(digits=17)
write.csv(datasets::npk, file.path(out,"npk.csv"), row.names=FALSE)
h46 <- tryCatch({
  fit <- lm(yield ~ block + N + P + K, data=datasets::npk)
  x <- model.matrix(fit); cc <- summary(fit)$coefficients; a <- anova(fit)
  list(X=unname(x), assign=attr(x,"assign"), y=unname(datasets::npk$yield),
    coef=unname(cc[,1]), se=unname(cc[,2]), p_t=unname(cc[,4]),
    anova=list(df=a$Df[-nrow(a)], ss=a$`Sum Sq`[-nrow(a)],
      f=a$`F value`[-nrow(a)], p=a$`Pr(>F)`[-nrow(a)]))
}, error=function(e) list(error=conditionMessage(e)))
h47 <- tryCatch({
  a <- oa.design(nlevels=c(4,rep(2,12)), randomize=FALSE)
  unname(sapply(a,function(col) as.integer(factor(col,levels=levels(col)))-1L))
}, error=function(e) list(error=conditionMessage(e)))
jsonlite::write_json(list(H46=h46,H47=h47),file.path(out,"r.json"),
  digits=NA,auto_unbox=TRUE,pretty=TRUE,na="null")
