library(lmerTest)
out <- Sys.getenv("R_ORACLE_OUT"); options(digits=17)
cake <- lme4::cake; cake$temp <- factor(cake$temperature, ordered=FALSE)
write.csv(cake,file.path(out,"cake.csv"),row.names=FALSE)
write.csv(datasets::stackloss,file.path(out,"stackloss.csv"),row.names=FALSE)
h49 <- tryCatch({
  fit <- lmerTest::lmer(angle ~ recipe * temp + (1 | replicate) + (1 | replicate:recipe),data=cake,REML=TRUE)
  x <- model.matrix(fit); cc <- summary(fit)$coefficients; vc <- as.data.frame(VarCorr(fit))
  list(X=unname(x),columns=colnames(x),fixed=unname(cc[,1]),SE=unname(cc[,2]),df=unname(cc[,3]),
       variance=vc$vcov[match(c("replicate","replicate:recipe","Residual"),vc$grp)],logLik=as.numeric(logLik(fit)))
},error=function(e) list(error=conditionMessage(e)))
h50 <- tryCatch({
  fit <- lm(stack.loss ~ Air.Flow + Water.Temp + Acid.Conc.,data=datasets::stackloss)
  point <- data.frame(Air.Flow=60,Water.Temp=21,Acid.Conc.=87)
  list(confidence=predict(fit,point,interval="confidence",se.fit=TRUE,level=.95),prediction=predict(fit,point,interval="prediction",level=.95))
},error=function(e) list(error=conditionMessage(e)))
jsonlite::write_json(list(H49=h49,H50=h50),file.path(out,"r.json"),digits=NA,auto_unbox=TRUE,pretty=TRUE)
