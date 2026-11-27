library(rsm)
library(desirability)
options(digits=17)
out <- Sys.getenv("R_ORACLE_OUT")
pred <- function(fit, points) {
  ci <- predict(fit, points, interval="confidence", level=.95, se.fit=TRUE)
  pi <- predict(fit, points, interval="prediction", level=.95, se.fit=TRUE)
  list(points=points, model_matrix=model.matrix(delete.response(terms(fit)), points,
    contrasts.arg=fit$contrasts), coef=coef(fit), confidence=ci, prediction=pi)
}
wb <- pred(lm(breaks ~ wool + tension, warpbreaks),
  expand.grid(wool=levels(warpbreaks$wool), tension=levels(warpbreaks$tension)))
CR <- coded.data(ChemReact, x1 ~ (Time - 85)/5, x2 ~ (Temp - 175)/5)
stationary <- xs(rsm(Yield ~ Block + SO(x1, x2), CR))
cr <- pred(lm(Yield ~ Block + x1 + x2 + x1:x2 + I(x1^2) + I(x2^2), CR),
  data.frame(Block=factor(c("B1", "B1"), levels=levels(CR$Block)),
    x1=c(0, stationary[1]), x2=c(0, stationary[2])))
# Evaluate only the published example chunks from the installed, pinned vignette.
vignette <- readLines(system.file("doc", "desirability.R", package="desirability"))
markers <- grep("^### code chunk number", vignette)
chunk <- function(number) {
  start <- markers[grep(paste0("number ", number, ":"), vignette[markers])]
  end <- min(c(markers[markers > start], length(vignette) + 1)) - 1
  eval(parse(text=vignette[(start+1):end]), envir=.GlobalEnv)
}
for (i in c(2, 6, 7, 12, 13)) chunk(i)
points <- rbind(optimum=best$par, centre=c(0,0,0),
  as.matrix(expand.grid(time=c(-1.682,1.682), temperature=c(-1.682,1.682),
    catalyst=c(-1.682,1.682)))[1:3,])
responses <- t(apply(points, 1, function(x) c(conversion=conversionPred(x), activity=activityPred(x))))
ds <- list(source="desirability.Rnw, sections 4 An Example and 5 Optimization; square region",
  models=list(conversion=deparse(conversionPred), activity=deparse(activityPred)),
  objects=list(conversion=unclass(conversionD), activity=unclass(activityD), overall=unclass(overallD)),
  points=points, responses=responses, individual=list(conversion=predict(conversionD, responses[,1]),
    activity=predict(activityD, responses[,2])), overall=predict(overallD, as.data.frame(responses)),
  optimization=list(method="optim default Nelder-Mead, control=list(fnscale=-1), square box +/-1.682",
    starts=searchGrid, result=best))
tcrit <- qt(.975, wb$confidence$df)
ends <- wb$confidence$fit[1,1] + c(-1,1)*tcrit*wb$confidence$se.fit[1]
stopifnot(max(abs(ends-wb$confidence$fit[1,2:3])) <= 1e-12)
jsonlite::write_json(list(warpbreaks=wb, ChemReact=cr, stationary=stationary, desirability=ds,
  consistency=list(tcrit=tcrit, manual=ends, R=wb$confidence$fit[1,2:3])),
  file.path(out,"r.json"), digits=NA, auto_unbox=TRUE, pretty=TRUE)
