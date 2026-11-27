library(rsm)
bootstrap <- function(fit, seed) {
  X <- model.matrix(fit)
  n <- nrow(X)
  r <- residuals(fit) / sqrt(1 - hatvalues(fit))
  rt <- r - mean(r)
  set.seed(seed)
  indices <- matrix(sample.int(n, n * 999, replace=TRUE), n)
  replicates <- t(vapply(seq_len(999), function(b)
    coef(lm.fit(X, fitted(fit) + rt[indices[, b]])), numeric(ncol(X))))
  delta <- max(vapply(seq_len(999), function(b)
    max(abs(replicates[b, ] - (coef(fit) +
      solve(crossprod(X), crossprod(X, rt[indices[, b]]))))), numeric(1)))
  ends <- apply(replicates, 2, quantile, probs=c(.025, .975), type=7)
  list(model_matrix=X, columns=colnames(X), y=model.response(model.frame(fit)),
    fitted=fitted(fit), hat=hatvalues(fit), rt=rt, indices=indices,
    indices_layout="n x B, 1-based", B=999, seed=seed, replicates=replicates,
    lower=ends[1, ], upper=ends[2, ], consistency_max_abs=delta)
}
CR <- coded.data(ChemReact, x1 ~ (Time - 85)/5, x2 ~ (Temp - 175)/5)
cases <- list(stackloss=bootstrap(lm(stack.loss ~ ., datasets::stackloss), 20261127),
  ChemReact=bootstrap(lm(Yield ~ Block + x1 + x2 + x1:x2 + I(x1^2) + I(x2^2), CR), 20261128))
print(vapply(cases, function(x) x$consistency_max_abs, numeric(1)))
jsonlite::write_json(cases, file.path(Sys.getenv("R_ORACLE_OUT"), "r.json"),
  digits=NA, auto_unbox=TRUE)
stopifnot(all(vapply(cases, function(x) x$consistency_max_abs <= 1e-10, logical(1))))
