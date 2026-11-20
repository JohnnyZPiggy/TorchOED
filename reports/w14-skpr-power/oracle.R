out <- Sys.getenv("R_ORACLE_OUT")
results <- list()
for (name in c("D8", "D12", "D16")) {
  design <- read.csv(file.path("reports/w14-skpr-power", paste0(name, ".csv")))
  result <- skpr::eval_design(design, model = ~ (x1 + x2 + x3)^2, alpha = 0.05, effectsize = 2)
  results[[name]] <- result
  print(result)
}
jsonlite::write_json(results, file.path(out, "reference.json"), dataframe="rows", auto_unbox=TRUE, pretty=TRUE, digits=17)
