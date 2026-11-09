#!/usr/bin/env Rscript
# C10 registered secondary analysis: the three lme4 models of Workbook 2, Ch. 7 ("Mixed-effects analysis").
#
#   (A) failures per episode, Poisson:  failures ~ configuration + offset(log(steps)) + (1 | task)
#       refit as negative binomial (glmer.nb) when the Pearson dispersion of the Poisson fit exceeds 1.5
#   (B) step-level failure, binomial:   failure ~ precision * step_type + precision * turn_depth + (1 | task/episode)
#   (C) episode success, binomial:      success ~ configuration + (1 | task)
#
# Convergence fallback for every model: default optimizer -> bobyqa (maxfun 2e5) -> drop the episode
# intercept (model B only) -> report the failure. Singular fits count as converged and are flagged.
#
# Input: steps / episodes tables as written by testbed/logger.py (steps.parquet, episodes.parquet), or CSV
# exports of them. Parquet needs the R 'arrow' package; testbed/mixed_effects.py exports CSV instead.
# Output: one JSON file with estimates (log scale and exponentiated), Wald 95% CIs and p-values.
#
# Usage:
#   Rscript analysis/mixed_effects.R --steps steps.csv --episodes episodes.csv --out mixed.json \
#       [--configuration variant] [--ref bf16] [--filter backend=cuda ...] [--depth-breaks 0,1,2,4] \
#       [--dispersion-threshold 1.5] [--models A,B,C]
#
# Column derivations (a column already present in the input wins):
#   failure    = failure_class in classifier.FAILURES (testbed/classifier.py)
#   step_type  = "call" if the step emitted a tool call block, "end" if failure_class == "end_turn", else "text"
#   turn_depth = turn (0-based) cut at --depth-breaks (lower bounds): 0,1,2,4 -> t0, t1, t2-3, t4+
#   precision  = 16bit / 8bit / 4bit from the variant name
#   success    = valid
#   episode    = run_id;  task = task_id

SCRIPT_VERSION <- "1.0.0"
FAILURES <- c("truncated", "malformed", "unknown_tool", "invalid_args", "exec_error", "forced_quit", "missing_call")
CALL_CLASSES <- c("malformed", "unknown_tool", "invalid_args", "exec_error", "ok_call")

suppressPackageStartupMessages(library(lme4))

# ---------------------------------------------------------------- arguments
parse_args <- function(argv) {
  opt <- list(configuration = "variant", ref = NA, filter = character(0), depth_breaks = "0,1,2,4",
              dispersion_threshold = 1.5, models = "A,B,C", steps = NA, episodes = NA, out = NA)
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if (!startsWith(a, "--")) stop("unexpected argument: ", a)
    kv <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
    key <- gsub("-", "_", kv[1])
    if (length(kv) > 1) {
      val <- paste(kv[-1], collapse = "=")
    } else {
      i <- i + 1
      if (i > length(argv)) stop("missing value for ", a)
      val <- argv[i]
    }
    if (!key %in% names(opt)) stop("unknown option --", kv[1])
    if (key == "filter") opt$filter <- c(opt$filter, val) else opt[[key]] <- val
    i <- i + 1
  }
  for (k in c("steps", "episodes", "out")) if (is.na(opt[[k]])) stop("--", k, " is required")
  opt$dispersion_threshold <- as.numeric(opt$dispersion_threshold)
  opt$depth_breaks <- as.integer(strsplit(opt$depth_breaks, ",")[[1]])
  opt$models <- toupper(strsplit(opt$models, ",")[[1]])
  opt
}

# ---------------------------------------------------------------- input
read_table <- function(path) {
  if (grepl("\\.parquet$", path)) {
    if (!requireNamespace("arrow", quietly = TRUE))
      stop("reading Parquet needs the R 'arrow' package; export CSV instead (testbed/mixed_effects.py does this)")
    return(as.data.frame(arrow::read_parquet(path)))
  }
  read.csv(path, stringsAsFactors = FALSE)
}

to_bool <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x)) return(x != 0)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes")
}

apply_filters <- function(df, filters) {
  for (f in filters) {
    kv <- strsplit(f, "=", fixed = TRUE)[[1]]
    if (kv[1] %in% names(df)) df <- df[as.character(df[[kv[1]]]) == kv[2], , drop = FALSE]
  }
  df
}

precision_of <- function(variant) {
  v <- tolower(as.character(variant))
  ifelse(v %in% c("bf16", "f16", "fp16", "f32", "fp32"), "16bit",
         ifelse(grepl("q8", v), "8bit", ifelse(grepl("q4", v), "4bit", v)))
}

depth_bins <- function(turn, breaks) {
  labs <- vapply(seq_along(breaks), function(i) {
    lo <- breaks[i]
    if (i == length(breaks)) return(sprintf("t%d+", lo))
    hi <- breaks[i + 1] - 1
    if (hi == lo) sprintf("t%d", lo) else sprintf("t%d-%d", lo, hi)
  }, "")
  factor(labs[findInterval(turn, breaks)], levels = labs)
}

pick_ref <- function(levels, ref) {
  if (!is.na(ref)) {
    if (!ref %in% levels) stop("reference level '", ref, "' not in data: ", paste(levels, collapse = ", "))
    return(ref)
  }
  hit <- intersect(c("bf16", "f16", "fp16", "16bit", "f32"), levels)
  if (length(hit)) hit[1] else sort(levels)[1]
}

# ---------------------------------------------------------------- fitting with the registered fallback
fit_once <- function(formula, data, family, nb, control) {
  warns <- character(0)
  fit <- tryCatch(withCallingHandlers({
    if (nb) glmer.nb(formula, data = data, control = control)
    else glmer(formula, data = data, family = family, control = control)
  }, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
     message = function(m) invokeRestart("muffleMessage")),
  error = function(e) { warns <<- c(warns, paste("error:", conditionMessage(e))); NULL })
  conv_msgs <- if (is.null(fit)) character(0) else fit@optinfo$conv$lme4$messages
  bad <- is.null(fit) || length(conv_msgs) > 0 || any(grepl("converge|Hessian|eigenvalue|error", warns))
  list(fit = fit, converged = !bad, warnings = unique(c(warns, conv_msgs)))
}

fit_with_fallback <- function(formula, data, family, nb = FALSE, reduced = NULL) {
  steps <- list(
    list(step = "default", formula = formula, control = glmerControl()),
    list(step = "bobyqa", formula = formula,
         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))))
  if (!is.null(reduced))
    steps[[3]] <- list(step = "drop_episode_intercept", formula = reduced,
                       control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  attempts <- list()
  for (s in steps) {
    r <- fit_once(s$formula, data, family, nb, s$control)
    attempts[[length(attempts) + 1]] <- list(step = s$step, formula = deparse1(s$formula),
                                             converged = r$converged, warnings = as.list(r$warnings))
    if (r$converged) return(list(fit = r$fit, step = s$step, formula = s$formula, attempts = attempts))
  }
  list(fit = NULL, step = "failed", formula = formula, attempts = attempts)
}

pearson_dispersion <- function(fit) sum(residuals(fit, type = "pearson")^2) / df.residual(fit)

fixed_table <- function(fit) {
  b <- fixef(fit)
  se <- sqrt(diag(as.matrix(vcov(fit))))
  z <- b / se
  lapply(seq_along(b), function(i) list(
    term = names(b)[i], estimate = b[[i]], se = se[[i]], z = z[[i]], p = 2 * pnorm(-abs(z[[i]])),
    ci_low = b[[i]] - qnorm(0.975) * se[[i]], ci_high = b[[i]] + qnorm(0.975) * se[[i]],
    exp_estimate = exp(b[[i]]), exp_ci_low = exp(b[[i]] - qnorm(0.975) * se[[i]]),
    exp_ci_high = exp(b[[i]] + qnorm(0.975) * se[[i]])))
}

# Wald estimate of a sum of fixed effects (e.g. the 4-bit effect on text steps = main + interaction)
lincomb <- function(fit, terms, label) {
  b <- fixef(fit); V <- as.matrix(vcov(fit))
  if (!all(terms %in% names(b))) return(list(label = label, terms = as.list(terms), estimate = NA))
  L <- as.numeric(names(b) %in% terms)
  est <- sum(L * b); se <- sqrt(drop(t(L) %*% V %*% L)); z <- est / se
  list(label = label, terms = as.list(terms), estimate = est, se = se, z = z, p = 2 * pnorm(-abs(z)),
       ci_low = est - qnorm(0.975) * se, ci_high = est + qnorm(0.975) * se,
       exp_estimate = exp(est), exp_ci_low = exp(est - qnorm(0.975) * se), exp_ci_high = exp(est + qnorm(0.975) * se))
}

describe <- function(res, family_name, scale, data, groups) {
  out <- list(status = if (is.null(res$fit)) "failed" else "ok", family = family_name, scale = scale,
              formula = deparse1(res$formula), fallback_step = res$step, attempts = res$attempts,
              n_obs = nrow(data), n_groups = lapply(groups, function(g) length(unique(data[[g]]))))
  if (!is.null(res$fit)) {
    out$singular <- isSingular(res$fit)
    out$aic <- AIC(res$fit)
    out$fixed <- fixed_table(res$fit)
    out$random_sd <- lapply(VarCorr(res$fit), function(v) attr(v, "stddev")[[1]])
  }
  out
}

# ---------------------------------------------------------------- JSON (base R, no dependency)
json <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    if (!is.null(names(x)) && length(x))
      return(paste0("{", paste0(vapply(names(x), json_str, ""), ":", vapply(x, json, ""), collapse = ","), "}"))
    return(paste0("[", paste0(vapply(x, json, ""), collapse = ","), "]"))
  }
  if (length(x) != 1) return(json(as.list(unname(x))))
  if (is.factor(x)) x <- as.character(x)
  if (is.na(x)) return("null")
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.numeric(x)) return(if (is.finite(x)) format(x, digits = 15, scientific = -2) else "null")
  json_str(as.character(x))
}
json_str <- function(s) {
  s <- gsub("\\\\", "\\\\\\\\", s); s <- gsub("\"", "\\\\\"", s)
  s <- gsub("\n", "\\\\n", s); s <- gsub("\t", "\\\\t", s); s <- gsub("\r", "\\\\r", s)
  paste0("\"", s, "\"")
}

# ---------------------------------------------------------------- main
main <- function() {
  opt <- parse_args(commandArgs(trailingOnly = TRUE))
  notes <- list()
  steps <- apply_filters(read_table(opt$steps), opt$filter)
  eps <- apply_filters(read_table(opt$episodes), opt$filter)
  cfg <- opt$configuration
  if (!cfg %in% names(eps)) stop("configuration column '", cfg, "' not in episodes")

  # episodes
  eps$episode <- as.character(eps$run_id)
  eps$task <- as.character(eps$task_id)
  eps$success <- as.integer(to_bool(if ("success" %in% names(eps)) eps$success else eps$valid))
  ref <- pick_ref(unique(as.character(eps[[cfg]])), opt$ref)
  eps$configuration <- relevel(factor(as.character(eps[[cfg]])), ref = ref)

  # steps (only steps of the kept episodes)
  steps$episode <- as.character(steps$run_id)
  steps <- steps[steps$episode %in% eps$episode, , drop = FALSE]
  steps$task <- as.character(steps$task_id)
  steps$failure <- as.integer(if ("failure" %in% names(steps)) to_bool(steps$failure)
                              else steps$failure_class %in% FAILURES)
  if (!"step_type" %in% names(steps)) {
    has_call <- if ("n_tool_call_blocks" %in% names(steps)) steps$n_tool_call_blocks > 0 else FALSE
    steps$step_type <- ifelse(has_call | steps$failure_class %in% CALL_CLASSES, "call",
                              ifelse(steps$failure_class == "end_turn", "end", "text"))
  }
  steps$turn_depth <- if ("turn_depth" %in% names(steps)) factor(steps$turn_depth)
                      else depth_bins(steps$turn, opt$depth_breaks)
  if (!"precision" %in% names(steps)) steps$precision <- precision_of(steps$variant)

  models <- list()

  # (A) failures per episode
  if ("A" %in% opt$models) {
    fails <- tapply(steps$failure, steps$episode, sum)
    agg <- data.frame(episode = names(fails), failures = as.integer(fails),
                      steps = as.integer(table(steps$episode)[names(fails)]), stringsAsFactors = FALSE)
    a <- merge(eps[, c("episode", "task", "configuration")], agg, by = "episode")
    if (nrow(a) < nrow(eps)) notes[[length(notes) + 1]] <- sprintf(
      "model A: %d episode(s) without step rows dropped", nrow(eps) - nrow(a))
    fA <- failures ~ configuration + offset(log(steps)) + (1 | task)
    pois <- fit_with_fallback(fA, a, poisson())
    disp <- if (is.null(pois$fit)) NA else pearson_dispersion(pois$fit)
    resA <- describe(pois, "poisson", "rate_ratio", a, c("task"))
    resA$poisson_dispersion <- disp
    resA$dispersion_threshold <- opt$dispersion_threshold
    if (!is.na(disp) && disp > opt$dispersion_threshold) {
      nb <- fit_with_fallback(fA, a, NULL, nb = TRUE)
      poisson_attempts <- resA$attempts
      resA <- describe(nb, "negative_binomial", "rate_ratio", a, c("task"))
      resA$poisson_dispersion <- disp
      resA$dispersion_threshold <- opt$dispersion_threshold
      resA$poisson_attempts <- poisson_attempts
      if (!is.null(nb$fit)) resA$nb_theta <- getME(nb$fit, "glmer.nb.theta")
    }
    models$A <- resA
  }

  # (B) step-level failure
  if ("B" %in% opt$models) {
    b <- steps
    # a step type or depth bin whose steps all fail or all succeed is not estimable (e.g. "end" steps
    # cannot fail under classifier v0); drop it and say so
    for (v in c("step_type", "turn_depth")) {
      r <- tapply(b$failure, as.character(b[[v]]), mean)
      degenerate <- names(r)[r == 0 | r == 1]
      if (length(degenerate)) {
        notes[[length(notes) + 1]] <- sprintf("model B: %s level(s) %s dropped (failure rate %s)", v,
          paste(degenerate, collapse = ", "), paste(round(r[degenerate], 3), collapse = ", "))
        b <- b[!as.character(b[[v]]) %in% degenerate, , drop = FALSE]
      }
    }
    b$precision <- relevel(factor(b$precision), ref = pick_ref(unique(b$precision), NA))
    st <- names(sort(table(b$step_type), decreasing = TRUE))
    b$step_type <- factor(b$step_type, levels = if ("call" %in% st) c("call", setdiff(st, "call")) else st)
    b$turn_depth <- droplevels(factor(b$turn_depth))
    fB <- failure ~ precision * step_type + precision * turn_depth + (1 | task / episode)
    rB <- failure ~ precision * step_type + precision * turn_depth + (1 | task)
    res <- fit_with_fallback(fB, b, binomial(), reduced = rB)
    resB <- describe(res, "binomial", "odds_ratio", b, c("task", "episode"))
    resB$reference <- list(precision = levels(b$precision)[1], step_type = levels(b$step_type)[1],
                           turn_depth = levels(b$turn_depth)[1])
    if (!is.null(res$fit)) {
      combos <- list()
      for (p in levels(b$precision)[-1]) {
        pt <- paste0("precision", p)
        for (s in levels(b$step_type)) combos[[length(combos) + 1]] <- lincomb(res$fit,
          c(pt, if (s != levels(b$step_type)[1]) paste0(pt, ":step_type", s)),
          sprintf("%s vs %s | step_type=%s, turn_depth=%s", p, levels(b$precision)[1], s, levels(b$turn_depth)[1]))
        for (d in levels(b$turn_depth)[-1]) combos[[length(combos) + 1]] <- lincomb(res$fit,
          c(pt, paste0(pt, ":turn_depth", d)),
          sprintf("%s vs %s | step_type=%s, turn_depth=%s", p, levels(b$precision)[1], levels(b$step_type)[1], d))
      }
      resB$precision_effects <- combos
    }
    models$B <- resB
  }

  # (C) episode success
  if ("C" %in% opt$models) {
    res <- fit_with_fallback(success ~ configuration + (1 | task), eps, binomial())
    models$C <- describe(res, "binomial", "odds_ratio", eps, c("task"))
  }

  out <- list(
    script = "analysis/mixed_effects.R", script_version = SCRIPT_VERSION,
    r_version = R.version.string, lme4_version = as.character(packageVersion("lme4")),
    inputs = list(steps = opt$steps, episodes = opt$episodes, n_step_rows = nrow(steps), n_episodes = nrow(eps),
                  configuration = cfg, reference = ref, levels = as.list(levels(eps$configuration)),
                  filters = as.list(opt$filter), depth_breaks = as.list(opt$depth_breaks)),
    ci_method = "Wald 95%", notes = notes, models = models)
  writeLines(json(out), opt$out)
  failed <- names(models)[vapply(models, function(m) m$status != "ok", TRUE)]
  cat(sprintf("wrote %s (models: %s%s)\n", opt$out, paste(names(models), collapse = ","),
              if (length(failed)) paste0("; FAILED: ", paste(failed, collapse = ",")) else ""))
}

main()
