# This is for IIH patients study. 
# The dataset constitute three modalities, T1, T2, and T2eye

# Setting the Project Path ------------------------------------------------
# Clean the workspace
rm(list=ls())
proj_path = '/Users/getang/Documents/EarthResearch/IIH'
source(file.path(proj_path, 'src', 'functions_IIH.R'))

# Loading the dataset -----------------------------------------------------
df <- read.csv(file.path(proj_path, 'data', 'withinfo_IIH_Metrics_20231127.csv'))

# Convert the birthdate to Age
df['Age'] <- sapply(df$birthdate, calculate_age)
df['BMI'] <- as.numeric(df$Weight) / ((df$Height/100) **2)
df['R_eyeball_radius'] <- nthroot(df$R_eyeball,3)
df['L_eyeball_radius'] <- nthroot(df$L_eyeball,3)

# Rename the right and left to a consistent way
names(df) <- gsub("(.*)_(R|L)", "\\2_\\1", names(df))

# Set the variables for multiple plots if you want
info_variables <- c('id', 'ses', 'sub', 'modality', 'fn_root', 'group', 'Gender', 'birthdate', 'Age', 'BMI', 'Height', 'Weight')
all_variables <- setdiff(names(df), c(info_variables))

# Make the group variables as factorical
df[, setdiff(info_variables, c('id', 'sub', 'fn_root', 'birthdate', 'Age', 'BMI', 'Height', 'Weight'))] <- 
  lapply(df[, setdiff(info_variables, c('id', 'sub', 'fn_root', 'birthdate', 'Age', 'BMI', 'Height', 'Weight'))], factor)

# Make the group variables as numerical
df[, setdiff(info_variables, c('id', 'ses', 'sub', 'modality', 'fn_root', 'group', 'Gender', 'birthdate'))] <- 
  lapply(df[, setdiff(info_variables, c('id', 'ses', 'sub', 'modality', 'fn_root', 'group', 'Gender', 'birthdate'))], as.numeric)

# The group with the side difference of eye
df_long_eye <- df %>%
  pivot_longer(
    cols = matches("^(R_|L_).*|.*(_R|_L)$"),
    names_to = c("side", ".value"),
    names_pattern = "([RL])_(.*)"
  )
df_long_eye$groupsecond = paste(df_long_eye$group, df_long_eye$side, sep = '.')
df_long_eye$groupsecond = as.factor(df_long_eye$groupsecond)

# Demographics ------------------------------------------------------------
test = FALSE
temp_df <- df %>% filter(modality == 'T1'&ses=='ses-01')

# Age
demographic_cols <- c('Age', 'BMI')
temp_df <- temp_df[setdiff(info_variables, c('id', 'ses', 'fn_root', 'Height', 'Weight'))]
temp_df <- unique(temp_df)

demographic_summarize <- temp_df %>%
  group_by(group) %>%
  add_count() %>% 
  summarise(count = n(), across(demographic_cols, list(mean = ~mean(.x, na.rm = TRUE), 
                                                       sd = ~sd(.x, na.rm = TRUE))))
if (!test){
  print(demographic_summarize)
}
            
# violin plots for visualization ------------------------------------------
save = TRUE
test = TRUE
temp_df <- df_long_eye

if (test){
  iter = 1e1
} else {
  iter = 1e5
}

# To order the x axis in desired way
order <- c("control.L", "patient.L", "control.R", "patient.R")

name_p1 <- 'compare_between_control_patient_ONS.pdf'
(p1 <- temp_df %>% 
    mutate(groupsecond = factor(groupsecond, levels = order)) %>% 
    ggplot(aes(x = groupsecond, y = on_sheath_with_nerve)) +
    geom_jitter(width = 0.1) + 
    geom_violin(draw_quantiles = c(0.25, 0.5, 0.75), fill = 'transparent') +
    stat_summary(fun.y=mean, geom="point", shape=20, size=5, color="red", fill="red") +
    stat_summary(fun.data = mean_cl_boot, fun.args = list(B = iter, conf.int=0.95), geom="errorbar", width = 0.3, size = 1, color="red") +
    scale_y_continuous(breaks = c(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)) +
    theme(aspect.ratio = 1.1) +
    ggtitle(sub("\\.pdf$", "", name_p1))
)

name_p2 <- 'compare_between_control_patient_d3.pdf'
(p2 <- temp_df %>% 
    mutate(groupsecond = factor(groupsecond, levels = order)) %>% 
    ggplot(aes(x = groupsecond, y = d3)) +
    geom_point(aes(color = modality), position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.3)) + 
    geom_violin(draw_quantiles = c(0.25, 0.5, 0.75), fill = 'transparent') +
    stat_summary(aes(group = modality), fun.y=mean, geom="point", shape=20, size=4, position = position_dodge(0.9), stroke = 2) +
    stat_summary(fun.y=mean, geom="point", shape=20, size=5, color="red", fill="red") +
    stat_summary(fun.data = mean_cl_boot, fun.args = list(B = 1e5, conf.int=0.95), geom="errorbar", width = 0.4, size = 1, color="red") +
    stat_summary(aes(group = modality), fun.data = mean_cl_boot, fun.args = list(B = iter, conf.int=0.95), geom="errorbar", width = 0.4, size = 1, position = position_dodge(0.9)) +
    scale_y_continuous(breaks = c(15, 16, 17, 18, 19, 20, 21, 22, 23)) +
    theme(aspect.ratio = 1.1) +
    ggtitle(sub("\\.pdf$", "", name_p2))
)

if (save){
  for (plotnum in (c(1, 2))){
    filepath = file.path(proj_path, 'results', 'plots', get(paste('name_p', plotnum, sep = '')))
    print(paste('Saving files to ...', filepath))
    ggsave(filepath, get(paste('p', plotnum, sep = ''))) 
  }
}

# Computer ICC value for T1 and T2 ----------------------------------------
temp_df <- df
modalities <- unique(temp_df$modality)

# Configuration
save = TRUE
tablename <- 'icc_summary'
file_extension <- '.csv'

# Remove the empty connecting 
tablename <- paste(tablename, file_extension, sep = '')
filepath <- file.path(proj_path, 'results', 'tables', tablename)

# Functions
perform_paired_t_test <- function(pair) {
  if (!(all(is.na(pair[[1]]))|all(is.na(pair[[2]])))) {
    t.test(pair[[1]], pair[[2]], paired = TRUE)
  } 
}

compare_list <- list()
modalities <- unique(temp_df$modality)
for (var in all_variables){
  df_paired <- cbind(temp_df[temp_df$modality == modalities[1], ][, var], temp_df[temp_df$modality == modalities[2], ][, var])
  colnames(df_paired) <- c(modalities[1], modalities[2])
  compare_list[[var]] <- as.data.frame(df_paired)
}

summary_icc <- sapply(compare_list, icc, unit = 'single', type = "agreement", model = "twoway")
summary_icc <- as.data.frame(apply(summary_icc, 2, unlist))
summary_icc_char <- data.frame(lapply(summary_icc, as.character), stringsAsFactors = FALSE)
summary_icc_transposed <- as.data.frame(t(summary_icc_char), stringsAsFactors = FALSE)
colnames(summary_icc_transposed) <- rownames(summary_icc)
summary_icc_transposed_part <- summary_icc_transposed[, c('value', 'lbound', 'ubound', 'p.value')]
summary_icc_transposed_part <- summary_icc_transposed_part[!apply(summary_icc_transposed_part, 1, function(row) all(is.na(row))), ]

t_test_results <- sapply(compare_list, perform_paired_t_test, simplify = FALSE)
t_test_results_df <- do.call(rbind, lapply(t_test_results, function(x) {
  data.frame(
    t_estimate = x$estimate,
    t_p_value = x$p.value,
    t_statistic = x$statistic,
    t_conf_int_lower = x$conf.int[1],
    t_conf_int_upper = x$conf.int[2],
    t_degree_of_freedom = x$parameter
  )
}))
t_test_results_part <- t_test_results_df[, c('t_estimate', 't_conf_int_lower', 't_conf_int_upper')]
summary_table <- cbind(summary_icc_transposed_part, t_test_results_part)

if (save) {
  print(paste('Saving files to ...', filepath))
  write.csv2(summary_table, file = filepath)
}


# t test for control and patient ------------------------------------------
# Setting the parameters
file_extension <- '.csv'
type_of_test <- 'Welch_ttest' # choice: 'Paired_ttest', 'Wilcox_test'
save = FALSE
temp_df <- df
modalities <- unique(temp_df$modality)

# Set the table names
tablename <- paste(type_of_test, file_extension, sep = '')
filepath = file.path(proj_path, 'results', 'tables', tablename)

# t test
t_res <- c()
for (i in modalities){
  temp_df <- df %>% filter(modality==i)
  temp_df <- remove_na(temp_df)
  
  for (var in intersect(names(temp_df), all_variables)){
    res <- t.test(temp_df[temp_df$group=='patient',][,var], 
                  temp_df[temp_df$group=='control',][,var], paired = FALSE)
    t_res <- rbind(t_res, res_dataframe(res, i, 'patient - control', var))
  }
}
# Add the significant level
symp <- sig_symbol(t_res$p.value)
t_res <- cbind(t_res, Signif = symp)

if (save){
  print(paste('Saving files to ...', filepath))
  write.csv2(t_res, file = filepath)
}


# Testing -----------------------------------------------------------------




