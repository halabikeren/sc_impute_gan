library(ggplot2)
library(tidyr)
library(data.table)
library(patchwork)


create_stats_plot<- function(df_path, title, out_path, subset_versions=NULL){
  df <- read.csv(df_path)
  rownames(df) <- df$X
  df$X <- NULL
  columns <- colnames(df)
  colors_to_be_used <- c("ground.truth" = "#E69F00", "data.before.imputation" = "#56B4E9", "scIGANs" = "#009E73", 
                         "scIGANs2_only_BE" = "#F0E442", "scIGANs2.BE.DAI_sample" = "#CC79A7", "scIGANs2.BE.DAI_sample.noise" = "chocolate4", 
                         "scIGANs2.BE.DAI_p0" = "#0072B2",
                         "scIGANs2.BE.DAI_p1" = "#999999",  "scIGANs2.BE.DAI_p2" = "darkmagenta")
  if (is.null(subset_versions)){
    columns_order <- c("ground.truth", "data.before.imputation", "scIGANs", "scIGANs2_only_BE", "scIGANs2.BE.DAI_sample", "scIGANs2.BE.DAI_sample.noise",
                       "scIGANs2.BE.DAI_p0", "scIGANs2.BE.DAI_p1", "scIGANs2.BE.DAI_p2")
    
    
  }else{
    columns_order <- subset_versions

  }

  
  existing_names <- c()
  colors <- c()
  for (k in 1:length(columns_order)){
    if (columns_order[k] %in% columns){
      existing_names <- c(existing_names, columns_order[k])
      colors <- c(colors, unname(colors_to_be_used[columns_order[k]]))
    }
  }
  df <- df[,existing_names]

  

                     
  
  df_t <- transpose(df)
  rownames(df_t) <- colnames(df)
  colnames(df_t) <- rownames(df)
  df_t["Method"] = colnames(df)
  df_tall <- df_t %>% gather(key = Metrics, value = Value, MI:"F-score")
  p <- ggplot(df_tall, aes(Metrics, Value, fill = Method)) + geom_col(position=position_dodge(0.5), width=0.5)+scale_fill_manual(values=colors) + 
    ggtitle(title)  + coord_flip()
  ggsave(out_path)
  
  
}




dir_dfs <- "D:\\ItayMNB9\\Documents\\unsupervised_learning\\final_project\\presentation"
dirs_datasets <- c("De-noised_100G_3T_300cPerT_dynamics_8_DS8_3_tech_partitions_stats", "De-noised_100G_9T_300cPerT_4_DS1_3_tech_partitions_stats", "empirical")
names_datasets <- c("simulation 1", "simulation 2", "empirical data")
partitions <- c("cell types", "sequencing technology")
partitions_suffixes <- c(".csv", "_tech.csv")
partitions_paths <- c("cell.png", "tech.png")
subset_methods <-  c("ground.truth", "data.before.imputation", "scIGANs", "scIGANs2_only_BE", "scIGANs2.BE.DAI_sample")
for (i in 1:length(dirs_datasets)){
  dataset_path <- paste(c(dir_dfs, dirs_datasets[i]), collapse = "\\")
  dataset_name <- names_datasets[i]
  for (j in 1:length(partitions)){
    table_path <- paste(c(dataset_path, "\\summary_stats", partitions_suffixes[j]), collapse = "")
    subplot_title <- paste(c(dataset_name, partitions[j]), collapse = ": ")
    out_path <- paste(c(dataset_path, partitions_paths[j]), collapse = "\\")
    create_stats_plot(table_path, subplot_title, out_path, subset_methods)
    
  }
  

}
ggsave(out_path)


