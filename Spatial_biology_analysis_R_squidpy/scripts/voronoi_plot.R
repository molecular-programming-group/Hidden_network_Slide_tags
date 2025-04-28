#' Test to make Voronoi representation from the reconstructed network
#' 
#' Found help here: https://stackoverflow.com/questions/61207179/setting-a-maximum-size-for-voronoi-cell-using-ggplot2

#' Install necessary packages
install.packages("ggforce")

#' Prep
library(ggforce)
library(ggplot2)

set.seed(123)

DIR_DATA <- file.path(getwd(), "data")
DIR_RES <- file.path(getwd(), "results")

colors_labels <- c("#EDC787", "#9EBBD5", "#CED1A0", "#776E69", "#C6CFD1", "#D2BBA1", "#CDDFDA")

#' Read data
pos_df <- read.csv(file.path(DIR_DATA, "reconstructed_positions_idx.csv")) 
pos_df$test_labs <- paste0("cell_", sample(1:5, nrow(pos_df), replace = T))


#' Plot
arr <- list(x = -3, y = -3, x_len = 3, y_len = 3)
pt_size <- 1
txt_size <- 12
max_rad <- 1

for(max_rad in c("0.25", "0.50", "0.75", "1.00")) {
  message(max_rad)
  p <- ggplot(pos_df, aes(x, y, group = -1L)) +
    geom_voronoi_tile(aes(fill=test_labs),
                      max.radius = as.numeric(max_rad),
                      colour = "black") +
    geom_point(aes(x=x,y=y), size=pt_size) +
    scale_fill_manual(values = colors_labels) +
    coord_fixed() +
    annotate("segment", 
             x = arr$x, xend = arr$x + c(arr$x_len, 0), 
             y = arr$y, yend = arr$y + c(0, arr$y_len), 
             arrow = arrow(type = "closed", length = unit(5, 'pt'))) +
    # theme_void()
    theme_minimal() +
    theme(panel.grid = element_blank(), 
          axis.ticks = element_blank(), 
          axis.text = element_blank(),
          axis.title = element_blank(),
          legend.title = element_blank(),
          legend.text = element_text(size=txt_size, color="black"))
  
  rad_txt <- stringr::str_pad(gsub("\\.", "", max_rad), 3, pad = "0")
  pdf(file = file.path(DIR_RES, paste0("voronoi_test_rad", rad_txt, ".pdf")), width = 8, height = 8)
  print(p)
  dev.off()
}


#' Save plot
pdf(file = file.path(DIR_RES, paste0("voronoi_test.pdf")), width = 8, height = 8);p;dev.off()

