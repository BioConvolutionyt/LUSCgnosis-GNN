### For each split of the features, perform WGCNA and generate adjacency matrix. 

# Replace with local input/output directories before running
data_folder = "REPLACE_WITH_INPUT_FEATURE_DIR"
output_folder = "REPLACE_WITH_OUTPUT_ADJ_DIR"


# Replace with the actual training set filename before running
data_file = paste(data_folder, "REPLACE_WITH_TRAINING_CSV_FILENAME", sep='/')

# WGCNA parameters
wgcna_power = 6
data = read.csv(data_file, header=F) # each row is a patient
geneExp = as.matrix(data[2:dim(data)[1], 2:321])

# gene as columns for WGCNA
# geneExp = t(geneExp)
dim(geneExp)

## imputate the NA by zero values.
geneExp[is.na(geneExp)]<-0
geneExp <- matrix(as.numeric(as.vector(geneExp)), nrow = nrow(geneExp), ncol = ncol(geneExp))
geneExp <- log2(geneExp + 1)
colnames(geneExp) <- data[1, ][2:321]

library(WGCNA)
adjacency = adjacency(geneExp, power = wgcna_power)
write.csv(adjacency,file=paste(output_folder, "adjacency_matrix.csv", sep='/'),quote=F,row.names = F)

