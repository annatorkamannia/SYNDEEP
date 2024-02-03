Overview:
SYNDEEP is a deep neural network model developed for predicting drug synergy in cancer cell lines. In comparison to existing methods, SYNDEEP integrates a diverse set of features, including drug-target interactions, protein-protein interactions, protein-metabolite interactions, genomic features (gene expression, mutations, and differential methylation), chemical structures, and cell lines. The model excels in accurately predicting synergistic effects, showcasing its potential in optimizing cancer treatment strategies. The prediction accuracy and AUC metrics for this model were 92.21% and 97.32% in 10-fold cross-validation.
1.	Materials and Methods:
o	Thorough explanation of data acquisition from the NCI-ALMANAC dataset.
o	Feature extraction details from drug-target interactions, protein-protein interactions, and more.
o	Construction of a network of features and feature groups.
2.	Construction of Deep Neural Network Model:
o	In-depth insight into the Multi-layer perceptron (MLP) architecture.
o	Hyperparameter settings, activation functions, and loss function.
3.	Evaluation Criteria:
o	Explanation of the 10-fold cross-validation.
o	Key evaluation metrics such as accuracy, sensitivity, specificity, precision, F-Score, MCC, and Cohen's kappa.
4.	Computational Equipment:
o	Overview of the software environment, including Python 3.7, Keras, Scikit-learn, and the usage of Google Colab.
Usage:
1.	Requirements:
o	Python 3.7 or later.
o	Required libraries: Keras, Scikit-learn.
2.	Running SYNDEEP:
o	Ensure you have the necessary dependencies installed.
o	Execute the provided code in a suitable environment.
3.	Customization:
o	You can customize the model based on your specific dataset and requirements.
o	Adjust hyperparameters and feature groups for optimal performance.
Citation
If you used our work and found the provided data helpful please cite:
Torkamannia, A., Omidi, Y. & Ferdousi, R. SYNDEEP: a deep learning approach for the prediction of cancer drugs synergy. Sci Rep 13, 6184 (2023). https://doi.org/10.1038/s41598-023- 33271-3.


