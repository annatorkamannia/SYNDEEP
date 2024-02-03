<h1>Overview</h1>
SYNDEEP is a deep neural network model developed for predicting drug synergy in cancer cell lines. In comparison to existing methods, SYNDEEP integrates a diverse set of features, including drug-target interactions, protein-protein interactions, protein-metabolite interactions, genomic features (gene expression, mutations, and differential methylation), chemical structures, and cell lines. The model excels in accurately predicting synergistic effects, showcasing its potential in optimizing cancer treatment strategies. The prediction accuracy and AUC metrics for this model were 92.21% and 97.32% in 10-fold cross-validation.
<h1>Materials and Methods</h1>
<ul>
  <li>Thorough explanation of data acquisition from the NCI-ALMANAC dataset.</li>
  <li>Feature extraction details from drug-target interactions, protein-protein interactions, and more.</li>
  <li>Construction of a network of features and feature groups.</li>
</ul>
<h1>Construction of Deep Neural Network Model:</h1>
<ul>
   <li>In-depth insight into the Multi-layer perceptron (MLP) architecture.</li>
   <li>Hyperparameter settings, activation functions, and loss function.</li>
</ul>
<h1>Evaluation Criteria</h1>
<ul>
   <li>Explanation of the 10-fold cross-validation.</li>
   <li>Key evaluation metrics such as accuracy, sensitivity, specificity, precision, F-Score, MCC, and Cohen's kappa.</li>
</ul>
<h1>Computational Equipment</h1>
<ul>
   <li>	Overview of the software environment, including Python 3.7, Keras, Scikit-learn, and the usage of Google Colab.</li>
</ul>
<h1>Usage</h1>
<h3>Requirements</h3>
<ul>
   <li>Python 3.7 or later</li>
   <li>Required libraries: Keras, Scikit-learn.</li>
</ul>
<h3>Running SYNDEEP</h3>
<ul>
   <li>Ensure you have the necessary dependencies installed.</li>
   <li>Execute the provided code in a suitable environment.</li>
</ul>
<h3>Customization</h3>
<ul>
   <li>You can customize the model based on your specific dataset and requirements.</li>
   <li>Adjust hyperparameters and feature groups for optimal performance.</li>
</ul>
<h3>Citation</h3>
<ul>
   <li>If you used our work and found the provided data helpful please cite:</li>
   <p>Torkamannia, A., Omidi, Y. & Ferdousi, R. SYNDEEP: a deep learning approach for the prediction of cancer drugs synergy. Sci Rep 13, 6184 (2023).https://doi.org/10.1038/s41598-023-33271-3</p>
</ul>
