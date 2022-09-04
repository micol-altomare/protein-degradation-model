*Disclaimer: this is an overview of a project that is still in progress and is confidential. The potential paper this work contributes to is not yet published.*

# Designing a Predictive Deep Learning Model for Degradation Rate Based on Primary Sequence
Micol Altomare | Garton Lab | May-August 2022 

Over the last summer, I learned a lot about deep learning and being a researcher. From first being introduced to the field with the book [_Deep Learning Illustrated_](https://deeplearningillustrated.com) and learning Pytorch, to participating in weekly lab meetings to meeting other research students in the Undergraduate Summer Research Program (USRP), I have witnessed what it is like to contribute to a pressing field and gained skills that will help me in my biomedical engineering career.


## Goal
Being able to predict the degradation rate (aka. turnover rate, half-life) of a protein solely based on its primary sequence can improve protein engineering and design. Knowing an (engineered) protein's degradation rate means:
* incorporating protein lifespan into the protein design process
* being able to control the concentration of a given protein in the cell
* better resource management and minimizing cell burden


## Data 
### May 2022
This research project focused on the natural human proteome. After gaining familiarity with the deep learning field and reading the literature on candidate protein turnover datasets, I chose to use HeLa cell protein turnover rates from [Systems-wide proteomic analysis in mammalian cells reveals conserved, functional protein turnover](https://pubmed.ncbi.nlm.nih.gov/22050367/) ([dataset](https://acs.figshare.com/articles/dataset/Systems_wide_Proteomic_Analysis_in_Mammalian_Cells_Reveals_Conserved_Functional_Protein_Turnover/2576524)). 

In this paper, Cambridge et al. use high-resolution mass spectrometry to measure the half-life of proteins in the human (HeLa cell) and mouse (C2C12 cell) proteomes. 

*Note:* In this dataset, proteins and many of their isoforms are grouped together with the same half-life. This is due to the fact that it is not possible to distinguish which of the isoforms had that half-life. For a more robust model, other datasets should be explored. 



## Parsing & Preprocessing 
### June-July 2022
Given the HeLa cell dataset, I removed any proteins whose sequence was not in the [Uniprot](https://www.uniprot.org) database or whose secondary accession number was already in the dataset (ie. duplicate proteins).

To obtain the proteins' amino acid sequences, I retrieved the FASTA files for all the given HeLa cell proteins using their Uniprot ID from [UniprotKB](https://www.uniprot.org/id-mapping). Using the [Biopython](http://biopython.org/DIST/docs/tutorial/Tutorial.html) package, I parsed  through the FASTA files and extracted the amino acid sequences while removing the few sequences that contained amino acids (O, U, B, Z, X) besides those in humans.

<img width="1283" alt="seq length distrib" src="https://user-images.githubusercontent.com/97775581/188311252-243ada4a-bfb9-4bdc-8f75-0dfb2421d26e.png">

Lastly, I used one-hot encoding to convert the amino acid sequences into a binary matrix and zero-padding the sequences to a any maximum length. Setting the maximum length to 1200 amino acids long allowed for a training and testing set of +10 000 proteins collectively.


## Model 
### August 2022
Having learned a lot about deep learning and models currently used in molecular biology, I decided to adapt the multiple convolutional neural network model from [MCNN: multiple convolutional neural networks for RNA-protein binding sites prediction](https://www.computer.org/csdl/journal/tb/5555/01/09763385/1CT4PemsFfW) to the HeLa cell dataset ([code](https://github.com/biomg/MCNN#mcnn-multiple-convolutional-neural-networks-for-rnaprotein-binding-sites-prediction)). Using RNA sequences as input, the model predicted RBP binding sites to an average AUC of 0.95. Using a convolutional neural network is beneficial for learning features (more relevantly in this case, local sequence patterns) and combining serveral in a MCNN (where each CNN has a different window size) allowed for even more complex features to be learned. 

The CNN is composed of 2 layers and has the following characteristics:
* loss: cross-entropy
* optimizer: Adam
* metrics: accuracy 
* activation: relu
* dropout: 0.25
* scheduler: ReduceLROnPlateau

*Note:* This CNN model takes a categorical approach where each half-life (continuous data: hours to one decimal place) is rounded to become an integer. Each integer then becomes its own category(as in: labels = [8, 9, 10, 11, 12, ..., 100]). (Note that the dataset can be trimmed to contain sequences that have up to a specific half-life). During testing, the accuracy is calculated by generating a prediction (up to 1) for each category. The model then chooses the index corresponding to the highest prediction as its actual prediction. The accuracy is then the percentage of correctly-guessed predictions.

While a categorical approach accompanying crossentropy loss function can reduce the number of possibilities for prediction, it has several weaknesses. For instance, categories don't contain information about the relationship between categories (as opposed to ordinal classification). Further testing with numerical (eg. means squared error) loss functions can be used to improve the model's performance.

## Results
The model was trained in various rounds of epochs up to 500 epochs. As the weights and biases were initialized randomly and the previous model's latest model was loaded at the start of each round of epochs, the model had varying degrees of performance ranging from 0-39% accuracy. 

<img width="702" alt="round after the successful training round" src="https://user-images.githubusercontent.com/97775581/188311356-691da083-1fce-431c-a32d-bc53ea78f91e.png">

<img width="712" alt="best training round" src="https://user-images.githubusercontent.com/97775581/188311685-856e1be0-2045-4d76-8d03-eaa35dd4fc06.png">

<img width="546" alt="Screen Shot 2022-08-31 at 6 59 02 PM" src="https://user-images.githubusercontent.com/97775581/188311317-4271cbde-f66f-483f-b86c-8fc94ff6461f.png">

Training for more epochs at a time and automatically using the prior round's learning rate can improve the model's performance.


## Onward
With further improvements in the dataset and model, it is becoming more possible to predict the degradation rate of a protein. Applying this model to engineered proteins as well as to other proteomes can expand the range of possibilities for deep learning and protein design.

 
