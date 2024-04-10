# MPeMSupervisedSurv

The advent of Deep Learning initiated a new era in which neural networks relying solely on Whole-Slide Images can estimate the survival time of cancer patients. Remarkably, despite deep learning’s potential in this domain, no prior research has been conducted on image-based survival analysis specifically for peritoneal mesothelioma. 
We introduce MPeMSupervisedSurv, a Convolutional Neural Network designed to predict the survival time of patients diagnosed with this disease.  MPeMSupervisedSurv demonstrates improvements over comparable methods with full patch supervision.

Using our proposed model, we performed patient stratification to assess the impact of clinical variables on survival time. Notably, the inclusion of information regarding adjuvant chemotherapy significantly enhances the model’s predictive prowess. Therefore, our findings indicate that treatment by adjuvant chemotherapy could be a factor affecting survival time.

The article was published in MDPI:
```bash
@Article{biomedinformatics4010046,
author = {Papadopoulos, Kleanthis Marios and Barmpoutis, Panagiotis and Stathaki, Tania and Kepenekian, Vahan and Dartigues, Peggy and Valmary-Degano, Séverine and Illac-Vauquelin, Claire and Avérous, Gerlinde and Chevallier, Anne and Laverriere, Marie-Hélène and Villeneuve, Laurent and Glehen, Olivier and Isaac, Sylvie and Hommell-Fontaine, Juliette and Ng Kee Kwong, Francois and Benzerdjeb, Nazim},
TITLE = {Overall Survival Time Estimation for Epithelioid Peritoneal Mesothelioma Patients from Whole-Slide Images},
journal = {BioMedInformatics},
volume = {4},
year = {2024},
number = {1},
pages = {823--836},
url = {https://www.mdpi.com/2673-7426/4/1/46},
issn = {2673-7426},
abstract = {Background: The advent of Deep Learning initiated a new era in which neural networks relying solely on Whole-Slide Images can estimate the survival time of cancer patients. Remarkably, despite deep learning’s potential in this domain, no prior research has been conducted on image-based survival analysis specifically for peritoneal mesothelioma. Prior studies performed statistical analysis to identify disease factors impacting patients’ survival time. Methods: Therefore, we introduce MPeMSupervisedSurv, a Convolutional Neural Network designed to predict the survival time of patients diagnosed with this disease. We subsequently perform patient stratification based on factors such as their Peritoneal Cancer Index and on whether patients received chemotherapy treatment. Results: MPeMSupervisedSurv demonstrates improvements over comparable methods. Using our proposed model, we performed patient stratification to assess the impact of clinical variables on survival time. Notably, the inclusion of information regarding adjuvant chemotherapy significantly enhances the model’s predictive prowess. Conversely, repeating the process for other factors did not yield significant performance improvements. Conclusions: Overall, MPeMSupervisedSurv is an effective neural network which can predict the survival time of peritoneal mesothelioma patients. Our findings also indicate that treatment by adjuvant chemotherapy could be a factor affecting survival time.},
doi = {10.3390/biomedinformatics4010046}
}
```
MPeMSupervisedSurv is a CNN inspired from EE-Surv:
```bash
@InProceedings{pmlr-v156-ghaffari-laleh21a,
  title = 	 {Deep Learning for interpretable end-to-end survival (E-ESurv) prediction in gastrointestinal cancer histopathology},
  author =       {Ghaffari Laleh, Narmin and Echle, Amelie and Muti, Hannah Sophie and Hewitt, Katherine Jane and Volkmar, Schulz and Kather, Jakob Nikolas},
  booktitle = 	 {Proceedings of the MICCAI Workshop on Computational Pathology},
  pages = 	 {81--93},
  year = 	 {2021},
  editor = 	 {Atzori, Manfredo and Burlutskiy, Nikolay and Ciompi, Francesco and Li, Zhang and Minhas, Fayyaz and Müller, Henning and Peng, Tingying and Rajpoot, Nasir and Torben-Nielsen, Ben and van der Laak, Jeroen and Veta, Mitko and Yuan, 
  Yinyin and Zlobec, Inti},
  volume = 	 {156},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {27 Sep},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v156/ghaffari-laleh21a/ghaffari-laleh21a.pdf},
  url = 	 {https://proceedings.mlr.press/v156/ghaffari-laleh21a.html},
  abstract = 	 {This paper demonstrates and validates EE-Surv, a powerful yet algorithmically simple method to predict survival directly from whole slide images which we validate in colorectal and gastric cancer, two clinically relevant and markedly different tumor types.}
}
```
The code contained in MPeMSupervisedSurv also draws inspiration from the Google Colab notebook found in the link below:
https://colab.research.google.com/github/sebp/survival-cnn-estimator/blob/master/tutorial_tf2.ipynb#scrollTo=azrczYYVvEQb.
