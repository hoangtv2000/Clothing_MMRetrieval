# Clothing Retrieval

**Abstract.** 
This work introduces and implements ComposeTransformers, the end-to-end recommender system for woman clothing via image-text content retrieval.

**Keywords**: Vision Transformer, BERT, multi-modal search.



## A. Paper and Seminar meterial

#### ⭐ For detail of report, watch [this article](https://github.com/hoangtv2000/Clothing_MMRetrieval/blob/main/retrieval.pdf).

#### ⭐ For slide of the seminar, watch [here](https://github.com/hoangtv2000/Clothing_MMRetrieval/blob/main/presentation.pptx).


## B. Technial tool

Annotations for modules of source code:
+ [requirements](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/requirements.txt): includes necessary libraries.
+ [config](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/config): a configuration file used for both training and inference phase.
+ Fashion200k: The folder containing all data and annotations, it is not available at the moment.
+ [dataloader](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/dataloader): code for dataloader (including image and text pre-processing).
+ [img_text_composition_model](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/img_text_composition_model): containing image-text composition module.
+ [logger](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/logger): logger of the training phase.
+ [tester](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/tester): for testing performance of the retrieval model.
+ [trainer](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/trainer): code for training phase with the ability to track loss and evaluation metrics during this progress.
+ [triplet_loss](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/triplet_loss): soft triplet loss module.
+ [utils](https://github.com/hoangtv2000/Clothing_MMRetrieval/tree/main/utils): utility functions.
+ [ComposeTransformers_Notebook](https://github.com/hoangtv2000/Clothing_MMRetrieval/blob/main/ComposeTransformers_Notebook.ipynb): notebook for training & evaluation and inference.
+ [IMAGE_FTRS](https://github.com/hoangtv2000/Clothing_MMRetrieval/blob/main/IMAGE_FTRS.npz): including extracted feature and path for all images in the sub-dataset.


## C. Results
### The pre-trained model
**Ask me in issue if you look foward to the pre-trained model**

### Evaluation results
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="4">Precision (%) </th>
  </tr>
  <tr>
    <td class="tg-7btt">P@1</td>
    <td class="tg-7btt">P@10</td>
    <td class="tg-7btt">P@50</td>
    <td class="tg-7btt">P@100</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.25</td>
    <td class="tg-c3ow">5.5</td>
    <td class="tg-c3ow">31.6</td>
    <td class="tg-c3ow">58.0</td>
  </tr>
</table></div>


<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="4">Recall (%) </th>
  </tr>
  <tr>
    <td class="tg-7btt">R@1</td>
    <td class="tg-7btt">R@10</td>
    <td class="tg-7btt">R@50</td>
    <td class="tg-7btt">R@100</td>
  </tr>
  <tr>
    <td class="tg-c3ow">4.9</td>
    <td class="tg-c3ow">22.6</td>
    <td class="tg-c3ow">55.4</td>
    <td class="tg-c3ow">75.7</td>
  </tr>
</table></div>


## D. Demo


<div align='center'>	
<a link href ='https://user-images.githubusercontent.com/58163069/154830995-0d26c5c0-e877-483f-b2db-78d0bd061628.mp4'>Demo </a>
</div>

<div align='center'>
<b> Full demo video </b>	
</div>
