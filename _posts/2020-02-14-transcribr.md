
##### A Transformer-based handwriting recognition architecture

![[[Image Source](https://chimpsnw.org/2019/08/in-honor-of-odd-man-inn/)]](https://cdn-images-1.medium.com/max/2172/1*BSaSpJ4e1znSO4aD_GaYrg.jpeg)
<span class='caption'>*[[Image Source](https://chimpsnw.org/2019/08/in-honor-of-odd-man-inn/)]*</span>

Digitizing handwritten documents to improve storage, access, search, and analysis is a compelling challenge. Prior to the deep learning revolution, no clear path existed towards achieving such a goal in a scalable way.

In light of advancements in computer vision and language processing, reliable and automated handwriting recognition is within reach. Towards that end, I endeavored to design a practical application which balances accuracy, generalizability, and inference speed. This post is a retrospective on that attempt as well as an explanation of design choices and training procedures. Check out a demo [here](https://transcribr.onrender.com).

Reading handwritten text is uniquely difficult. There are extreme variations between styles (e.g. cursive vs print vs block lettering), size, spacing, embellishments, and legibility. Misspellings, cross outs, and omissions are also common.

Many approaches to this problem divide the task into 3 separate components: segmentation, feature extraction, and classification. *[1,13,2]*

1. Detect and segment areas of text within an image.

1. Extract high dimensional features from each text segment.

1. Classify those features from a given vocabulary (e.g. words or characters).

Given the heterogeneity of handwriting, automatic text detection and segmentation can be error prone and often require custom preprocessing. An alternative approach is to frame text recognition as a sequence to sequence problem. A 2-dimensional image input with a 1-dimensional text output. The best performing sequence transduction models in NLP involve an encoder-decoder architecture, often including an attention mechanism. Perhaps the most revolutionary of these is the Transformer architecture *[3]* which is unique in that it relies solely on attention to encode representations of the input and output without resorting to any form of recurrence or convolution. An excellent explanation of the Transformer architecture can be found here: [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

This is not the first attempt to replace segmentation with attention in an end-to-end approach. *[5]* However, instead of augmenting the traditional recurrent layers with attention modules, I leveraged the Transformer architecture to surpass previous state-of-the-art results. Try the [Transcribr app](https://transcribr.onrender.com) for yourself!

***

## **ARCHITECTURE**

The Transformer architecture is based on layers of multi-head attention (“scaled dot-product”) followed by position-wise fully connected networks. Dot-product, or multiplicative, attention is faster (more computationally efficient) than additive attention though less performant in larger dimensions. Scaling helps to adjust for the shrinking gradients from multiplication. As per the paper, “multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.” Both the encoder and decoder utilize self-attention layers. The decoder additionally includes source-attention layers over the encoder outputs.

### Image Adaptor

The glaring difference between machine translation, for which the Transformer was designed, and handwriting recognition is that images must be manipulated into a sequential representation before being input into a Transformer.

Towards this end, a pre-trained convolutional neural network (ResNet18) is truncated prior to the final max-pooling layer to output a spatial feature map. The feature map is flattened along spatial dimensions, normalized, and transposed to create a rough 1-dimensional sequence of representations. (*Note*: ResNet18, ResNet34, and Xception models were tested without a significant difference in accuracy. I chose the smallest of the 3.)

### Encoder

Self-attention in the Transformer encoder helps parse semantic (and sequential) relationships from the flattened, spatial representations output by the Image Adaptor.

### Decoder

The Transformer decoder is auto-regressive, meaning previously generated tokens are taken into account via self-attention. To speed up training, “previously generated tokens” are mocked by the actual target sequences (ala Teacher Forcing). The decoder can then use more efficient batch matrix multiplication to generate all output tokens at one time rather than the time-consuming sequential decoding necessary for inference. A byte-mask is used within the attention computation to prevent the decoder from attending to positions in advance of what is being generated.

### 2-dimensional Spatial Encoding for Sequences

Because the Transformer does not employ any convolution or recurrence, the model contains no built-in spatial or sequential awareness. Positional encoding is required. In the original paper, a fixed sinusoidal encoding was used. The authors surmised that it would help the model learn relative positions as well as extrapolate to sequence lengths longer than those seen during training.

CNNs have implicit positional awareness so no additional encoding is necessary for the input images. The target sequences, however, do require positional encoding. Interestingly, multi-line paragraphs have a latent 2nd dimension when line-breaks (‘\n’) are included. Leveraging this line-break character, I used a learned 2-dimensional spatial encoding scheme based on the distance from the most recent line-break (approximate width) and the number of line-break characters since the start of sequence (approximate height).

(*Note: *Language models were also experimented with as an addition to the model. A pre-trained AWD-LSTM *[6]* language model yielded accuracy improvements but at too great an inference cost. A bi-directional Transformer encoder *[7]* language model was another potential choice. BERT incorporates context from both sides of a token and is non auto-regressive meaning inference time increases are minimal. Due to GPU limitations, I experimented with a distilled version *[8]*, but saw no accuracy improvements even after pre-training.)

***

## **DATA**

Much of the training for this task was based on the popular [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). This dataset is composed of 1,539 pages of scanned, handwritten text by 657 different writers. The IAM is based on a corpus (Lancaster-Oslo/Bergen) of British English texts first compiled in the 1970s. The content of these texts span genres including press reporting, scientific writing, essays, and popular lore.

Understanding that deep learning algorithm performance is bound by the quality of training data; this dataset is not ideal for a modern, American English application. However, as the largest publicly available compilation of annotated handwriting images, it is the most popular dataset for research applications in this space and offers a good baseline comparison with previous architectures.

In addition to 1,539 pages of text, the dataset is further segmented into ~13k lines and ~115k isolated words. These additional segmentations were critical to expand the training dataset.

Before expansion, a thorough examination of the data and manual correction of annotation/segmentation errors was necessary. A test set of 15 pages or approximately 1% of the total data was removed. The small test size is not ideal but is necessitated by the small overall size of the dataset.

### Word Combinations

A second dataset was created by combining randomly chosen images from the word segmentation list. 50k new images were created in a random configuration from a single word up to 4 lines of 4 words each.

### Line Concatenations

Another dataset was created by concatenating randomly chosen line images (normalized by height) from 3 to 13 lines in length. 20k of these images were created.

### Synthetic fonts

Another strategy was to use google fonts to create images of handwriting-like text. Text from Wikipedia, IMDB movie reviews, and open source books were rendered using 95 different handwriting fonts in variable sizes to create ~129k images of varying lengths. Background noise, blurring, random padding/cropping, pixel skewing, etc. was added to reflect the organic irregularities of the primary dataset as well as handwritten text in general.

### Downloaded handwriting samples

In order to improve the generalization performance of the algorithm beyond the IAM dataset, another dataset was constructed of 161 manually annotated images, publicly available on the internet. These images were treated with 11 different combinations of image modulation resulting in a final dataset containing 1771 images.

### Tokenization

For tasks involving text sequences, Tokenization is critical. Character tokenization has a number of benefits. Character sizes are relatively standard. Vocabulary size is small with few out-of-vocabulary tokens. However, character inference is slow (auto-regressive sequential decoding of 1000+ characters takes time…) and as mentioned above, the heterogeneity of handwriting means characters are often overlapping, illegible, or even omitted.

Word tokenization is intuitive as words seem to have primacy over characters in human reading*. Inference time is much faster than with characters. However, vocabulary size must be very large in order to limit out-of-vocabulary tokens.

A fixed-length subword tokenizer, [SentencePiece](https://github.com/google/sentencepiece), was used as a compromise. Using a learned, unigram language model encoding *[9],* I created a vocabulary of 10k subword units. Critically, additional special tokens were added to represent spaces, line-break characters, letter and word capitalization indicators, and punctuation marks. (*Note*: 10k, 30k, and 50k vocabularies were tested with the 10k being the most performant as well as keeping the model footprint small. Win win!)

Subword tokenization offers good inference speed, modest vocabulary size, and few out-of-vocabulary tokens. However, text images are not obviously divisible into subword units. I found that training with both character and subword tokens together helped encode a more robust feature representation and improved accuracy for both.

*[[A fanscinaitg dgsiesrion itno pchsyo-liigntusics via a ppoular ietnernt mmee](https://www.mrc-cbu.cam.ac.uk/people/matt.davis/cmabridge/). Tl;dr: While words are recognized as chunks; human reading (especially in difficult conditions) involves both word and character processes.]*

***

## **TRAINING**

The datasets were combined into 3 sequential training groups. The model was first trained on the word combination dataset (50k images at 256² pixels) to compare architectural design decisions, test hyper-parameter settings, and pre-train for later groups. The model was then trained on the synthetic font generated data (129k images at 512² pixels) to read long, well-formatted text sequences. The final handwriting dataset (25k images at 512² pixels) consisting of the primary IAM page images, concatenated lines, and downloaded text images was used to fine-tune the model for handwritten text.

Data augmentation included: random variations in rotation, warp, contrast, and brightness.

### Transcribr Parameters

* model dimensions: 512

* Activation function: GELU [10]

* Number of layers: 4

* Attention Heads: 8

* Dropout: 0.1

* AdamW optimizer [11]: (fixes weight decay for adaptive gradient algorithms)

* Maximum Learning Rate: 1e–3 (varied according to “1 cycle” policy [12])

* Betas: (0.95, 0.99), epsilon: 1e–8

* Momentum: (0.95, 0.85)

* Weight Decay: 1e–2

* Label Smoothing: 0.1

***

## **RESULTS**

The following comparison with previously published architectures is not scientifically rigorous as the aim of this project was to build a practical tool rather than publish an academic paper. As such, train/test splits were not standard across all architectures and results for other architectures were based on the pre-segmented line-level dataset while the Transcribr architecture was measured against the full page dataset. However, using published results as a loose comparison, the Transcribr architecture proves very competitive.

![](https://cdn-images-1.medium.com/max/2252/1*bkzC8zLgJw6-nBFHxd6SiQ.png)

![Transcribr (wordpiece token) results on 4 images from the IAM handwriting test dataset](https://cdn-images-1.medium.com/max/2260/1*M0aVaY8DVECBcH6LrylM7g.png)
<span class='caption'>*Transcribr (wordpiece token) results on 4 images from the IAM handwriting test dataset*</span>

***

## **LIMITATIONS**

I labored under several constraints during this project. The primary handicap was budget which directly impacts computational resources. Training was done on an NVIDIA P6000 (24GB RAM, 3840 CUDA cores). Inference for the [Transcribr app](https://transcribr.onrender.com) is performed on a paltry 1CPU with 1GB RAM and transcribes images at a pitiful rate of ~3 tokens/sec:(

(*Note: *Testing inference locally on a 2015 MacBook Pro with 4 cores and 16GB RAM, yields a good transcription rate of ~30 tokens/sec.)

Amount and quality of data was another factor. Synthetic data and augmentation were crucial in achieving these results but remain poor substitutes for diverse, high-quality handwriting samples. When tested on a test set containing diverse, out-of-dataset images, Transcribr performed rather poorly.

![](https://cdn-images-1.medium.com/max/2256/1*sNxOnomW11HTDeJDNdgq4w.png)

![Transcribr (wordpiece token) results on 4 non-dataset test images](https://cdn-images-1.medium.com/max/2260/1*fMKK_DdDZ2RqLMFV3d8maA.png)
<span class='caption'>*Transcribr (wordpiece token) results on 4 non-dataset test images*</span>

Architectural choices prioritized inference time above other factors, including accuracy. To this end, the model was kept as lean as possible, weighing in at a trim, ~50.8M parameters. Greedy decoding was used instead of the more accurate but costly, beam search. The [Transcribr app](https://transcribr.onrender.com) uses the less accurate but faster wordpiece tokenization scheme.

***

## **CONCLUSION**

I fell well short of the initial goal of building a useful (fast, generalizable, and accurate) tool. Transcribr is neither fast nor generalizable:/ Techniques such as quantization and distillation could be used to further reduce the model size and converting the python model to C++ would speed up inference. Gathering more handwriting images from a greater variety of sources and contexts would improve generalization accuracy.

However, judging within the narrow confines of an academic dataset, my results are encouraging. Some of the novel techniques I used include:

* Using a Transformer architecture for handwriting recognition

* 2d learned positional encoding based on line-breaks (‘\n’)

* SentencePiece tokenization rather than words or characters

* Training both a token and character decoder at the same time

* Generating synthetic data using handwriting fonts

These techniques allowed Transcribr to out-perform previously published results and shows that even shallow-pocketed independent researchers can help push the (research, if not practical:) boundaries of deep learning.

***

## **LINKS**

* [Transcribr App](https://transcribr.onrender.com/)

* [Github Notebook](https://github.com/ahs8w/Handwriting/blob/master/1--Transcribr.ipynb)

* [Pytorch](https://pytorch.org/)

* [Fast.ai](https://fast.ai/)

* [DistilBert Library](https://github.com/huggingface/transformers)

* [SentencePiece Tokenizer](https://github.com/google/sentencepiece)

***

* [1] J. Chung & T. Delteil, “[A Computationally Efficient Pipeline Approach to Full Page Offline Handwritten Text Recognition](https://arxiv.org/pdf/1910.00663.pdf)”, 2019

* [2] C. Wigington, et al., “[Start, Follow, Read: End-to-End Full Page Handwriting Recognition](http://openaccess.thecvf.com/content_ECCV_2018/papers/Curtis_Wigington_Start_Follow_Read_ECCV_2018_paper.pdf)”, 2018

* [3] A. Vaswani, et al., “[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)”, 2017

* [4] T. Bluche, “[Joint Line Segmentation and Transcription for End-to-End Handwritten Paragraph Recognition](https://arxiv.org/pdf/1604.08352.pdf)”, 2016

* [5] T. Bluche, et al., “[Scan, Attend and Read: End-to-End Handwritten Paragraph Recognition with MDLSTM Attention](https://arxiv.org/pdf/1604.03286.pdf)”, 2016

* [6] S. Merity, et al., “[Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)”, 2017

* [7] J. Devlin, et al., “[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)”, 2019

* [8] V. Sanh, et al., “[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf)”, 2020

* [9] T. Kudo, “[Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959.pdf)”, 2018

* [10] D. Hendrycks & K. Gimpel, “[Gaussian Error Linear Units (GELUS)](https://arxiv.org/pdf/1606.08415.pdf)”, 2018

* [11] I. Loshchilov & F. Hutter, “[Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf)”, 2019

* [12] L. Smith, “[A Disciplined Approach To Neural Network Hyper-parameters: Part 1 — Learning Rate, Batch Size, Momentum, And Weight Decay](https://arxiv.org/pdf/1803.09820.pdf)”, 2018

***

* [13] [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/)

* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

* [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

* [Mmee Dicuissson](https://www.mrc-cbu.cam.ac.uk/people/matt.davis/cmabridge/)
