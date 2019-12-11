MSC-RNN - Multi-Scale, Cascaded RNN
==========

MSC-RNN is a new RNN architecture proposed in the paper,
[One Size Does Not Fit All: Multi-Scale, Cascaded RNNs for Radar Classification](https://arxiv.org/abs/1909.03082), 
which won the **Best Paper Runner-Up** Award at *BuildSys 2019*.

MSC-RNN is created using EMI-RNN and FastGRNN from the EdgeML repository. 
It comprises of an EMI-FastGRNN for clutter discrimination at a lower tier and a more complex FastGRNN 
classifier for source classifcation at the upper-tier and is trained using a novel joint-training routine.

MSC-RNN holistically improves the accuracy and per-class recalls over ML models suitable for radar inferencing. 
Notably, MSC-RNN outperforms cross-domain handcrafted feature engineering with time-domain deep feature learning, 
while also being up to ∼3× more efficient than the competitive SVM based solutions.

# Resources

**Paper** - [pdf](/docs/publications/MSCRNN.pdf) | [arXiv](https://arxiv.org/pdf/1909.03082.pdf) | [ACM DL](https://dl.acm.org/citation.cfm?id=3360860)

**Code** - https://github.com/dhruboroy29/MSCRNN

**Dataset** - https://doi.org/10.5281/zenodo.3451408
