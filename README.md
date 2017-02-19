# Dual-Low-rank-Decompostion
# AAAI 16 work
# min ||Zi||_*+||Zp||_*+\lambda||E||_1+\alpha G(Li,Lp,Zi,Zp), s.t., P'X = P'XZi+P'XZp+E.
# for more detail, please check our AAAI-16 paper
# @inproceedings{ding2016robust,
#   title={Robust Multi-view Subspace Learning through Dual Low-rank Decompositions},
#   author={Ding, Zhengming and Fu, Yun},
#   booktitle={The Thirtieth AAAI Conference on Artificial Intelligence },
#   year={2016}
#}


main_2v.m is the main function

RMSL.m is our key function for dual low-rank decompostion

constructW.m is used to construct the weight of graph

cvKnn.m and cvEudist.m is the kNN classifier.

2view.mat is the evaluation data.
