# title

Routing and Distillation Enhanced Multimodal Fusion for Robust Drug-Target Binding Affinity Prediction

## overview

![overview](fig1.png)

## abstract

Predicting drug–target binding affinity is a key challenge in computational drug discovery. It requires robust integration of multimodal molecular information. Existing multimodal methods often suffer from modality imbalance, where stronger modalities suppress weaker ones. To address this issue, we propose a dynamic routing and distillation enhanced multimodal fusion framework for robust drug-target binding affinity prediction, termed MRD-DTA. Dynamic routing adaptively assigns sample-specific weights to each modality, enabling flexible fusion of different modal representations. Uni-directional distillation further transfers knowledge from the fused embedding to individual modalities, reinforcing weaker modalities and enhancing cross-view consistency. On benchmark datasets, MRD-DTA demonstrates outstanding performance, reducing the mean square error by 8.8\% on the Davis. Ablation studies indicate the complementary contributions of routing and distillation mechanisms. By bridging modality imbalance in multimodal molecular representation, MRD-DTA provides an interpretable and generalizable framework for affinity prediction, demonstrating practical potential to accelerate drug discovery.

### dataset

You can get the data from this link: https://drive.google.com/file/d/1u-uDPgK4Bm8dW4KvC7jUnbEhEeFTdlpt/view?usp=sharing

### quickly start


```bash
python main.py
```

## file

```
.
├── config.py
├── egnn.py
├── main.py
├── model.py
├── utils.py
├── dataset.py
├── build_vocab.py
└── __pycache__/
```

## cite
```
coming soon
```

