### 應用不同注意力機制之混合深度學習模型於股價預測準確度之探討：以 Google 股價為例

第 37 屆國際資訊管理學術研討會 (ICIM 2026) 口頭發表論文之相關程式碼。
本人擔任通訊作者與發表人，**負責 Time Aware Attention 機制的實作**，四支基準模型為團隊共同程式。

#### 研究簡介

以 Alphabet Inc. (GOOG)2010–2016 年歷史交易資料 (Open、High、Low、Close、Volume)，
預測次一交易日收盤價。研究分兩階段：

1. **基準模型比較**：CNN、RNN、LSTM、GRU，選出 RNN 作為後續基底
2. **混合模型與注意力機制**：RNN 搭配 1–3 層 CNN，並結合 6 種注意力機制，共 28 組模型

#### 檔案說明

| 檔案 | 說明 |
|---|---|
| `basecnn.py`、`basernn.py`、`baselstm.py`、`basegru.py` | 第一階段基準模型 |
| `TimeAwareAttention_RNN.py` | RNN + Time Aware Attention |
| `TimeAwareAttention_RNN_CNN.py` | 1–3 Layer CNN+RNN + Time Aware Attention |

#### Time Aware Attention 實作重點

引入時間衰減函數，使越接近預測日的交易資料獲得越高權重，反映股市近期資訊影響較大的特性。

#### 實驗結果 (Time Aware Attention)

| 模型 | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| RNN | 9.8159 | 7.3957 | 0.0095 | 0.4686 |
| **CNN+RNN** | **7.9819** | **5.8911** | **0.0076** | **0.6516** |
| 2-Layer CNN+RNN | 8.5699 | 6.4495 | 0.0083 | 0.5962 |
| 3-Layer CNN+RNN | 9.3322 | 7.0346 | 0.0091 | 0.5228 |

R² 呈倒 U 型趨勢：單層卷積能與時間感知機制良好搭配，過深的卷積層則會稀釋原始時序訊號。

#### 技術與資料

- Python、TensorFlow (Keras)、scikit-learn (MinMaxScaler)
- 資料前處理：Min-Max 正規化、滑動視窗 (視窗長度 3 日)
- 資料來源：Kaggle 開源資料集 https://www.kaggle.com/code/benjibb/lstm-stock-prediction-20170507/input?select=prices-split-adjusted.csv

#### 備註

本研究為國立彰化師範大學資訊管理學系團隊合作成果，指導教授：翁政雄博士。
