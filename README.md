# Proposal-for-Optimizing-Number-of-Servers-in-Closed-BCMP-Queueing-Network

本プログラムは，平均値解析法に対して遺伝アルゴリズムを用いた窓口数の最適化計算を行い，結果からシミュレーションを用いてネットワーク内の動的な情報を得ることができる．

## Usage

### Optimization

BCMP_GA_Class_std_v4.pyでは，遺伝アルゴリズムを用いた窓口数の最適化を行う．プログラム内でBCMP_MVA_v2を呼び出し，平均値解析法により同一の推移確率に対する様々な窓口数での平均系内人数を算出する．目的関数による評価が最も良かったモデルについて平均系内人数，窓口数，推移確率，拠点情報，評価値を出力する．

BCMP_GA_Class_std_v4.pyの実行は，以下のように実行時にコマンドラインで引数を指定する．

```bash
mpiexec -n (並列数) python3 BCMP_GA_Class_std_v4.py (拠点数) (客クラス数) (網内客数) (遺伝子数) (世代数) (最大窓口数)
```

出力ファイル一覧
- ga_transition_std.png : 目的関数が収束する様子
- distance_std.csv : 距離行列
- ga_L_std.csv : クラス別平均系内人数の理論値
- ga_Node_std.csv : 各拠点の窓口数
- ga_Object_std.csv : 各世代のベストスコア
- ga_P_std.csv : 推移確率行列
- popularity_std.csv : 各拠点の人気度
- position_std.csv : 各拠点の位置

### Simulation

BCMP_Simulation_v3.pyでは，最適化計算によって得られた情報を利用してシミュレーションを行い，時系列で動的な情報を確認する．

結果を出力するためのフォルダとして，csv，plot，processを作成する．

BCMP_Simulation_v3.pyの実行は，以下のように実行時にコマンドラインで引数を指定する．最適化計算の結果として得られた推移確率行列，クラス別平均系内人数の理論値，窓口数を使用する．

```bash
mpiexec -n (並列数) python3 BCMP_Simulation_v3.py (拠点数) (客クラス数) (網内客数) (最大窓口数) (シミュレーション時間) (推移確率行列) (理論値) (窓口数) > result.txt
```

出力ファイル一覧
- csv/avg_L.csv : 平均系内人数の平均
- csv/avg_Lc.csv : クラス別平均系内人数の平均
- csv/avg_Q.csv : 平均待ち人数の平均
- csv/avg_Qc.csv : クラス別平均待ち人数の平均
- csv/avg_RMSE.csv : 各シミュレーション時間のRMSEの平均
- csv/L.csv : 平均系内人数
- csv/Lc.csv : クラス別平均系内人数
- csv/Q.csv : 平均待ち人数
- csv/Qc.csv : クラス別平均待ち人数
- csv/RMSE.csv : 各シミュレーション時間のRMSE
- plot/avg_RMSE.png : 各シミュレーション時間のRMSEの平均
- plot/box_L.png : 平均系内人数
- plot/box_L_Time5000.png : 5000時間までの平均系内人数
- plot/length.png : 系内人数の推移
- plot/RMSE.png : 各シミュレーション時間のRMSE
