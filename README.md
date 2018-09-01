# ジャンケン予測問題
いずれのファイルも4つの特徴量，教師データあり3クラス分類を行う．

「4Feature_BPTT」は Backpropagation Through Time を用いて学習モデルを構築する．
「4Feature_LSTM」は Long Short _ Term Memory を用いて学習モデルを構築する．
「4Feature_BiRNN」は Bidirectional RNN を用いて学習モデルを構築する．

上記3つのフォルダにはデータセットである「TeachingData」とpythonのプログラムファイルがある．
それぞれのフォルダにあるpythonデータをpython3系で実行することにより，学習モデルの構築と予測制度の算出ができる．
データセットのディレクトリを変更するとプログラムが動作しなくなってしまうことに注意する必要がある．
