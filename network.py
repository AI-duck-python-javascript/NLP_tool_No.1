import MeCab #メカブ用
import pandas as pd #計算したデータをpandasを使って表形式に格納して、整理するために必要
import matplotlib.pyplot as plt #描画用のライブラリ
import japanize_matplotlib  # 描画用のライブラリを日本語にした時に文字化けしないようにするために必要
import itertools  # 共起語の組合せ作成に必要
import networkx as nx  # 共起ネットワーク作成に必要
from collections import Counter #単語の数え上げに使用
from datetime import datetime #できたファイル名に自動で日付をつけるために必要
import time

# メイン関数の定義======================================================================================================================
def main():

    # UPPER_N以上の頻出度のものを描画するために設定(定数)
    UPPER_N=10
   
    # text.txtを読み込む関数を実行
    text_read()
   
    # 読み込んだテキストを分割する関数を実行
    text_split_by_sentence()
   
    # 分割したテキストを形態素解析する関数を実行
    morphological_analysis()
   
    # 形態素解析後、2語の組み合わせを作る関数を実行
    make_combinations(UPPER_N)

    # 数え上げと共起ネットワークを計算をする関数を実行
    count_and_calc(UPPER_N)

    # グラフを描画し、データ一覧をシェル上に表示する関数を実行
    make_graphs(UPPER_N)
 

# テキスト読み込み関数======================================================================================================================
def text_read():
    with open('text.txt', 'r') as f:
        words_original = f.read()
    return words_original

# 読み込んだテキストを行単位「。」で分割する関数=================================================================================================
def text_split_by_sentence():
    words_original=text_read()
    words = words_original.split("。")
    return words

# 形態素解析、結果を行単位で格納する===========================================================================================================
def morphological_analysis():
    words = text_split_by_sentence()
    lines = []    
    t = MeCab.Tagger()

    for word in words:
        datas = []
        node = t.parseToNode(word)

        while node:

            # 表層形
            surface = node.surface

            # 品詞が活用情報など
            feature = node.feature.split(',')

            # 品詞、原形
            pos, origin = feature[0], feature[6]

            # 指定品詞のみ(名詞、形容詞) 
            if pos in ['名詞', '形容詞']:
                # 原形があるなら原形、ない場合は表層形にしたい場合は以下。でもバグの原因に
                # word = origin if origin != '*' else surface
                word = surface
                datas.append(word)

            node = node.next
        lines.append(datas)
    lines.pop(len(lines)-1)
    return lines

# 各行における２語の組み合わせを作る(変数cmb_lines)=============================================================================================
def make_combinations(UPPER_N):
    lines = morphological_analysis()
  
    cmb_lines = [list(itertools.combinations(line, 2)) for line in lines]
 
    words = []
    for cmb_line in cmb_lines:
        words.extend(cmb_line)

    # (の,の)など重複を削除
    for n in range(0,len(words))[::-1]:
        if words[n][0]==words[n][1]:
            words.pop(n)

    #　UPPER_N以上のみを格納する
    words_list=[]
    for i in words:
        if words.count(i) >=UPPER_N:
            words_list.append(i)
    return words_list
    
# 単語Aと単語Bの積集合カウント================================================================================================================
def count_and_calc(UPPER_N):
    words_list = make_combinations(UPPER_N)
    words_original= text_read()
    word_count = Counter(words_list)

    word_A = [k[0] for k in word_count.keys()]
    word_B = [k[1] for k in word_count.keys()]
    intersection_cnt = list(word_count.values())

    df = pd.DataFrame({'WORD_A': word_A, 'WORD_B': word_B, 'ITS_CNT': intersection_cnt})
    df.head()

    #単語A、単語Bのカウント 単語A、単語Bの和集合の数((単語Aの登場回数カウント + 単語Bの登場回数カウント) – (単語A、単語Bの積集合))

    # ワードAをカウントする
    word_A_cnt = df['WORD_A'].value_counts()
    for n in range(0,len(word_A_cnt)):
        word_A_cnt[n] =words_original.count(word_A_cnt.index.values[n])
    df1 = word_A_cnt.reset_index()
    df1.rename(columns={'index': 'WORD_A', 'WORD_A': 'WORD_A_CNT'}, inplace=True)

    # ワードBをカウントする
    word_B_cnt = df['WORD_B'].value_counts()
    for n in range(0,len(word_B_cnt)):
        word_B_cnt[n] =words_original.count(word_B_cnt.index.values[n])

    df2 = word_B_cnt.reset_index()
    df2.rename(columns={'index': 'WORD_B', 'WORD_B': 'WORD_B_CNT'}, inplace=True)

    # pandas左外部結合
    df = pd.merge(df, df1, how='left', on='WORD_A')
    df = pd.merge(df, df2, how='left', on='WORD_B')
    df.head()

    # 単語A、単語Bの和集合カウント
    df['UNION_CNT'] = (df['WORD_A_CNT'] + df['WORD_B_CNT']) - df['ITS_CNT']
    df.head()

    # Jaccard係数df['UNION_CNT']AとBの和集合：df['ITS_CNT']AとBの席集合
    df['JACCARD'] = df['ITS_CNT'] / df['UNION_CNT']

    df.head()
    return df

# グラフを描く関数===========================================================================================================================
def make_graphs(UPPER_N):
    df=count_and_calc(UPPER_N)
    # グラフオブジェクト作成 
    G = nx.Graph()

    # 単語A、単語Bの重複排除して直列化
    nodes = list(set(df['WORD_A'].tolist() + df['WORD_B'].tolist()))

    # node（円）
    for node in nodes:
        G.add_node(node)

    # edge（線）
    for i in range(len(df)):
        row = df.iloc[i]
        G.add_edge(row['WORD_A'], row['WORD_B'], weight=row['JACCARD'])

    # グラフレイアウト
    pos = nx.spring_layout(G, k=0.3)  # k = node間反発係数

    # 重要度決定アルゴリズム（重要なnodeを見つけ分析する）
    pr = nx.pagerank(G)

    # nodeの描画の設定
    nx.draw_networkx_nodes(
        G, pos, node_color=list(pr.values()),
        cmap='prism',
        alpha=0.7,
        node_size=[70000 * v for v in pr.values()],
    )

    # edge（線）
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.5,
        edge_color="lightgrey",
        width=[d["weight"] * 20 for (u, v, d) in G.edges(data=True)])

    # 日本語ラベル
    nx.draw_networkx_labels(G, pos, font_size=6, font_family='IPAexGothic', font_weight="bold")

    # 結果出力等
    plt.axis('off')
    plt.tight_layout()
    
    # 表も表示
    print(df)
    
    # ファイルの作成
    today = datetime.now().strftime('%Y%m%d')
    file_name = f'共起ネット_result_{today}.png'
    plt.savefig(file_name, dpi=600)
    plt.show()

# メイン関数の実行==============================================================================================================================
if __name__ == "__main__":

    # 時間計測開始
    time_sta = time.perf_counter()
    
    # メイン関数の実行
    main()
    
    # 時間計測終了
    time_end = time.perf_counter()
    
    # 経過時間（秒）
    time = time_end- time_sta   
    print(time)