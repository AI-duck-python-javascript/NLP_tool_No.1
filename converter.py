
import pandas as pd

# 個数指定# UPPER=len(lst)が最大 200行で20秒ぐらい。100ぐらいがちょうど良い
UPPER=200


# 配列に一行ずつ入れる
new_list=[]
lst = pd.read_csv("/Users/eigo/Desktop/develop2022/uec_project_lang/data_after.csv").values.tolist()

# 100ぐらいまでサクサク動く
for n in range(0,UPPER):
    content=str(lst[n]).replace("\r\n", "").replace("\n", "").replace("['","").replace("']","").replace("\\u3000","").replace("\\\\","").replace("f","").replace("&","").replace("<","").replace("x","").replace("ﾘ","").replace("ﾉﾟ","").replace("(","").replace(")","").replace("0","").replace("\\","").replace("0","").replace(":","")
    if content[-1]!="。":
        content=str(content)+ "。"
    new_list.append(content)

# print(content[-1])
# print(lst)
# print(count)
# print(new_list[0])
print(new_list[0])

# テキスト化

with open('text.txt', 'w') as f:
    f.writelines(new_list)