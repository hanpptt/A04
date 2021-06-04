import re
import joblib
import numpy as np
import pandas as pd
from os import path, remove
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for
from flask import send_from_directory


def read_csv(filename):
    data = pd.read_csv(filename,
                       encoding='utf-8')  # 读取文件
    return data


def time_cal(data):
    data['seg_dep_time_loc'] = pd.to_datetime(data['seg_dep_time'])
    print(data['seg_dep_time'])
    max_date = pd.Timestamp(data['seg_dep_time_loc'].max()).timestamp()
    date_df = data['seg_dep_time_loc'].apply(
        lambda x: pd.Timestamp(x).timestamp())
    data['seg_dep_time_interval'] = data['seg_dep_time_loc'].apply(
        lambda x: x.hour)
    data.drop('seg_dep_time', axis=1, inplace=True)
    data['seg_dep_time_loc'] = max_date-date_df
    data['seg_dep_time_gap'] = data['seg_dep_time_loc']/(24*60*60)


def flt_cal(data, data2):
    flt_col = []
    for col in data.columns:
        res_y3 = re.search(r'^flt_bag_cnt', col)
        if res_y3:
            # print(res.group())
            flt_col.append(col)
    flt_df = pd.DataFrame(data[flt_col])
    for i in range(flt_df.shape[0]):
        for j in range(flt_df.shape[1]-1):
            if flt_df.iloc[i, j+1] < flt_df.iloc[i, j]:
                flt_df.iloc[i, j+1] = flt_df.iloc[i, j]
    res3 = flt_df[flt_col[0]]*0.6+(flt_df[flt_col[1]]-flt_df[flt_col[0]]) * \
        0.3+(flt_df[flt_col[2]]-flt_df[flt_col[1]])*0.1
    data2['flt_bag_cnt_y1'] = res3


def avg_cal(data, data2):
    flt_col = []
    for col in data.columns:
        res_y3 = re.search(r'^avg_dist_cnt', col)
        if res_y3:
            # print(res.group())
            flt_col.append(col)
    flt_df = pd.DataFrame(data[flt_col])
    for i in range(flt_df.shape[0]):
        for j in range(flt_df.shape[1]-1):
            if flt_df.iloc[i, j+1] < flt_df.iloc[i, j]:
                flt_df.iloc[i, j+1] = flt_df.iloc[i, j]
    res3 = flt_df[flt_col[0]]*0.6+(flt_df[flt_col[1]]-flt_df[flt_col[0]]) * \
        0.3+(flt_df[flt_col[2]]-flt_df[flt_col[1]])*0.1
    data2['avg_dist_cnt_y1'] = res3


def char2num(data2):
    object_col = ['seg_cabin']
    for col in object_col:
        dict_t = np.load('./dict/dict_{}.npy'.format(col),
                         allow_pickle=True).item()
        dict_now = data2[col].value_counts(1).to_dict()
        # print(dict_t,dict_now)
        for i in dict_now:
            print(i)
            if i not in dict_t:
                dict_t = 0
        data2[col].replace(dict_t, inplace=True)
    data2 = data2.iloc[:, :2]
    return data2


# 实例化产生一个Flask对象
dropzone = Dropzone()
app = Flask(__name__)
dropzone.init_app(app)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/ml')
def ml():
    data = read_csv(a_filename)
    print(a_filename)
    time_cal(data)
    data2 = data[['seg_dep_time_gap', 'pax_fcny',
                  'seg_dep_time_interval', 'seg_cabin']]
    flt_cal(data, data2)
    avg_cal(data, data2)
    data2 = char2num(data2)
    data2 = (data2-data2.mean())/data2.std()
    data2 = (data2-data2.min())/(data2.max()-data2.min())
    model = joblib.load('./a04.model')
    y_pred = model.predict(data2)  # 对新的样本Z做预测
    y_ab = model.predict_proba(data2)
    answer = pd.DataFrame(data['pax_passport'])
    answer['y_pred'] = y_pred
    df_ab = pd.DataFrame(y_ab)
    answer['y_ab'] = df_ab.iloc[:, 1]
    print(answer)
    remove(a_filename)
    answer.to_csv('test-web/answer.csv', index=False)
    answer = answer.sort_values(by='y_ab', ascending=False)
    answer = answer.iloc[:30, :]
    return render_template('ml.html', data=answer.values)


@app.route('/download')
def download():
    return send_from_directory(r"", filename="answer.csv")


a_filename = ''
@app.route("/upload", methods=['GET', 'POST'])
def upload():
    UPLOAD_FILE_TYPE = ['csv']
    if request.method == 'POST':
        f = request.files["file"]
        ext = path.splitext(f.filename)[1]
        if ext.split('.')[-1] not in UPLOAD_FILE_TYPE:
            return '   csv only!', 400
        base_path = path.abspath(path.dirname(__file__))
        upload_path = path.join(base_path, '')
        file_name = upload_path + secure_filename(f.filename)
        f.save(file_name)
        global a_filename
        a_filename = file_name
        return redirect(url_for('upload'))
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
