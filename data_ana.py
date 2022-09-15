# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random
import joblib

def get_padding(data_dict):
    min_x = min(data_dict.keys())
    max_x = data_x_max
    print(f"min_x:{min_x} max_x:{max_x}")
    count_1 = len([data for data in list(data_dict.values()) if data>=min_x and data<=padding_x_2])

    step = int((padding_x_2-min_x)/count_1)
    print(f"step {step}")

    for x in range(padding_x_1,min_x):
        data_dict[x] = padding_data
    for x in range(padding_x_2,max_x,step):
        data_dict[x] = padding_data

def get_x_y(file_name):
    data_dict = {}
    with open(file_name,'r') as f:
        for row in f:
            try:
                x = int(row.replace("\n","").replace(" ",""))
                if x<=data_x_max:
                    if x in data_dict:
                        data_dict[x] += 1
                    else:
                        data_dict[x] = 1
            except:
                pass
    print(f"data_dict:{data_dict}")
    count_all = sum(data_dict.values())
    print(f"count_all_1 {count_all}")
    get_padding(data_dict)
    count_all = sum(data_dict.values())
    print(f"count_all_2 {count_all}")
    data_x = list(data_dict.keys())
    data_y = [round(data_dict[x] / count_all, 5) for x in data_x]
    data_y_count = [data_dict[x] for x in data_x]
    print(f"data_dict sorted {sorted(data_dict.items(),key=lambda x:x[1])}")
    return data_x,data_y,data_y_count


def get_linear_pipeline(data_x,data_y):
    poly_reg = Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
    ])
    data_x_2 = np.array(data_x).reshape(-1,1)
    data_y_2 = np.array(data_y).reshape(-1,1)
    ploy_reg_fit = poly_reg.fit(data_x_2,data_y_2)
    joblib.dump(ploy_reg_fit,f"{degree}_{file_name}.joblib")


def get_sampling(poly_reg,data_y_max,sample_n):
    u_p = 1/(data_x_max-data_x_min+1)
    c = round(data_y_max/u_p,5)
    def accept():
        # 拒绝采样
        while True:
            x_1 = random.randint(data_x_min,data_x_max)
            poly_reg_y_1 = poly_reg.predict(np.array([x_1]).reshape(-1,1)).reshape(-1).tolist()[0]
            count_temp =0
            while True:
                y = random.uniform(0, c * u_p)
                if y<= poly_reg_y_1:
                   count_temp +=1
                else:
                    break
                if count_temp>=judge_n:
                    return x_1


    samples = []
    while True:
        x_1 = accept()
        samples.append(x_1)
        if len(samples)%1000==0:
            print(f"samples len:{len(samples)}")
        if len(samples)>= sample_n:
            break
    return samples






def get_samples_x_y(samples):
    dict_temp = {}
    for data in samples:
        if data in dict_temp:
            dict_temp[data] +=1
        else:
            dict_temp[data] = 1
    count_all = sum(dict_temp.values())
    sample_x = list(dict_temp.keys())
    sample_y = [round(dict_temp[x]/count_all,5) for x in sample_x]
    sample_y_count = [dict_temp[x] for x in sample_x]
    print(f"sample count_all {count_all}")
    a = sorted(dict_temp.items(),key = lambda x:x[1])
    print(f"samples_dict {a}")
    return sample_x,sample_y,sample_y_count



def write_samples(samples):
    with open(f"{data_x_min}_{data_x_max}_{degree}_{file_name}_{sample_n}_sample.txt","w") as f:
        for i in range(len(samples)):
            if i == len(samples)-1:
                f.write(str(samples[i]))
            else:
                f.write(str(samples[i])+"\n")

def get_list_element_count(list_temp):
    dict_temp = {}
    for data in list_temp:
        if data in dict_temp:
            dict_temp[data] += 1
        else:
            dict_temp[data] = 1
    data_x = list(dict_temp.keys())
    data_y = [dict_temp[data] for data in data_x]
    return data_x,data_y

def draw_samples(sample_split):
    draw_n = 1
    start_index = 0
    while True:
        end_index = start_index+sample_split
        print(f"draw_n:{draw_n},start_index:{start_index},end_index:{end_index}")
        samples_temp  = samples[start_index:end_index]
        data_x, data_y = get_list_element_count(samples_temp)
        fig = plt.figure()
        plt.scatter(data_x,data_y,s=dotsize,c='red',label=f"draw_samples_{draw_n}")
        plt.savefig(f"{data_x_min}_{data_x_max}_{degree}_{file_name}_{sample_n}_draw_samples_{draw_n}.png")
        draw_n += 1
        start_index +=  sample_split
        if end_index >= len(samples)-1 :
            break








if __name__ == '__main__':
    degree = 20
    data_x_min = 19
    data_x_max = 1000
    sample_n = 5000
    dotsize = 5
    judge_n = 2


    padding_data = 1
    padding_x_1 = -10
    padding_x_2 = 1000


    file_name = "905869418_ts"
    data_x,data_y,data_y_count = get_x_y(f"{file_name}.log")
    print(f"data_x:{data_x}")
    print(f"data_y:{data_y}")
    get_linear_pipeline(data_x,data_y)
    poly_reg = joblib.load(f"{degree}_{file_name}.joblib")
    # 拟合数据
    data_y_list = poly_reg.predict(np.array(range(data_x_min, data_x_max + 1)).reshape(-1, 1)).reshape(-1).tolist()
    data_y_max = max(data_y_list)
    print(f"data_y_list:{data_y_list}")
    print(f"data_y_max:{data_y_max}")
    samples = get_sampling(poly_reg,data_y_max,sample_n)
    print(f"samples len:{len(samples)}")

    sample_x,sample_y,sample_y_count = get_samples_x_y(samples)
    #
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(30,10))
    print("draw info")
    print(f"len data_x:{len(data_x)},len data_y:{len(data_y)}")
    ax1.scatter(data_x,data_y,s=dotsize,c='blue',label="data_y")
    data_x_list = list(range(data_x_min,data_x_max+1))
    print(f"len data_x_list:{len(data_x_list)},len data_y_list:{len(data_y_list)}")
    ax1.scatter(data_x_list,data_y_list,s=dotsize,c='green',label="data_y_list")
    print(f"len sample_x:{len(sample_x)},len sample_y:{len(sample_y)}")
    ax1.scatter(sample_x,sample_y,s=dotsize,c='red',label="sample_y")
    ax1.set_title(f"real-fit-sampling{sample_n}-percentAna")
    ax1.set_xlabel("data")
    ax1.set_ylabel("data percent")
    ax1.legend()
    ax2.scatter(data_x,data_y_count,s=dotsize,c='blue',label="data_y_count")
    ax2.scatter(sample_x,sample_y_count,s=dotsize,c='red',label="sample_ycount")
    ax2.set_title(f"real-sampling{sample_n}-countAna")
    ax2.set_xlabel("data")
    ax2.set_ylabel("data count")
    ax2.legend()
    plt.savefig(f"{data_x_min}_{data_x_max}_{degree}_{file_name}_{sample_n}.png")
    plt.show()
    write_samples(samples)

    # 针对采样数据，选取连续的个数，观察分布
    sample_split = 5000
    draw_samples(sample_split)

