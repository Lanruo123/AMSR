import re
import os
import shutil
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder

def get_nodes_edges(filepath):
    readfile = open(filepath)
    all_nodes = []
    all_edges = []
    for line in readfile:
        edge = line.split()
        if int(edge[0]) not in all_nodes:
            all_nodes.append(int(edge[0]))
        if int(edge[1]) not in all_nodes:
            all_nodes.append(int(edge[1]))
        all_edges.append([int(edge[0]), int(edge[1])])
    readfile.close()
    return all_nodes, all_edges


def get_graph(nodes, edges):
    print('get_graph_nodes_number',len(nodes))
    print('get_graph_edges_number',len(edges))
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    # nx.draw(graph, with_labels=True)
    # plt.show()
    return graph


def load_data(filepath):
    all_nodes, all_edges = get_nodes_edges(filepath)
    graph = get_graph(all_nodes, all_edges)
    return graph


def get_edge_embs(graph, savepath):  # node2vec生成边向量表征(64维)
    # num_walks为产生多少个随机游走序列，walk_length为游走序列长度
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit()
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    edges_kv = edges_embs.as_keyed_vectors()
    edges_kv.save_word2vec_format(savepath)
    # print(edges_embs[('2', '7')])
    # print('edges embedding finish')


def get_node_index(str):  # 正则匹配出节点index
    result = re.findall(r"\d+", str)
    return int(result[0])


def write_infro(file, link_label, network_label, edge_infor):
    edge_infor[0] = link_label  # 边存在标签
    edge_infor[1] = network_label  # 网络来源标签
    # print(edge_infor)  # 写入的每行信息
    file.write(edge_infor[0])  # 边存在标签
    for infor in edge_infor[1:]:
        file.write(',' + infor)  # 网络来源标签+边向量表征(64维)
    file.write('\n')

def get_labels(graph, temppath, writepath, network_label):
    readfile = open(temppath)  # temppath保存node2vec生成的边向量表征(64维)
    read_infor = readfile.readline()  # 第一行是整体信息
    print("edge_embedding_vector, number * dimension: ", read_infor)

    writefile = open(writepath, 'a', encoding='utf-8')

    for line in readfile:
        edge_infor = line.split()
        # print(edge_infor)
        u = get_node_index(edge_infor[0])
        v = get_node_index(edge_infor[1])
        if u == v or graph.has_edge(u, v) or graph.has_edge(v, u):
            write_infro(writefile, '1', str(network_label), edge_infor)  # 写入正样本
        else:
            write_infro(writefile, '0', str(network_label), edge_infor)  # 写入负样本

    readfile.close()
    writefile.close()

def pro_data_main(dataset):
    path = 'data/' + dataset + '_data/'  # 原始数据文件夹
    temppath = 'node2vec/' + dataset + '_temp.txt'  # 中间临时文件,保存的是node2vec生成的边向量表征(64维)
    savepath = 'node2vec/' + dataset + '.txt' # 最终数据
    if os.path.exists(savepath):
        os.remove(savepath)
    network_label = 0
    path_list = os.listdir(path)
    for filename in path_list:
        readpath = os.path.join(path, filename)  # 原始数据
        print('---', readpath, '---')
        print('network_label',network_label)
        graph = load_data(readpath)  # 构图
        get_edge_embs(graph, temppath)  # node2vec得到边向量表征(64维)
        get_labels(graph, temppath, savepath, network_label)  # 边存在标签+网络来源标签+边向量表征(64维)
        network_label += 1
    # 删除中间生成的临时文件
    if os.path.exists(temppath):
        os.remove(temppath)
    print('network numbers:', network_label)
    return network_label

if __name__ == '__main__':
    datasets=['Aarhus','Enron','London','TF']
    for dataset in datasets:
        pro_data_main(dataset)

