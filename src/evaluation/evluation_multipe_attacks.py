import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from incdbscan import IncrementalDBSCAN
import time

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import ast
import wandb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gps_global_model import NNConv_model
from get_base_pattern import get_base_pattern,base_pattern_parse
import data_loader
from config import DEVICE, GRAPH_DATA_DIR, CHECKPOINT_DIR, KNOWLEDGE_DIR, EMBEDDING_DIR
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Global hyperparameters
coo_graph_train_path = str(GRAPH_DATA_DIR / "correct" / "graph_train.csv")
coo_graph_test_path = str(GRAPH_DATA_DIR / "correct" / "graph_test.csv")
# Min alert count per base pattern
min_base_pattern_graph_num = 1
node_out_feature = 30
graph_out_feature = 2

id_label_map = {
    0: "BENIGN",
    1: "Bot",
    2: "PortScan",
    3: "DDoS",
    4: "SSH-Patator",
    5: "DoS Hulk",
    6: "Web Attack-Brute Force",
    7: "Web Attack-XSS",
    8: "DoS slowloris",
    9: "Infiltration",
    10: "DoS Slowhttptest",
    11: "DoS GoldenEye",
    12: "FTP-Patator",
    13: "Web Attack-Sql Injection"
}
# Label processing function
def parse_and_strip_benign(label_str):
    label_set = ast.literal_eval(label_str)  # Safely parse string as set
    new_set = label_set - {'BENIGN'}

    return str(new_set) if new_set else str(label_set)

def get_base_pattern_data(coo_graph_df, embedding_data_df):
    """
    Get base pattern profiles, merge with embeddings and signatures。

    参数:
        coo_graph_df (DataFrame): 包含图的COO格式数据的DataFrame。
        embedding_data_df (DataFrame): 包含嵌入数据的DataFrame。
    返回:
        DataFrame: 合并后的DataFrame，包含图的基本模式画像、嵌入数据和编码签名数据。
    """
    encoded_signature_path = str(KNOWLEDGE_DIR / "encoded_signature.csv")
    encoded_signature_df = pd.read_csv(encoded_signature_path)
    # Extract base pattern and graph metadata
    coo_graph_df['base_profile'] = ''
    coo_graph_df['base_pattern_parse'] = ''
    coo_graph_df['base_topologic'] = ''

    for i in range(len(coo_graph_df)):
        now_graph = coo_graph_df.iloc[i]
        # Get base pattern profile
        base_pattern = get_base_pattern(now_graph)
        # Parse base pattern
        parse = base_pattern_parse(base_pattern, encoded_signature_df)
        # Center node count, topology type
        base_topologic = (base_pattern[0], base_pattern[3])
        # Add pattern, parse, topology to DataFrame
        coo_graph_df.loc[i, 'base_profile'] = str(base_pattern)
        coo_graph_df.loc[i, 'base_pattern_parse'] = str(parse)
        coo_graph_df.loc[i, 'base_topologic'] = str(base_topologic)

    # Merge graph and embedding data
    graph_all_data = pd.merge(coo_graph_df, embedding_data_df, on=['id'], how='left')
    # Count alerts per cluster
    edges_label = graph_all_data['edge_label'].to_list()
    warn_num_list = []
    for one_edge in edges_label:
        one_edge_label = eval(one_edge)
        warn_num = sum(one_edge_label.values())
        warn_num_list.append(warn_num)
    graph_all_data['warn_num'] = warn_num_list

    # Transform labels
    graph_all_data['group_label'] = graph_all_data['group_label'].apply(parse_and_strip_benign)
    condition = (graph_all_data['group_label'] == "{'BENIGN'}")

    # Binary classification only
    graph_all_data['group_label_easy'] = "{'BENIGN'}"
    graph_all_data.loc[~condition,'group_label_easy'] = "{'High Risk'}"
    # Record binary label
    graph_all_data['risk_level'] = 0
    graph_all_data.loc[~condition,'risk_level'] = 1
    label_map = {
    "{'BENIGN'}": 0,
    "{'Bot'}": 1,
    "{'PortScan'}": 2,
    "{'DDoS'}": 3,
    "{'SSH-Patator'}": 4,
    "{'DoS Hulk'}": 5,
    "{'Web Attack ?Brute Force'}": 6,
    "{'Web Attack ?XSS'}": 7,
    "{'DoS slowloris'}": 8,
    "{'Infiltration'}": 9,
    "{'DoS Slowhttptest'}": 10,
    "{'DoS GoldenEye'}": 11,
    "{'FTP-Patator'}": 12,
    "{'Web Attack ?Sql Injection'}": 13
    }
    # Record multi-class label
    graph_all_data['label_id'] = graph_all_data['group_label'].map(label_map)
    return graph_all_data


def one_base_cluster(one_base_graph_data,IncrementalDBSCAN,base_graphId:dict,base_embedding:dict,base_risk:dict):
    """Cluster alerts within a base pattern
        
    Args:
        one_base_graph_data: All alert clusters in one base pattern，格式如graph_data_all一般
        IncrementalDBSCAN: IncrementalDBSCAN for one base pattern
    """
    one_base_graph_data = one_base_graph_data.copy()
    base_pattern_parse = one_base_graph_data['base_pattern_parse'].unique()[0]
    embedding_reduce = one_base_graph_data['embedding_reduce']
    embedding_reduce_list = embedding_reduce.to_list()
    embedding_reduce_array = embedding_reduce_list
    
    IncrementalDBSCAN.insert(embedding_reduce_array)
    cluster_labels = IncrementalDBSCAN.get_cluster_labels(embedding_reduce_array)
    cluster_labels = list(cluster_labels)
    
    base_graphId[base_pattern_parse] = one_base_graph_data['id'].to_list()
    # base_risk[base_pattern_parse] = one_base_graph_data['risk_level'].to_list()
    base_risk[base_pattern_parse] = one_base_graph_data['label_id'].to_list()

    base_embedding[base_pattern_parse] = embedding_reduce_array

    #                                                                                #
    #                                                                                #
    ##################################################################################
    #pd.set_option('mode.chained_assignment', None)
    assert len(cluster_labels) == len(one_base_graph_data), "cluster_labels length does not match DataFrame rows"
    one_base_graph_data.loc[:, 'HDBSCAN_LABEL'] = cluster_labels
    cluster_distri = one_base_graph_data.groupby(['HDBSCAN_LABEL', 'group_label']) \
        .agg({'id': 'count', 'warn_num': 'sum'}) \
        .reset_index() \
        .rename(columns={'id': 'graph_count'})
    grouped_id = one_base_graph_data.groupby(['HDBSCAN_LABEL', 'group_label'])['id'].apply(list).reset_index()

    result_df = pd.merge(cluster_distri, grouped_id, on=['HDBSCAN_LABEL', 'group_label'])
    result_df['base_pattern_parse'] = base_pattern_parse
    result_df['HDBSCAN_LABEL'] = cluster_distri['HDBSCAN_LABEL'].astype(int)
    
    return result_df

from math import comb
def get_supsicous_prob(data:pd.DataFrame, k):
    """ Get suspicious-identification probability per cluster
    
    Args:
        data: 单个集群对应的数据表
        k: 采样个数
        
    Return:
        find_supsicous_prob: 识别该集群为supsicous的概率的概率
    """
    find_supsicous_prob = 0
    no_find_supsicous_prob = 0
    # If max risk count < k, always identified as suspicious
    #print(data)
    group_label_max_num = data['graph_count'].max()
    graph_sum_num = data['graph_count'].sum()
    if group_label_max_num < k:
        find_supsicous_prob = 1
        return find_supsicous_prob
    # Otherwise find groups with count >= k
    else:
        for index_count in range(len(data)):
            graph_count = data.loc[index_count,'graph_count']
            if graph_count >= k:
                no_find_supsicous_prob += comb(graph_count, k)/comb(graph_sum_num, k)
        find_supsicous_prob = 1 - no_find_supsicous_prob
        return find_supsicous_prob

def get_sample_num(unique_values_count:pd.DataFrame,clsuter_ok_distri,cluster_single_risk_percent):
    """ Get sample count per cluster

    Args:
        data: 单个集群对应的数据表
        k: 采样个数

    Return:
        sample_num: 采样个数
    """
    for n_samples in range(1,100):
        supicious_clusters = unique_values_count[unique_values_count['group_label_num']>1]
        supicious_clusters = supicious_clusters.reset_index(drop=True)
        supicious_cluster_name = supicious_clusters['cluster_name'].unique()

        supsicous_prob_list = []
        for cluster_name in supicious_cluster_name:
            cluster_data = clsuter_ok_distri.loc[clsuter_ok_distri['cluster_name']==cluster_name]
            cluster_data = cluster_data.reset_index(drop=True)
            supsicous_prob = get_supsicous_prob(cluster_data,k=n_samples)
            supsicous_prob_list.append(supsicous_prob)
    
        supsicous_prob_mean = sum(supsicous_prob_list)/len(supsicous_prob_list)
        overall_supsicous_prob = supsicous_prob_mean * (1-cluster_single_risk_percent) + 1*cluster_single_risk_percent
        if overall_supsicous_prob>=0.96:
            return n_samples
    return {'error':'sample count too large'}

def evaluation(train_embedding_path, test_embedding_path, eps=0.10, min_pts=3):
    """Evaluation function
    Args:
        train_embedding_path (str): 训练embedding数据的所在路径, 为csv格式
        test_embedding_path (str): Testembedding数据的所在路径, 为csv格式
        model_param (str): 模型参数, 默认为''
        eps (float): Incremental clustering的eps参数
        min_pts (int): Incremental clustering的min_pts参数
    Returns:
        None, 但写入res.txt: 记录此次Test结果信息
    """
    coo_graph_df = pd.read_csv(coo_graph_train_path)
    embedding_data_df = pd.read_csv(train_embedding_path)
    embedding_data_df['embedding_data'] = embedding_data_df['embedding_data'].apply(ast.literal_eval)
    graph_all_data = get_base_pattern_data(coo_graph_df,embedding_data_df)
    graph_all_data['embedding_reduce'] = graph_all_data['embedding_data']

    # Filter graphs meeting min base pattern count
    pattern_classification = graph_all_data.groupby(['base_pattern_parse'])['id'].apply(list).reset_index()
    pattern_classification = pattern_classification.rename(columns={'id': 'id_list'})
    pattern_classification['count'] = pattern_classification['id_list'].apply(len)
    limit_base_pattern = pattern_classification.loc[pattern_classification['count']>=min_base_pattern_graph_num]
    graph_all_data_select = pd.merge(graph_all_data, limit_base_pattern, on=['base_pattern_parse'])
    base_profile_support  = graph_all_data_select['base_pattern_parse'].unique()
    # wandb.log(
    #     {   "eps":eps,
    #         "handle_stage/base_pattern_num": len(base_profile_support)}
    # )
    cluster_distri_list = []

    # Incremental clustering
    base_dbscan = {}
    base_graphId = {}
    base_embedding = {}
    base_risk = {}
    for i,one_base in enumerate(base_profile_support):
        one_base_graph_data = graph_all_data_select[graph_all_data_select['base_pattern_parse'] == one_base]
        base_dbscan[one_base] = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
        cluster_distri= one_base_cluster(one_base_graph_data,base_dbscan[one_base],base_graphId,base_embedding,base_risk)
        cluster_distri_list.append(cluster_distri)

    cluster_distri_res = pd.concat(cluster_distri_list, ignore_index=True)
    
    # Manual-stage evaluation
    # Count noise points (label -1)
    clsuter_ok_distri = cluster_distri_res[cluster_distri_res['HDBSCAN_LABEL']!=-1]
    clsuter_ok_distri = clsuter_ok_distri.reset_index(drop=True)
    cluster_noise_distri = cluster_distri_res[cluster_distri_res['HDBSCAN_LABEL']==-1]
    cluster_noise_num, subgraph_noise_num, alert_noise_num = len(cluster_noise_distri), cluster_noise_distri['graph_count'].sum(), cluster_noise_distri['warn_num'].sum()
    wandb.log(
        {
            "eps":eps,
            "handle_stage/train_cluster_num": len(cluster_distri_res),
            "handle_stage/train_subgraph_num": cluster_distri_res['graph_count'].sum(),
            "handle_stage/train_alert_num": cluster_distri_res['warn_num'].sum(),
            "handle_stage/noise_train_cluster_num": cluster_noise_num,
            "handle_stage/noise_train_subgraph_num": subgraph_noise_num,
            "handle_stage/noise_train_alert_num": alert_noise_num,
        }
    )
    # Compute coverage
    no_select_subgraph = len(graph_all_data) -  len(graph_all_data_select)
    subgraph_coverage = 1 - (no_select_subgraph + subgraph_noise_num) / len(graph_all_data)
    # Compute single-risk proportion
    clsuter_ok_distri['cluster_name'] = list(zip(clsuter_ok_distri['base_pattern_parse'],clsuter_ok_distri['HDBSCAN_LABEL']))
    unique_values_count = clsuter_ok_distri.groupby(['cluster_name']) \
        .agg({'group_label': ['nunique', list], 'graph_count':'sum', 'warn_num':'sum'}) \
        .reset_index()
    unique_values_count.columns = ['cluster_name', 'group_label_num', 'group_label_list', 'graph_count', 'warn_num']
    one_single_risk_data = unique_values_count[unique_values_count['group_label_num']==1]
    cluster_single_risk_num = len(one_single_risk_data)
    cluster_single_risk_percent = cluster_single_risk_num / len(unique_values_count)
    subgraph_single_risk_num = one_single_risk_data['graph_count'].sum()
    subgraph_single_risk_percent = subgraph_single_risk_num / unique_values_count['graph_count'].sum()
    alert_single_risk_num = one_single_risk_data['warn_num'].sum()
    alert_single_risk_percent = alert_single_risk_num / unique_values_count['warn_num'].sum()
    print('---------------------------------Workload reduction RESULTS----------------------------------------------------')
    print("唯一事件label集群占比: {:.2%}, Corresponding subgraph proportion: {:.2%}, Alert proportion{:.2%}".format(cluster_single_risk_percent,subgraph_single_risk_percent,alert_single_risk_percent))
    wandb.log(
        {
            "eps": eps,
            "handle_stage/cluster_single_risk_percent": cluster_single_risk_percent,
            "handle_stage/subgraph_single_risk_percent": subgraph_single_risk_percent,
            "handle_stage/alert_single_risk_percent": alert_single_risk_percent}
    )
    # Compute sample count & reduction rates
    sample_num = get_sample_num(unique_values_count,clsuter_ok_distri,cluster_single_risk_percent)
    print(f"Sample count:: {sample_num}")
    wandb.log(
        {   "eps": eps,
            "handle_stage/sample_num": sample_num}
    )
    grouped = clsuter_ok_distri.groupby(['HDBSCAN_LABEL', 'base_pattern_parse'])
    num_groups = grouped.ngroups
    cluster_hanlde_sugraph_num = sample_num * num_groups
    subgraph_reduction = 1 - cluster_hanlde_sugraph_num / clsuter_ok_distri['warn_num'].sum()
    total_subgraph_reduction = 1 - (no_select_subgraph + subgraph_noise_num + cluster_hanlde_sugraph_num) / len(graph_all_data)
    total_alert_reduction = 1 - (no_select_subgraph + subgraph_noise_num + cluster_hanlde_sugraph_num) / graph_all_data['warn_num'].sum()
    print("Alert cluster coverage:: {:.2%}, Reduction in covered portion:: {:.2%}, Total reduction:{:.2%}，Total alert log reduction:{:.2%}".format(subgraph_coverage,subgraph_reduction,total_subgraph_reduction,total_alert_reduction))
    wandb.log(
        {   
            "eps": eps,
            "handle_stage/subgraph_coverage": subgraph_coverage,
            "handle_stage/subgraph_coverage_reduction": subgraph_reduction,
            "handle_stage/total_subgraph_reduction": total_subgraph_reduction,
            "handle_stage/total_alert_reduction": total_alert_reduction}
    )

    # Automatic-stage evaluation
    coo_test_df = pd.read_csv(coo_graph_test_path)
    embedding_test_df = pd.read_csv(test_embedding_path)
    embedding_test_df['embedding_data'] = embedding_test_df['embedding_data'].apply(ast.literal_eval)
    graph_test_data = get_base_pattern_data(coo_test_df,embedding_test_df)
    # Log basic info
    wandb.log(
        {   
            "eps":eps,
            "auto_stage/test_graph_num": len(graph_test_data),
            "auto_stage/test_alert_num": graph_test_data['warn_num'].sum()
        }
    )
    # Match
    graph_test_data['match_label'] = '未Match'
    graph_test_data['match_risk'] = -1
    for graph_index in range(len(graph_test_data)):
        new_cluster_flag = False
        embedding_data = graph_test_data.loc[graph_index,'embedding_data']
        embedding_data_array = [embedding_data]
        base_pattern_parse = graph_test_data.loc[graph_index,'base_pattern_parse']
        graph_id = graph_test_data.loc[graph_index,'id']
        # Get known clusters
        base_dbscan_now = base_dbscan.get(base_pattern_parse,0)
        # New cluster
        if base_dbscan_now == 0:
            base_dbscan_now = IncrementalDBSCAN(eps=eps, min_pts=min_pts)
            base_dbscan_now.insert(embedding_data_array)
            labels_now = base_dbscan_now.get_cluster_labels(embedding_data_array)
            labels_now = labels_now.item()
            labels_now = int(labels_now) 
            #labels_now = int(labels_now)
            
            base_dbscan[base_pattern_parse] = base_dbscan_now
            base_risk[base_pattern_parse] = [-1]
            base_embedding[base_pattern_parse]=embedding_data_array
            base_graphId[base_pattern_parse]=[graph_id]
            
            new_cluster_flag = True
        else:
            base_dbscan_now.insert(embedding_data_array)
            label_now = base_dbscan_now.get_cluster_labels(embedding_data_array)
            label_now = int(label_now[0])
            # Noise point
            if label_now == -1:
                base_risk[base_pattern_parse].append(-1)
                base_embedding[base_pattern_parse].append(embedding_data_array[0])
                
            else:
                match_risk = -1
                # Get historical labels
                labels_past = base_dbscan_now.get_cluster_labels(base_embedding[base_pattern_parse])
                now_base_labels = labels_past.tolist()
                now_base_risk = base_risk[base_pattern_parse]
     
                match_risk_list = []
                # for i,label in enumerate(now_base_labels):
                #     if int(label) == label_now:
                #         match_risk = max(match_risk,now_base_risk[i])
                for i,label in enumerate(now_base_labels):
                    if int(label) == label_now:
                        match_risk_list.append(now_base_risk[i])
                if len(set(match_risk_list)) >=2:
                    match_risk_list = [x for x in match_risk_list if x != 0]
                    match_risk = max(set(match_risk_list), key=match_risk_list.count)
                else:
                    match_risk = match_risk_list[0]
                graph_test_data.loc[graph_index,'match_risk'] = match_risk
                graph_test_data.loc[graph_index,'match_label'] = '成功Match'
                base_embedding[base_pattern_parse].append(embedding_data_array[0])
                base_risk[base_pattern_parse].append(match_risk)
            base_graphId[base_pattern_parse].append(graph_id)

    succeed_match = graph_test_data[graph_test_data['match_risk']!=-1]
    succeed_match = succeed_match[succeed_match['risk_level'] != -1]
    succeed_match_data = succeed_match.reset_index(drop=True)
    now_fine_tune_num = test_embedding_path.split('/')[-2]
    output_succed_log_path_pre = str(EMBEDDING_DIR / f"predict_question_log/{now_fine_tune_num}")
    if not os.path.exists(output_succed_log_path_pre):
        os.makedirs(output_succed_log_path_pre)
    eps_out_name = round(eps, 3)
    output_succed_log_path = f"{output_succed_log_path_pre}/succeed_match_{eps_out_name}.xlsx"
    succeed_match_data = succeed_match_data[['id','group_num','edge_index','node_attr','edge_label','group_label','nodes','start_time','end_time','base_profile','base_pattern_parse','embedding_data','warn_num','group_label_easy','risk_level','label_id','match_label','match_risk']]
    succeed_match_data.to_excel(output_succed_log_path,index=False)
    graph_test_data = graph_test_data[graph_test_data['risk_level']!=-1]
    graph_test_data = graph_test_data.reset_index(drop=True)
    # Compute result metrics
    succeed_match_prob = len(succeed_match_data) / len(graph_test_data)
    succeed_log_match_prob = succeed_match_data['warn_num'].sum() / graph_test_data['warn_num'].sum()
    print('---------------------------------Semi-automatic RESULTS----------------------------------------------------\n')
    print('eventsMatch成功率为：{:.2%}，日志Match成功率为：{:.2%}'.format(succeed_match_prob,succeed_log_match_prob))
    wandb.log(
        {
            "eps": eps,
            "auto_stage/succeed_graph_match_prob": succeed_match_prob,
            "auto_stage/succeed_log_match_prob": succeed_log_match_prob
            }
    )
    all_labels = sorted(set(succeed_match_data['label_id']) | set(succeed_match_data['match_risk']))
    label_idx_map = {label: idx for idx, label in enumerate(all_labels)}
    idx_label_map = {idx: label for label, idx in label_idx_map.items()}

    # Graph-level (unweighted)
    report_graph = classification_report(
        succeed_match_data['label_id'],
        succeed_match_data['match_risk'],
        output_dict=True,
        zero_division=0
    )
    cm_graph= confusion_matrix(succeed_match_data['label_id'], succeed_match_data['match_risk'], labels=all_labels)

    # Alert-level (weighted)
    report_alarm = classification_report(
        succeed_match_data['label_id'],
        succeed_match_data['match_risk'],
        sample_weight=succeed_match_data['warn_num'],
        output_dict=True,
        zero_division=0
    )

    cm_alarm = confusion_matrix(succeed_match_data['label_id'], succeed_match_data['match_risk'], sample_weight=succeed_match_data['warn_num'], labels=all_labels)

    wandb_metrics = {}
    # Log graph-level metrics
    for class_label, metrics in report_graph.items():
        if isinstance(metrics, dict):
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            support = metrics['support']
            print("  Precision: {:.2%}".format(precision))
            print("  Recall: {:.2%}".format(recall))
            print("  F1 Score: {:.2%}".format(f1))
            print("  Support: {:n}".format(support))

            class_label_name = ''
            if isinstance(class_label, str) and class_label.isdigit() and 0 <= int(class_label) <= 13:
                class_label_name = id_label_map[int(class_label)]

            if class_label_name!='':
                safe_label = str(class_label_name).replace(" ", "_")
            else:
                safe_label = str(class_label).replace(" ", "_")
            wandb_metrics[f"auto_stage_graph_{safe_label}/precision"] = precision
            wandb_metrics[f"auto_stage_graph_{safe_label}/recall"] = recall
            wandb_metrics[f"auto_stage_graph_{safe_label}/f1_score"] = f1
            wandb_metrics[f"auto_stage_graph_{safe_label}/support"] = support
            if isinstance(class_label, str) and class_label.isdigit() and int(class_label) in label_idx_map.keys():
                #print("FPR HELLO---------------------------------------------------")
                
                idx = label_idx_map[int(class_label)]
                TP = cm_graph[idx, idx]
                FP = cm_graph[:, idx].sum() - TP
                FN = cm_graph[idx, :].sum() - TP
                TN = cm_graph.sum() - (TP + FP + FN)
                fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
                wandb_metrics[f"auto_stage_graph_{safe_label}/fpr"] = fpr
                wandb_metrics[f"auto_stage_graph_{safe_label}/TP"] = TP
                wandb_metrics[f"auto_stage_graph_{safe_label}/FP"] = FP
                wandb_metrics[f"auto_stage_graph_{safe_label}/FN"] = FN
                wandb_metrics[f"auto_stage_graph_{safe_label}/TN"] = TN
        else:
            wandb_metrics["auto_stage/graph_accuracy"] = metrics

    # Log alert-level metrics (weighted)
    for class_label, metrics in report_alarm.items():
        print(f"[Alarm] Class {class_label}:")
        if isinstance(metrics, dict):
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            support = metrics['support']
            print("  Precision: {:.2%}".format(precision))
            print("  Recall: {:.2%}".format(recall))
            print("  F1 Score: {:.2%}".format(f1))
            print("  Support: {:n}".format(support))
            class_label_name = ''
            if isinstance(class_label, str) and class_label.isdigit() and 0 <= int(class_label) <= 13:
                class_label_name = id_label_map[int(class_label)]

            if class_label_name!='':
                safe_label = str(class_label_name).replace(" ", "_")
            else:
                safe_label = str(class_label).replace(" ", "_")
            print(f"[Alarm] Class_label {safe_label}")
            wandb_metrics[f"auto_stage_alarm_{safe_label}/precision"] = precision
            wandb_metrics[f"auto_stage_alarm_{safe_label}/recall"] = recall
            wandb_metrics[f"auto_stage_alarm_{safe_label}/f1_score"] = f1
            wandb_metrics[f"auto_stage_alarm_{safe_label}/support"] = support

            if isinstance(class_label, str) and class_label.isdigit() and int(class_label) in label_idx_map.keys():
                idx = label_idx_map[int(class_label)]
                TP = cm_alarm[idx, idx]
                FP = cm_alarm[:, idx].sum() - TP
                FN = cm_alarm[idx, :].sum() - TP
                TN = cm_alarm.sum() - (TP + FP + FN)
                fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
                wandb_metrics[f"auto_stage_alarm_{safe_label}/fpr"] = fpr
                wandb_metrics[f"auto_stage_alarm_{safe_label}/TP"] = TP
                wandb_metrics[f"auto_stage_alarm_{safe_label}/FP"] = FP
                wandb_metrics[f"auto_stage_alarm_{safe_label}/FN"] = FN
                wandb_metrics[f"auto_stage_alarm_{safe_label}/TN"] = TN
        else:
            wandb_metrics["auto_stage/alarm_accuracy"] = metrics

    # Attack class recall
    # Compute non-zero class recall
    attack_labels = [k for k in report_alarm.keys() if isinstance(k, str) and k.isdigit() and int(k) != 0]
    # Weighted attack class recall
    total_attack_support_alarm = sum(report_alarm[k]['support'] for k in attack_labels)
    weighted_attack_recall_alarm = sum(
        report_alarm[k]['recall'] * report_alarm[k]['support'] / total_attack_support_alarm
        for k in attack_labels
    )
    wandb_metrics["auto_stage/alarm_weighted_attack_recall"] = weighted_attack_recall_alarm
    print("Alert-level weighted attack recall: {:.2%}".format(weighted_attack_recall_alarm))

    total_attack_support_graph = sum(report_graph[k]['support'] for k in attack_labels)
    weighted_attack_recall_graph = sum(
        report_graph[k]['recall'] * report_graph[k]['support'] / total_attack_support_graph
        for k in attack_labels
    )
    print("Graph-level weighted attack recall: {:.2%}".format(weighted_attack_recall_graph))
    wandb_metrics["auto_stage/graph_weighted_attack_recall"] = weighted_attack_recall_graph

    # Compute under-estimation flag per row
    low_estimation = (succeed_match['risk_level'] > succeed_match['match_risk']).astype(int)
    # Compute under-estimation count per row
    weighted_low_estimation = low_estimation * succeed_match['warn_num']
    # Compute overall under-estimation rate
    total_warn_num = succeed_match['warn_num'].sum()
    total_weighted_low_estimation = weighted_low_estimation.sum()
    total_low_estimation_rate = total_weighted_low_estimation / total_warn_num
    print("Overall under-estimation rate::{:.4%}".format(total_low_estimation_rate))
    wandb_metrics["total_low_estimation_rate"] = total_low_estimation_rate
    wandb_metrics['eps'] = eps
    wandb.log(wandb_metrics)

def get_embedding(input_data:list, best_model_path:str, output_path:str):
    """ Generate embedding for each subgraph

        Generate embedding for each subgraph

    Args:
        input_data: List of subgraphs to embed
        best_model_path: Model checkpoint path
        output_path: Output embedding path
    """

    if len(input_data) == 0:
        return 0
    
    node_in_feature = input_data[0].num_node_features
    edge_in_feature = input_data[0].num_edge_features
    graph_in_feature = input_data[0].graph_attr.shape[0]
    model = NNConv_model(node_in_feature, edge_in_feature, graph_in_feature,node_out_feature=node_out_feature,graph_out_feature=graph_out_feature).to(DEVICE)

    if best_model_path !='':
        print('Loading trained model')
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        print('Random initialization')

    model.eval()

    data_loader = DataLoader(input_data, batch_size=1)

    rows = []
    with torch.no_grad():
        for i, batchData in enumerate(data_loader):
            # NNConv
            embedding_data = model(batchData)
            embedding_data = embedding_data.squeeze(0).tolist()
            rows.append(
                {
                    'id': i+1,
                    'embedding_data': embedding_data,
                }
            )
            if (i+1) % 20 == 0:
                print("process has gone {} row".format(i+1))
    res_df = pd.DataFrame(data = rows,columns=['id', 'embedding_data'])
    res_df.to_csv(output_path,index=False)

if __name__ == "__main__":
    # Model version
    # model_select_ver = 'only_tune'
    # model_select_ver = 'only_pre'
    #model_select_ver = 'random'
    #model_select_ver = 'best'
    model_select_ver = 'pre_and_tune'
    min_pts = 2

    #ratio_list = [0.01,0.05,0.10,0.20,0.50,0.70,1.00]
    #ratio_list = [0]
    ratio_list = [0.10,0.20,1.00]
    #ratio_list = [0.20,1.00]
    for ratio in ratio_list:
        fine_tune_num = int(ratio * 100)
        # Model selection
        if model_select_ver == 'only_tune':
            model_path = str(CHECKPOINT_DIR / f"fine-tune-model2/{fine_tune_num}/gps_model_only_tune.pt")
        elif model_select_ver == 'only_pre':
            model_path = str(CHECKPOINT_DIR / "pre-train-model/gps_global_model.pt")

        elif model_select_ver == 'pre_and_tune':
            model_path = str(CHECKPOINT_DIR / "fine-tune-model2/{fine_tune_num}/gps_model.pt")
        
        elif model_select_ver == 'random':
            model_path = ''

       
        elif model_select_ver == 'best':
            model_path = ''

        train_embedding_pre_path = str(EMBEDDING_DIR / f"{model_select_ver}/train/{fine_tune_num}")
        if not os.path.exists(train_embedding_pre_path):
            os.makedirs(train_embedding_pre_path)
        train_embedding_path = f"{train_embedding_pre_path}/train_embedding.csv"

        test_embedding_pre_path = str(EMBEDDING_DIR / f"{model_select_ver}/test/{fine_tune_num}")
        if not os.path.exists(test_embedding_pre_path):
            os.makedirs(test_embedding_pre_path)
        test_embedding_path = f"{test_embedding_pre_path}/test_embedding.csv"

        # Log experiment config
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", ""),
            project="CIC-IDS2017",
            name=f"question-debug-multiple-class-fpr-{fine_tune_num}%",
            tags=['question','multiple-class','debug'],
            config={
                "model_ver": model_select_ver,
                "GNN_model_path": model_path,
                "coo_graph_train_path": coo_graph_train_path,
                "coo_graph_test_path": coo_graph_test_path,
                "train_embedding_path": train_embedding_path,
                "test_embedding_path": test_embedding_path,
                "node_out_feature":node_out_feature,
                "graph_out_feature":graph_out_feature,
                "min_pts":min_pts,

            })
        
        # Get graph embeddings
        get_embedding_start_time = time.time()
        if not os.path.exists(train_embedding_path):
            print("get_train_embedding")
            data_train = data_loader.get_all_data(coo_graph_train_path,PE=True)
            get_embedding(data_train, model_path, train_embedding_path)
        if not os.path.exists(test_embedding_path):
            data_test = data_loader.get_all_data(coo_graph_test_path,PE=True)
            get_embedding(data_test, model_path, test_embedding_path)
        get_embedding_end_time = time.time()
        print(f"get_embedding time: {get_embedding_end_time - get_embedding_start_time}s")
        wandb.log({
            "get_embedding_time": get_embedding_end_time - get_embedding_start_time,
        })

        # Ultra-fine eps sweep
        # for eps in np.arange(0.01, 0.10, 0.01):
        #     evaluation(train_embedding_path, test_embedding_path, eps=eps, min_pts=min_pts)

        # Fine-grained eps sweep
        for eps in np.arange(0.5, 1.1, 0.05):
            evaluation(train_embedding_path, test_embedding_path, eps=eps, min_pts=min_pts)


        wandb.finish()
        # for eps in np.arange(1.0, 3.0, 0.5):
        #     evaluation(train_embedding_path, test_embedding_path, model_param, eps=eps, min_pts=3)

        # Custom eps (uncomment to use)
        # for eps in np.arange(0.7, 0.8, 0.01):
        #     evaluation(train_embedding_path, test_embedding_path, model_param, eps=eps, min_pts=3)