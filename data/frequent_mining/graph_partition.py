import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import igraph as ig
import pandas as pd
import time

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import ast
from config import FREQUENT_RULES_FILE, ALERT_CORRECTED_DIR, GRAPH_DATA_DIR, FREQUENT_MINING_DIR

class Log2graph():
    """Undirected community detection using (sip, dip, warn_num)
        
        Algorithm: Louvainhttps://python-louvain.readthedocs.io/en/latest/api.html

    """
    def __init__(self, csv_name="train.csv", start_time="", end_time="", out_path=None, time_interval=60*10):
        self.csv_path = str(ALERT_CORRECTED_DIR / csv_name)
        self.start_time = start_time
        self.end_time = end_time
        # Time window: 60*60s (1 hour)
        # self.time_interval = 60 * 60
        # Time window: 60*10s (10 minutes)
        self.time_interval = time_interval
        self.out_path = out_path

    def format_data2csv(self):
        # Read all data into memory
        input_data_df = pd.read_csv(self.csv_path)
        statr_time = self.start_time
        start_time_new = time.strftime('%Y-%m-%d %H:%M:%S',
                                time.localtime(int(time.mktime(time.strptime(statr_time, '%Y-%m-%d %H:%M:%S'))) + self.time_interval))
        count_graph = 0
        # Buffer per-window data, write at end
        out_graph_data = pd.DataFrame(columns=['id','group_num', 'edge_index', 'node_attr', 'edge_attr', 'group_attr','edge_label','group_label', '', 'start_time', 'end_time'])
        while int(time.mktime(time.strptime(start_time_new,'%Y-%m-%d %H:%M:%S')))<=int(time.mktime(time.strptime(self.end_time,'%Y-%m-%d %H:%M:%S'))):
            print('start:', statr_time,'-', 'end:', start_time_new)
            # Filter by time window
            data_df = input_data_df[(input_data_df['real_time'] >= statr_time) & (input_data_df['real_time'] < start_time_new)]
            # Partition into normalized subgraph data
            data_dic = self.getGraph_withLog(input_data_df=data_df)
            if len(data_dic) == 0:
                statr_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(int(time.mktime(time.strptime(statr_time, '%Y-%m-%d %H:%M:%S'))) + self.time_interval))
                start_time_new=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(time.mktime(time.strptime(start_time_new,'%Y-%m-%d %H:%M:%S'))) + self.time_interval))
                continue
            rows = []
            for group_num, values in data_dic.items():
                count_graph += 1
                rows.append({'id': count_graph, 'group_num': group_num, 'edge_index': values['edge_index'], 'node_attr': values['node_attr'], 'edge_attr':values['edge_attr'],'group_attr': values['group_attr'], 'edge_label': values['edge_label'],'group_label': values['group_label'], 'nodes': values['nodes'],'start_time':statr_time, 'end_time':start_time_new})
            
            out_graph_data = pd.concat([out_graph_data, pd.DataFrame(rows)], ignore_index=True)

            statr_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(int(time.mktime(time.strptime(statr_time, '%Y-%m-%d %H:%M:%S'))) + self.time_interval))
            start_time_new=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(time.mktime(time.strptime(start_time_new,'%Y-%m-%d %H:%M:%S'))) + self.time_interval))
             
    def getGraph_withLog(self, input_data_df: pd.DataFrame, out_data_path=None):
        """Get subgraph cluster data

            Get partitioned subgraphs with nodes, edges, indices, features

            
        Args:
            input_data_df: 给定的待划分数据，dataframe格式数据,    \n

        Return:
            写出划分后的结果           \n
        """
        # Merge alert logs
        if len(input_data_df) == 0:
            return {}
        merge_log = {}
        louvain_edge = {}
        for log in input_data_df.itertuples(index=False):
            id,real_time,sip_str,dip_str,sport,dport,protocol,category_standard,category_standard_id,signature_priority,category_label = log
            # Remove missing IPs
            if sip_str == None or dip_str == None:
                continue

            # real datasets
            edge_key = (sip_str,dip_str,dport,protocol,category_standard_id,signature_priority)

            if edge_key not in merge_log.keys():
                merge_log[edge_key] = {
                    "warn_num": 1,
                    "rep": [(id,real_time,category_label,sport)]
                }
            else:
                merge_log[edge_key]["warn_num"] += 1
                merge_log[edge_key]["rep"].append((id,real_time,category_label,sport))

            # louvain edge
            if ((sip_str,dip_str) not in louvain_edge.keys()) and ((dip_str,sip_str) not in louvain_edge.keys()):
                louvain_edge[(sip_str,dip_str)] = 1
            else:
                if (sip_str,dip_str) in louvain_edge.keys():
                    louvain_edge[(sip_str,dip_str)] += 1
                else:
                    louvain_edge[(dip_str,sip_str)] += 1
        
       # Create igraph graph
        g = ig.Graph()
        # Get unique nodes
        all_nodes = set()
        for edge in louvain_edge.keys():
            all_nodes.add(edge[0])
            all_nodes.add(edge[1])
        # Add nodes
        node_index_map = {}
        for i, node in enumerate(all_nodes):
            g.add_vertex(name=node)
            node_index_map[node] = i
    
        # Add weighted edges
        for (sip, dip), weight in louvain_edge.items():
            source_index = node_index_map[sip]
            target_index = node_index_map[dip]
            g.add_edge(source_index, target_index, weight=weight)
        # Louvain community detection
        partition = g.community_multilevel(weights=g.es["weight"])

        # Convert igraph partition to dict
        partition_dict = {g.vs[i]["name"]: partition.membership[i] for i in range(len(g.vs))}
        partition_edges = [[] for _ in range(max(partition.membership) + 1)]

        for edge_key in merge_log.keys():
            sip_str, dip_str, _, _, _, _ = edge_key
            sip_community = partition_dict[sip_str]
            dip_community = partition_dict[dip_str]
            warn_num = merge_log[edge_key]['warn_num']
            if sip_community == dip_community:
                for _ in range(warn_num):
                    partition_edges[sip_community].append(edge_key[-2])
        file_path = out_data_path if out_data_path else str(FREQUENT_MINING_DIR / 'frequent_mining_edges.txt')
        try:
            with open(file_path, 'a') as file:
                for sublist in partition_edges:
                    line = str(sublist)
                    file.write(line + '\n')
            print(f"Data written successfully {file_path}")
        except Exception as e:
            print(f"Error writing file: {e}")


        group_format_data = {}

        return group_format_data
   
if __name__=="__main__":

    start_time = time.time()
    csv_name = 'train.csv'
    out_path = str(FREQUENT_MINING_DIR / 'frequent_mining_edges.txt')
    log2graph = Log2graph(csv_name=csv_name, start_time="2017-07-03 08:00:00", end_time="2017-07-07 17:00:00",out_path=out_path)
    data = log2graph.format_data2csv() 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Community partition time on CIC-IDS2017，{csv_name}上的runtime: {elapsed_time} sec")

