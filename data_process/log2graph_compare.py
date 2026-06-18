
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from node_attribute_tools import ip2Label, get_is_centerNode
from collections import Counter

import os
import edge_attribute_tools as edgeAttribute_tools
import igraph as ig
import pandas as pd
import time
from itertools import permutations
from config import FREQUENT_RULES_FILE, ALERT_CORRECTED_DIR, GRAPH_DATA_DIR
import leidenalg as la

# Pre-load frequent pattern rules
category_rule = None
rule_file = str(FREQUENT_RULES_FILE)
try:
    with open(rule_file, 'r') as file:
        content = file.read()
        content = content.strip('[]')
        tuples_str = content.split('), (')
        category_rule = set()
        for tuple_str in tuples_str:
            tuple_str = tuple_str.strip('()\n')
            elements = tuple_str.split(', ')
            elements = elements[:-1]
            itemset = []
            for element in elements:
                element = element.replace('frozenset({', '').replace('})', '').replace("'", "")
                itemset.extend(element.split(', '))
            for perm in permutations(itemset, 2):
                category_rule.add(perm)
except FileNotFoundError:
    print(f"WARNING: Frequent rules file not found at {rule_file}")
    print("Run 'python data/frequent_mining/generate_rules.py' first.")
    category_rule = set()

class Log2graph():
    """Undirected community detection using (sip, dip, warn_num)
        
        Algorithm: Louvainhttps://python-louvain.readthedocs.io/en/latest/api.html

    """
    def __init__(self, csv_name="monday-alert-label", start_time="", end_time="", out_path=None, time_interval=60*10, partition_method='our'):
        self.csv_path = str(ALERT_CORRECTED_DIR / csv_name)
        self.start_time = start_time
        self.end_time = end_time
        self.time_interval = time_interval
        self.out_path = out_path
        self.partition_method = partition_method

    def format_data2csv(self):
        # Read all data into memory
        input_data_df = pd.read_csv(self.csv_path)
        # input_data_df['real_time'] = pd.to_datetime(input_data_df['real_time'], format='%Y-%m-%d %H:%M:%S')
        statr_time = self.start_time
        start_time_new = time.strftime('%Y-%m-%d %H:%M:%S',
                                time.localtime(int(time.mktime(time.strptime(statr_time, '%Y-%m-%d %H:%M:%S'))) + self.time_interval))
        count_graph = 0
        # Buffer per-window data, write at end
        out_graph_data = pd.DataFrame(columns=['id','group_num', 'edge_index', 'node_attr', 'edge_attr', 'group_attr','edge_label','group_label', 'nodes', 'start_time', 'end_time'])
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
             

        # Partition complete, writing data
        print('Writing data.........')
        out_graph_data.reset_index(drop=True, inplace=True)
        out_graph_data.to_csv(self.out_path, header=True, index=False)

    def getGraph_withLog(self, input_data_df:pd.DataFrame):
        """Get subgraph cluster data

            Get partitioned subgraphs with nodes, edges, indices, features

            
        Args:
            input_data_df: 给定的待划分数据，dataframe格式数据,    \n

        Return:
            # Get normalized node/edge data from logs
            group_nodes: Node data by community           \n
            group_edges: Edge data by community             \n
            group_graph: Graph data by community             \n

        Examples:
            nodes = {
                "id":node_index,    \n
                "ip":ip,            \n
                "group":group       \n
            }                       

            group_edges[group_num] = {                      \n
                    'edges': [edge],
                    'edge_index': [[sip_index, dip_index]], \n
                    'edge_attr': [edge_attr],               \n
                }
    
        """

        # Merge alert logs
        if len(input_data_df) == 0:
            return {}
        merge_log = {}
        louvain_edge = {}
        for log in input_data_df.itertuples(index=False):
            id,real_time,sip_str,dip_str,sport,dport,protocol,category_standard,category_standard_id,signature_priority,category_label = log
            #print("id,real_time,sip_str,dip_str,sport,dport,protocol,category_standard,category_standard_id,signature_priority,category_label:")
            #print(id,real_time,sip_str,dip_str,sport,dport,protocol,category_standard,category_standard_id,signature_priority,category_label)
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
        if self.partition_method == 'Louvain':
            partition = g.community_multilevel(weights=g.es["weight"])
        elif self.partition_method == 'Leiden':
            partition = la.find_partition(g, la.ModularityVertexPartition)
        elif self.partition_method == 'GN':
            dendrogram = g.community_edge_betweenness(directed=True)
            partition = dendrogram.as_clustering()
        elif self.partition_method == 'LPA':
            partition = g.community_label_propagation(weights=g.es["weight"])
        elif self.partition_method == 'Second_Louvain':
            partition = g.community_multilevel(weights=g.es["weight"])
        else:
            raise ValueError("Invalid partition method")
        
        if self.partition_method in ('Louvain','Leiden','GN','LPA'):
            partition_dict = {g.vs[i]["name"]: partition.membership[i] for i in range(len(g.vs))}
            partition_edges = [[] for _ in range(max(partition.membership) + 1)]
        elif self.partition_method == 'Second_Louvain':
            partition_dict = second_level_community_detection(g=g,first_partition=partition,algorithm='louvain')
            partition_edges = [[] for _ in range(max(partition_dict.values()) + 1)]
        else:
            raise ValueError("Invalid partition method")
        

        for edge_key in merge_log.keys():
            sip_str, dip_str, _, _, _, _ = edge_key
            sip_community = partition_dict[sip_str]
            dip_community = partition_dict[dip_str]
            if sip_community == dip_community:
                partition_edges[sip_community].append(edge_key)
 
        group_nodes = {}
        group_edges = {}
        group_graph = {}
        group_num = 0
        for one_group_key_list in partition_edges:
            group_nodes_list = []
            for one_edge_key in one_group_key_list:
                sip_str,dip_str,dport,protocol,signature,signature_priority = one_edge_key
                values = merge_log[one_edge_key]
                if group_num not in group_nodes.keys():
                    id = 0
                    group_nodes[group_num] = [
                        {
                            'id': id,
                            'ip': sip_str,
                        },
                        {
                            'id': id + 1,
                            'ip': dip_str,
                        }
                    ]
                    group_nodes_list = [sip_str,dip_str]
                    sip_index, dip_index = id, id+1
                else:
                    if sip_str not in group_nodes_list:
                        id = len(group_nodes[group_num])
                        group_nodes[group_num].append(
                            {
                                'id': id,
                                'ip': sip_str,
                            }
                        )
                        group_nodes_list.append(sip_str)
                        sip_index = id
                    else:
                        sip_index = group_nodes_list.index(sip_str)

                    if dip_str not in group_nodes_list:
                        id = len(group_nodes[group_num])
                        group_nodes[group_num].append(
                            {
                                'id': id,
                                'ip': dip_str,
                            }
                        )
                        group_nodes_list.append(dip_str)
                        dip_index = id
                    else:
                        dip_index = group_nodes_list.index(dip_str)

                group_edgeIndex = group_edges.get(group_num,0)
                if group_edgeIndex != 0:
                    group_edgeIndex = len(group_edgeIndex['edges'])

                warn_num = values['warn_num']
                rep = values['rep']
                edge = {
                    'id': group_edgeIndex,
                    'sip': sip_str,
                    'sip_id': sip_index,
                    'dip': dip_str,
                    'dip_id': dip_index,
                    'dport': dport,
                    'protocol': protocol,
                    'signature': signature,
                    'signature_priority':signature_priority,
                    'warn_num': warn_num,
                    'rep': rep,
                }
                edge_category_label = {}
                group_category_label = set()
                for one_alert in rep:
                    
                    if (group_edgeIndex,one_alert[2]) not in edge_category_label.keys():
                        edge_category_label[(group_edgeIndex,one_alert[2])] = 1
                    else:
                        edge_category_label[(group_edgeIndex,one_alert[2])] += 1

                    group_category_label.add(one_alert[2])
                edge_attr = [dport, protocol, signature, signature_priority, warn_num]
                
                if group_num not in group_edges.keys():
                    group_edges[group_num] = {
                        'edges': [edge],
                        'edge_index': [[sip_index, dip_index]],
                        'edge_attr': [edge_attr],
                    }
                    group_graph[group_num] = {
                        'edge_label': edge_category_label,
                        'group_label': group_category_label,
                    }
                else:
                    group_edges[group_num]['edges'].append(edge)
                    group_edges[group_num]['edge_index'].append([sip_index,dip_index])
                    group_edges[group_num]['edge_attr'].append(edge_attr)
                    group_graph[group_num]['edge_label'] = dict(Counter(group_graph[group_num]['edge_label']) + Counter(edge_category_label))
                    group_graph[group_num]['group_label'] = group_graph[group_num]['group_label'] | group_category_label
                    # print('group_graph_label->',group_graph[group_num]['edge_label'])
            group_num += 1
        group_format_data = {}
        for group_num,values in group_edges.items():
            
            edge_index = values['edge_index']
            node_attr = self.get_node_attr(group_nodes[group_num],edge_index)
            edge_attr = self.get_edge_attr(values['edge_attr'])
            edge_label = group_graph[group_num]['edge_label']
            group_label = group_graph[group_num]['group_label']
            group_attr = self.get_graph_attr(node_attr, edge_index)

            group_format_data[group_num] = {
                'edge_index': edge_index,
                'node_attr': node_attr,
                'edge_attr': edge_attr,
                'group_attr': group_attr,
                'edge_label': edge_label,
                'group_label': group_label,
                'nodes': group_nodes[group_num],
            }
        return group_format_data
   
    def get_node_attr(self, nodes:list, edge_index:list):
# [Translated from Chinese]


        Args:
            nodes: 节点列表 
            edge_index: 事件簇边列表
        Examples:
            nodes = [
                {
                    'id': 0,                \n
                    'ip': '192.168.12.13',    \n
                },                          \n
                ...
            ]
            edge_index = [[sip_id,dip_id],...]
        
        Returns:
            node_attr: 每个节点特征编码组成的集合, 前两维代表是否为中心ip, 后面的维度为资产属性组成的one-hot向量；总共2+8位
        Examples:
            node_attr = [[0,1,0,1,0,..],[1,0,0,0,1,...],...]

        Note:
            需要注意输入的nodes需要按照id的顺序升序排列
        """
        node_attr = []
        node_centerIp_dic = get_is_centerNode(edge_index)
        for node in nodes:
            id,ip = node['id'],node['ip']

            node_center = node_centerIp_dic[id]['center_ip']
            ip_label = ip2Label(ip)
            #print('iplabel=',ip_label)
            node_attr.append(node_center+ip_label)

        return node_attr
    
    def get_edge_attr(self, edge_attr:list)->list:
# [Translated from Chinese]


        按照[dport,protocol,signature,sig_priority,warn_num]对每条边依次升序进行编码（与edge_index对应）

        Args:
            edge_attr: 一个簇所有的边特征,[[dport, protocol, signature, signature_priority, warn_num],...]

        Returns:
            edge_attr_format: 簇所有按照独特编码的边特征数据。 627 + 20 + 178 + 1 + 1= 827维，分别为目的端口、协议、攻击类型、攻击类型优先等级、攻击数量
        """

        edge_attr_format = []
        for one_attr in edge_attr:
            one_edge_attr = []
            dport,protocol,signature,signature_priority,warn_num = one_attr
            dport_hot = edgeAttribute_tools.get_dport(dport)
            protocol_hot = edgeAttribute_tools.get_protocol(protocol)
            signature_hot = edgeAttribute_tools.get_signature(signature)

            
            one_edge_attr.extend(dport_hot)
            one_edge_attr.extend(protocol_hot)
            one_edge_attr.extend(signature_hot)
            one_edge_attr.append(signature_priority)
            one_edge_attr.append(warn_num)

            edge_attr_format.append(one_edge_attr)

        return edge_attr_format
    
    def get_graph_attr(self, node_attr:list, edge_index:list, topology_threshold1=0.5, topology_threshold2=0.8):
# [Translated from Chinese]


        包括中心节点个数，13种三节点子图个数，拓扑类型、时间特征编码，共1+13+7+31维特征
        
        Args:
            node_attr: 事件簇节点特征编码列表, 第一维代表是否维中心ip, 第二维代表是否为内外网（资产属性）,
            edge_index: 事件簇边索引列表集合,

        Returns:
            group_attr: 事件簇的图级别属性特征，包括中心节点个数，13种三节点子图个数 共1+13 = 14维特征
        """
        graph_attr = []
        center_num = 0
        for one_node in node_attr:
            # print(one_node)
            center_num += one_node[0]
        graph_attr.append(center_num)

        graph_now = ig.Graph(edges=edge_index, directed=True)
        tmp_three_nodes_motifs = graph_now.motifs_randesu(size=3)
        three_nodes_motifs = [tmp_three_nodes_motifs[2]] + tmp_three_nodes_motifs[4:]
        sum_num_motifs = sum(three_nodes_motifs)
        if sum_num_motifs != 0:
            three_nodes_motifs_percent = [quantity / sum_num_motifs for quantity in three_nodes_motifs]
        else:
            three_nodes_motifs_percent = three_nodes_motifs
        graph_attr.extend(three_nodes_motifs_percent)

        return graph_attr


def split(partition, merge_log):
    """
    partition为best_community聚类结果，merge_log为所有数据，用于拆分时索引属性
    """
    data = []
    community_dict = {}
    for i, edge_key in enumerate(merge_log.keys()):
        data.append(edge_key)
        sip_str, dip_str, dport, protocol, category, signature_priority = edge_key
        group_id = partition[sip_str]
        community_dict.setdefault(group_id, []).append(i)
    community = list(community_dict.values())
    community_split = []

    for community_i in community:
        community_split_one = []
        community_i_category = []
        nodes2id = {}
        edges2id = {}
        edges_list = []
        attribute_list = []
        i = 0
        for data_i in community_i:
            src_ip = data[data_i][0]
            dst_ip = data[data_i][1]
            if src_ip not in nodes2id:
                nodes2id[src_ip] = i
                i += 1
            if dst_ip not in nodes2id:
                nodes2id[dst_ip] = i
                i += 1
            edges2id[data[data_i]] = data_i
            edges_list.append([nodes2id[src_ip], nodes2id[dst_ip]])
            attribute_list.append(data[data_i])

        g = ig.Graph(directed=True, n=len(nodes2id), edges=edges_list)
        g.es["attribute"] = attribute_list
        centre_list = []

        in_degrees = g.degree(mode='in')
        out_degrees = g.degree(mode='out')
        while max(in_degrees) > 1 or max(out_degrees) > 1:
            max_in = max(in_degrees)
            max_out = max(out_degrees)
            community_split_i = []

            if max_in > max_out:
                centre = in_degrees.index(max_in)
                neighbors_category = {}
                edges_neighbors = g.es.select(_to=centre)
                for edge in edges_neighbors:
                    category = edge['attribute'][4]
                    neighbors_category[category] = neighbors_category.get(category, 0) + 1
                centre_list.append([edges_neighbors[0]['attribute'][1]])
                max_neighbors_category = max(neighbors_category, key=neighbors_category.get)
                community_i_category.append(max_neighbors_category)
                edges_del = []
                for edge in edges_neighbors:
                    if edge['attribute'][4] == max_neighbors_category:
                        community_split_i.append(edges2id[edge['attribute']])
                        edges_del.append(edge)
            else:
                centre = out_degrees.index(max_out)
                neighbors_category = {}
                edges_neighbors = g.es.select(_from=centre)
                for edge in edges_neighbors:
                    category = edge['attribute'][4]
                    neighbors_category[category] = neighbors_category.get(category, 0) + 1
                centre_list.append([edges_neighbors[0]['attribute'][1]])
                max_neighbors_category = max(neighbors_category, key=neighbors_category.get)
                community_i_category.append(max_neighbors_category)
                edges_del = []
                for edge in edges_neighbors:
                    if edge['attribute'][4] == max_neighbors_category:
                        community_split_i.append(edges2id[edge['attribute']])
                        edges_del.append(edge)

            community_split_one.append(community_split_i)
            g.delete_edges(edges_del)
            in_degrees = g.degree(mode='in')
            out_degrees = g.degree(mode='out')

        for er in range(g.ecount()):
            community_split_one.append([edges2id[g.es[er]['attribute']]])
            community_i_category.append(g.es[er]['attribute'][4])
            centre_list.append([g.es[er]['attribute'][0], g.es[er]['attribute'][1]])

        # Match
        rule_dict = {}
        for one_rule in category_rule:
            rule_dict.setdefault(one_rule[0], []).append(one_rule[1])

        community_split_rule = []
        centre_dict = {}
        community_flag = [False] * len(community_split_one)

        for i, category_i in enumerate(community_i_category):
            if category_i in rule_dict:
                for j, category_j in enumerate(community_i_category):
                    if j > i and category_j in rule_dict[category_i]:
                        a_ip_cache = {data[a][0] for a in community_split_one[i]} | {data[a][1] for a in community_split_one[i]}
                        is_connected = False
                        for b in community_split_one[j]:
                            if data[b][0] in a_ip_cache or data[b][1] in a_ip_cache:
                                is_connected = True
                                break
                        if is_connected and not community_flag[i] and not community_flag[j]:
                            community_flag[i] = True
                            community_flag[j] = True
                            combined_community = community_split_one[i] + community_split_one[j]
                            community_split_rule.append(combined_community)
                            combined_centre = tuple(centre_list[i] + centre_list[j])
                            centre_dict.setdefault(combined_centre, []).append(len(community_split_rule) - 1)

        for i in range(len(community_flag)):
            if not community_flag[i]:
                community_split_rule.append(community_split_one[i])
                centre_key = tuple(centre_list[i])
                centre_dict.setdefault(centre_key, []).append(len(community_split_rule) - 1)

        community_split_centre = []
        for value in centre_dict.values():
            if len(value) == 1:
                community_split_centre.append(community_split_rule[value[0]])
            else:
                combined_community = []
                for i in value:
                    combined_community += community_split_rule[i]
                community_split_centre.append(combined_community)

        community_split.extend(community_split_centre)

    community_edge = []
    for community_i in community_split:
        community_i_edge = [data[alert_i] for alert_i in community_i]
        community_edge.append(community_i_edge)
    return community_edge

def second_level_community_detection(g, first_partition, algorithm='louvain'):
    """
    进行第二次社区划分





    """
    final_partition = []
    community_counter = 0
    for community_index in range(max(first_partition.membership) + 1):
        community_nodes = [i for i, mem in enumerate(first_partition.membership) if mem == community_index]
        subgraph = g.subgraph(community_nodes)
        if len(subgraph.es) > 0:
            if algorithm == 'louvain':
                second_partition = subgraph.community_multilevel(weights=subgraph.es["weight"])
            elif algorithm == 'label_propagation':
                second_partition = subgraph.community_label_propagation()
            elif algorithm == 'gn':
                dendrogram = subgraph.community_edge_betweenness(directed=False)
                second_partition = dendrogram.as_clustering()
            else:
                raise ValueError("Unsupported algorithm. Choose 'louvain', 'label_propagation', or 'gn'.")

            for sub_node_index, sub_community in enumerate(second_partition.membership):
                original_node_index = community_nodes[sub_node_index]
                new_community_index = community_counter + sub_community
                final_partition.append((original_node_index, new_community_index))
            community_counter += max(second_partition.membership) + 1
        else:
            for node_index in community_nodes:
                new_community_index = community_counter
                final_partition.append((node_index, new_community_index))
            community_counter += 1

    final_partition_dict = {g.vs[node]["name"]: community for node, community in final_partition}
    return final_partition_dict


if __name__=="__main__":
    csv_name_list = ['train.csv','test.csv']
    partition_method_list = ['LPA','Second_Louvain']
    #partition_method = 'LPA'
    for partition_method in partition_method_list:
        print("partition_method is:",partition_method)
    
        out_path_pre = str(ALERT_CORRECTED_DIR / partition_method)
        if not os.path.exists(out_path_pre):
            os.makedirs(out_path_pre)
        for csv_name in csv_name_list:
            start_time = time.time()
            out_path = f'{out_path_pre}/graph_{csv_name}'
            # out_path = str(ALERT_CORRECTED_DIR / f'graph_{csv_name}')
            log2graph = Log2graph(csv_name=csv_name, start_time="2017-07-03 08:00:00", end_time="2017-07-07 20:00:00",out_path=out_path,partition_method=partition_method)
            data = log2graph.format_data2csv() 
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            print(f"{partition_method}Graph partition + I/O time on {csv_name}: {elapsed_time:.2f} min: {elapsed_time} 分钟")
            with open('partition_time.log', 'a') as file:
                file.write(f"{partition_method}社区划分（含读入读出数据），方法在CIC-IDS2017，{csv_name}上的运行时间: {elapsed_time} 分钟\n")
            
