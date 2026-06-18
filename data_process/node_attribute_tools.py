import igraph as ig

def ip2Label(ip:str)->list:
# [Translated from Chinese]


    获取ip的资产属性标签
    Args:
        ip: 规范的ip
    Return:
        label: ip属性标签的one-hot向量(8位)
    """
    ip = ip.strip(b'\x00'.decode())
    label_map = {
        "Firewall": {
            "205.174.165.80",
            "172.16.0.1"
        },

        "DNS+DC Server": {
            "192.168.10.3"
        },

        "Web Server": {
            "192.168.10.50",
            "205.174.165.68"
        },

        "Ubuntu Server": {
            "192.168.10.51",
            "205.174.165.66"
        },

        "Ubuntu Client": {
            "192.168.10.19",  # Ubuntu 14.4, 32B
            "192.168.10.17",  # Ubuntu 14.4, 64B
            "192.168.10.16",  # Ubuntu 16.4, 32B
            "192.168.10.12"   # Ubuntu 16.4, 64B
        },

        "Windows Client": {
            "192.168.10.9",   # Win 7 Pro, 64B
            "192.168.10.5",   # Win 8.1, 64B
            "192.168.10.8",   # Win Vista, 64B
            "192.168.10.14",  # Win 10, Pro 32B
            "192.168.10.15"   # Win 10, 64B
        },

        "MAC": {
            "192.168.10.25"
        }
    }

    onehot_map = {
        "Firewall":        [1, 0, 0, 0, 0, 0, 0, 0],
        "DNS+DC Server":   [0, 1, 0, 0, 0, 0, 0, 0],
        "Web Server":      [0, 0, 1, 0, 0, 0, 0, 0],
        "Ubuntu Server":   [0, 0, 0, 1, 0, 0, 0, 0],
        "Ubuntu Client":   [0, 0, 0, 0, 1, 0, 0, 0],
        "Windows Client":  [0, 0, 0, 0, 0, 1, 0, 0],
        "MAC":             [0, 0, 0, 0, 0, 0, 1, 0],
        "Outsider":        [0, 0, 0, 0, 0, 0, 0, 1]
    }


    for label, ip_set in label_map.items():
        if ip in ip_set:
            return onehot_map[label]
    return onehot_map["Outsider"]

def get_is_centerNode(edge_index:list, threhold = 0.50)->dict:
# [Translated from Chinese]


    中心IP是指在整个簇中，度占比满足大于等于阈值的节点

    Args:
        edge_index: 边索引列表
        threhold: 度占比阈值
    
    Returns:
        nodes_dic: 节点属性字典，key为ip的id，values为度&是否为中心ip，
    Examples:
        nodeRes_dic[i] = {
            'degree': degrees[i],   \n
            'center_ip': [1,0],         \n
        }
    """
    if len(edge_index) == 0:
        error_message = {
            'error': '传入数据非法！传入空图'
        }
        print(error_message)
        return error_message


    g = ig.Graph(edges=edge_index, directed=True)
    degrees = g.degree()
    
    total_degree = 0
    nodeRes_dic = {}
    for i in range(len(degrees)):
        nodeRes_dic[i] = {
            'degree': degrees[i],
            'center_ip': [0,1],
        }
        
        total_degree += degrees[i]


    sorted_degree = sorted(nodeRes_dic.items(),key=lambda x:x[1]['degree'],reverse=True)
    
    edge2num = {}
    for edge in edge_index:
        edge_key = tuple(edge)
        if edge_key not in edge2num.keys():
            edge2num[edge_key] = 1
        else:
            edge2num[edge_key] += 1

    if edge2num.keys() == 1:
        center_id = edge_index[0][1]
        nodeRes_dic[center_id]['center_ip'] = [1,0]
        return nodeRes_dic
    else:
        sum_degree = 0
        center_id_list = []
        for one in sorted_degree:
            if sum_degree < threhold*total_degree:
                newDegree = one[1]['degree']
                repeat = 0
                for center_id in center_id_list:
                    center_edge1 = (one[0],center_id)
                    center_edge2 = (center_id,one[0])
                    repeat = repeat + edge2num.get(center_edge1, 0) + edge2num.get(center_edge2, 0)

                center_id_list.append(one[0])
                sum_degree = sum_degree + newDegree - repeat
        for center_id in center_id_list:
            nodeRes_dic[center_id]['center_ip'] = [1,0]

    return nodeRes_dic


if __name__ == '__main__':

    # edge_index = [[0,1],[0,1],[1,2]]
    # node_dic = get_is_centerNode(edge_index)
    #print(node_dic)

    ip = '202.100.20.98'
    
    print(ip2Label(ip))