"""Extract base pattern profiles from alert subgraphs.

Computes a structural "base pattern" for each alert subgraph, including:
- Center/margin IP types and counts
- Graph topology type (simple, divergent, convergent, etc.)
- Attack types and destination port types

Used as a grouping key before incremental clustering in the evaluation pipeline.
"""

import json

import pandas as pd


def get_base_pattern(input_graph, topology_threshold1=0.5) -> tuple:
    """Extract base pattern profile from a graph in COO format.

    Args:
        input_graph: pd.Series: one graph row from graph CSV, with columns['id','group_num', 'edge_index', 'node_attr', 'edge_attr', 'group_attr','group_label', 'nodes', 'start_time', 'end_time']
    
    Returns:
        base_pattern: Base pattern: (center_num,cental_ip_label,no_central_ip_label,topo_type,attack_type,dport_type)

    """
    node_attr = json.loads(input_graph['node_attr'])
    center_num = 0
    cental_ip_label = set()
    no_central_ip_label = set()
    for node in node_attr:
        ip_is_center,ip_label = node[0],node[2:]  
        ip_label =  ip_label.index(1)
        center_num += ip_is_center
        if ip_is_center == 1:
            cental_ip_label.add(ip_label)
        else:
            no_central_ip_label.add(ip_label)

    cental_ip_label = list(cental_ip_label)
    cental_ip_label.sort()
    no_central_ip_label = list(no_central_ip_label)
    no_central_ip_label.sort()

    if len(cental_ip_label) >= 4:
        cental_ip_label = "复合Center IP types"
    if len(no_central_ip_label) >= 4:
        no_central_ip_label = "复合Margin IP types"
        
    graph_attr = json.loads(input_graph['group_attr'])
    nomal_three_nodes_motifs = graph_attr[1:14]
    if len(node_attr) <= 2 or sum(nomal_three_nodes_motifs)==0:
        topology_encode = [1,0,0,0,0,0,0]   
    else:        
        if nomal_three_nodes_motifs[3] > topology_threshold1:
            topology_encode = [0,1,0,0,0,0,0]
        elif nomal_three_nodes_motifs[0] > topology_threshold1:
            topology_encode = [0,0,1,0,0,0,0]

        elif nomal_three_nodes_motifs[7] >= topology_threshold1:
            topology_encode = [0,0,0,1,0,0,0]

        elif nomal_three_nodes_motifs[6] >= topology_threshold1:
            topology_encode = [0,0,0,0,1,0,0]

        elif nomal_three_nodes_motifs[2] >= topology_threshold1:
            topology_encode = [0,0,0,0,0,1,0]

        else:
            topology_encode = [0,0,0,0,0,0,1]
    topo_type = topology_encode.index(1)

    try:
        edge_attr = json.loads(input_graph['edge_attr'])
    except Exception as e:
        print("id: {}, edge_attr: {}".format(input_graph['id'],input_graph['edge_attr']))
    attack_type = set()
    dport_type = set()
    
    for edge in edge_attr:
        protocol_len = 20
        protocol_end = 627+protocol_len
        dprot,protocol,signature = edge[0:627],edge[627:protocol_end],edge[protocol_end:-2]
        #print(protocol)
        signature_num = signature.index(1)
        attack_type.add(signature_num)
        
        dprot_type_num = dprot.index(1)
        dport_type.add(dprot_type_num)
        
    attack_type = list(attack_type)
    attack_type.sort()
    dport_type = list(dport_type)
    dport_type.sort()

    if len(attack_type) >= 4:
        attack_type = "复合攻击类型"
    if len(dport_type) >= 4:
        dport_type = "复合端口"
        
    base_pattern = (center_num,cental_ip_label,no_central_ip_label,topo_type,attack_type,dport_type)

    return base_pattern

def base_pattern_parse(base_pattern,encoded_signature_df):
    """Get human-readable base pattern format.

        Get human-readable base pattern format.
       
    Args:
        base_pattern: Base pattern: (center_num,cental_ip_label,no_central_ip_label,topo_type,attack_type,dport_type)
    Returns:
        base_pattern_parse: Human-readable base pattern.
    """
    center_num,cental_ip_label,no_central_ip_label,topo_type,attack_type,dport_type = base_pattern
    # cental_ip_label,no_central_ip_label

    # ipLabel_dict = {
    #     "Internal user": [1,0],
    #     "unknown ip": [0, 1]
    # }
    # ═══ IP-TO-ROLE MAPPING — must match data_process/node_attribute_tools.py ═══
    # Update both files when adapting to a new network topology.
    ipLabel_dict = {
        "Firewall":        [1, 0, 0, 0, 0, 0, 0, 0],
        "DNS+DC Server":   [0, 1, 0, 0, 0, 0, 0, 0],
        "Web Server":      [0, 0, 1, 0, 0, 0, 0, 0],
        "Ubuntu Server":   [0, 0, 0, 1, 0, 0, 0, 0],
        "Ubuntu Client":   [0, 0, 0, 0, 1, 0, 0, 0],
        "Windows Client":  [0, 0, 0, 0, 0, 1, 0, 0],
        "MAC":             [0, 0, 0, 0, 0, 0, 1, 0],
        "Outsider":        [0, 0, 0, 0, 0, 0, 0, 1]
    }
    attack_type_num = 177
    # ip
    ipLabel_list = list(ipLabel_dict.keys())
    no_central_ip_label_parse,cental_ip_label_parse = [],[]
    if cental_ip_label=='复合Center IP types':
        #cental_ip_label_parse = cental_ip_label
        cental_ip_label_parse = 'Multi-type center'
    else:
        for one_ip in cental_ip_label:
            cental_ip_label_parse.append(ipLabel_list[one_ip])
            
    if no_central_ip_label == '复合Margin IP types':
        # no_central_ip_label_parse = no_central_ip_label
        no_central_ip_label_parse = 'Multi-type margin'
    else:
        for one_ip in no_central_ip_label:
            no_central_ip_label_parse.append(ipLabel_list[one_ip])
    
    # topo_type
    topo_type_list = [' simple ', 'divergent', 'convergent', 'two-way type', 'generalized divergent', 'generalized convergent', 'other types']
    topo_type_parse = topo_type_list[topo_type]
    # attack_type
    attack_type_parse = []
    if attack_type == '复合攻击类型':
        # attack_type_parse = attack_type
        attack_type_parse = 'Multiple attack types'
    else:
        for one_attack_id in attack_type:
            one_attack_id += 1
            #print(one_attack_id)
            if one_attack_id>attack_type_num:
                attack_type_parse.append('Unknown attack type')
            else:
                matching_rows = encoded_signature_df.loc[encoded_signature_df['ID'] == one_attack_id]
                
                # detail_attack_type = matching_rows['class'].to_list()[0]
                # # detail_attack_type_en = matching_rows['category_standard_en'].to_list()[0]
                # rough_attack_type = detail_attack_type.split('+')[0]
                
                rough_attack_type = matching_rows['class'].to_list()[0]
                attack_type_parse.append(rough_attack_type)


    base_pattern_parse = (center_num,cental_ip_label_parse,no_central_ip_label_parse,topo_type_parse,attack_type_parse)
    #base_pattern_parse_en = (center_num, createRequest(cental_ip_label_parse), createRequest(no_central_ip_label_parse), createRequest(topo_type_parse), createRequest(attack_type_parse))
    return base_pattern_parse