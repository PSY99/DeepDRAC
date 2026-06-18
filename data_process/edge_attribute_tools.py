import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime
import json
from config import KNOWLEDGE_DIR

def check_range(row, dport):
    
    port_value = row['Port']
    if '–' in port_value:
        start, end = map(int, port_value.split('–'))
        return dport >= start and dport <= end

    elif '-' in port_value:
        start, end = map(int, port_value.split('-'))
        return dport >= start and dport <= end

    elif '[' in port_value:
        port_value = port_value.split('[')[0]
        return int(port_value) == dport

    else:
        return int(port_value) == dport

def get_dport(dport:int)->list:
    """ Get destination port one-hot encoding

    Args:
        dport: Destination port number
    
    Returns:
        Destination port one-hot encoding，627-dim

    """
    encoded_port_path = str(KNOWLEDGE_DIR / "encoded_port_service.csv")
    encoded_port_df = pd.read_csv(encoded_port_path)

    try:
        if dport == 'other' or dport == '':
            dport = -1
        dport = int(dport)

    except Exception as e:
        error = {"error message": "dport:{} invalid format,{}".format(dport,e)}
        print(error)
        return error

    matching_rows = encoded_port_df[encoded_port_df.apply(lambda row: check_range(row, dport), axis=1)]
    if len(matching_rows) != 0:
        encode_one_hot = matching_rows['one_hot'].to_list()[0]
        encode_one_hot = json.loads(encode_one_hot)
        encode_one_hot.append(0)
    else:
        # Match
        last_one_hot = encoded_port_df.iloc[-1]['one_hot']
        encode_one_hot = json.loads(last_one_hot)
        encode_one_hot.pop()
        encode_one_hot.extend([0,1])

    return encode_one_hot

def get_protocol(protocol:str)->list:
# [Translated from Chinese]


    1.Encode application/transport protocol first
    2.Otherwise use network layer (TCP/UDP)

    Args:
        protocol: Protocol type
    
    Returns:
        Protocol type one-hot encoding
    
    """
    # Normalize protocol to lowercase
    protocol = protocol.lower()
    protocol_dict = {
        "http": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "tcp": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "tls": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "dns": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "ssh": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "udp": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "mysql": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "ssdp": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "bittorrent": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "snmp": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
        "ftp": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0,0],
        "gh0st": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0],
        "pop3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0],
        "rdp": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0],
        "socks": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0],
        "jabber": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "https": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "icmp": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "netbios": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "other": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    return protocol_dict.get(protocol,protocol_dict['other'])

def get_signature(signature)->list:
    """ Get attack type one-hot encoding

    Args:
        signature: Attack signature ID
    
    Returns:
        Attack type one-hot (148-dim)

    """
    
    encoded_signature_path = str(KNOWLEDGE_DIR / "encoded_signature.csv")
    encoded_signature_df = pd.read_csv(encoded_signature_path)

    matching_rows = encoded_signature_df.loc[encoded_signature_df['rule'] == signature]
    if len(matching_rows):
        encode_one_hot = matching_rows['one_hot'].to_list()[0]
        encode_one_hot = json.loads(encode_one_hot)
        encode_one_hot.append(0)
    else:
        print(f'hello-->{signature}')
        last_one_hot = encoded_signature_df.iloc[-1]['one_hot']
        encode_one_hot = json.loads(last_one_hot)
        encode_one_hot.pop()
        encode_one_hot.extend([0,1])

    return encode_one_hot



    

if __name__ == '__main__':

    print('hello')