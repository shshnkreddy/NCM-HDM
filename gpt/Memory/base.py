from utils.util import *
from Prompting.prompt import *
import json

class Node:
    def __init__(self, node_id, creation, last_access, embedding, data):
        self.node_id = node_id
        self.creation = creation
        self.embedding = embedding
        if(last_access is None):
            self.last_access = creation
        else:
            self.last_access = last_access
        self.data = data

    def get_node_dict(self):
        node_dict = {
            'node_id': self.node_id,
            'creation': self.creation,
            'data': self.data,
            'embedding': convert_numpy_to_list(self.embedding),
            'last_access': self.last_access
        }
        return node_dict

    def __str__(self):
        return f'ID: {self.node_id} Creation: {self.creation} LastAcc: {self.last_access} Data: {self.data}\n'

class Memory:
    def __init__(self, llm, tokenizer, sampling_params, embedding_layer='first', llm_api='hf'):
        self.nodes = {}
        self.llm = llm
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.sampling_params = sampling_params
        self.llm_api = llm_api

    def prompt_llm(self, prompts):
        if(self.llm_api=='hf'):
            return prompt_llama_hf(self.llm, self.tokenizer, prompts)

        return prompt_llama_vllm(self.llm, self.sampling_params, prompts)

    def _get_embedding(self, description, layer='first'):
        if(self.llm_api=='vllm'):
            return np.random.rand(4096)
        embedding = get_embedding_hf(self.llm, self.tokenizer, [description], layer=self.embedding_layer)[0]
        return embedding.cpu().detach().numpy()
    
    # def _get_relevance(self, key, query):
    #     return cosine_similarity(key, query)

    def _insert(self, node, node_id):
        if(node_id in self.nodes):
            assert "Memory node with current node id already present."
        self.nodes[node_id] = node
    
    def delete(self, node_id):
        if(node_id in self.nodes):
            del self.nodes[node_id]

    def insert(self, node_id, creation, data, embedding=None):
        if(node_id is None):
            node_id = len(self.nodes)

        if(embedding is None):
            embedding = self._get_embedding(description=data, layer=self.embedding_layer)
        node = Node(node_id, creation, None, embedding, data) 
        self._insert(node, node_id)

    def get_n(self):
        return len(self.nodes)
    
    def retrieve(self, query, t, n=3, alpha_data=1.0, alpha_t=1.0, query_embedding=None, update_last_access=True, ret_least_score=False, rel_func=cosine_similarity):

        n = min(n, len(self.nodes))
        if(n==0):
            if(ret_least_score):
                return [], [], [], []
            return [], []
        
        relevance = []
        recency = []
        node_ids = []

        if(query_embedding is None):
            query_embedding = self._get_embedding(query)
        
        for node_id in self.nodes.keys():
            node_ids.append(node_id)
            node = self.nodes[node_id]
            recency.append(node.last_access)
            # relevance.append(self._get_relevance(node.embedding, query_embedding))
            relevance.append(rel_func(node.embedding, query_embedding))
    
        node_ids = np.array(node_ids)
        recency = np.array(recency)
        relevance = np.array(relevance)

        #sort nodes by recency
        idxs = np.flip(np.argsort(recency))
        recency = recency[idxs]
        recency = decay(np.arange(len(recency)), 0.995)
        recency = minmax_normalize(recency) 
        relevance = minmax_normalize(relevance[idxs])
        
        scores = alpha_t*recency + alpha_data*relevance
        idxs_filter = np.flip(np.argsort(scores))
        bad_node_ids = node_ids[idxs][idxs_filter].tolist()[-n:]
        node_ids = node_ids[idxs][idxs_filter].tolist()[:n]
        

        infos = np.hstack((scores.reshape(-1, 1), recency.reshape(-1, 1), relevance.reshape(-1, 1)))
        bad_infos = infos[idxs_filter][-n:]
        infos = infos[idxs_filter][:n]

        ret_nodes = []
        
        for node_id in node_ids:
            if(update_last_access):
                self.nodes[node_id].last_access = t
            ret_nodes.append(self.nodes[node_id])

        if(ret_least_score):
            bad_nodes = []
            for node_id in bad_node_ids:
                bad_nodes.append(self.nodes[node_id])
            return ret_nodes, infos, bad_nodes, bad_infos

        return ret_nodes, infos

    def __str__(self):
        _str = ""
        for node_id in self.nodes.keys():
            _str += self.nodes[node_id].__str__()
            _str += '\n'
        return _str
    
    def get_dict(self):
        save_dict = {}
        for node_id in self.nodes.keys():
            save_dict[node_id] = self.nodes[node_id].get_node_dict()

        return save_dict

    def load_from_json(self, file_path):
        with open(file_path, 'r') as json_file:
            load_dict = json.load(json_file)
        self.nodes = {}
        for node_id in load_dict.keys():
            l_dict = load_dict[node_id]
            self.nodes[node_id] = Node(**l_dict)


    