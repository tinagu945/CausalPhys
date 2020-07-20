from torch.utils.data import Dataset

class ControlDataset(Dataset):
    def __init__(self, data, control_nodes, variations=5, start_node=2, edges=None):
        """
        Assume the nodes to be controlled are the top k nodes, and the target nodes are at bottom
        control_nodes: #nodes needs to be controlled, equals #nodes in data - #target nodes
        variations: #values each node takes in the dataset. Currently needs be the same for all nodes. 
                    Training batch size must equal to this.       
        """
        self.data = data
        self.edges = edges
        self.control_nodes = control_nodes
        self.variations = variations
        self.start_node=start_node

    def __len__(self):
        return self.data.shape[0]*self.control_nodes

    def __getitem__(self, idx):
        #data[j*(5**x)+i]
        which_node, which_variation = divmod(idx, self.data.shape[0])
#         which_node += self.start_node
        i, j = divmod(which_variation, self.variations)
        idx = j*(self.variations**which_node)+i
#         print(idx)
        return self.data[idx], self.edges[0], which_node