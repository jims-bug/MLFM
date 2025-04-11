'''
Description:  
Author: Song Jin
Date: 2024-07-21 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MLFMBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(100, opt.polarities_dim)

    def forward(self, inputs):
        outputs1 = self.gcn_model(inputs)
        logits = self.classifier(outputs1)

        return logits, None


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, deprel_adj, asp_start, asp_end, src_mask, aspect_mask= inputs
        h = self.gcn(inputs)    
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100)  
        outputs1 = (h*aspect_mask).sum(dim=1) / asp_wn
        return outputs1   

class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.attdim = 100
        self.W = nn.Linear(self.attdim,self.attdim)
        self.Wx= nn.Linear(self.attention_heads+self.attdim*2, self.attention_heads)
        self.Wxx = nn.Linear(self.bert_dim, self.attdim)
        self.Wi = nn.Linear(self.attdim,50)
        self.aggregate_W = nn.Linear(self.attdim*2, self.attdim)  

        self.attn = MultiHeadAttention(opt.attention_heads, self.attdim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        
        # NodeAttention
        # self.nodeattention = AspectNeighborAttention(in_dim = opt.rnn_hidden * 2 , dep_embed_dim = opt.dep_embed_dim, opt = opt)
        # self.nodalattention = NodalAttention(in_dim = opt.rnn_hidden * 2 , dep_embed_dim = opt.dep_embed_dim, opt = opt)
        self.aspectNeighborAttention = AspectNeighborAttention(in_dim = opt.rnn_hidden * 2 , dep_embed_dim = opt.dep_embed_dim, opt = opt)
        self.edge_embeddings = nn.Embedding(num_embeddings=opt.dep_size,
                                            embedding_dim=opt.dep_embed_dim,
                                            padding_idx=0)
        self.dep_embed_linear = nn.Linear(opt.dep_embed_dim, opt.dynamic_tree_attn_head)
        
        # AdaptiveResidualFusion
        self.adaptiveResidualFusion = AdaptiveResidualFusion(embedding_dim = opt.rnn_hidden * 2)
        self.adaptiveCrossFusion_dropout = nn.Dropout(opt.adaptiveCrossFusion_dropout) # 0.1   
        self.dualCrossAttentionFusion = DualCrossAttentionFusion(embedding_dim = opt.rnn_hidden * 2, num_heads = opt.CrossAttentionFusion_head, opt = opt, num_iterations = opt.adaptiveCrossFusion_num_iterations)
        
    def GCN_Layer(self,weight_adj, gcn_outputs, i, maxlen):
            gcn_outputs = gcn_outputs.unsqueeze(1).expand(gcn_outputs.shape[0], self.attention_heads, maxlen, self.attdim)   # 注意力是5个头 
            Ax = torch.matmul(weight_adj, gcn_outputs)  
            Ax = Ax.mean(dim=1)  
    
            Ax = self.W(Ax)   
            weights_gcn_outputs = F.selu(Ax)

            gcn_outputs = weights_gcn_outputs      

            # gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs 
                

            weight_adj=weight_adj.permute(0, 2, 3, 1).contiguous()  
            node_outputs1 = gcn_outputs.unsqueeze(1).expand(gcn_outputs.shape[0], maxlen, maxlen, self.attdim)   
            node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous() 

            node = torch.cat([node_outputs1, node_outputs2], dim=-1) 
            edge_n=torch.cat([weight_adj,node], dim=-1)
            edge = self.Wx(edge_n) 
            edge = self.gcn_drop(edge) if i < self.layers - 1 else edge 
            weight_adj=edge.permute(0,3,1,2).contiguous()
            
            return weight_adj, gcn_outputs
    def forward(self, inputs): 
        text_bert_indices, bert_segments_ids, attention_mask, deprel_adj, asp_start, asp_end, src_mask, aspect_mask = inputs
        src_mask = src_mask.unsqueeze(-2) 
        batch = src_mask.size(0)
        len = src_mask.size()[2]

        _temp = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output, pooled_output = _temp[0], _temp[1]
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)  
        pooled_output = self.pooled_drop(pooled_output)

        gcn_inputs = self.Wxx(gcn_inputs)
        
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100) 
        aspect = (gcn_inputs*aspect_mask).sum(dim=1) / asp_wn   
        
        # crop
        dep_type_matrix_mask = (deprel_adj != 0)
        # Embedding operations on non-zero elements
        nonzero_indices = deprel_adj[dep_type_matrix_mask]
        embedded_nonzero = self.edge_embeddings(nonzero_indices.long())

        dep_type_adj = torch.zeros(deprel_adj.shape[0], deprel_adj.shape[1], deprel_adj.shape[2], self.opt.dep_embed_dim).to(self.opt.device)
        dep_type_adj[dep_type_matrix_mask] = embedded_nonzero       
        #######################################################################################################################
        gcn_inputs = self.aspectNeighborAttention(gcn_inputs, dep_type_adj, inputs)
        
                
        dep_type_adj_short = self.dep_embed_linear(dep_type_adj).squeeze(-1)
        dep_type_adj_short = dep_type_adj_short.transpose(1, 3)
        
        attn_tensor_L, attn_tensor_R = self.attn(gcn_inputs, gcn_inputs, dep_type_adj_short, aspect, src_mask)   
        
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor_L, 1, dim=1)]
        multi_head_list = []
        outputs_dep = None
        adj_ag = None
        
        weight_adj_L, weight_adj_R=attn_tensor_L, attn_tensor_R   
        gcn_outputs_L, gcn_outputs_R=gcn_inputs, gcn_inputs   
        layer_list = [gcn_inputs]
     

        for i in range(self.layers):
            weight_adj_L, gcn_outputs_L = self.GCN_Layer(weight_adj_L, gcn_outputs_L, i, len)
            weight_adj_R, gcn_outputs_R = self.GCN_Layer(weight_adj_R, gcn_outputs_R, i, len)


        # outputs = torch.cat(layer_list, dim=-1)
        # node_outputs=self.Wi(gcn_outputs)
        # gcn_outputs = torch.cat((gcn_outputs_L, gcn_outputs_R), dim=-1)
        # gcn_outputs=self.aggregate_W(gcn_outputs)
        gcn_outputs = self.adaptiveCrossFusion_dropout(self.dualCrossAttentionFusion(gcn_outputs_L, gcn_outputs_R))

        
        
        gcn_outputs=F.relu(gcn_outputs)

        return gcn_outputs




def attention(query, key, short, aspect, weight_m, bias_m, mask=None, dropout=None):   
    d_k = query.size(-1)   
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    batch=len(scores)  
    p=weight_m.size(0)
    max=weight_m.size(1)
    weight_m=weight_m.unsqueeze(0).repeat(batch,1,1,1)

    aspect_scores = torch.tanh(torch.add(torch.matmul(aspect, key.transpose(-2, -1)), bias_m))  
    
    scores_L=torch.add(scores, aspect_scores)
    
    

    if mask is not None:
        scores_L = scores_L.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask == 0, -1e9)
    # scores=torch.add(scores, short)
    scores_R=torch.add(scores, short)
    
    p_attn_L = F.softmax(scores_L, dim=-1)
    p_attn_R = F.softmax(scores_R, dim=-1)
    if dropout is not None:
        p_attn_L = dropout(p_attn_L)
        p_attn_R = dropout(p_attn_R)

    return p_attn_L, p_attn_R

 
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  
        self.d_k = d_model // h  
        self.h = h    
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k)) 
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)
    

    def forward(self, query, key, short, aspect, mask=None):   
        mask = mask[:, :, :query.size(1)]  
        if mask is not None:
            mask = mask.unsqueeze(1)  
        
        nbatches = query.size(0)  
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        
        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)    
        aspect = self.dense(aspect) 
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k)
        attn_L, attn_R = attention(query, key,short,aspect, self.weight_m, self.bias_m, mask=mask, dropout=self.dropout)  
        return attn_L, attn_R
    

class NodeAttention(nn.Module):
    def __init__(self, in_dim, dep_embed_dim, opt):
        super(NodeAttention, self).__init__()
        self.in_dim = in_dim
        self.dep_embed_dim = dep_embed_dim
        self.attention = nn.Linear(in_dim + dep_embed_dim + in_dim, 1, bias=False)
        self.nodalAttention_dropout = nn.Dropout(opt.nodalAttention_dropout)

    def forward(self, features, aspect_onehot, adj_matrix):
        batch_size, num_nodes, _ = features.size()

        updated_features = features.clone()
        for b in range(batch_size):

            aspect_indices = aspect_onehot[b].nonzero(as_tuple=False).squeeze(1).tolist()
            for aspect_idx in aspect_indices:

                connected_indices = adj_matrix[b, aspect_idx].nonzero(as_tuple=False).squeeze(1)
            
                if connected_indices.numel() == 0:
                    continue

                valid_connected_indices = connected_indices[connected_indices < num_nodes]
             
                if valid_connected_indices.numel() == 0:
                    continue
       
                aspect_feature = features[b, aspect_idx].unsqueeze(0)  # (1, in_dim)
            
                connected_features = features[b, valid_connected_indices]  # (num_connected_nodes, in_dim)
                connected_dep_features = adj_matrix[b, aspect_idx, valid_connected_indices]  # (num_connected_nodes, dep_embed_dim)
      
                aspect_repeated = aspect_feature.expand(valid_connected_indices.size(0), -1)  # (num_connected_nodes, in_dim)
                concatenated_features = torch.cat([connected_features, connected_dep_features, aspect_repeated], dim=1)  # (num_connected_nodes, in_dim + dep_embed_dim + in_dim)
                attention_scores = self.attention(concatenated_features).squeeze(1)  # (num_connected_nodes,)
        
                attention_weights = F.softmax(attention_scores, dim=0)  # (num_connected_nodes,)
  
                aggregated_feature = torch.matmul(attention_weights, connected_features)  # (in_dim,)
                
          
                updated_features[b, aspect_idx] = self.nodalAttention_dropout(aggregated_feature)
        return updated_features

class NodalAttention(nn.Module):
    def __init__(self, in_dim, dep_embed_dim, opt):
        super(NodalAttention, self).__init__()
        self.in_dim = in_dim
        self.dep_embed_dim = dep_embed_dim
        self.z_linear = nn.Linear(in_dim, in_dim)
        self.attention_layer = nn.Linear(in_dim + dep_embed_dim + in_dim, 1, bias=False)        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.w_m = torch.nn.Linear(in_dim + dep_embed_dim, in_dim, bias=False)
        self.w_h = torch.nn.Linear(in_dim + in_dim, in_dim, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.nodalAttention_dropout = nn.Dropout(opt.nodalAttention_dropout)

    def forward(self, features, dep_type_adj, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, deprel_adj, asp_start, asp_end, src_mask, aspect_mask = inputs

        batch_size, num_nodes, _ = features.size()

        updated_features = features.clone()
        features = self.z_linear(features)
            
        for b in range(batch_size):
            indices = torch.nonzero(deprel_adj[b], as_tuple=True) 
            for i in range(asp_start[b], asp_end[b]):
                gcn_inputs_aspect = features[0][i+1].unsqueeze(0)
                index1 = (torch.tensor([idx[0] for idx in zip(*indices) if idx[0] == i], dtype=torch.long), 
                        torch.tensor([idx[1] for idx in zip(*indices) if idx[0] == i], dtype=torch.long),)
                index2 = (torch.tensor([idx[0] for idx in zip(*indices) if idx[1] == i], dtype=torch.long), 
                        torch.tensor([idx[1] for idx in zip(*indices) if idx[1] == i], dtype=torch.long))
                
                result_dim1 = torch.cat((dep_type_adj[b][index1], features[b][index1[1]+1], gcn_inputs_aspect.repeat(dep_type_adj[b][index1].shape[0], 1)), dim=1)               
                result_dim2 = torch.cat((dep_type_adj[b][index2], features[b][index2[0]+1], gcn_inputs_aspect.repeat(dep_type_adj[b][index2].shape[0], 1)), dim=1)
                combined_result1 = torch.cat((result_dim1, result_dim2), dim=0)
                
                result_dim3 = torch.cat((dep_type_adj[b][index1], features[b][index1[1]+1]), dim=1)               
                result_dim4 = torch.cat((dep_type_adj[b][index2], features[b][index2[0]+1]), dim=1)               
                combined_result2 = torch.cat((result_dim3, result_dim4), dim=0)
                

                attention_scores = self.leaky_relu(self.attention_layer(combined_result1))  # shape: (3, 1)
                attention_weights = F.softmax(attention_scores, dim=0)  # shape: (3, 1)
                input_tensor = self.relu1(self.w_m(combined_result2))
                attention_fused = torch.sum(input_tensor * attention_weights, dim=0, keepdim=True)  # shape: (1, 220)
                aspect_feature = torch.cat((attention_fused, gcn_inputs_aspect), dim=1)
                aspect_feature = self.relu2(self.w_h(aspect_feature))
                                
                updated_features[b][i+1] = aspect_feature

            
        updated_features = self.nodalAttention_dropout(updated_features)
                               
        return updated_features
    
class AspectNeighborAttention(nn.Module):
    def __init__(self, in_dim, dep_embed_dim, opt):
        super().__init__()
        self.in_dim = in_dim
        self.dep_embed_dim = dep_embed_dim

        self.z_linear = nn.Linear(in_dim, in_dim)

        self.attention_weights = nn.Sequential(
            nn.Linear(in_dim * 2 + dep_embed_dim, 1)
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.dep_type_fusion = nn.Linear(in_dim + dep_embed_dim, in_dim, bias=False)
        self.w_h = torch.nn.Linear(in_dim + in_dim, in_dim, bias=False)

        self.nodalAttention_dropout = nn.Dropout(opt.nodalAttention_dropout)
        
    def forward(self, bert_hidden_states, dep_type_adj, inputs):
        batch_size, seq_len, hidden_size = bert_hidden_states.size()
        text_bert_indices, bert_segments_ids, attention_mask, deprel_adj, asp_start, asp_end, src_mask, aspect_mask = inputs
        # Initialize output features as copies of input features
        output_hidden_states = bert_hidden_states.clone()
        bert_hidden_states = self.z_linear(bert_hidden_states)
        for b in range(batch_size):
            adj_start = asp_start[b]
            adj_end = asp_end[b]
            
            for asp_idx in range(adj_start, adj_end + 1):
                # Find neighboring nodes
                adj_mask = deprel_adj[b, asp_idx, :] > 0
                # Corresponding dependency types are embedded
                dep_type_embeds = dep_type_adj[b, asp_idx, adj_mask, :]
                adj_mask = torch.roll(adj_mask, shifts = 1, dims=0)
                # Representation of neighboring nodes
                neighbor_nodes = bert_hidden_states[b, adj_mask, :]

                current_asp_repr = bert_hidden_states[b, asp_idx + 1, :]

                if len(neighbor_nodes) == 0:
                    continue

                repeated_asp_repr = current_asp_repr.repeat(len(neighbor_nodes), 1)

                combined = torch.cat([
                    repeated_asp_repr, 
                    neighbor_nodes, 
                    dep_type_embeds
                ], dim=-1)
                # Calculating Attention Weights
                attention_weight = F.softmax(
                    self.leaky_relu(self.attention_weights(combined)), 
                    dim=0
                )
                neighbor = torch.cat([neighbor_nodes, dep_type_embeds], dim=-1)
                neighbor = self.dep_type_fusion(neighbor)

                weighted_node = attention_weight * neighbor

                neighbor_repr = weighted_node.sum(dim=0)
                temp = self.w_h(torch.cat([neighbor_repr, current_asp_repr], dim=-1))

                output_hidden_states[b, asp_idx + 1, :] = temp
        
        return output_hidden_states

class AdaptiveResidualFusion(nn.Module):
    def __init__(self, embedding_dim):
        super(AdaptiveResidualFusion, self).__init__()
        self.embedding_dim = embedding_dim

        self.resNet1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),

            
        )
        self.resNet2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),

            
        )
        
        # Initialize alpha and beta as learnable parameters
        self.weight1 = nn.Linear(embedding_dim, 1)
        self.weight2 = nn.Linear(embedding_dim, 1)
        
        self.layer_norm1 = nn.LayerNorm(embedding_dim)

    def forward(self, x1, x2):
        # Process the first input feature
        # out1 = self.fc1(x1)
        # out1 = self.relu(out1)
        # out1 = self.fc2(out1)
        out1 = self.resNet1(x1)
        out1 = self.layer_norm1(out1 + x1)
        
        weight1 = torch.sigmoid(self.weight1(out1))
        # Process the second input feature
        # out2 = self.fc2(x2)
        # out2 = self.relu(out2)
        # out2 = self.fc2(out2)
        out2 = self.resNet2(x2)
        out2 = self.layer_norm1(out2 + x2)
        
        weight2 = torch.sigmoid(self.weight2(out2))#
        
        weight1 = weight1/(weight1+weight2)
        weight2= 1-weight1

        # Apply adaptive residual connection to combine x1 and x2
        # out = self.alpha * out1 + self.beta * out2
        out = weight1 * out1 + weight2 * out2
        
        return out

class DualCrossAttentionFusion(nn.Module):
    def __init__(self, embedding_dim, num_heads, opt ,num_iterations = 2):
        super(DualCrossAttentionFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_iterations = num_iterations
        self.adaptiveResidualFusion = AdaptiveResidualFusion(embedding_dim = self.embedding_dim)
        # self.cross_attention11 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        # self.cross_attention21 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        self.layers1 = nn.ModuleList([nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True) for _ in range(num_iterations)])
        self.layers2 = nn.ModuleList([nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True) for _ in range(num_iterations)])
            
        self.weight1 = nn.Linear(embedding_dim, 1)
        self.weight2 = nn.Linear(embedding_dim, 1)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.adaptiveResidualFusion_drop = nn.Dropout(opt.adaptiveResidualFusion_dropout)

    def forward(self, x1, x2):
        
        x = self.adaptiveResidualFusion_drop(self.adaptiveResidualFusion(x1, x2))

        first_iteration = True
        for layer1, layer2 in zip(self.layers1, self.layers2):
            if first_iteration:
                attn_output1, _ = layer1(x, x1, x1)
                attn_output2, _ = layer2(x, x2, x2)
                first_iteration = False
                attn_output1 = self.layer_norm(x1 + attn_output1)
                attn_output2 = self.layer_norm(x2 + attn_output2)
            else:                 
                attn_output1, _ = layer1(attn_output2, attn_output1, attn_output1)
                attn_output2, _ = layer2(attn_output1, attn_output2, attn_output2)
                attn_output1 = self.layer_norm(attn_output1)
                attn_output2 = self.layer_norm(attn_output2)

                
        
        weight1 = torch.sigmoid(self.weight1(attn_output1))
        weight2 = torch.sigmoid(self.weight2(attn_output2))
        weight1 = weight1/(weight1+weight2)
        weight2= 1-weight1
        # Adaptive Fusion
        output = weight1 * attn_output1 + weight2 * attn_output2
        return output
