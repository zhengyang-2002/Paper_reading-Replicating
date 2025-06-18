import torch
import torch.nn as nn

class Bi_Encoder(nn.Module):
    name = 'Bi_Encoder'
    def __init__(self, cfg):
        super(Bi_Encoder, self).__init__()
        def __create_xh(embedding_size, hidden_size):
            return nn.Sequential(
                nn.Linear(embedding_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, hidden_size)
            )
        def __create_hh(hidden_size):
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, hidden_size)
            )
    
        self.xh = __create_xh(cfg.embedding_size, cfg.hidden_size)
        self.hh = __create_hh(cfg.hidden_size)
        self.xh_ = __create_xh(cfg.embedding_size, cfg.hidden_size)
        self.hh_ = __create_hh(cfg.hidden_size)
        
        self.tanh = nn.Tanh()

        self.cfg = cfg

    def forward(self, seq, input_lengths):
        batch_size, seq_len, embedding_size = seq.size()
        mask = torch.arange(seq_len, device=self.cfg.device).expand(batch_size, -1) < input_lengths.unsqueeze(1)
        
        hidden_state = torch.zeros(batch_size, self.cfg.hidden_size, device=self.cfg.device)
        hidden_state_ = torch.zeros(batch_size, self.cfg.hidden_size, device=self.cfg.device)

        forward_hidden_states = torch.zeros(batch_size, seq_len, self.cfg.hidden_size, device=self.cfg.device)
        backward_hidden_states = torch.zeros(batch_size, seq_len, self.cfg.hidden_size, device=self.cfg.device)

        for t in range(seq_len):
            token, token_ = seq[:,t,:], seq[:,seq_len-t-1,:]
            current_mask, current_mask_ = mask[:, t].unsqueeze(1), mask[:, seq_len-t-1].unsqueeze(1)
            
            temp_hidden_state = self.tanh(self.xh(token)+self.hh(hidden_state))
            temp_hidden_state_ = self.tanh(self.xh_(token_)+self.hh_(hidden_state_))
            
            hidden_state = torch.where(current_mask, temp_hidden_state, hidden_state) # batch_size, embedding_size
            hidden_state_ = torch.where(current_mask_, temp_hidden_state_, hidden_state_)

            forward_hidden_states[:, t, :] = hidden_state
            backward_hidden_states[:, seq_len-t-1, :] = hidden_state_

        annotations = torch.concatenate([forward_hidden_states, backward_hidden_states], dim=-1)
        return hidden_state, annotations


#初始版本
class Attention_Decoder(nn.Module):
    name = 'Attention_Decoder'
    def __init__(self, cfg):
        super(Attention_Decoder, self).__init__()
        # Output Matrix
        self.Wg = nn.Sequential(
            nn.Linear(cfg.embedding_size+cfg.hidden_size+cfg.hidden_size*2, cfg.embedding_size//2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size//2, cfg.embedding_size)
        )
        # Hidden Matrix
        self.Wf = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.embedding_size+cfg.hidden_size*2, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        # Alignment Matrix
        self.Wa = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)

        self.cfg = cfg
          
    def g(self, y_prev, s_i, c_i):
        temp_state = torch.concatenate([y_prev, s_i, c_i], dim=-1)
        predication = self.Wg(temp_state)
        return predication
        
    def f(self, s_prev, y_prev, c_i):
        temp_state = torch.concatenate([s_prev, y_prev, c_i], dim=-1)
        new_hidden_state = self.Wf(temp_state)
        return new_hidden_state

    def a(self, s_prev, annotation):
        temp_state = torch.concatenate([s_prev, annotation], dim=-1)
        energy = self.Wa(temp_state).squeeze(-1)  # consecutive dim_size = 1 could squeeze into one dim_size = 1
        return energy

    def generate_context(self, s_prev, annotations):
        # At this moment, s_prev: batch_size, hidden_size; annotations: batch_size, annotation_number, annotation_size
        batch_size, annotation_number, annotation_size = annotations.size()
        s_prev = s_prev.unsqueeze(1).expand(batch_size, annotation_number, -1)

        energys = self.a(s_prev, annotations) 
        attention_weight_expand = torch.nn.functional.softmax(energys, dim=1).unsqueeze(-1)
        context = torch.sum(annotations*attention_weight_expand, dim=1) # batch_size, annotation_size
        return context

    def forward(self, hidden_state, decode_length, annotations):
        #hidden_state = torch.zeros_like(hidden_state)
        batch_size, _ = hidden_state.size()
        outputs = torch.zeros(batch_size, decode_length, self.cfg.embedding_size, device=self.cfg.device)
        y = torch.zeros(batch_size, self.cfg.embedding_size, device=self.cfg.device)
        for t in range(decode_length):
            y_prev = y
            c_i = self.generate_context(hidden_state, annotations)
            hidden_state = self.f(hidden_state, y_prev, c_i)
            y = self.g(y_prev, hidden_state, c_i)
            outputs[:,t,:] = y
        outputs = self.fc(outputs)
        return outputs



#4.1
class Attention_Decoder_41(nn.Module):
    name = 'Attention_Decoder_41'
    def __init__(self, cfg):
        super(Attention_Decoder_41, self).__init__()
        # Output Matrix
        self.Wg = nn.Sequential(
            nn.Linear(cfg.embedding_size+cfg.hidden_size+cfg.hidden_size*2, cfg.embedding_size//2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size//2, cfg.embedding_size)
        )
        # Hidden Matrix
        self.Wf = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.embedding_size+cfg.hidden_size*2, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        # Alignment Matrix
        self.Wa = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)

        self.cfg = cfg
          
    def g(self, y_prev, s_i, c_i):
        temp_state = torch.concatenate([y_prev, s_i, c_i], dim=-1)
        predication = self.Wg(temp_state)
        return predication
        
    def f(self, s_prev, y_prev, c_i):
        temp_state = torch.concatenate([s_prev, y_prev, c_i], dim=-1)
        new_hidden_state = self.Wf(temp_state)
        return new_hidden_state

    def a(self, s_prev, annotation):
        temp_state = torch.concatenate([s_prev, annotation], dim=-1)
        energy = self.Wa(temp_state).squeeze(-1)  # consecutive dim_size = 1 could squeeze into one dim_size = 1
        return energy

    def generate_context(self, s_prev, annotations):
        # At this moment, s_prev: batch_size, hidden_size; annotations: batch_size, annotation_number, annotation_size
        batch_size, annotation_number, annotation_size = annotations.size()
        s_prev = s_prev.unsqueeze(1).expand(batch_size, annotation_number, -1)

        energys = self.a(s_prev, annotations) 
        attention_weight_expand = torch.nn.functional.softmax(energys, dim=1).unsqueeze(-1)
        context = torch.sum(annotations*attention_weight_expand, dim=1) # batch_size, annotation_size
        return context

    def forward(self, hidden_state, decode_length, annotations):
        hidden_state = torch.zeros_like(hidden_state)
        batch_size, _ = hidden_state.size()
        outputs = torch.zeros(batch_size, decode_length, self.cfg.embedding_size, device=self.cfg.device)
        y = torch.zeros(batch_size, self.cfg.embedding_size, device=self.cfg.device)
        for t in range(decode_length):
            y_prev = y
            c_i = self.generate_context(hidden_state, annotations)
            hidden_state = self.f(hidden_state, y_prev, c_i)
            y = self.g(y_prev, hidden_state, c_i)
            outputs[:,t,:] = y
        outputs = self.fc(outputs)
        return outputs



#4.2
class Attention_Decoder_42(nn.Module):
    name = 'Attention_Decoder_42'
    def __init__(self, cfg):
        super(Attention_Decoder_42, self).__init__()
        # Output Matrix
        self.Wg = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.embedding_size//2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size//2, cfg.embedding_size)
        )
        # Hidden Matrix
        self.Wf = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.embedding_size+cfg.hidden_size*2, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        # Alignment Matrix
        self.Wa = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)

        self.cfg = cfg
          
    def g(self, s_i):
        predication = self.Wg(s_i)
        return predication
        
    def f(self, s_prev, y_prev, c_i):
        temp_state = torch.concatenate([s_prev, y_prev, c_i], dim=-1)
        new_hidden_state = self.Wf(temp_state)
        return new_hidden_state

    def a(self, s_prev, annotation):
        temp_state = torch.concatenate([s_prev, annotation], dim=-1)
        energy = self.Wa(temp_state).squeeze(-1)  # consecutive dim_size = 1 could squeeze into one dim_size = 1
        return energy

    def generate_context(self, s_prev, annotations):
        # At this moment, s_prev: batch_size, hidden_size; annotations: batch_size, annotation_number, annotation_size
        batch_size, annotation_number, annotation_size = annotations.size()
        s_prev = s_prev.unsqueeze(1).expand(batch_size, annotation_number, -1)

        energys = self.a(s_prev, annotations) 
        attention_weight_expand = torch.nn.functional.softmax(energys, dim=1).unsqueeze(-1)
        context = torch.sum(annotations*attention_weight_expand, dim=1) # batch_size, annotation_size
        return context

    def forward(self, hidden_state, decode_length, annotations):
        #hidden_state = torch.zeros_like(hidden_state)
        batch_size, _ = hidden_state.size()
        outputs = torch.zeros(batch_size, decode_length, self.cfg.embedding_size, device=self.cfg.device)
        y = torch.zeros(batch_size, self.cfg.embedding_size, device=self.cfg.device)
        for t in range(decode_length):
            y_prev = y
            c_i = self.generate_context(hidden_state, annotations)
            hidden_state = self.f(hidden_state, y_prev, c_i)
            y = self.g(hidden_state)
            outputs[:,t,:] = y
        outputs = self.fc(outputs)
        return outputs


#4.3
class Attention_Decoder_43(nn.Module):
    name = 'Attention_Decoder_43'
    def __init__(self, cfg):
        super(Attention_Decoder_43, self).__init__()
        # Output Matrix
        self.Wg = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.embedding_size//2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size//2, cfg.embedding_size)
        )
        # Hidden Matrix
        self.Wf = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        # Alignment Matrix
        self.Wa = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)

        self.cfg = cfg
          
    def g(self, s_i, c_i):
        temp_state = torch.concatenate([s_i, c_i], dim=-1)
        predication = self.Wg(temp_state)
        return predication
        
    def f(self, s_prev, c_i):
        temp_state = torch.concatenate([s_prev, c_i], dim=-1)
        new_hidden_state = self.Wf(temp_state)
        return new_hidden_state

    def a(self, s_prev, annotation):
        temp_state = torch.concatenate([s_prev, annotation], dim=-1)
        energy = self.Wa(temp_state).squeeze(-1)  # consecutive dim_size = 1 could squeeze into one dim_size = 1
        return energy

    def generate_context(self, s_prev, annotations):
        # At this moment, s_prev: batch_size, hidden_size; annotations: batch_size, annotation_number, annotation_size
        batch_size, annotation_number, annotation_size = annotations.size()
        s_prev = s_prev.unsqueeze(1).expand(batch_size, annotation_number, -1)

        energys = self.a(s_prev, annotations) 
        attention_weight_expand = torch.nn.functional.softmax(energys, dim=1).unsqueeze(-1)
        context = torch.sum(annotations*attention_weight_expand, dim=1) # batch_size, annotation_size
        return context

    def forward(self, hidden_state, decode_length, annotations):
        #hidden_state = torch.zeros_like(hidden_state)
        batch_size, _ = hidden_state.size()
        outputs = torch.zeros(batch_size, decode_length, self.cfg.embedding_size, device=self.cfg.device)
        y = torch.zeros(batch_size, self.cfg.embedding_size, device=self.cfg.device)
        for t in range(decode_length):
            y_prev = y
            c_i = self.generate_context(hidden_state, annotations)
            hidden_state = self.f(hidden_state, c_i)
            y = self.g(hidden_state, c_i)
            outputs[:,t,:] = y
        outputs = self.fc(outputs)
        return outputs


#4.4
class Attention_Decoder_44(nn.Module):
    name = 'Attention_Decoder_44'
    def __init__(self, cfg):
        super(Attention_Decoder_44, self).__init__()
        # Output Matrix
        self.Wg = nn.Sequential(
            nn.Linear(cfg.embedding_size+cfg.hidden_size+cfg.hidden_size*2, cfg.embedding_size//2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size//2, cfg.embedding_size)
        )
        # Hidden Matrix
        self.Wf = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        # Alignment Matrix
        self.Wa = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)

        self.cfg = cfg
          
    def g(self, y_prev, s_i, c_i):
        temp_state = torch.concatenate([y_prev, s_i, c_i], dim=-1)
        predication = self.Wg(temp_state)
        return predication
        
    def f(self, s_prev):
        #temp_state = torch.concatenate([s_prev, y_prev, c_i], dim=-1)
        new_hidden_state = self.Wf(s_prev)
        return new_hidden_state

    def a(self, s_prev, annotation):
        temp_state = torch.concatenate([s_prev, annotation], dim=-1)
        energy = self.Wa(temp_state).squeeze(-1)  # consecutive dim_size = 1 could squeeze into one dim_size = 1
        return energy

    def generate_context(self, s_prev, annotations):
        # At this moment, s_prev: batch_size, hidden_size; annotations: batch_size, annotation_number, annotation_size
        batch_size, annotation_number, annotation_size = annotations.size()
        s_prev = s_prev.unsqueeze(1).expand(batch_size, annotation_number, -1)

        energys = self.a(s_prev, annotations) 
        attention_weight_expand = torch.nn.functional.softmax(energys, dim=1).unsqueeze(-1)
        context = torch.sum(annotations*attention_weight_expand, dim=1) # batch_size, annotation_size
        return context

    def forward(self, hidden_state, decode_length, annotations):
        #hidden_state = torch.zeros_like(hidden_state)
        batch_size, _ = hidden_state.size()
        outputs = torch.zeros(batch_size, decode_length, self.cfg.embedding_size, device=self.cfg.device)
        y = torch.zeros(batch_size, self.cfg.embedding_size, device=self.cfg.device)
        for t in range(decode_length):
            y_prev = y
            c_i = self.generate_context(hidden_state, annotations)
            hidden_state = self.f(hidden_state)
            y = self.g(y_prev, hidden_state, c_i)
            outputs[:,t,:] = y
        outputs = self.fc(outputs)
        return outputs

#---------------------------------------------------------------------------------------------------------------------------------------------
class Luong_Encoder(nn.Module):
    name = 'Luong_Encoder'
    def __init__(self, cfg):
        super(Luong_Encoder, self).__init__()
        self.cfg = cfg
        self.xh = nn.Sequential(
            nn.Linear(cfg.embedding_size, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        self.hh = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        self.tanh = nn.Tanh()

    def forward(self, seq, input_lengths):
        batch_size, seq_len, embedding_size = seq.size()
        mask = torch.arange(seq_len, device=self.cfg.device).expand(batch_size, -1) < input_lengths.unsqueeze(1)

        hidden_state = torch.zeros(batch_size, self.cfg.hidden_size, device=self.cfg.device)
        hidden_state_records = torch.zeros(batch_size, seq_len, self.cfg.hidden_size, device=self.cfg.device)

        for t in range(seq_len):
            token = seq[:,t,:]
            current_mask = mask[:, t].unsqueeze(1)
            
            temp_hidden_state = self.tanh(self.xh(token)+self.hh(hidden_state))
            hidden_state = torch.where(current_mask, temp_hidden_state, hidden_state)
            hidden_state_records[:, t, :] = hidden_state
            
        return hidden_state_records

class Luong_Decoder_Dot(nn.Module):
    name = 'Luong_Encoder'
    def __init__(self, cfg):
        super(Luong_Encoder, self).__init__()
        self.Wp = nn.Linear()
        
        # Output Matrix
        self.Wg = nn.Sequential(
            nn.Linear(cfg.embedding_size+cfg.hidden_size+cfg.hidden_size*2, cfg.embedding_size//2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size//2, cfg.embedding_size)
        )
        # Hidden Matrix
        self.Wf = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.embedding_size+cfg.hidden_size*2, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        # Alignment Matrix
        self.Wa = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)
        
        self.cfg = cfg
          
    def g(self, y_prev, s_i, c_i):
        temp_state = torch.concatenate([y_prev, s_i, c_i], dim=-1)
        predication = self.Wg(temp_state)
        return predication
        
    def f(self, s_prev, y_prev, c_i):
        temp_state = torch.concatenate([s_prev, y_prev, c_i], dim=-1)
        new_hidden_state = self.Wf(temp_state)
        return new_hidden_state

    def a(self, s_prev, annotation):
        temp_state = torch.concatenate([s_prev, annotation], dim=-1)
        energy = self.Wa(temp_state).squeeze(-1)  # consecutive dim_size = 1 could squeeze into one dim_size = 1
        return energy

    def generate_context(self, s_prev, annotations):
        # At this moment, s_prev: batch_size, hidden_size; annotations: batch_size, annotation_number, annotation_size
        batch_size, annotation_number, annotation_size = annotations.size()
        s_prev = s_prev.unsqueeze(1).expand(batch_size, annotation_number, -1)

        energys = self.a(s_prev, annotations) 
        attention_weight_expand = torch.nn.functional.softmax(energys, dim=1).unsqueeze(-1)
        context = torch.sum(annotations*attention_weight_expand, dim=1) # batch_size, annotation_size
        return context

    def forward(self, hidden_state, decode_length, annotations):
        #hidden_state = torch.zeros_like(hidden_state)
        batch_size, _ = hidden_state.size()
        outputs = torch.zeros(batch_size, decode_length, self.cfg.embedding_size, device=self.cfg.device)
        y = torch.zeros(batch_size, self.cfg.embedding_size, device=self.cfg.device)
        for t in range(decode_length):
            y_prev = y
            c_i = self.generate_context(hidden_state, annotations)
            hidden_state = self.f(hidden_state, y_prev, c_i)
            y = self.g(y_prev, hidden_state, c_i)
            outputs[:,t,:] = y
        outputs = self.fc(outputs)
        return outputs



#---------------------------------------------------------------------------------------------------------------------------------------------
class Luong_Encoder(nn.Module):
    name = 'Luong_Encoder'
    def __init__(self, cfg):
        super(Luong_Encoder, self).__init__()
        self.cfg = cfg
        self.xh = nn.Sequential(
            nn.Linear(cfg.embedding_size, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        self.hh = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        self.tanh = nn.Tanh()

    def forward(self, seq, input_lengths):
        batch_size, seq_len, embedding_size = seq.size()
        mask = torch.arange(seq_len, device=self.cfg.device).expand(batch_size, -1) < input_lengths.unsqueeze(1)

        hidden_state = torch.zeros(batch_size, self.cfg.hidden_size, device=self.cfg.device)
        hidden_state_records = torch.zeros(batch_size, seq_len, self.cfg.hidden_size, device=self.cfg.device)

        for t in range(seq_len):
            token = seq[:,t,:]
            current_mask = mask[:, t].unsqueeze(1)
            
            temp_hidden_state = self.tanh(self.xh(token)+self.hh(hidden_state))
            hidden_state = torch.where(current_mask, temp_hidden_state, hidden_state)
            hidden_state_records[:, t, :] = hidden_state
            
        return hidden_state_records

class Luong_Decoder_Dot(nn.Module):
    name = 'Luong_Decoder_Dot'
    def __init__(self, cfg):
        super(Luong_Decoder_Dot, self).__init__()
        # Anchor Compute Parameter
        self.Wp = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.Vp = nn.Parameter(torch.rand(cfg.hidden_size))
        
        # Output Matrix
        self.WPred = nn.Sequential(
            nn.Linear(cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.embedding_size)
        )

        # RNN Matrix
        self.hh = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )

        # embedding 2 vocab
        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.cfg = cfg

    def _generate_pt(self, ht, S):
        x = self.Wp(ht)
        x = self.tanh(x)
        x = torch.sum(self.Vp * x, dim=-1)
        x = self.softmax(x)
        pt = (S*x).unsqueeze(-1)
        pt = torch.round(pt).to(torch.int64)
        return pt

    def _Gauss_weights(self, pt, S):
        batch_size = pt.size(0)
        D=self.cfg.max_length//4
        s_range = torch.arange(-D, D, device=self.cfg.device).unsqueeze(0).expand(batch_size, -1) + pt
        s_range = torch.clamp(s_range, min=0, max=S-1)
        diff = s_range - pt
        gauss_weights = torch.exp(-(pow(diff,2)) / (2*pow(D/2,2))) # batch_size, scope(2D)
        return s_range, gauss_weights

    def _dot_align(self, hidden_state_records, s_range, pt, ht):
        '''
        ht: batch, embedding
        '''
        ht_expanded = ht.unsqueeze(-1)
        batch_size, _ ,last_dim = hidden_state_records.size()
        index_tensor = s_range.unsqueeze(-1).expand(-1, -1, last_dim) #batch, scope, embedding
        subset = torch.gather(hidden_state_records, 1, index_tensor) #batch, scope, embedding
        dot_score = torch.bmm(subset, ht_expanded).squeeze(-1) #batch, scope(2D)
        return subset, dot_score

    def generate_context(self, hidden_state_records, ht):
        '''
        hidden_state_records: batch, number_of_record, embedding
        context: batch, embedding
        '''
        S = hidden_state_records.size(1)
        pt = self._generate_pt(ht, S)
        s_range, gauss_weights = self._Gauss_weights(pt, S)
        subset, dot_score = self._dot_align(hidden_state_records, s_range, pt, ht)
        final_weights = (dot_score * gauss_weights).unsqueeze(-1)
        context = torch.mean(subset * final_weights, dim=1).squeeze(1)
        return context

    def forward(self, hidden_state_records, decode_length):
        '''
        hidden_state_records: batch_size, input_steps ,embedding_size
        '''
        batch_size, input_steps, hidden_size = hidden_state_records.size()
        
        hidden_state = hidden_state_records[:,-1,:]
        hidden_states = [hidden_state_records]

        outputs = torch.zeros(batch_size, decode_length, self.cfg.embedding_size, device=self.cfg.device)
        for t in range(decode_length):
            # 总的来说module里面的参数或者手动设置require_grad = True的张量都会被pytorch识别并且计算梯度
            # 而如果一个张量在本地修改，version1，version2.。。并不会被记录到计算图中，也就会产生计算图断裂
            hidden_state = self.hh(hidden_state)
            hidden_states.append(hidden_state.unsqueeze(1))
            hidden_state_records_expanded = torch.cat(hidden_states, dim=1)

            context = self.generate_context(hidden_state_records_expanded, hidden_state)
            pred_vector = torch.cat((hidden_state, context), dim=-1)
            
            output = self.WPred(pred_vector)
            outputs[:,t,:] = output
        outputs = self.fc(outputs)
        return outputs