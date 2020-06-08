import logging
import torch
from torch import nn
from torch.nn import functional as F
from modules import CrossmodalTransformer
from fairseq.modules import PositionalEmbedding

logger = logging.getLogger(__name__)

class MULTModel(nn.Module):
    def __init__(
        self,
        only_audio,
        merge_how,
        orig_d_a,
        orig_d_t,
        n_head,
        n_cmlayer,
        d_out,
        only_audio_dim = 40,
        d_model=40,
        emb_dropout=0.25,
        attn_dropout=0.1,
        attn_dropout_audio=0.0,
        attn_dropout_vision=0.0,
        relu_dropout=0.1,
        res_dropout=0.1,
        out_dropout=0.0,
        max_position=128,
        attn_mask=True,
        scale_embedding=True,
    ):
        super(MULTModel, self).__init__()


        
        self.only_audio = only_audio
        self.merge_how = merge_how
        self.d_model = d_model
        self.emb_dropout = emb_dropout
        self.out_dropout = out_dropout

        combined_dim = 2 * d_model 

        
        if self.only_audio:
            
            self.audio_layers = CrossmodalTransformer(
                only_audio_dim,
                n_head,
                emb_dropout,
                attn_dropout,
                res_dropout,
                relu_dropout,
                n_cmlayer,
                attn_mask,
            )

            self.fc_layer1 = nn.Linear(only_audio_dim, only_audio_dim)
            self.fc_layer2 = nn.Linear(only_audio_dim, only_audio_dim)
            self.out_layer = nn.Linear(only_audio_dim, d_out)  
            
        else:    
            # Input Encoder (Temporal convolution layers) -> (B, orig_d, L) => (B, d, L)
            self.audio_encoder = nn.Conv1d(orig_d_a, d_model, kernel_size=3, padding=0, bias=False) # 5
            self.text_encoder = nn.Conv1d(orig_d_t, d_model, kernel_size=3, padding=0, bias=False)


            self.audio_layers_with_text = CrossmodalTransformer(
                    d_model,
                    n_head,
                    emb_dropout,
                    attn_dropout_audio,
                    res_dropout,
                    relu_dropout,
                    n_cmlayer,
                    attn_mask,
            )

            self.text_layers_with_audio = CrossmodalTransformer(
                    d_model,
                    n_head,
                    emb_dropout,
                    attn_dropout,
                    res_dropout,
                    relu_dropout,
                    n_cmlayer,
                    attn_mask,
            )



            self.audio_layers = CrossmodalTransformer(
                d_model,
                n_head,
                emb_dropout,
                attn_dropout,
                res_dropout,
                relu_dropout,
                n_cmlayer,
                attn_mask,
            )

            self.text_layers = CrossmodalTransformer(
                d_model,
                n_head,
                emb_dropout,
                attn_dropout,
                res_dropout,
                relu_dropout,
                n_cmlayer,
                attn_mask,
            )

            # Projection layers
            self.fc_layer1 = nn.Linear(combined_dim, combined_dim)
            self.fc_layer2 = nn.Linear(combined_dim, combined_dim)
            self.out_layer = nn.Linear(combined_dim, d_out)
        


    def forward(self, x_audio, x_text =None,
                      a_mask =None,
                      t_mask =None,
                       ):
        """
            Args:
        x_vision, x_audio, x_text : input tensor -> (B, L, d)
        """
        if self.only_audio:
            #  (B, L, d) -> (L, B, d)  
            x_audio = self.audio_layers(x_audio,
                                        x_key_padding_mask=None)
            
            if self.merge_how=='average':
                # (bat, dim)
                
                features = x_audio.mean(dim=0)
            else:
                # (bat, dim)
                features = x_audio[-1]
        else:
            # for conv, (B, L, D) => (B, D, L)
            x_audio = x_audio.transpose(1, 2)
            x_text = F.dropout(x_text.transpose(1, 2), self.emb_dropout, self.training)

            # (B, D, L) => (B, L, D)
            x_audio = self.audio_encoder(x_audio).transpose(1, 2)   
            x_text = self.text_encoder(x_text).transpose(1, 2)


            # Crossmodal Attention
            # out: (seq, bat, dim) 
            # key masking was already applied to BERT model
            x_audio_with_text = self.audio_layers_with_text(x_audio,
                                                            x_text)
            # out: (seq, bat, dim)
            x_text_with_audio = self.text_layers_with_audio(x_text,
                                                            x_audio,
                                                            x_key_padding_mask=a_mask)


            # bat, seq, dim -> seq, bat, dim
            x_audio2 = x_audio_with_text.transpose(0, 1)
            x_text2 = x_text_with_audio.transpose(0, 1)


            x_audio2 = self.audio_layers(x_audio2)
            x_text2 = self.text_layers(x_text2)

            if self.merge_how=='average':

                # (bat, 2*dim)
                features = torch.cat([x_audio2.mean(dim=0), x_text2.mean(dim=0)], dim=1)
            else:
                # (bat, 2*dim)
                features = torch.cat([x_audio2[-1], x_text2[-1]], dim=1)
        

        #--------------------    
            
        out = F.relu(self.fc_layer1(features))
        out = self.fc_layer2(F.dropout(out, p=self.out_dropout, training=self.training))
        out = out + features

        out = self.out_layer(out)
        
        return out, features

    
