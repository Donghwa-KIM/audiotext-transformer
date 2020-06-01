# audiotext-transformer
cross-modal model between audio(MFCC) and text(KoBERT)

- `main.py`
  - `../../split_dataset_vat`: [data.zip](https://drive.google.com/open?id=1GmhSGaDiGVE4LuTHay-n17kIC2EF79eF)
  - `./korbert_vat/best_model`: [best_model](https://drive.google.com/open?id=1Pbyx8Lwss27HGiW3-3jfzVnqOwH-bwTo)
- `dataset.py`
  - `korbert_vat.pretrained_model`: [pretrained_model](https://drive.google.com/open?id=1rPtnqyKkME_6ZZUrvgkalfzIqedMQYf4)

```
usage: main.py [-h] [--do_audio] [--merge_how MERGE_HOW] [--model MODEL]
               [--data_name DATA_NAME] [--data_path DATA_PATH]
               [--bert_config_path BERT_CONFIG_PATH]
               [--bert_args_path BERT_ARGS_PATH]
               [--bert_model_path BERT_MODEL_PATH]
               [--attn_dropout ATTN_DROPOUT] [--attn_dropout_a ATTN_DROPOUT_A]
               [--attn_dropout_v ATTN_DROPOUT_V] [--relu_dropout RELU_DROPOUT]
               [--embed_dropout EMBED_DROPOUT] [--res_dropout RES_DROPOUT]
               [--out_dropout OUT_DROPOUT] [--layers LAYERS]
               [--d_model D_MODEL] [--d_out D_OUT] [--num_heads NUM_HEADS]
               [--attn_mask] [--batch_size N] [--clip CLIP] [--lr LR]
               [--optim OPTIM] [--num_epochs NUM_EPOCHS] [--when WHEN]
               [--batch_chunk BATCH_CHUNK] [--local_rank LOCAL_RANK]
               [--warmup_percent WARMUP_PERCENT]
               [--max_len_for_text MAX_LEN_FOR_TEXT]
               [--hidden_size_for_bert HIDDEN_SIZE_FOR_BERT]
               [--max_len_for_audio MAX_LEN_FOR_AUDIO]
               [--sample_rate SAMPLE_RATE] [--resample_rate RESAMPLE_RATE]
               [--n_fft_size N_FFT_SIZE] [--n_mfcc N_MFCC]
               [--logging_steps LOGGING_STEPS] [--seed SEED] [--no_cuda]
               [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --do_audio            whether to use of only audio (Transformer, etc.)
  --merge_how MERGE_HOW
                        how to merge the sequence for a modal
  --model MODEL         name of the model to use (Transformer, etc.)
  --data_name DATA_NAME
                        dataset to use (default: vat)
  --data_path DATA_PATH
                        path for storing the dataset
  --bert_config_path BERT_CONFIG_PATH
                        bert_config_path
  --bert_args_path BERT_ARGS_PATH
                        bert_args_path
  --bert_model_path BERT_MODEL_PATH
                        bert_model_path (pretrained & finetuned by a sentiment
                        task)
  --attn_dropout ATTN_DROPOUT
                        attention dropout
  --attn_dropout_a ATTN_DROPOUT_A
                        attention dropout (for audio)
  --attn_dropout_v ATTN_DROPOUT_V
                        attention dropout (for visual)
  --relu_dropout RELU_DROPOUT
                        relu dropout
  --embed_dropout EMBED_DROPOUT
                        embedding dropout
  --res_dropout RES_DROPOUT
                        residual block dropout
  --out_dropout OUT_DROPOUT
                        output layer dropout
  --layers LAYERS       number of layers in the network (default: 5)
  --d_model D_MODEL     dimension of layers in the network (default: 30)
  --d_out D_OUT         dimension of target dimension in the network (default:
                        7 for multi)
  --num_heads NUM_HEADS
                        number of heads for the transformer network (default:
                        5)
  --attn_mask           use attention mask for Transformer (default: true)
  --batch_size N        batch size (default: 24)
  --clip CLIP           gradient clip value (default: 0.8)
  --lr LR               initial learning rate (default: 1e-3)
  --optim OPTIM         optimizer to use (default: Adam)
  --num_epochs NUM_EPOCHS
                        number of epochs (default: 40)
  --when WHEN           when to decay learning rate (default: 20)
  --batch_chunk BATCH_CHUNK
                        number of chunks per batch (default: 1)
  --local_rank LOCAL_RANK
                        For distributed training
  --warmup_percent WARMUP_PERCENT
                        Linear warmup over warmup_percent.
  --max_len_for_text MAX_LEN_FOR_TEXT
                        Maximum sequence length for text
  --hidden_size_for_bert HIDDEN_SIZE_FOR_BERT
                        hidden size used to a fine-tuned BERT
  --max_len_for_audio MAX_LEN_FOR_AUDIO
                        Maximum sequence length for audio
  --sample_rate SAMPLE_RATE
                        sampling rate for audio
  --resample_rate RESAMPLE_RATE
                        resampling rate to reduce audio sequence
  --n_fft_size N_FFT_SIZE
                        time widnow for fourier transform
  --n_mfcc N_MFCC       low frequency range (from 0 to n_mfcc)
  --logging_steps LOGGING_STEPS
                        frequency of result logging (default: 30)
  --seed SEED           random seed
```
