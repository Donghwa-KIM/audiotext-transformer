from pytorch_transformers import optimization


def layerwise_decay_optimizer(model, lr, layerwise_decay=None):

    # optimizer and lr scheduler
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    if layerwise_decay is True:
        optimizer_grouped_parameters = []
        for i in range(12):
            tmp = [{'params': [p for n, p in model.named_parameters()
                               if 'bert.encoder.layer.'+str(i)+'.' in n
                               and not any(nd in n for nd in no_decay)],
                    'lr': lr * (layerwise_decay ** (11-i)),
                    'weight_decay': 0.01},

                   {'params': [p for n, p in model.named_parameters()
                               if 'bert.encoder.layer.' + str(i) + '.' in n
                               and any(nd in n for nd in no_decay)],
                    'lr': lr * (layerwise_decay ** (11-i)),
                    'weight_decay': 0}
                   ]
            optimizer_grouped_parameters += tmp

        tmp = [{'params': [p for n, p in model.named_parameters()
                           if 'bert.encoder.layer.' not in n
                           and not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01},

               {'params': [p for n, p in model.named_parameters()
                           if 'bert.encoder.layer.' not in n
                           and any(nd in n for nd in no_decay)],
                'weight_decay': 0}
               ]
        optimizer_grouped_parameters += tmp

    optimizer = optimization.AdamW(
        optimizer_grouped_parameters, lr=lr, correct_bias=False)

    return optimizer
